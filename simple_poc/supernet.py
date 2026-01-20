import sys
import os
import pickle
import torch
import torch.nn as nn
import json

# --- SETUP DE RUTAS (CRÍTICO) ---
# Forzamos a Python a ver el paquete local 'third_package' y la raíz
sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

# Importamos utilidades del repo
from blocklize import MODEL_BLOCKS, MODEL_ZOO
# --- CORRECCIÓN: Quitamos 'network_to_module' que no existe aquí ---
from simlarity.feature_extraction import create_sub_network 
from simlarity.utils import Block, Block_Assign

# --- CLASE: STITCH LAYER (LA COSTURA) ---
class StitchLayer(nn.Module):
    """
    Adapta dimensiones entre dos bloques heterogéneos.
    Maneja cambio de canales (Conv1x1) y cambio de resolución (Pooling/Stride).
    """
    def __init__(self, in_channels, out_channels, in_res, out_res):
        super().__init__()
        
        ops = []
        
        # 1. Adaptación de Resolución (Downsampling)
        if in_res > out_res:
            stride = in_res // out_res
            if stride > 1:
                # Usamos AvgPool para bajar resolución suavemente sin añadir parámetros
                ops.append(nn.AvgPool2d(kernel_size=stride, stride=stride))
        elif in_res < out_res:
            scale = out_res // in_res
            ops.append(nn.Upsample(scale_factor=scale, mode='nearest'))

        # 2. Adaptación de Canales
        if in_channels != out_channels:
            ops.append(nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False))
            ops.append(nn.BatchNorm2d(out_channels))
            ops.append(nn.ReLU(inplace=True))
            
        self.adapter = nn.Sequential(*ops)

    def forward(self, x):
        return self.adapter(x)

# --- CLASE: SUPER NETWORK (EL ENSAMBLAJE) ---
class SuperNetwork(nn.Module):
    def __init__(self, plan_path, num_classes=10, input_size=224):
        super().__init__()
        print(f">>> Ensamblando SuperNetwork desde {plan_path}...")
        
        # 1. Cargar el Plan
        with open(plan_path, 'rb') as f:
            self.plan = pickle.load(f)
        
        # Estructura del plan: self.plan.center2block es una lista de listas
        # Stage 0: [BlockA, BlockB, BlockC...]
        self.num_stages = len(self.plan.center2block)
        self.choices_per_stage = [len(opts) for opts in self.plan.center2block]
        
        # Contenedores de Módulos (ModuleList de ModuleLists)
        self.stages = nn.ModuleList()
        self.stitches = nn.ModuleList() 
        
        # Metadatos para calcular stitches
        stage_infos = [] 

        # --- 2. CONSTRUIR BLOQUES (STAGES) ---
        dummy_input = torch.randn(1, 3, input_size, input_size)
        
        for i, stage_blocks in enumerate(self.plan.center2block):
            print(f"  -> Construyendo Stage {i} ({len(stage_blocks)} opciones)...")
            current_stage_modules = nn.ModuleList()
            current_stage_infos = []
            
            for block_obj in stage_blocks:
                # Extraer el sub-modelo real
                sub_model = self._extract_submodule(block_obj)
                current_stage_modules.append(sub_model)
                
                try:
                    # Recuperemos input shape del JSON de tools
                    with open("tools/MODEL_INOUT_SHAPE.json") as f:
                        shapes = json.load(f)
                    
                    # Nombre del nodo de entrada
                    start_node_name = MODEL_BLOCKS[block_obj.model_name][block_obj.node_list[0]]
                    in_ch, in_res = shapes[block_obj.model_name]['in_size'][start_node_name]
                    
                    # Test pass para confirmar salida
                    # Nota: si in_res es muy grande (ej 224) y el modelo es pequeño, esto confirma la logica
                    test_t = torch.randn(1, in_ch, in_res, in_res)
                    with torch.no_grad():
                        out_t = sub_model(test_t)
                    
                    out_ch = out_t.shape[1]
                    out_res = out_t.shape[2]
                    
                    current_stage_infos.append({
                        'in': (in_ch, in_res),
                        'out': (out_ch, out_res)
                    })
                    
                except Exception as e:
                    print(f"Error analizando bloque {block_obj}: {e}")
                    raise e

            self.stages.append(current_stage_modules)
            stage_infos.append(current_stage_infos)

        # --- 3. CONSTRUIR STITCHES (COSTURAS) ---
        for i in range(self.num_stages - 1):
            print(f"  -> Tejiendo costuras entre Stage {i} y {i+1}...")
            stage_stitches = nn.ModuleList() 
            
            src_infos = stage_infos[i]
            dst_infos = stage_infos[i+1]
            
            for src_idx, src_info in enumerate(src_infos):
                row_stitches = nn.ModuleList()
                for dst_idx, dst_info in enumerate(dst_infos):
                    stitch = StitchLayer(
                        in_channels=src_info['out'][0],
                        out_channels=dst_info['in'][0],
                        in_res=src_info['out'][1],
                        out_res=dst_info['in'][1]
                    )
                    row_stitches.append(stitch)
                stage_stitches.append(row_stitches)
            self.stitches.append(stage_stitches)

        # --- 4. CLASIFICADOR FINAL (HEAD) ---
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.heads = nn.ModuleList()
        last_stage_infos = stage_infos[-1]
        for info in last_stage_infos:
            out_ch = info['out'][0]
            self.heads.append(nn.Linear(out_ch, num_classes))

    def _extract_submodule(self, block_obj):
        """Recrea el modelo y extrae las capas específicas"""
        model_name = block_obj.model_name
        
        # Instanciar modelo base
        if 'resnet' in model_name or 'squeezenet' in model_name or 'densenet' in model_name:
            import torchvision.models as models
            # Fix para versiones nuevas de torchvision que piden 'weights' en vez de 'pretrained'
            try:
                base_model = getattr(models, model_name)(weights='DEFAULT')
            except:
                base_model = getattr(models, model_name)(pretrained=True)
        else:
            import timm
            base_model = timm.create_model(model_name, pretrained=True)
            
        # Obtener nombres de nodos
        node_indices = block_obj.node_list
        start_idx = node_indices[0]
        end_idx = node_indices[-1]
        
        all_nodes = MODEL_BLOCKS[model_name]
        input_node_name = all_nodes[start_idx]
        output_node_name = all_nodes[end_idx]
        
        # Si es el nodo 0, el input es la imagen, no necesitamos cortar el inicio.
        input_args = [input_node_name] if start_idx > 0 else []
        
        # Usamos la función nativa de PyTorch (que parcheamos en feature_extraction.py)
        sub_model = create_sub_network(base_model, input_args, [output_node_name])
        
        return sub_model

    def forward(self, x, path=None):
        if path is None:
            # Random sampling
            path = [torch.randint(0, c, (1,)).item() for c in self.choices_per_stage]
        
        # --- Stage 0 ---
        choice_0 = path[0]
        out = self.stages[0][choice_0](x)
        
        # --- Middle Stages ---
        for i in range(1, self.num_stages):
            choice_prev = path[i-1]
            choice_curr = path[i]
            
            # Aplicar Stitch: src -> dst
            out = self.stitches[i-1][choice_prev][choice_curr](out)
            
            # Aplicar Bloque
            out = self.stages[i][choice_curr](out)
            
        # --- Head ---
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        choice_last = path[-1]
        out = self.heads[choice_last](out)
        
        return out

# --- MAIN DE PRUEBA ---
if __name__ == '__main__':
    print(">>> Test de Ensamblaje de SuperNetwork <<<")
    
    pkl_path = "network_plan.pkl"
    
    if not os.path.exists(pkl_path):
        print(f"Error: No existe {pkl_path}. Corre partition.py primero.")
        sys.exit(1)
        
    try:
        # Instanciar para CIFAR-10
        supernet = SuperNetwork(pkl_path, num_classes=10)
        
        device = 'cuda' if torch.cuda.is_available() else 'cpu'
        supernet.to(device)
        print(f"\nSuperNetwork cargada en {device}.")
        
        print("\nEjecutando Dummy Pass (Random Path)...")
        x = torch.randn(2, 3, 224, 224).to(device)
        
        for i in range(3):
            y = supernet(x)
            print(f"  Intento {i+1}: Output Shape {y.shape} (Esperado: [2, 10]) -> OK")
            
        print("\n¡ÉXITO! La Super-Red está viva y conectada.")
        
    except Exception as e:
        print(f"\nFALLO FATAL:\n{e}")
        import traceback
        traceback.print_exc()