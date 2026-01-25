import sys
import os
import pickle
import torch
import torch.nn as nn
import json

# --- SETUP DE RUTAS ---
sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

from blocklize import MODEL_BLOCKS
from simlarity.feature_extraction import create_sub_network
from simlarity.utils import Block, Block_Assign

# --- HELPER: DETECTOR DE TIPO ---
def get_block_type(model_name):
    transformers = ['vit', 'swin', 'deit', 'tnt']
    for t in transformers:
        if t in model_name.lower():
            return 'TRANS'
    return 'CNN'

# --- CLASE: STITCH LAYER ---
class StitchLayer(nn.Module):
    def __init__(self, src_cfg, dst_cfg):
        super().__init__()
        mode = f"{src_cfg['type']}-to-{dst_cfg['type']}"
        in_ch = src_cfg['ch']
        out_ch = dst_cfg['ch']
        in_res = src_cfg['res']
        out_res = dst_cfg['res']
        layers = []

        # 1. Alineación Espacial
        if in_res > out_res:
            stride = in_res // out_res
            if stride > 1:
                layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride))
        elif in_res < out_res:
            scale = out_res // in_res
            layers.append(nn.Upsample(scale_factor=scale, mode='nearest'))

        # 2. Adaptación de Canales (Tabla 5)
        if mode == 'CNN-to-CNN':
            layers.append(nn.BatchNorm2d(in_ch))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        elif mode == 'CNN-to-TRANS':
            layers.append(nn.BatchNorm2d(in_ch))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            layers.append(nn.Flatten(2))
        elif mode == 'TRANS-to-CNN':
            class Unflatten(nn.Module):
                def __init__(self, h, w):
                    super().__init__()
                    self.h, self.w = h, w
                def forward(self, x):
                    return x.view(x.shape[0], x.shape[1], self.h, self.w)
            layers.append(Unflatten(out_res, out_res))
            layers.append(nn.BatchNorm2d(in_ch))
            layers.append(nn.Conv2d(in_ch, out_ch, kernel_size=1, bias=False))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        elif mode == 'TRANS-to-TRANS':
            layers.append(nn.LayerNorm(in_ch))
            layers.append(nn.Linear(in_ch, out_ch))
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
        else:
            raise NotImplementedError(f"Modo desconocido: {mode}")

        self.op = nn.Sequential(*layers)

    def forward(self, x):
        return self.op(x)

# --- CLASE: SUPER NETWORK ---
class SuperNetwork(nn.Module):
    def __init__(self, plan_path, num_classes=10, input_size=224):
        super().__init__()
        print(f">>> Ensamblando SuperNetwork desde {plan_path}...")
        
        with open(plan_path, 'rb') as f:
            self.plan = pickle.load(f)
        
        self.num_stages = len(self.plan.center2block)
        self.choices_per_stage = [len(opts) for opts in self.plan.center2block]
        
        self.stages = nn.ModuleList()
        self.stitches = nn.ModuleList()
        
        # Cargar metadatos globales
        with open("tools/MODEL_INOUT_SHAPE.json") as f:
            shapes_db = json.load(f)

        stage_infos = [] 
        dummy_input = torch.randn(1, 3, input_size, input_size)

        # 1. CONSTRUIR BLOQUES
        for i, stage_blocks in enumerate(self.plan.center2block):
            print(f"  -> Construyendo Stage {i} ({len(stage_blocks)} opciones)...")
            current_stage_modules = nn.ModuleList()
            current_stage_infos = []
            
            for block_obj in stage_blocks:
                try:
                    # A. Extraer sub-modelo
                    sub_model = self._extract_submodule(block_obj)
                    
                    # B. Obtener input real esperado (CON UNPACKING SEGURO)
                    start_node = MODEL_BLOCKS[block_obj.model_name][block_obj.node_list[0]]
                    raw_shape = shapes_db[block_obj.model_name]['in_size'][start_node]
                    
                    # --- FIX: Lógica para extraer Canales y Resolución de una lista de 3 o 4 elementos ---
                    if len(raw_shape) == 4: # Formato [Batch, Channels, H, W] -> ej: [1, 16, 112, 112]
                        real_in_ch = raw_shape[1]
                        real_in_res = raw_shape[2]
                    elif len(raw_shape) == 3: # Formato [Channels, H, W] -> ej: [16, 112, 112]
                        real_in_ch = raw_shape[0]
                        real_in_res = raw_shape[1]
                    else:
                        # Fallback raro: asumimos [Ch, Res]
                        real_in_ch, real_in_res = raw_shape
                    
                    b_type = get_block_type(block_obj.model_name)

                    # C. AUTO-STEM (Parche de Entrada)
                    if i == 0:
                        if real_in_ch != 3 or real_in_res != input_size:
                            # print(f"     [Auto-Stem] Adaptando input: Imagen(3, {input_size}) -> Bloque({real_in_ch}, {real_in_res})")
                            
                            src_cfg = {'type': 'CNN', 'ch': 3, 'res': input_size}
                            dst_cfg = {'type': b_type, 'ch': real_in_ch, 'res': real_in_res}
                            
                            stem = StitchLayer(src_cfg, dst_cfg)
                            sub_model = nn.Sequential(stem, sub_model)
                            
                            real_in_ch = 3
                            real_in_res = input_size

                    current_stage_modules.append(sub_model)
                    
                    # D. Calcular Output Info (Dummy Pass)
                    if i == 0:
                        test_in = dummy_input
                    else:
                        prev_res = input_size // (2**i) 
                        test_in = torch.randn(1, real_in_ch, real_in_res, real_in_res)

                    with torch.no_grad():
                        out_t = sub_model(test_in)
                    
                    info = {
                        'type': b_type,
                        'ch': out_t.shape[1],
                        'res': out_t.shape[2]
                    }
                    current_stage_infos.append(info)
                    
                except Exception as e:
                    print(f"Error analizando bloque {block_obj.model_name}: {e}")
                    raise e

            self.stages.append(current_stage_modules)
            stage_infos.append(current_stage_infos)

        # 2. CONSTRUIR STITCHES INTER-ETAPAS
        for i in range(self.num_stages - 1):
            print(f"  -> Tejiendo costuras entre Stage {i} y {i+1}...")
            stage_stitches = nn.ModuleList() 
            src_infos = stage_infos[i]
            dst_infos = stage_infos[i+1]
            
            for src_cfg in src_infos:
                row_stitches = nn.ModuleList()
                for dst_cfg in dst_infos:
                    stitch = StitchLayer(src_cfg, dst_cfg)
                    row_stitches.append(stitch)
                stage_stitches.append(row_stitches)
            self.stitches.append(stage_stitches)

        # 3. HEAD
        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.heads = nn.ModuleList()
        for info in stage_infos[-1]:
            self.heads.append(nn.Linear(info['ch'], num_classes))

    def _extract_submodule(self, block_obj):
        model_name = block_obj.model_name
        
        # ResNet, SqueezeNet, DenseNet -> Usar Torchvision
        if any(x in model_name for x in ['resnet', 'squeezenet', 'densenet']):
            import torchvision.models as models
            try:
                base_model = getattr(models, model_name)(weights='DEFAULT')
            except:
                base_model = getattr(models, model_name)(pretrained=True)
        else:
            # MobileNet, ShuffleNet, EfficientNet -> Usar TIMM
            import timm
            base_model = timm.create_model(model_name, pretrained=True)
            
        node_indices = block_obj.node_list
        start_idx = node_indices[0]
        end_idx = node_indices[-1]
        all_nodes = MODEL_BLOCKS[model_name]
        input_node_name = all_nodes[start_idx]
        output_node_name = all_nodes[end_idx]
        
        input_args = [input_node_name] if start_idx > 0 else []
        return create_sub_network(base_model, input_args, [output_node_name])

    def forward(self, x, path=None):
        if path is None:
            path = [torch.randint(0, c, (1,)).item() for c in self.choices_per_stage]
        
        out = self.stages[0][path[0]](x)
        
        for i in range(1, self.num_stages):
            prev_idx = path[i-1]
            curr_idx = path[i]
            out = self.stitches[i-1][prev_idx][curr_idx](out)
            out = self.stages[i][curr_idx](out)
            
        out = self.global_pool(out)
        out = torch.flatten(out, 1)
        out = self.heads[path[-1]](out)
        return out

if __name__ == '__main__':
    print(">>> Test de Stitching Riguroso + Auto-Stem <<<")
    pkl_path = "network_plan.pkl"
    if os.path.exists(pkl_path):
        try:
            supernet = SuperNetwork(pkl_path, num_classes=10)
            print("\nSuperNetwork construida exitosamente.")
            
            x = torch.randn(2, 3, 224, 224)
            y = supernet(x)
            print(f"Output shape del dummy pass: {y.shape}")
            
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()