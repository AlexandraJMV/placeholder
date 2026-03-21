import sys
import os
import pickle
import torch
import torch.nn as nn
import json
import math

sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

from blocklize import MODEL_BLOCKS
from simlarity.feature_extraction import create_sub_network
from simlarity.utils import Block, Block_Assign

def get_block_type(model_name):
    transformers = ['vit', 'swin', 'deit', 'tnt']
    for t in transformers:
        if t in model_name.lower():
            return 'TRANS'
    return 'CNN'

class OutputUnwrapper(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module
    
    def forward(self, x):
        out = self.module(x)
        if isinstance(out, tuple): out = out[0]
        if isinstance(out, dict): out = list(out.values())[0]
        if isinstance(out, tuple): out = out[0]
        return out

class StitchLayer(nn.Module):
    def __init__(self, src_cfg, dst_cfg, init_mode='ls', weight_matrix=None):
        super().__init__()
        self.mode = f"{src_cfg['type']}-to-{dst_cfg['type']}"
        self.in_ch = src_cfg['ch']
        self.out_ch = dst_cfg['ch']
        in_res = src_cfg['res']
        out_res = dst_cfg['res']
        
        layers = []

        if in_res > out_res:
            stride = in_res // out_res
            if stride > 1:
                layers.append(nn.AvgPool2d(kernel_size=stride, stride=stride))
        elif in_res < out_res:
            scale = out_res // in_res
            layers.append(nn.Upsample(scale_factor=scale, mode='nearest'))

        if self.mode in ['CNN-to-CNN', 'CNN-to-TRANS']:
            layers.append(nn.BatchNorm2d(self.in_ch))
            self.conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, bias=False)
            layers.append(self.conv)
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            if self.mode == 'CNN-to-TRANS': layers.append(nn.Flatten(2))
            
        elif self.mode == 'TRANS-to-CNN':
            class Unflatten(nn.Module):
                def __init__(self, h, w):
                    super().__init__()
                    self.h, self.w = h, w
                def forward(self, x):
                    return x.view(x.shape[0], x.shape[1], self.h, self.w)
            layers.append(Unflatten(out_res, out_res))
            layers.append(nn.BatchNorm2d(self.in_ch))
            self.conv = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1, bias=False)
            layers.append(self.conv)
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            
        elif self.mode == 'TRANS-to-TRANS':
            layers.append(nn.LayerNorm(self.in_ch))
            self.linear = nn.Linear(self.in_ch, self.out_ch)
            layers.append(self.linear)
            layers.append(nn.LeakyReLU(negative_slope=0.1, inplace=True))
            self.conv = None 
        else:
            raise NotImplementedError(f"Modo desconocido: {self.mode}")

        self.op = nn.Sequential(*layers)
        
        self.apply_initialization(init_mode, weight_matrix)

    def apply_initialization(self, mode, weight_matrix):
        target_layer = self.conv if hasattr(self, 'conv') and self.conv else getattr(self, 'linear', None)
        if target_layer is None: return

        if mode == 'random':
            nn.init.kaiming_normal_(target_layer.weight, mode='fan_out', nonlinearity='leaky_relu')
            
        elif mode == 'ls':
            if weight_matrix is not None:
                with torch.no_grad():
                    if target_layer.weight.dim() == 4:
                        if weight_matrix.dim() == 2:
                            weight_matrix = weight_matrix.view(self.out_ch, self.in_ch, 1, 1)
                    target_layer.weight.copy_(weight_matrix)
            else:
                nn.init.orthogonal_(target_layer.weight)
                with torch.no_grad():
                    target_layer.weight.mul_(math.sqrt(2)) 

    def forward(self, x):
        return self.op(x)

class SuperNetwork(nn.Module):
    def __init__(self, plan_path, num_classes=10, input_size=224, stitch_init_mode='ls', matrices_path=None):
        """
        Args:
            stitch_init_mode (str): 'ls' (Least Squares / Default) o 'random'.
            matrices_path (str): Ruta a un archivo .pkl con diccionario de matrices { (src, dst): matrix }.
        """
        super().__init__()
        print(f">>> Ensamblando SuperNetwork (Init: {stitch_init_mode})...")
        
        self.stitch_init_mode = stitch_init_mode
        self.matrices = {}
        
        if stitch_init_mode == 'ls' and matrices_path and os.path.exists(matrices_path):
            print(f"   Cargando matrices de transformación desde {matrices_path}...")
            with open(matrices_path, 'rb') as f:
                self.matrices = pickle.load(f)
        
        with open(plan_path, 'rb') as f:
            self.plan = pickle.load(f)
        
        self.num_stages = len(self.plan.center2block)
        self.choices_per_stage = [len(opts) for opts in self.plan.center2block]
        
        self.stages = nn.ModuleList()
        self.stitches = nn.ModuleList()
        
        with open("tools/MODEL_INOUT_SHAPE.json") as f:
            shapes_db = json.load(f)

        stage_infos = [] 
        dummy_input = torch.randn(1, 3, input_size, input_size)

        for i, stage_blocks in enumerate(self.plan.center2block):
            print(f"  -> Construyendo Stage {i} ({len(stage_blocks)} opciones)...")
            current_stage_modules = nn.ModuleList()
            current_stage_infos = []
            
            for block_obj in stage_blocks:
                try:
                    raw_sub_model = self._extract_submodule(block_obj)
                    sub_model = OutputUnwrapper(raw_sub_model)
                    
                    start_node = MODEL_BLOCKS[block_obj.model_name][block_obj.node_list[0]]
                    raw_shape = shapes_db[block_obj.model_name]['in_size'][start_node]
                    
                    if len(raw_shape) == 4: real_in_ch, real_in_res = raw_shape[1], raw_shape[2]
                    elif len(raw_shape) == 3: real_in_ch, real_in_res = raw_shape[0], raw_shape[1]
                    else: real_in_ch, real_in_res = raw_shape
                    
                    if block_obj.node_list[0] == 0:
                        real_in_ch = 3
                        real_in_res = input_size

                    b_type = get_block_type(block_obj.model_name)

                    if i == 0:
                        if real_in_ch != 3 or real_in_res != input_size:
                            src_cfg = {'type': 'CNN', 'ch': 3, 'res': input_size}
                            dst_cfg = {'type': b_type, 'ch': real_in_ch, 'res': real_in_res}
                            
                            matrix = self._get_matrix('input', block_obj.model_name)
                            
                            stem = StitchLayer(src_cfg, dst_cfg, 
                                             init_mode=self.stitch_init_mode, 
                                             weight_matrix=matrix)
                            sub_model = nn.Sequential(stem, sub_model)
                            real_in_ch = 3
                            real_in_res = input_size

                    current_stage_modules.append(sub_model)
                    
                    # D. Dummy Pass
                    if i == 0: test_in = dummy_input
                    else:
                        prev_res = input_size // (2**i) 
                        test_in = torch.randn(1, real_in_ch, real_in_res, real_in_res)

                    with torch.no_grad():
                        out_t = sub_model(test_in)
                    
                    info = {
                        'type': b_type,
                        'in_ch': real_in_ch,
                        'in_res': real_in_res,
                        'out_ch': out_t.shape[1],
                        'out_res': out_t.shape[2],
                        'model_name': block_obj.model_name # Guardar ID para buscar matriz luego
                    }
                    current_stage_infos.append(info)
                    
                except Exception as e:
                    print(f"Error analizando bloque {block_obj.model_name}: {e}")
                    raise e

            self.stages.append(current_stage_modules)
            stage_infos.append(current_stage_infos)

        for i in range(self.num_stages - 1):
            print(f"  -> Tejiendo costuras entre Stage {i} y {i+1}...")
            stage_stitches = nn.ModuleList() 
            src_infos = stage_infos[i]
            dst_infos = stage_infos[i+1]
            
            for src_info in src_infos:
                row_stitches = nn.ModuleList()
                for dst_info in dst_infos:
                    s_cfg = {'type': src_info['type'], 'ch': src_info['out_ch'], 'res': src_info['out_res']}
                    d_cfg = {'type': dst_info['type'], 'ch': dst_info['in_ch'], 'res': dst_info['in_res']}
                    
                    matrix = self._get_matrix(src_info['model_name'], dst_info['model_name'])
                    
                    stitch = StitchLayer(s_cfg, d_cfg, 
                                       init_mode=self.stitch_init_mode, 
                                       weight_matrix=matrix)
                    row_stitches.append(stitch)
                stage_stitches.append(row_stitches)
            self.stitches.append(stage_stitches)

        self.global_pool = nn.AdaptiveAvgPool2d(1)
        self.heads = nn.ModuleList()
        for info in stage_infos[-1]:
            self.heads.append(nn.Linear(info['out_ch'], num_classes))

    def _get_matrix(self, src_name, dst_name):
        """Busca la matriz de transformación en el diccionario cargado."""
        if not self.matrices: return None
        key = f"{src_name}->{dst_name}"
        return self.matrices.get(key, None)

    def _extract_submodule(self, block_obj):
        model_name = block_obj.model_name
        if any(x in model_name for x in ['resnet', 'squeezenet', 'densenet', 'shufflenet']):
            import torchvision.models as models
            try:
                base_model = getattr(models, model_name)(weights='DEFAULT')
            except:
                base_model = getattr(models, model_name)(pretrained=True)
        else:
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
    print(">>> Test: Stitching con Init LS vs Random <<<")
    pkl_path = "network_plan.pkl"
    if os.path.exists(pkl_path):
        try:
            supernet = SuperNetwork(pkl_path, num_classes=10, stitch_init_mode='ls')
            print("\nSuperNetwork (LS) construida exitosamente.")
            
            x = torch.randn(2, 3, 224, 224)
            y = supernet(x)
            print(f"Output shape: {y.shape}")
            
        except Exception as e:
            print(e)
            import traceback
            traceback.print_exc()