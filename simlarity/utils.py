import json
# import mmcv  <-- REMOVED
import numpy as np
import os
import pickle
import timm
import torch
from .compare_functions import (cca_torch, cka_linear_torch, cka_rbf_torch, lr_torch)
from .feature_extraction import (create_sub_network,
                                create_sub_network_transformer,
                                get_graph_node_names, graph_to_table)
from torchvision import models

import third_package.timm as mytimm
from blocklize import MODEL_BLOCKS, MODEL_STATS, MODEL_ZOO, MODEL_PRINT
from blocklize.block_meta import MODEL_INOUT_SHAPE
# import mmcls <-- REMOVED
from blocklize.block_meta import MODEL_INOUT_SHAPE as _DEFAULT_SHAPES

# Initialize with default, but allow update
MODEL_INOUT_SHAPE = _DEFAULT_SHAPES

def update_global_shapes(json_path):
    """Updates the global MODEL_INOUT_SHAPE with data from a JSON file.
    
    Updates BOTH the local utils module reference AND blocklize.block_meta
    to ensure all import paths see the same data.
    """
    global MODEL_INOUT_SHAPE
    import blocklize.block_meta as bm
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Rebind both module-level references so Block.get_inout_size() and
    # any direct importer of blocklize.block_meta see the same shape dict.
    MODEL_INOUT_SHAPE = data
    bm.MODEL_INOUT_SHAPE = data
    print(f"Global MODEL_INOUT_SHAPE updated from {json_path}")
    
    
def create_feature_dict(path):
    result_dict = dict()
    for name in os.listdir(path):
        # tmp_dict = mmcv.load(os.path.join(path, name)) <-- REPLACED
        with open(os.path.join(path, name), 'rb') as f:
            tmp_dict = pickle.load(f)
            
        for ink in tmp_dict.keys():
            if ink in result_dict.keys():
                for outk in tmp_dict[ink].keys():
                    result_dict[ink][outk] = torch.cat(
                        [tmp_dict[ink][outk], result_dict[ink][outk]], dim=0)
            else:
                result_dict[ink] = tmp_dict[ink]
    return result_dict


def similarity_pair(pickle1, pickle2):
    # feat1 = mmcv.load(pickle1) <-- REPLACED
    with open(pickle1, 'rb') as f: feat1 = pickle.load(f)
    
    # feat2 = mmcv.load(pickle2) <-- REPLACED
    with open(pickle2, 'rb') as f: feat2 = pickle.load(f)
    
    num_layer1 = len(feat1.keys())
    num_layer2 = len(feat2.keys())
    print(f'number of layers in {pickle1} is {num_layer1}')
    print(f'number of layers in {pickle2} is {num_layer2}')

    cka_map = torch.zeros((num_layer1, num_layer2))
    # prog_bar = mmcv.ProgressBar(num_layer1) <-- REMOVED (Use simple loop or tqdm if needed)
    
    for i, (k1, v1) in enumerate(feat1.items()):
        for j, (k2, v2) in enumerate(feat2.items()):
            cka_from_examples = cka_linear_torch(v1, v2)
            cka_map[i, j] = cka_from_examples
            print(f'layers {i} in {pickle1} and layers {j} in {pickle2}')
        # prog_bar.update() <-- REMOVED
    print(cka_map)


def network_to_module(model_name, block_name, backend):
    if backend == 'timm':
        backbone = timm.create_model(
            model_name, pretrained=True, scriptable=True)
    elif backend == 'pytorch':
        backbone = getattr(models, model_name)(pretrained=True)

    for name, module in backbone.named_modules():
        if name == block_name:
            return module


class NpEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        # if isinstance(obj, int64):
        return super(NpEncoder, self).default(obj)


def network_to_module_subnet(model_name, block_input, block_output, backend):
    if backend == 'timm':
        backbone = timm.create_model(
            model_name, pretrained=True)
    elif backend == 'mytimm':
        backbone = mytimm.create_model(
            model_name, pretrained=True)
    elif backend == 'mmcv':
        # This block was for loading configs using mmcv. Since we are avoiding mmcv,
        # we will raise an error if 'mmcv' backend is requested, or fallback to pytorch/timm.
        # Given your setup, you likely won't hit this if MODEL_STATS are configured correctly.
        raise NotImplementedError("MMCV backend is not supported in this lightweight version.")
        # config = MODEL_STATS[model_name]['cfg']
        # cfg = mmcv.Config.fromfile(config)
        # backbone = mmcls.build_backbone(cfg.model.backbone)
    elif backend == 'pytorch':
        backbone = getattr(models, model_name)(pretrained=True)

    if isinstance(block_input, str):
        block_input = [block_input]
    elif isinstance(block_input, tuple):
        block_input = list(block_input)
    elif isinstance(block_input, list):
        block_input = block_input
    else:
        TypeError('Block input should be a string or tuple or list')

    if isinstance(block_output, str):
        block_output = [block_output]
    elif isinstance(block_output, tuple):
        block_output = list(block_output)
    elif isinstance(block_output, list):
        block_output = block_output
    else:
        TypeError('Block output should be a string or tuple or list')

    if model_name.startswith('swin_') or model_name.startswith('vit'):
        subnet = create_sub_network_transformer(backbone, model_name, block_input, block_output)
    else:
        subnet = create_sub_network(backbone, block_input, block_output)
    return subnet


class Block:
    def __init__(self, model_name, block_index, node_list):
        assert isinstance(model_name, str)
        assert isinstance(node_list, list)
        for i in range(len(node_list)-1):
            assert node_list[i+1] - node_list[i] == 1, node_list
        self.model_name = model_name
        self.block_index = block_index
        self.node_list = node_list
        # print(model_name)
        self.value = 0  # MODEL_STATS[self.model_name]['top1']
        self.size = 0
        self.group_id = None

    def print_split(self):
        start = self.node_list[0]
        end = self.node_list[-1]
        return [MODEL_STATS[self.model_name]['arch'],
                MODEL_BLOCKS[self.model_name][start],
                MODEL_BLOCKS[self.model_name][end],
                MODEL_STATS[self.model_name]['backend']]

    def get_inout_size(self):
        start = self.node_list[0]
        end = self.node_list[-1]
        start_name = MODEL_BLOCKS[self.model_name][start]
        end_name = MODEL_BLOCKS[self.model_name][end]
        self.in_size = MODEL_INOUT_SHAPE[self.model_name]['in_size'][start_name]
        self.out_size = MODEL_INOUT_SHAPE[self.model_name]['out_size'][end_name]

    def __len__(self):
        return len(self.node_list)

    def get_model_size(self):
        model, block_input, block_output, backend = self.print_split()
        block = network_to_module_subnet(
            model, block_input, block_output, backend)
        self.size = sum(p.numel() for p in block.parameters())/1e6

    def __eq__(self, other):
        if isinstance(other, Block):
            return (self.model_name == other.model_name and
                    self.block_index == other.block_index and
                    self.node_list == other.node_list)
        else:
            return False

    def __repr__(self) -> str:
        return json.dumps(self.__dict__, cls=NpEncoder, indent=2)

    def __str__(self):
        nodes_in = str(self.node_list[0]) 
        node_out = str(self.node_list[-1])
        return f'{MODEL_PRINT[self.model_name]}:{nodes_in}-{node_out} Stage-{self.block_index}'
        # return f'Model Name:{self.model_name}\tNode list:{self.node_list}\tBlock Index: {self.block_index}\t Size:{self.size}'


class Block_Sim:
    def __init__(self, sim_dict):
        self.sim_dict = sim_dict

    def get_sim(self, block1, block2):
        if not (isinstance(block1, Block) and isinstance(block2, Block)):
            raise TypeError('block1 and block2 must be Block instances')

        key = f'{block1.model_name}.{block2.model_name}'
        if key not in self.sim_dict:
            return 0

        if block1 == block2:
            return 1
        # FIX: was `block1.model_name == block1.model_name` (tautology).
        # Correct logic: zero out cross-stage similarity within the same model
        # to prevent a block being assigned to a functionally misaligned stage.
        if block1.model_name == block2.model_name and \
                block1.block_index != block2.block_index:
            return 0

        sim_map = self.sim_dict[key]
        # FIX: convert numpy.int64 indices to Python int to guarantee
        # compatibility with both numpy matrix and torch.Tensor indexing.
        i0, i1 = int(block1.node_list[0]),  int(block1.node_list[-1])
        j0, j1 = int(block2.node_list[0]),  int(block2.node_list[-1])
        try:
            block_sim = float(sim_map[i0, j0]) + float(sim_map[i1, j1])
        except (IndexError, KeyError):
            raise AssertionError(
                f'Functional similarity cannot be computed for blocks '
                f'{block1} and {block2}: indices ({i0},{j0}),({i1},{j1}) '
                f'out of range for sim_map shape {sim_map.shape}'
            )
        return block_sim        
    
class Block_Assign:
    def __init__(self, assignment_index, block_split_dict, centers):
        self.block2center = dict()
        self.center2block = [[] for _ in centers]
        self.centers = centers

        for m, model_name in enumerate(MODEL_ZOO):
            self.block2center[model_name] = dict()
            for j, block in enumerate(block_split_dict[model_name]):
                center_index = assignment_index[m, j]
                block.group_id = center_index
                self.block2center[model_name][j] = centers[center_index]
                self.center2block[center_index].append(block)
        # Ensure each group contains its designated center block exactly once
        for idx, center in enumerate(centers):
            if center not in self.center2block[idx]:
                self.center2block[idx].insert(0, center)
        
        
    def get_center(self, block):
        return self.block2center[block.model_name][block.block_index]

    def print_center(self):
        return '.'.join([str(c) for c in self.centers])

    def print_assignment(self):
        results = ''
        for i, group in enumerate(self.center2block):
            results += 'Center {}\n'.format(str(self.centers[i]))
            results += '\n'.join(['\t'+str(c) for c in group])
            results += '\n'
        print(results)

    def get_size(self):
        for group in self.center2block:
            for block in group:
                # block.get_model_size()
                block.get_inout_size()
                print(block)

    def save_assignment(self, out_file):
        with open(out_file, 'wb') as file:
            pickle.dump(self, file)


if __name__ == '__main__':
    with open(f'test.pickle', 'rb') as file2:
        s1 = pickle.load(file2)
    s1.save_assignment()