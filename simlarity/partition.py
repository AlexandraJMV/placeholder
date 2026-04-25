import sys
import os

# Inject project root to sys.path for proper module resolution
sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))

import argparse
import copy
import pickle
import numpy as np
import json
import traceback
from itertools import combinations
from tqdm import tqdm
from scipy.optimize import linear_sum_assignment

# Relative imports
from .utils import Block, Block_Assign, Block_Sim, update_global_shapes
from blocklize import MODEL_BLOCKS, MODEL_ZOO

def parse_args():
    parser = argparse.ArgumentParser(description='DeRy Partitioning - Exhaustive Optimization Mode')
    parser.add_argument('--sim_path', default='./simlarity/out/sim_cka/', help='Path to similarity matrices')
    parser.add_argument('--K', type=int, default=4, help='Number of discrete stages (K equivalence sets)')
    parser.add_argument('--trial', type=int, default=200, help='Number of independent random starts')
    parser.add_argument('--out', default='./simlarity/out/assignment')
    parser.add_argument('--eps', type=float, default=0.2, help='Relaxation parameter for block capacity limit')
    parser.add_argument('--num_iter', type=int, default=200, help='Max coordinate descent iterations per trial')
    parser.add_argument('--save_path', type=str, default='network_plan.pkl', help='Serialized output path')
    parser.add_argument('--shape_path', type=str, required=True, help='Path to tensor dimension metadata')
    parser.add_argument('--feat_path', type=str, default='') 
    parser.add_argument('--input_size', type=int, default=160, help='Spatial input resolution')
    parser.add_argument('--num_stages', type=int, default=4, help='Alias for K')
    parser.add_argument('--penalty', type=float, default=0.1)

    args = parser.parse_args()
    if args.num_stages:
        args.K = args.num_stages
    return args

def get_all_sim(args):
    """
    Loads pairwise geometric similarity matrices computed via Centered Kernel Alignment (CKA).
    Utilizes O(1) dictionary lookups for O(N^2) similarity extractions during the optimization loop.
    """
    all_sim_dict = dict()
    with open(args.shape_path, 'r') as f:
        shapes = json.load(f)
    
    available_models = [m for m in shapes.keys() if m in MODEL_ZOO]
    comb = list(combinations(available_models, 2))
    comb += [(m, m) for m in available_models]
    
    loaded_count = 0
    for pair in tqdm(list(comb), desc="Loading Similarity Tensors"):
        a, b = pair                                                 
        pickle1 = os.path.join(args.sim_path, f'{a}.{b}.pkl')       
        pickle2 = os.path.join(args.sim_path, f'{b}.{a}.pkl')       

        data = None
        is_transpose = False

        if os.path.exists(pickle1):
            with open(pickle1, 'rb') as f: data = pickle.load(f)
        elif os.path.exists(pickle2):
            with open(pickle2, 'rb') as f: data = pickle.load(f)
            is_transpose = True
        
        if data is not None:
            sim_matrix = data['sim']
            if not is_transpose:
                all_sim_dict[f'{a}.{b}'] = sim_matrix
                all_sim_dict[f'{b}.{a}'] = sim_matrix.T
            else:
                all_sim_dict[f'{a}.{b}'] = sim_matrix.T
                all_sim_dict[f'{b}.{a}'] = sim_matrix
            loaded_count += 1
    
    if loaded_count == 0:
        raise ValueError("Critical Initialization Failure: Similarity matrices undefined at target path.")

    block_sims = Block_Sim(all_sim_dict)
    return block_sims

def sanitize_assignment(assignment, block_split_dict, centers):
    """
    Forces strict 1:1 mapping between the centroid array and the equivalence set blocks.
    Overrides the original utility constructor to prevent temporal state duplication.
    """
    assignment.center2block = [[] for _ in centers]
    for model_name in MODEL_ZOO:
        if model_name in block_split_dict:
            for j, block in enumerate(block_split_dict[model_name]):
                assignment.center2block[block.group_id].append(block)
    return assignment

def recenter(args, block_split_dict, block_sims, assignment):
    """
    Computes the geometric median of each equivalence set.
    The node with the highest aggregate similarity to all other members becomes the new cluster centroid.
    """
    new_centers = []
    for c_id, group in enumerate(assignment.center2block):
        num_in_group = len(group)
        if num_in_group == 0:
            continue
        group_sim = np.zeros((num_in_group, num_in_group))
        for b1_id in range(num_in_group):
            for b2_id in range(num_in_group):
                block1 = group[b1_id]
                block2 = group[b2_id]
                group_sim[b1_id, b2_id] = block_sims.get_sim(block1, block2)
        new_center_index = np.argmax(group_sim.sum(0))
        new_centers.append(group[new_center_index])
    assignment.centers = new_centers
    return assignment

def reassign(args, block_split_dict, block_sims, assignment):
    """
    Computes bipartite matching between model blocks and cluster centroids via the Hungarian Algorithm.
    Mathematically ensures Eq. (3) (Bijective Mapping Constraint).
    """
    num_model = len(MODEL_ZOO)
    centers = assignment.centers
    block_sim_map = np.zeros((args.K, num_model, args.K))
    
    for i, center_block in enumerate(centers):
        for m, other_model_name in enumerate(MODEL_ZOO):
            for j, block in enumerate(block_split_dict[other_model_name]):
                block_sim_map[i, m, j] = block_sims.get_sim(center_block, block)

    assignment_index = np.zeros((num_model, args.K), dtype=int)
    
    # Linear Sum Assignment Problem (LSAP) execution
    for m in range(num_model):
        cost_matrix = -block_sim_map[:, m, :]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for center_idx, block_idx in zip(row_ind, col_ind):
            assignment_index[m, block_idx] = center_idx

    new_assignment = Block_Assign(assignment_index=assignment_index,
                                  block_split_dict=block_split_dict,
                                  centers=centers)
    return sanitize_assignment(new_assignment, block_split_dict, centers)

def compute_cost(block, all_assignmemt, block_sims):
    center_block = all_assignmemt.get_center(block)
    return block_sims.get_sim(center_block, block)

def total_cost(assignment, block_sims):
    """
    Objective function execution.
    Calculates the sum of CKA similarities of all non-duplicate block pairs within all sets.
    """
    total_sim = 0
    for group in assignment.center2block:
        num_in_group = len(group)
        group_sim = np.zeros((num_in_group, num_in_group))
        for b1_id in range(num_in_group):
            for b2_id in range(b1_id+1, num_in_group):
                block1 = group[b1_id]
                block2 = group[b2_id]
                group_sim[b1_id, b2_id] = block_sims.get_sim(block1, block2)
        total_sim += np.sum(group_sim)
    return total_sim

def repartition(args, block_split_dict, block_sims, all_assignmemt):
    """
    Local contiguous boundary search.
    Iteratively swaps sequential node boundaries between adjacent blocks, constrained by 
    strict block capacity limits.
    """
    improved = False
    for m_id, model_name in enumerate(MODEL_ZOO):
        iter_block_split = copy.deepcopy(block_split_dict[model_name])
        max_node_model = block_split_dict[f'{model_name}_max_node']
        
        for b_id in range(len(iter_block_split)-1):
            block1 = iter_block_split[b_id]
            block2 = iter_block_split[b_id+1]
            len1, len2 = len(block1), len(block2)

            best_cost = (compute_cost(block1, all_assignmemt, block_sims) +
                         compute_cost(block2, all_assignmemt, block_sims))
            concat_nodes = (block1.node_list + block2.node_list)
            
            # Forward Shift: Block 1 absorbs the first node of Block 2
            if len2 > block_split_dict['min_node'] and len1 < max_node_model:
                block1_new = Block(model_name, b_id, concat_nodes[:len1+1])
                block2_new = Block(model_name, b_id+1, concat_nodes[len1+1:])
                new_cost = (compute_cost(block1_new, all_assignmemt, block_sims) +
                            compute_cost(block2_new, all_assignmemt, block_sims))
                if new_cost > best_cost:
                    improved = True
                    best_cost = new_cost
                    iter_block_split[b_id] = block1_new
                    iter_block_split[b_id+1] = block2_new

            # Backward Shift: Block 2 absorbs the terminal node of Block 1
            if len1 > block_split_dict['min_node'] and len2 < max_node_model:
                block1_new = Block(model_name, b_id, concat_nodes[:len1-1])
                block2_new = Block(model_name, b_id+1, concat_nodes[len1-1:])
                new_cost = (compute_cost(block1_new, all_assignmemt, block_sims) +
                            compute_cost(block2_new, all_assignmemt, block_sims))
                if new_cost > best_cost:
                    improved = True
                    best_cost = new_cost
                    iter_block_split[b_id] = block1_new
                    iter_block_split[b_id+1] = block2_new

        block_split_dict[model_name] = iter_block_split
    return block_split_dict, improved

def init_partition(args):
    """
    Initializes continuous arrays of nodes via deterministic uniform splitting, 
    guaranteeing compliance with the (1+eps) capacity constraint at step zero.
    """
    block_split_dict = dict()
    block_split_dict['min_node'] = 1
    
    for model_name in MODEL_ZOO:
        node_list = MODEL_BLOCKS[model_name]
        N = len(node_list)
        max_node_per_block = int(np.ceil(N/args.K) * (1+args.eps))
        
        block_split_dict[f'{model_name}_max_node'] = max_node_per_block
        block_split_dict[model_name] = []
        
        splits = np.array_split(np.arange(N), args.K)
        for k in range(args.K):
            block = Block(model_name, k, [int(x) for x in splits[k]])
            block_split_dict[model_name].append(block)

    return block_split_dict

def init_assign(args, block_split_dict, block_sims):
    all_blocks = []
    num_model = len(MODEL_ZOO)
    for model_name in MODEL_ZOO:
        for i in range(args.K):
            all_blocks.append(block_split_dict[model_name][i])
    
    centers = [all_blocks[i] for i in np.random.choice(len(all_blocks), args.K, replace=False)]
    block_sim_map = np.zeros((args.K, num_model, args.K))
    
    for i, center_block in enumerate(centers):
        for m, other_model_name in enumerate(MODEL_ZOO):
            for j, block in enumerate(block_split_dict[other_model_name]):
                block_sim_map[i, m, j] = block_sims.get_sim(center_block, block)

    assignment_index = np.zeros((num_model, args.K), dtype=int)
    
    for m in range(num_model):
        cost_matrix = -block_sim_map[:, m, :]
        row_ind, col_ind = linear_sum_assignment(cost_matrix)
        for center_idx, block_idx in zip(row_ind, col_ind):
            assignment_index[m, block_idx] = center_idx

    new_assignment = Block_Assign(assignment_index=assignment_index,
                                  block_split_dict=block_split_dict,
                                  centers=centers)
    return sanitize_assignment(new_assignment, block_split_dict, centers)

def print_partition(block_split_dict):
    print("\n--- Network Plan Final ---")
    for model_name in MODEL_ZOO:
        if model_name in block_split_dict:
            blocks_str = ' | '.join([str(b) for b in block_split_dict[model_name]])
            print(f"[{model_name}]: {blocks_str}")

def main():
    args = parse_args()
    update_global_shapes(args.shape_path)
    block_sims = get_all_sim(args)      
    all_sim = 0                         
    best_all_assignmemt = None          
    best_block_split_dict = None        

    print(f"Execution initialized: {args.trial} independent stochastic trials.")

    for k in tqdm(range(args.trial), desc="Trials"):
        try:
            block_split_dict = init_partition(args)
            all_assignmemt = init_assign(args, block_split_dict, block_sims)
            non_improved = 0
            
            # Coordinate Descent Execution Loop
            for i in range(args.num_iter):
                block_split_dict, improved = repartition(args, block_split_dict, block_sims, all_assignmemt)
                
                if improved:
                    all_assignmemt = reassign(args, block_split_dict, block_sims, all_assignmemt)
                
                all_assignmemt = recenter(args, block_split_dict, block_sims, all_assignmemt)
                all_assignmemt = reassign(args, block_split_dict, block_sims, all_assignmemt)

                current_sim = total_cost(all_assignmemt, block_sims)

                if current_sim > all_sim:
                    all_sim = current_sim
                    
                    # Pointer Atomicity: Isolate state to prevent temporal corruption
                    state_snapshot = copy.deepcopy({
                        'split': block_split_dict,
                        'assign': all_assignmemt
                    })
                    best_block_split_dict = state_snapshot['split']
                    best_all_assignmemt = state_snapshot['assign']
                    non_improved = 0
                else:
                    non_improved += 1

                if non_improved > 20:
                    break
        except Exception as e:
            print(f"\n[!] State calculation failed at trial {k}: {e}")
            traceback.print_exc()
            break 
    
    if best_block_split_dict is None:
        print("Failure: Objective space unnavigable. Check input tensors.")
        return

    print_partition(best_block_split_dict)

    # --- TOPOLOGICAL DAG SORTING ---
    # Computes absolute spatial volume to strictly enforce monotonic downsampling geometry
    stage_depths = []
    for i, group in enumerate(best_all_assignmemt.center2block):
        if len(group) > 0:
            for b in group:
                b.get_inout_size()
            
            hs = []
            for b in group:
                if hasattr(b, 'in_size') and isinstance(b.in_size, (list, tuple)) and len(b.in_size) >= 2:
                    hs.append(b.in_size[-2])
                else:
                    hs.append(1) 
            avg_H = np.mean(hs)
            
            depths = [b.node_list[0] / len(MODEL_BLOCKS[b.model_name]) for b in group]
            avg_depth = np.mean(depths)
        else:
            avg_H = -1
            avg_depth = float('inf')
        
        stage_depths.append((i, avg_H, avg_depth))
    
    # Sort hierarchy:
    # 1. Spatial height descending (larger tensor volume = early stage)
    # 2. Structural model depth fractional representation ascending
    stage_depths.sort(key=lambda x: (-x[1], x[2]))
    sorted_indices = [idx for idx, _, _ in stage_depths]
    
    # 1. Sort the physical arrays
    best_all_assignmemt.centers = [best_all_assignmemt.centers[i] for i in sorted_indices]
    best_all_assignmemt.center2block = [best_all_assignmemt.center2block[i] for i in sorted_indices]
    
    # 2. Synchronize internal group_ids for the Center Anchor Blocks
    for new_stage_idx, center_block in enumerate(best_all_assignmemt.centers):
        center_block.group_id = new_stage_idx

    # 3. Synchronize internal group_ids for the Member Blocks and the Origin Dictionary
    for new_stage_idx, group in enumerate(best_all_assignmemt.center2block):
        for block in group:
            block.group_id = new_stage_idx
            best_block_split_dict[block.model_name][block.block_index].group_id = new_stage_idx
            
    # 4. Rebuild the Block-to-Center topological map
    best_all_assignmemt.block2center = {m: dict() for m in MODEL_ZOO}
    for model_name in MODEL_ZOO:
        for j, block in enumerate(best_block_split_dict[model_name]):
            assigned_stage = block.group_id
            best_all_assignmemt.block2center[model_name][j] = best_all_assignmemt.centers[assigned_stage]
    # -------------------------------

    print(f"Artifact serialized: {args.save_path}")
    if os.path.dirname(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    with open(args.save_path, 'wb') as f:
        pickle.dump(best_all_assignmemt, f)

if __name__ == '__main__':
    main()