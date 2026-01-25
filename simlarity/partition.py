import sys
import os

# --- PARCHE 1: Arreglar conflicto de versiones de timm ---
# Esto fuerza a usar la carpeta local 'third_package'
sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))

import argparse
import copy
import pickle
import numpy as np
import json
from itertools import combinations
from random import sample
from tqdm import tqdm

# Importaciones relativas
from .utils import Block, Block_Assign, Block_Sim
from blocklize import MODEL_BLOCKS, MODEL_ZOO

def parse_args():
    parser = argparse.ArgumentParser(description='DeRy Partitioning')
    
    # --- Argumentos Originales ---
    parser.add_argument('--sim_path', default='./simlarity/out/sim_cka/', help='Path to similarity matrices')
    parser.add_argument('--K', type=int, default=4, help='Number of stages')
    parser.add_argument('--trial', type=int, default=200)
    parser.add_argument('--out', default='./simlarity/out/assignment')
    parser.add_argument('--eps', type=float, default=0.2)
    parser.add_argument('--num_iter', type=int, default=200)

    # --- Argumentos Nuevos (NECESARIOS para tu comando) ---
    parser.add_argument('--save_path', type=str, default='network_plan.pkl', help='Output file path')
    parser.add_argument('--shape_path', type=str, required=True, help='Path to JSON with model shapes')
    
    # Argumentos extra para compatibilidad (evitan errores si se pasan en el comando)
    parser.add_argument('--feat_path', type=str, default='') 
    parser.add_argument('--input_size', type=int, default=224)
    parser.add_argument('--num_stages', type=int, default=4, help='Alias for K')
    parser.add_argument('--penalty', type=float, default=0.1)

    args = parser.parse_args()
    
    # Sincronizar alias (num_stages -> K)
    if args.num_stages:
        args.K = args.num_stages

    return args

def get_all_sim(args):
    """Carga las matrices de similitud sin usar mmcv"""
    all_sim_dict = dict()
    
    # Leer modelos disponibles desde el JSON
    with open(args.shape_path, 'r') as f:
        shapes = json.load(f)
    available_models = [m for m in shapes.keys() if m in MODEL_ZOO]

    # Generar combinaciones
    comb = list(combinations(available_models, 2))
    comb += [(m, m) for m in available_models]
    
    print(f"Cargando matrices desde: {args.sim_path}")
    
    loaded_count = 0
    for pair in tqdm(list(comb), desc="Loading Matrices"):
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
        raise ValueError("¡No se cargó ninguna matriz! Revisa que 'mis_similitudes' tenga archivos .pkl")

    block_sims = Block_Sim(all_sim_dict)
    return block_sims

# --- Funciones de Optimización (Lógica Original) ---

def recenter(args, block_split_dict, block_sims, assignment):
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
    num_model = len(MODEL_ZOO)
    centers = assignment.centers
    block_sim_map = np.zeros((args.K, num_model, args.K))
    for i, center_block in enumerate(centers):
        for m, other_model_name in enumerate(MODEL_ZOO):
            for j, block in enumerate(block_split_dict[other_model_name]):
                block_sim = block_sims.get_sim(center_block, block)
                block_sim_map[i, m, j] = block_sim

    assignment_index = np.argmax(block_sim_map, axis=0)
    assignment = Block_Assign(assignment_index=assignment_index,
                              block_split_dict=block_split_dict,
                              centers=centers)
    return assignment

def compute_cost(block, all_assignmemt, block_sims):
    center_block = all_assignmemt.get_center(block)
    block_sim = block_sims.get_sim(center_block, block)
    return block_sim

def total_cost(assignment, block_sims):
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
    improved = False
    for m_id, model_name in enumerate(MODEL_ZOO):
        iter_block_split = copy.deepcopy(block_split_dict[model_name])
        for b_id in range(len(iter_block_split)-1):
            block1 = iter_block_split[b_id]
            block2 = iter_block_split[b_id+1]
            len1, len2 = len(block1), len(block2)

            best_cost = (compute_cost(block1, all_assignmemt, block_sims) +
                         compute_cost(block2, all_assignmemt, block_sims))
            concat_nodes = (block1.node_list + block2.node_list)
            
            if len2 > block_split_dict['min_node'] and len2 <= block_split_dict['max_node']:
                block1_new = Block(model_name, b_id, concat_nodes[:len1+1])
                block2_new = Block(model_name, b_id+1, concat_nodes[len1+1:])
                new_cost = (compute_cost(block1_new, all_assignmemt, block_sims) +
                            compute_cost(block2_new, all_assignmemt, block_sims))
                if new_cost > best_cost:
                    improved = True
                    best_cost = new_cost
                    iter_block_split[b_id] = block1_new
                    iter_block_split[b_id+1] = block2_new

            if len1 > block_split_dict['min_node'] and len1 <= block_split_dict['max_node']:
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
    block_split_dict = dict()
    block_split_dict['min_node'] = 1
    
    # Calcular nodos máximos
    for model_name in MODEL_ZOO:
        node_list = MODEL_BLOCKS[model_name]
        N = len(node_list)
        max_node_per_block = int(np.ceil(N/args.K) * (1+args.eps))
        block_split_dict['max_node'] = max_node_per_block
        
        block_split_dict[model_name] = []
        node_indexs = np.arange(N)
        
        # Iniciar cortes aleatorios
        possible_cuts = list(range(1, N))
        if len(possible_cuts) < args.K - 1:
            # Caso borde: modelo muy pequeño
            node_split = sorted(possible_cuts + [possible_cuts[-1]] * ((args.K-1) - len(possible_cuts)))
        else:
            node_split = sorted(sample(possible_cuts, args.K-1))
            
        node_split = [0] + node_split
        
        for k in range(args.K):
            i1 = node_split[k]
            if k == args.K-1:
                # Último bloque hasta el final
                block = Block(model_name, k, list(node_indexs[i1:]))
            else:
                i2 = node_split[k+1]
                block = Block(model_name, k, list(node_indexs[i1:i2]))
            block_split_dict[model_name].append(block)

    return block_split_dict

def init_assign(args, block_split_dict, block_sims):
    all_blocks = []
    num_model = len(MODEL_ZOO)
    for model_name in MODEL_ZOO:
        for i in range(args.K):
            block = block_split_dict[model_name][i]
            all_blocks.append(block)
    
    # Elegir centros iniciales aleatorios
    centers = sample(all_blocks, args.K)

    block_sim_map = np.zeros((args.K, num_model, args.K))
    for i, center_block in enumerate(centers):
        for m, other_model_name in enumerate(MODEL_ZOO):
            for j, block in enumerate(block_split_dict[other_model_name]):
                block_sim = block_sims.get_sim(center_block, block)
                block_sim_map[i, m, j] = block_sim

    assignment_index = np.argmax(block_sim_map, axis=0)
    assignment = Block_Assign(assignment_index=assignment_index,
                              block_split_dict=block_split_dict,
                              centers=centers)
    return assignment

def print_partition(block_split_dict):
    print("\n--- Network Plan Final ---")
    for model_name in MODEL_ZOO:
        if model_name in ['min_node', 'max_node']: continue
        blocks_str = ' | '.join([str(b) for b in block_split_dict[model_name]])
        print(f"[{model_name}]: {blocks_str}")

def main():
    args = parse_args()

    # 1. Cargar datos
    block_sims = get_all_sim(args)
    all_sim = 0
    best_all_assignmemt = None
    best_block_split_dict = None

    print(f"Iniciando búsqueda: {args.trial} intentos, {args.num_iter} iteraciones...")

    for k in tqdm(range(args.trial), desc="Trials"):
        try:
            block_split_dict = init_partition(args)
            all_assignmemt = init_assign(args, block_split_dict, block_sims)
            non_improved = 0
            
            for i in range(args.num_iter):
                # Paso 1: Reparticionar
                block_split_dict, _ = repartition(args, block_split_dict, block_sims, all_assignmemt)
                # Paso 2: Recentrar clusters
                all_assignmemt = recenter(args, block_split_dict, block_sims, all_assignmemt)
                # Paso 3: Reasignar
                all_assignmemt = reassign(args, block_split_dict, block_sims, all_assignmemt)

                current_sim = total_cost(all_assignmemt, block_sims)

                if current_sim > all_sim:
                    all_sim = current_sim
                    best_block_split_dict = copy.deepcopy(block_split_dict)
                    best_all_assignmemt = copy.deepcopy(all_assignmemt)
                    non_improved = 0
                else:
                    non_improved += 1

                if non_improved > 20:
                    break
        except Exception:
            continue # Si falla un trial aleatorio, probar otro
    
    if best_block_split_dict is None:
        print("ERROR: No se encontró solución. Revisa tus datos de entrada.")
        return

    print_partition(best_block_split_dict)

    # Guardar resultado sin mmcv
    print(f"Guardando plan en: {args.save_path}")
    if os.path.dirname(args.save_path):
        os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    with open(args.save_path, 'wb') as f:
        pickle.dump(best_all_assignmemt, f)

if __name__ == '__main__':
    main()
