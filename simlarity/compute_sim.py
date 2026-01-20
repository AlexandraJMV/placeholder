import argparse
import os
from itertools import combinations
import torch
import pickle
from tqdm import tqdm  # Opcional: para ver progreso real

from simlarity.compare_functions import SIM_FUNC

def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('--feat_path', type=str, help='path containing feature .pth files')
    parser.add_argument('--out', default='', help='output directory for similarity .pkl files')
    parser.add_argument('--sim_func', default='cka', choices=['cka', 'rbf_cka', 'lr'], help='metric function')
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    
    # --- FIX 1: Crear directorio de salida si no existe ---
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    # --- FIX 2: Filtrar solo archivos .pth ---
    # Si lees archivos de sistema o json por error, torch.load explota.
    files = [f for f in os.listdir(args.feat_path) if f.endswith('.pth')]
    full_paths = [os.path.join(args.feat_path, p) for p in files]
    
    # Generar combinaciones
    pkls_comb = list(combinations(full_paths, 2))
    # Agregar auto-comparaciones (A vs A)
    pkls_comb += [(p, p) for p in full_paths]
    
    print(f"Procesando {len(pkls_comb)} pares de modelos...")

    # Usamos tqdm para barra de progreso general (opcional, pero recomendado)
    for path1, path2 in tqdm(reversed(pkls_comb), total=len(pkls_comb)):
        
        # Cargar datos
        try:
            data1 = torch.load(path1, map_location='cuda' if torch.cuda.is_available() else 'cpu')
            data2 = torch.load(path2, map_location='cuda' if torch.cuda.is_available() else 'cpu')
        except Exception as e:
            print(f"Error cargando {path1} o {path2}: {e}")
            continue

        name1 = data1['model_name']
        name2 = data2['model_name']
        
        # Lógica legacy para train_strategy (mantener por compatibilidad)
        if 'train_strategy' in data1.keys(): name1 += data1['train_strategy']
        if 'train_strategy' in data2.keys(): name2 += data2['train_strategy']
        
        save_path = os.path.join(args.out, f'{name1}.{name2}.pkl')
        
        # Saltar si ya existe
        if os.path.exists(save_path):
            continue
        
        # Calcular Similitud
        # print(f'Computing {name1} vs {name2}...') 
        sim = SIM_FUNC[args.sim_func](data1, data2, bs=2048)
        
        # Guardar resultados
        results = dict(
            sim=sim, 
            model1=dict(arch=name1, model_name=name1),
            model2=dict(arch=name2, model_name=name2)
        )
        
        # --- FIX 3 (Ya lo tenías bien): Usar 'with open' ---
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Limpiar memoria GPU/RAM
        del data1
        del data2
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()