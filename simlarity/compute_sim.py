import argparse
import os
from itertools import combinations
import torch
import pickle
from tqdm import tqdm  # Opcional: para ver progreso real

from simlarity.compare_functions import SIM_FUNC


# Función para parsear argumentos de línea de comandos
def parse_args():
    
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('--feat_path', type=str, help='path containing feature .pth files')                     # Directorio con los archivos .pth de características
    parser.add_argument('--out', default='', help='output directory for similarity .pkl files')                 # Directorio de salida para los archivos .pkl de similitud
    parser.add_argument('--sim_func', default='cka', choices=['cka', 'rbf_cka', 'lr'], help='metric function')  # Función de similitud a usar (por defecto 'cka', opciones: 'cka', 'rbf_cka', 'lr')
    args = parser.parse_args()
    return args

# Función principal para calcular similitudes entre modelos a partir de sus características
def main():
    # Parsear argumentos
    args = parse_args()
    
    # Crear directorio de salida si no existe ---
    if args.out:
        os.makedirs(args.out, exist_ok=True)

    # Filtrar solo archivos .pth 
    files = [f for f in os.listdir(args.feat_path) if f.endswith('.pth')]       # Listar solo los archivos .pth en el directorio de características
    full_paths = [os.path.join(args.feat_path, p) for p in files]               # Obtener rutas completas de los archivos .pth
    
    # Crear combinaciones de archivos para comparación (incluyendo auto-comparaciones)
    pkls_comb = list(combinations(full_paths, 2))
    
    # Agregar auto-comparaciones (A vs A)
    pkls_comb += [(p, p) for p in full_paths]
    
    
    print(f"Procesando {len(pkls_comb)} pares de modelos...")

    # Usamos tqdm para barra de progreso general
    for path1, path2 in tqdm(reversed(pkls_comb), total=len(pkls_comb)):
        
        # Cargar datos
        try:
            data1 = torch.load(path1, map_location='cuda' if torch.cuda.is_available() else 'cpu')  # Cargar el primer archivo .pth (características del modelo 1)
            data2 = torch.load(path2, map_location='cuda' if torch.cuda.is_available() else 'cpu')  # Cargar el segundo archivo .pth (características del modelo 2)
        except Exception as e:
            print(f"Error cargando {path1} o {path2}: {e}")
            continue

        # Extraer nombres de modelos para nombrar el archivo de salida
        name1 = data1['model_name']
        name2 = data2['model_name']
        
        # Agregar estrategia de entrenamiento al nombre si está presente
        if 'train_strategy' in data1.keys(): name1 += data1['train_strategy']
        if 'train_strategy' in data2.keys(): name2 += data2['train_strategy']
        
        # Construir ruta de guardado para el archivo de similitud
        save_path = os.path.join(args.out, f'{name1}.{name2}.pkl')
        
        # Saltar si ya existe
        if os.path.exists(save_path):
            continue
        
        # Calcular similitud usando la función seleccionada
        sim = SIM_FUNC[args.sim_func](data1, data2, bs=2048)
        
        # Guardar resultados en un diccionario y luego en un archivo .pkl
        results = dict(
            sim=sim, 
            model1=dict(arch=name1, model_name=name1),
            model2=dict(arch=name2, model_name=name2)
        )
        
        # Guardar resultados en un archivo .pkl
        with open(save_path, 'wb') as f:
            pickle.dump(results, f)
        
        # Limpiar memoria GPU/RAM
        del data1
        del data2
        torch.cuda.empty_cache()

if __name__ == '__main__':
    main()