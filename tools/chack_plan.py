import sys
import os
import pickle

# Hack para que Python encuentre tus módulos locales
sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

# Necesitamos importar la clase Block para que pickle pueda leer el objeto
from simlarity.utils import Block, Block_Assign

def inspect_plan(pkl_path):
    if not os.path.exists(pkl_path):
        print("ERROR: No existe el archivo.")
        return

    print(f"--- INSPECCIONANDO: {pkl_path} ---\n")
    
    with open(pkl_path, 'rb') as f:
        plan = pickle.load(f)

    # El objeto plan tiene un atributo 'center2block' que es una lista de listas
    # Cada lista interna es una ETAPA (Stage)
    
    stages = plan.center2block
    
    for stage_idx, blocks in enumerate(stages):
        print(f"🔵 ETAPA {stage_idx} (Opciones intercambiables):")
        print(f"   La red puede elegir CUALQUIERA de estos caminos para esta etapa:")
        
        for b in blocks:
            # b.node_list son los índices de las capas originales
            print(f"    - Modelo: {b.model_name:20} | Bloques Originales: {b.node_list}")
            
        print("-" * 60)

    print("\n✅ CONCLUSIÓN:")
    print("Si ves que en la Etapa 0 hay bloques 'tempranos' (ej. 0-2)")
    print("y en la Etapa 3 hay bloques 'tardíos' (ej. 10-15), el plan es LÓGICO.")

if __name__ == '__main__':
    inspect_plan('network_plan.pkl')