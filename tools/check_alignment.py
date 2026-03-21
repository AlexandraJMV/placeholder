import sys
import os
import pickle
import argparse
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

try:
    from blocklize import MODEL_BLOCKS, MODEL_ZOO
except ImportError:
    print("❌ Error Fatal: No se pudo importar 'blocklize'. Asegúrate de correr esto desde la raíz del proyecto.")
    sys.exit(1)

def check_alignment(sim_path):
    print(f"🔍 Iniciando Auditoría de Alineación en: {sim_path}\n")
    print(f"{'MODELO':<30} | {'METADATA (Nodes)':<15} | {'MATRIZ (Dim)':<15} | {'ESTADO'}")
    print("-" * 80)
    
    issues_found = False
    
    for model_name in MODEL_ZOO:
        if model_name not in MODEL_BLOCKS:
            print(f"{model_name:<30} | {'MISSING':<15} | {'---':<15} | ❌ ERROR (No en MODEL_BLOCKS)")
            issues_found = True
            continue
            
        expected_n = len(MODEL_BLOCKS[model_name])
        
        pkl_name = f"{model_name}.{model_name}.pkl"
        pkl_path = os.path.join(sim_path, pkl_name)
        
        if not os.path.exists(pkl_path):
            print(f"{model_name:<30} | {expected_n:<15} | {'NOT FOUND':<15} | ⚠️  WARNING (Falta archivo .pkl)")
            continue
            
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            if 'sim' not in data:
                 print(f"{model_name:<30} | {expected_n:<15} | {'INVALID':<15} | ❌ ERROR (PKL corrupto)")
                 issues_found = True
                 continue

            sim_matrix = data['sim']
            matrix_n = sim_matrix.shape[0]
            
            if matrix_n != expected_n:
                status = "❌ MISMATCH"
                issues_found = True
            else:
                status = "✅ OK"
                
            print(f"{model_name:<30} | {expected_n:<15} | {matrix_n:<15} | {status}")
                
        except Exception as e:
            print(f"{model_name:<30} | {expected_n:<15} | {'ERROR':<15} | ❌ Lectura fallida: {e}")
            issues_found = True

    print("-" * 80)
    if not issues_found:
        print("\n🎉 ÉXITO: Todos los índices están perfectamente alineados. Puedes particionar con confianza.")
    else:
        print("\n⛔ PELIGRO: Se detectaron desalineaciones.")
        print("   Si procedes con partition.py, el algoritmo mezclará capas incorrectas.")
        print("   SOLUCIÓN: Regenera las matrices CKA o actualiza MODEL_BLOCKS en blocklize/block_meta.py")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Verifica que MODEL_BLOCKS coincida con las matrices CKA generadas.")
    parser.add_argument('--sim_path', type=str, default='mis_similitudes', help='Ruta a la carpeta con archivos .pkl')
    args = parser.parse_args()
    
    if not os.path.exists(args.sim_path):
        print(f"Error: La carpeta '{args.sim_path}' no existe.")
    else:
        check_alignment(args.sim_path)