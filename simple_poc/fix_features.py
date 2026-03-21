import os
import sys
import torch

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import simple_poc.poc_config as cfg

def fix_features():
    print(f">>> REPARANDO ARCHIVOS .PTH EN {cfg.FEAT_DIR} <<<")
    
    if not os.path.exists(cfg.FEAT_DIR):
        print("Error: No existe la carpeta de features.")
        return

    files = [f for f in os.listdir(cfg.FEAT_DIR) if f.endswith('.pth')]
    
    for filename in files:
        path = os.path.join(cfg.FEAT_DIR, filename)
        model_name = os.path.splitext(filename)[0]
        
        try:
            data = torch.load(path)
            
            if 'model_name' not in data:
                data['model_name'] = model_name
                torch.save(data, path)
                print(f"   [REPARADO] {filename} -> Se agregó model_name='{model_name}'")
            else:
                print(f"   [OK] {filename} ya está correcto.")
                
        except Exception as e:
            print(f"   [ERROR] Falló al procesar {filename}: {e}")

    print("\n¡Listo! Ahora compute_sim.py debería funcionar.")

if __name__ == '__main__':
    fix_features()