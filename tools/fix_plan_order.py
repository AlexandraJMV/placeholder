import sys
import os
import pickle
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

from simlarity.utils import Block, Block_Assign

def fix_order(pkl_path):
    if not os.path.exists(pkl_path):
        print(f"❌ Error: No se encontró el archivo {pkl_path}")
        return

    print(f"🔧 Abriendo {pkl_path} para auditoría y reparación...")
    
    with open(pkl_path, 'rb') as f:
        plan = pickle.load(f)

    original_stages = plan.center2block
    num_stages = len(original_stages)
    
    stage_scores = []
    
    print("\n📊 Analizando profundidad promedio de cada etapa actual:")
    
    for i, blocks in enumerate(original_stages):
        if not blocks:
            print(f"   -> Stage {i}: VACÍA (Warning!)")
            stage_scores.append((99999, i))
            continue
            
        start_indices = [b.node_list[0] for b in blocks]
        avg_depth = np.mean(start_indices)
        
        stage_scores.append((avg_depth, i))
        print(f"   -> Stage {i} (Original): Profundidad promedio = {avg_depth:.2f}")

    stage_scores.sort(key=lambda x: x[0])
    
    new_order_indices = [x[1] for x in stage_scores]
    
    is_sorted = (new_order_indices == list(range(num_stages)))
    
    if is_sorted:
        print("\n✅ El plan YA está ordenado correctamente. No se necesitan cambios.")
        return
    else:
        print(f"\n⚠️ DESORDEN DETECTADO. Reordenando etapas...")
        print(f"   Nuevo orden de índices originales: {new_order_indices}")
        print("   (La Stage 0 nueva será la que antes era la Stage " + str(new_order_indices[0]) + ", etc.)")

    new_center2block = [original_stages[i] for i in new_order_indices]
    
    if hasattr(plan, 'centers') and plan.centers:
        plan.centers = [plan.centers[i] for i in new_order_indices]
        
    plan.center2block = new_center2block
    
    for new_stage_idx, blocks in enumerate(plan.center2block):
        for block in blocks:
            block.stage_id = new_stage_idx

    with open(pkl_path, 'wb') as f:
        pickle.dump(plan, f)
    
    print(f"\n💾 Reparación completada. {pkl_path} ha sido sobrescrito con el orden topológico correcto.")

if __name__ == '__main__':
    fix_order('network_plan.pkl')