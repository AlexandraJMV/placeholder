import sys
import os
import pickle
import numpy as np

# --- SETUP DE IMPORTACIONES ---
# Necesario para que pickle reconozca las clases 'Block', 'Block_Assign', etc.
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

    # center2block es una lista de listas: [ [Bloques Etapa 0], [Bloques Etapa 1], ... ]
    original_stages = plan.center2block
    num_stages = len(original_stages)
    
    stage_scores = []
    
    print("\n📊 Analizando profundidad promedio de cada etapa actual:")
    
    for i, blocks in enumerate(original_stages):
        if not blocks:
            # Si una etapa está vacía por error, le damos score infinito para mandarla al final
            print(f"   -> Stage {i}: VACÍA (Warning!)")
            stage_scores.append((99999, i))
            continue
            
        # Calculamos la "profundidad" de la etapa promediando el índice de inicio 
        # de los bloques originales que contiene.
        # node_list[0] es el índice de la primera capa del bloque en el modelo original.
        start_indices = [b.node_list[0] for b in blocks]
        avg_depth = np.mean(start_indices)
        
        stage_scores.append((avg_depth, i))
        print(f"   -> Stage {i} (Original): Profundidad promedio = {avg_depth:.2f}")

    # --- LÓGICA DE REORDENAMIENTO ---
    # Ordenamos las etapas de MENOR profundidad (input) a MAYOR profundidad (output)
    stage_scores.sort(key=lambda x: x[0])
    
    new_order_indices = [x[1] for x in stage_scores]
    
    # Comprobar si ya estaba ordenado
    is_sorted = (new_order_indices == list(range(num_stages)))
    
    if is_sorted:
        print("\n✅ El plan YA está ordenado correctamente. No se necesitan cambios.")
        return
    else:
        print(f"\n⚠️ DESORDEN DETECTADO. Reordenando etapas...")
        print(f"   Nuevo orden de índices originales: {new_order_indices}")
        print("   (La Stage 0 nueva será la que antes era la Stage " + str(new_order_indices[0]) + ", etc.)")

    # Reconstruir las listas en el nuevo orden
    new_center2block = [original_stages[i] for i in new_order_indices]
    
    # Si el objeto plan tiene 'centers' (usado para reassign), también hay que ordenarlos
    if hasattr(plan, 'centers') and plan.centers:
        plan.centers = [plan.centers[i] for i in new_order_indices]
        
    # Actualizar la lista principal
    plan.center2block = new_center2block
    
    # Actualizar los índices internos de los bloques (b.stage_id)
    # Esto es crucial: cada bloque sabe a qué etapa pertenece. Hay que decírselo de nuevo.
    for new_stage_idx, blocks in enumerate(plan.center2block):
        for block in blocks:
            block.stage_id = new_stage_idx

    # --- GUARDADO ---
    with open(pkl_path, 'wb') as f:
        pickle.dump(plan, f)
    
    print(f"\n💾 Reparación completada. {pkl_path} ha sido sobrescrito con el orden topológico correcto.")

if __name__ == '__main__':
    # Puedes cambiar el path si tu archivo se llama distinto
    fix_order('network_plan.pkl')