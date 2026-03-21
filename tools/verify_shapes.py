import sys
import os
import json
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

try:
    from blocklize import MODEL_BLOCKS, MODEL_ZOO
except ImportError:
    print("❌ Error Fatal: No se pudo importar 'blocklize'. Ejecuta desde la raíz del proyecto.")
    sys.exit(1)

def verify_shapes(json_path):
    print(f"🕵️‍♂️ Auditando archivo de formas: {json_path}\n")
    
    if not os.path.exists(json_path):
        print(f"❌ Error: El archivo {json_path} no existe.")
        return

    with open(json_path, 'r') as f:
        try:
            shapes_db = json.load(f)
        except json.JSONDecodeError:
            print("❌ Error: El archivo no es un JSON válido.")
            return

    issues_found = False
    
    print(f"{'MODELO':<30} | {'NODOS ESPERADOS':<15} | {'ESTADO'}")
    print("-" * 70)
    
    for model_name in MODEL_ZOO:
        if model_name not in shapes_db:
            print(f"{model_name:<30} | {'---':<15} | ❌ MISSING (No está en el JSON)")
            issues_found = True
            continue
            
        model_data = shapes_db[model_name]
        expected_nodes = MODEL_BLOCKS.get(model_name, [])
        
        missing_nodes = []
        for node_name in expected_nodes:
            if 'in_size' not in model_data or node_name not in model_data['in_size']:
                missing_nodes.append(node_name)
        
        status = "✅ OK"
        if missing_nodes:
            status = f"⚠️ FALTAN {len(missing_nodes)} NODOS"
            issues_found = True
            
        print(f"{model_name:<30} | {len(expected_nodes):<15} | {status}")
        
        if missing_nodes:
            print(f"   -> Ejemplos faltantes: {missing_nodes[:3]}...")

        if expected_nodes:
            first_node = expected_nodes[0]
            if first_node in model_data.get('in_size', {}):
                shape = model_data['in_size'][first_node]
                if not isinstance(shape, list) or len(shape) not in [3, 4]:
                     print(f"   ❌ Formato de forma sospechoso en {first_node}: {shape}")
                     issues_found = True

    print("-" * 70)
    if not issues_found:
        print("\n🎉 VERIFICACIÓN EXITOSA: El JSON coincide perfectamente con la definición de tus modelos.")
    else:
        print("\n🚫 ERRORES ENCONTRADOS: El JSON está incompleto o desactualizado.")
        print("   Solución: Ejecuta nuevamente el script que genera este JSON (probablemente 'get_model_shape.py')")

if __name__ == "__main__":
    json_path = "tools/MODEL_INOUT_SHAPE.json" 
    verify_shapes(json_path)