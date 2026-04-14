import sys
import os
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

# Importar MODEL_BLOCKS y MODEL_ZOO desde blocklize.py, 
# si falla, se muestra un error claro.
try:
    from blocklize import MODEL_BLOCKS, MODEL_ZOO
except ImportError:
    print("❌ Error Fatal: Ejecuta desde la raíz del proyecto.")
    sys.exit(1)


# Función para cargar el modelo base, 
# detectando si es de Torchvision o Timm
def load_base_model(model_name):
    
    torchvision_models = ['resnet18', 'squeezenet1_1', 'densenet', 'shufflenet_v2_x0_5']
    
    if any(x in model_name for x in torchvision_models):
        import torchvision.models as models
        try:
            return getattr(models, model_name)(weights='DEFAULT')
        except:
            return getattr(models, model_name)(pretrained=True)
    else:
        import timm
        return timm.create_model(model_name, pretrained=True)

# Función para obtener un sub-módulo por su nombre de string (ej: 'layer1.0')
def get_module_by_name(model, name):
    modules = dict(model.named_modules())   # Crea un diccionario de todos los módulos con sus nombres
    return modules.get(name, None)          # Devuelve el módulo si existe, o None si no se encuentra    

# Función principal para generar el JSON con las formas de entrada 
# y salida de los nodos definidos en blocklize.py
def generate_json(output_path="tools/MODEL_INOUT_SHAPE.json"):
    
    print(f"🔄 INICIANDO MAPEO DE NODOS MAESTROS (blocklize)")
    print(f"📂 Salida: {output_path}")
    
    # Diccionario para almacenar los resultados finales
    new_db = {}
    
    # Iterar sobre cada modelo definido en MODEL_ZOO
    for model_name in MODEL_ZOO:
        print(f"\n{'='*40}")
        print(f"MODELO: {model_name}")
        print(f"{'='*40}")

        # Intentar cargar el modelo base, si falla, 
        # se muestra un error claro y se continúa con el siguiente modelo
        try:
            base_model = load_base_model(model_name)
            base_model.eval()
        except Exception as e:
            print(f"❌ Fallo crítico cargando modelo: {e}")
            continue
        
        # Obtener los nodos a mapear para este modelo desde MODEL_BLOCKS
        target_nodes = MODEL_BLOCKS.get(model_name, [])
        if not target_nodes:
            print("⚠️  SKIPPING: No hay nodos definidos en MODEL_BLOCKS.")
            continue
        
        print(f"📍 Nodos a mapear: {len(target_nodes)}")

        # Diccionario para almacenar las formas capturadas por los hooks
        captured_data = {}
        
        # Lista para almacenar los hooks registrados, 
        # para poder eliminarlos después
        hooks = []

        # Función para crear un hook que capture las formas de entrada y salida
        def make_hook(name):
            
            # El hook maneja diferentes tipos de salida (tensor, tupla, dict)
            def hook(module, input, output):
                in_tensor = input[0] if isinstance(input, tuple) else input
                
                # 
                if isinstance(output, tuple):
                    out_tensor = output[0]          
                elif isinstance(output, dict):
                    out_tensor = list(output.values())[0]
                else:
                    out_tensor = output
                
                if hasattr(in_tensor, 'shape') and hasattr(out_tensor, 'shape'):
                    captured_data[name] = {
                        'in': list(in_tensor.shape)[1:],
                        'out': list(out_tensor.shape)[1:]
                    }
            return hook

        # Registrar hooks para cada nodo objetivo, 
        # si el nodo no existe, se muestra una advertencia
        for node_name in target_nodes:
            module = get_module_by_name(base_model, node_name)          # Obtener el módulo por su nombre
            if module is not None:                                      # Si el módulo existe, registrar el hook        
                h = module.register_forward_hook(make_hook(node_name))  # Registrar el hook y guardarlo para eliminarlo después
                hooks.append(h)                                         # Agregar el hook a la lista de hooks registrados
            else:
                print(f"⚠️  Advertencia: El nodo '{node_name}' definido en blocklize NO existe en el modelo real.")

        if not hooks:
            print("❌ No se pudo registrar ningún hook. Revisa los nombres en blocklize.")
            continue

        # Realizar una pasada de inferencia con un tensor de entrada 
        # dummy para activar los hooks
        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                base_model(dummy_input)     # Ejecutar el modelo para activar los hooks y capturar las formas
            print("✅ Forward pass completado.")
        except Exception as e:
            print(f"❌ Error durante la ejecución: {e}")

        # Eliminar los hooks registrados para evitar efectos secundarios en futuras ejecuciones 
        for h in hooks: h.remove()

        # Construir la entrada para el nuevo JSON con las formas capturadas,
        model_entry = {"in_size": {}, "out_size": {}}
        success_count = 0
        
        # Iterar sobre los nodos objetivo y llenar el nuevo 
        # JSON con las formas capturadas
        for node_name in target_nodes:          # Para cada nodo objetivo, verificar si se capturaron datos
            if node_name in captured_data:
                model_entry["in_size"][node_name] = captured_data[node_name]['in']          # Llenar la forma de entrada en el nuevo JSON
                model_entry["out_size"][node_name] = captured_data[node_name]['out']        # Llenar la forma de salida en el nuevo JSON
                success_count += 1
            else:
                print(f"❌ {node_name}: No capturó datos (¿Código muerto o saltado?)")

        # Agregar la entrada del modelo al nuevo JSON
        new_db[model_name] = model_entry
        print(f"📊 Capturados {success_count}/{len(target_nodes)} nodos correctamente.")

    # Guardar el nuevo JSON con las formas capturadas en el archivo de salida
    with open(output_path, 'w') as f:
        json.dump(new_db, f, indent=4)
    print(f"\n💾 ¡Mapeo Finalizado! Archivo guardado en: {output_path}")

if __name__ == "__main__":
    generate_json()