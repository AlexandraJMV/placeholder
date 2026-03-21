import sys
import os
import json
import torch
import torch.nn as nn

sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

try:
    from blocklize import MODEL_BLOCKS, MODEL_ZOO
except ImportError:
    print("❌ Error Fatal: Ejecuta desde la raíz del proyecto.")
    sys.exit(1)

def load_base_model(model_name):
    """Carga el modelo base adecuado (Torchvision o Timm)"""
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

def get_module_by_name(model, name):
    """Busca un sub-módulo dentro del modelo usando su nombre de string (ej: 'layer1.0')"""
    modules = dict(model.named_modules())
    return modules.get(name, None)

def generate_json(output_path="tools/MODEL_INOUT_SHAPE.json"):
    print(f"🔄 INICIANDO MAPEO DE NODOS MAESTROS (blocklize)")
    print(f"📂 Salida: {output_path}")
    
    new_db = {}
    
    for model_name in MODEL_ZOO:
        print(f"\n{'='*40}")
        print(f"MODELO: {model_name}")
        print(f"{'='*40}")

        try:
            base_model = load_base_model(model_name)
            base_model.eval()
        except Exception as e:
            print(f"❌ Fallo crítico cargando modelo: {e}")
            continue

        target_nodes = MODEL_BLOCKS.get(model_name, [])
        if not target_nodes:
            print("⚠️  SKIPPING: No hay nodos definidos en MODEL_BLOCKS.")
            continue

        print(f"📍 Nodos a mapear: {len(target_nodes)}")

        captured_data = {}
        hooks = []

        def make_hook(name):
            def hook(module, input, output):
                in_tensor = input[0] if isinstance(input, tuple) else input
                
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

        for node_name in target_nodes:
            module = get_module_by_name(base_model, node_name)
            if module is not None:
                h = module.register_forward_hook(make_hook(node_name))
                hooks.append(h)
            else:
                print(f"⚠️  Advertencia: El nodo '{node_name}' definido en blocklize NO existe en el modelo real.")

        if not hooks:
            print("❌ No se pudo registrar ningún hook. Revisa los nombres en blocklize.")
            continue

        try:
            dummy_input = torch.randn(1, 3, 224, 224)
            with torch.no_grad():
                base_model(dummy_input)
            print("✅ Forward pass completado.")
        except Exception as e:
            print(f"❌ Error durante la ejecución: {e}")

        for h in hooks: h.remove()

        model_entry = {"in_size": {}, "out_size": {}}
        success_count = 0
        
        for node_name in target_nodes:
            if node_name in captured_data:
                model_entry["in_size"][node_name] = captured_data[node_name]['in']
                model_entry["out_size"][node_name] = captured_data[node_name]['out']
                success_count += 1
            else:
                print(f"❌ {node_name}: No capturó datos (¿Código muerto o saltado?)")

        new_db[model_name] = model_entry
        print(f"📊 Capturados {success_count}/{len(target_nodes)} nodos correctamente.")

    with open(output_path, 'w') as f:
        json.dump(new_db, f, indent=4)
    print(f"\n💾 ¡Mapeo Finalizado! Archivo guardado en: {output_path}")

if __name__ == "__main__":
    generate_json()