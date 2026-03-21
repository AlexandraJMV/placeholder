import sys
import os
import json
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

try:
    from blocklize import MODEL_BLOCKS, MODEL_ZOO
    from simlarity.feature_extraction import create_sub_network
except ImportError:
    print("❌ Error Fatal: No se pudo importar 'blocklize' o 'simlarity'. Ejecuta desde la raíz.")
    sys.exit(1)

def load_base_model(model_name):
    if any(x in model_name for x in ['resnet', 'squeezenet', 'densenet']):
        import torchvision.models as models
        try:
            return getattr(models, model_name)(weights='DEFAULT')
        except:
            return getattr(models, model_name)(pretrained=True)
    else:
        import timm
        return timm.create_model(model_name, pretrained=True)

def deep_verify(json_path):
    print(f"🔬 INICIANDO VERIFICACIÓN PROFUNDA DE FORMAS Y FÍSICA\n")
    print(f"Archivo: {json_path}")
    
    if not os.path.exists(json_path):
        print("❌ El archivo JSON no existe.")
        return

    with open(json_path, 'r') as f:
        shapes_db = json.load(f)

    for model_name in MODEL_ZOO:
        print(f"\n{'='*60}")
        print(f"MODELO: {model_name}")
        print(f"{'='*60}")
        
        if model_name not in shapes_db:
            print(f"❌ MISSING: No hay datos en el JSON para este modelo.")
            continue
            
        try:
            base_model = load_base_model(model_name)
        except Exception as e:
            print(f"⚠️  SKIPPING: No se pudo instanciar el modelo base ({e})")
            continue
            
        node_names = MODEL_BLOCKS.get(model_name, [])
        model_data = shapes_db[model_name]
        
        if not node_names:
            print("⚠️  No hay nodos definidos en MODEL_BLOCKS.")
            continue

        prev_out_shape = None
        issues_count = 0

        for i, node_name in enumerate(node_names):
            if node_name not in model_data['in_size'] or node_name not in model_data['out_size']:
                print(f"   [Node {i}: {node_name}] ❌ FALTA DATA en JSON")
                issues_count += 1
                continue
                
            in_shape_raw = model_data['in_size'][node_name]
            out_shape_raw = model_data['out_size'][node_name]
            
            def parse_shape(s):
                if len(s) == 4: return s[1], s[2], s[3] # C, H, W
                if len(s) == 3: return s[0], s[1], s[2]
                return s
            
            in_c, in_h, in_w = parse_shape(in_shape_raw)
            out_c, out_h, out_w = parse_shape(out_shape_raw)

            if prev_out_shape is not None:
                pc, ph, pw = prev_out_shape
                if (pc != in_c) or (ph != in_h) or (pw != in_w):
                    if pc != in_c:
                         print(f"   [Node {i}: {node_name}] ⚠️  DISCONTINUIDAD: Anterior sacó {pc} ch, este espera {in_c} ch.")
                         issues_count += 1
            
            try:
                input_node = node_names[i-1] if i > 0 else None
                input_args = [input_node] if input_node else []
                
                sub_model = create_sub_network(base_model, input_args, [node_name])
                
                dummy_in = torch.randn(1, in_c, in_h, in_w)
                
                with torch.no_grad():
                    dummy_out = sub_model(dummy_in)
                
                real_out_c = dummy_out.shape[1]
                real_out_h = dummy_out.shape[2]
                
                if real_out_c != out_c:
                    print(f"   [Node {i}: {node_name}] ❌ ERROR DE SALIDA: JSON dice {out_c} ch, Realidad sacó {real_out_c} ch.")
                    issues_count += 1
                elif real_out_h != out_h:
                    print(f"   [Node {i}: {node_name}] ⚠️  Info: Diferencia de resolución (JSON: {out_h}, Real: {real_out_h})")
                else:
                    if i == 0:
                         print(f"   [Node 0: {node_name}] ✅ INICIO CORRECTO (Física OK)")

            except RuntimeError as e:
                msg = str(e)
                if "channels" in msg:
                    print(f"   [Node {i}: {node_name}] ⛔ ERROR FÍSICO GRAVE: El JSON dice entrada {in_c} canales, pero el modelo rechazó ese tensor.")
                    print(f"       -> Mensaje PyTorch: {msg}")
                    issues_count += 1
                else:
                    print(f"   [Node {i}: {node_name}] ❌ Fallo de ejecución: {e}")
                    issues_count += 1
            except Exception as e:
                 print(f"   [Node {i}: {node_name}] ❌ Error inesperado: {e}")

            prev_out_shape = (out_c, out_h, out_w)

        if issues_count == 0:
            print(f"   ✅ Modelo {model_name}: 100% Consistente.")
        else:
            print(f"   🚫 Modelo {model_name}: Se encontraron {issues_count} problemas.")

if __name__ == "__main__":
    json_path = "tools/MODEL_INOUT_SHAPE.json"
    deep_verify(json_path)