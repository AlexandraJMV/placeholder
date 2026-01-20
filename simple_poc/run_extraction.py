import sys
import os
import json
import torch
import torch.nn as nn
import timm
from torchvision import datasets, transforms, models
from tqdm import tqdm

# Hack para importar módulos del directorio padre
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from blocklize.block_meta import MODEL_ZOO, MODEL_BLOCKS, MODEL_STATS
import simple_poc.poc_config as cfg

# --- CLASE HOOK MEJORADA ---
class FeatureHook:
    """Captura Features y Dimensiones (Input y Output)"""
    def __init__(self, name):
        self.name = name
        self.output = None
        self.in_shape = None  # Lista [C, H, W]
        self.out_shape = None # Lista [C, H, W]

    def hook_fn(self, module, input, output):
        # input es una tupla (tensor, ...). Generalmente input[0] es lo que buscamos.
        if self.in_shape is None:
            x = input[0]
            # Guardamos la forma excluyendo el Batch (índice 0)
            self.in_shape = list(x.shape[1:])
        
        if self.out_shape is None:
            # Guardamos la forma excluyendo el Batch
            self.out_shape = list(output.shape[1:])
            
        # Guardamos features desconectados para guardar en disco después
        self.output = output.detach().cpu()

def get_layer_by_name(model, name):
    parts = name.split('.')
    current = model
    for part in parts:
        if part.isdigit():
            current = current[int(part)]
        else:
            current = getattr(current, part)
    return current

def load_model(model_name):
    stats = MODEL_STATS[model_name]
    print(f"   Cargando {model_name} (Backend: {stats['backend']})...")
    
    if stats['backend'] == 'mytimm' or stats['backend'] == 'timm':
        model = timm.create_model(model_name, pretrained=True)
    else:
        try:
            # Versiones nuevas de Torchvision
            weights = getattr(models, f"{stats['arch'].title().replace('_','')}Weights").DEFAULT
            model = getattr(models, stats['arch'])(weights=weights)
        except:
            # Versiones viejas
            model = getattr(models, stats['arch'])(pretrained=True)
    
    return model.to(cfg.DEVICE).eval()

# --- MAIN ---
def run():
    print(f">>> INICIANDO EXTRACCIÓN (CORREGIDO) EN {cfg.DEVICE} <<<")
    
    if not os.path.exists(cfg.FEAT_DIR): os.makedirs(cfg.FEAT_DIR)
    if not os.path.exists(cfg.JSON_OUTPUT_DIR): os.makedirs(cfg.JSON_OUTPUT_DIR)

    # Preparar Datos (Resize es vital para medir las formas correctamente)
    transform = transforms.Compose([
        transforms.Resize((cfg.IMG_SIZE, cfg.IMG_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    # Usamos un dataset dummy si CIFAR no está descargado, o CIFAR si sí.
    # Para asegurar que funcione, forzamos CIFAR download.
    dataset = datasets.CIFAR10(root=cfg.DATASET_ROOT, train=False, download=True, transform=transform)
    loader = torch.utils.data.DataLoader(dataset, batch_size=cfg.BATCH_SIZE, shuffle=False, num_workers=cfg.NUM_WORKERS)

    # Estructura final del JSON
    json_data = {} 

    for model_name in MODEL_ZOO:
        print(f"\n--- Procesando: {model_name} ---")
        
        # Inicializar estructura del modelo en el JSON
        json_data[model_name] = {
            "in_size": {},
            "out_size": {}
        }

        # 1. Cargar Modelo
        try:
            model = load_model(model_name)
        except Exception as e:
            print(f"   [ERROR] No se pudo cargar {model_name}: {e}")
            continue

        # 2. Registrar Hooks
        blocks = MODEL_BLOCKS[model_name]
        hooks = []
        for block_name in blocks:
            try:
                layer = get_layer_by_name(model, block_name)
                h = FeatureHook(block_name)
                handle = layer.register_forward_hook(h.hook_fn)
                hooks.append({'handler': handle, 'obj': h, 'name': block_name})
            except AttributeError:
                print(f"   [ALERTA] Bloque no encontrado: {block_name}")

        # 3. Inferencia (Para capturar formas y features)
        all_features = {b: [] for b in blocks}
        
        print("   Midiendo formas y extrayendo features...")
        with torch.no_grad():
            for i, (images, _) in enumerate(tqdm(loader)):
                if i >= cfg.MAX_BATCHES: break
                
                images = images.to(cfg.DEVICE)
                model(images)
                
                for h in hooks:
                    # Guardar features para el .pth
                    all_features[h['name']].append(h['obj'].output)
                    
                    # Guardar formas en el diccionario JSON (Solo necesitamos hacerlo una vez)
                    # Sobrescribimos en cada iteración, no importa, es el mismo valor.
                    if h['obj'].in_shape and h['obj'].out_shape:
                        json_data[model_name]["in_size"][h['name']] = h['obj'].in_shape
                        json_data[model_name]["out_size"][h['name']] = h['obj'].out_shape

        # 4. Guardar .pth (Features)
        save_dict = {'size': {}, 'feat': {}}
        for b_name in blocks:
            if not all_features[b_name]: continue

            # Concatenar batches
            feat_tensor = torch.cat(all_features[b_name], dim=0)
            
            # Sub-sampling si es muy grande (Opcional, ahorra espacio)
            if feat_tensor.size(0) > 1000: feat_tensor = feat_tensor[:1000]
            
            save_dict['feat'][b_name] = feat_tensor
            
            # El .pth también necesita una clave 'size', aunque sea redundante con el JSON
            # Formato esperado en .pth: {block: (in_shape_tuple, out_shape_tuple)}
            # OJO: Aquí sí se mantiene el formato tupla, el JSON requiere listas.
            if b_name in json_data[model_name]["in_size"]:
                s_in = tuple(json_data[model_name]["in_size"][b_name])
                s_out = tuple(json_data[model_name]["out_size"][b_name])
                save_dict['size'][b_name] = (s_in, s_out)

        pth_path = os.path.join(cfg.FEAT_DIR, f"{model_name}.pth")
        torch.save(save_dict, pth_path)
        print(f"   Features guardados en: {pth_path}")

        # Limpieza
        del model
        for h in hooks: h['handler'].remove()
        torch.cuda.empty_cache()

    # 5. Guardar JSON FINAL
    print("\n>>> Guardando tools/MODEL_INOUT_SHAPE.json <<<")
    with open(cfg.JSON_OUTPUT_PATH, 'w') as f:
        json.dump(json_data, f, indent=4)
    
    print(f"¡ÉXITO! JSON generado correctamente en: {cfg.JSON_OUTPUT_PATH}")

if __name__ == '__main__':
    run()