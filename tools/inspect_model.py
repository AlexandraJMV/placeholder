import timm
import torch.nn as nn

# 1. Elige el modelo que quieres investigar
model_name = 'efficientnet_b0'  # Cambia esto por 'squeezenet1_1', 'mobilenetv3_small_050', etc.

print(f"--- Inspeccionando: {model_name} ---")

# 2. Cargar modelo
try:
    model = timm.create_model(model_name, pretrained=False)
    print("Loaded from timm")
except:
    # Fallback si no está en timm, intentar torchvision standard (raro, timm tiene casi todo)
    import torchvision.models as models
    model = getattr(models, model_name)()
    print("Loaded from torchvision.models")


# 3. Imprimir solo los bloques principales
# El truco es no imprimir todo (conv2d, relu), sino los "contenedores".
seen_blocks = set()

for name, module in model.named_modules():
    # Filtro heurístico: Buscamos nombres que parezcan bloques (blocks, features, layer, stage)
    # y que tengan cierta profundidad (x.y o x.y.z)
    parts = name.split('.')
    
    # Si el nombre es muy largo (ej: blocks.1.0.conv1.bn), es demasiado profundo. Ignorar.
    # Si es muy corto (ej: blocks), es demasiado general. Ignorar.
    if 2 <= len(parts) <= 3:
        # Evitar imprimir componentes internos como 'conv1', 'bn1'
        if not any(x in parts[-1] for x in ['conv', 'bn', 'act', 'pool', 'fc', 'classifier']):
            print(f"'{name}',") 

print("\n--- Fin de la lista ---")