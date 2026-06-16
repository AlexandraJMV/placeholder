import timm
import torch.nn as nn

model_name = 'resnet18' 
print(f"--- Inspeccionando: {model_name} ---")

try:
    model = timm.create_model(model_name, pretrained=False)
    print("Loaded from timm")
except:
    import torchvision.models as models
    model = getattr(models, model_name)()
    print("Loaded from torchvision.models")


seen_blocks = set()
for name, module in model.named_modules():
    parts = name.split('.')
    
    if 2 <= len(parts) <= 3:
        if not any(x in parts[-1] for x in ['conv', 'bn', 'act', 'pool', 'fc', 'classifier']):
            print(f"'{name}',") 
print("\n--- Fin de la lista ---")