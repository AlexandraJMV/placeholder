import argparse
import os
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import timm
from tqdm import tqdm

# Importa la definición de MODEL_BLOCKS desde block_meta.py
from blocklize.block_meta import MODEL_BLOCKS

# Define la clase GetFeatureHook para capturar las 
# características de las capas objetivo durante la inferencia
class GetFeatureHook:
    def __init__(self, module):
        # Registra un hook de avance en el módulo dado
        self.hook = module.register_forward_hook(self.hook_fn)
        
        # Inicializa una lista para almacenar las características capturadas, 
        # así como variables para los tamaños de entrada y salida
        self.feature = []
        self.in_size = None
        self.out_size = None

    def hook_fn(self, module, input, output):
        # Si los tamaños de entrada y salida no se han establecido, 
        # los obtiene del primer paso del hook
        if self.in_size is None:
            self.in_size = input[0].shape[1:] if isinstance(input, tuple) else input.shape[1:]
            self.out_size = output[0].shape[1:] if isinstance(output, tuple) else output.shape[1:]

        if isinstance(output, tuple):
            output = output[0]

        # Dependiendo de la forma de la salida, aplica un pooling adaptativo 
        # o un promedio para obtener una representación fija
        if len(output.shape) == 4:
            feat = F.adaptive_avg_pool2d(output, (1, 1))
        elif len(output.shape) == 3:
            feat = output.mean(dim=1, keepdim=True).transpose(1, 2)
        else:
            feat = output

        # Almacena las características capturadas, moviéndolas a la CPU y desconectándolas 
        # del grafo de cómputo para ahorrar memoria
        self.feature.append(feat.detach().cpu())

    # Método para concatenar todas las características capturadas en un solo tensor
    def concat(self):
        return torch.cat(self.feature, dim=0) if self.feature else torch.tensor([])
     # Método para eliminar el hook registrado, liberando recursos
    def close(self):
        self.hook.remove()

# Función para cargar el modelo, con manejo de excepciones 
# para modelos no encontrados en timm
def get_model(model_name, device):
    try:
        # Primero intentamos cargar el modelo desde timm
        model = timm.create_model(model_name, pretrained=True)
    except:
        # Si no se encuentra en timm, intentamos cargarlo desde torchvision
        import torchvision.models as models
        model = getattr(models, model_name)(weights='DEFAULT')
    return model.to(device).eval()

# Función para procesar un solo modelo, 
# encapsulando toda la lógica de extracción y guardado
def process_single_model(model_name, dataloader, device, args):
    
    print(f"\n Procesando modelo: {model_name}")
    
    if model_name not in MODEL_BLOCKS:
        print(f"Saltando {model_name}: No definido en MODEL_BLOCKS")
        return

    # Carga el modelo
    model = get_model(model_name, device)
    
    # Obtiene las capas objetivo para el modelo actual desde MODEL_BLOCKS
    target_blocks = MODEL_BLOCKS[model_name]
    
    # Diccionario para almacenar los hooks y sus características
    hooks = {}

    # Configura los hooks para las capas objetivo
    for name, module in model.named_modules():          # Recorre todas las capas del modelo
        if name in target_blocks:                       # Si el nombre de la capa está en los bloques objetivo  
            hooks[name] = GetFeatureHook(module)        # Crea un hook para esa capa y lo almacena en el diccionario

    # Itera sobre el dataloader y pasa las imágenes a través del 
    # modelo para activar los hooks y capturar las características        
    with torch.no_grad():
        for imgs, _ in tqdm(dataloader, desc=f"Inferencia {model_name}"):       # Itera sobre el dataloader con una barra de progreso
            model(imgs.to(device))                                              # Pasa las imágenes a través del modelo para activar los hooks

    # Después de procesar el dataset, concatena las características capturadas por 
    # cada hook y las almacena en un diccionario junto con sus tamaños de entrada y salida
    feat_dict = {}
    size_dict = {}
    
    for block_name, hook in hooks.items():
        concatenated_features = hook.concat()                                                   # Concatena las características capturadas por el hook
        feat_dict[block_name] = concatenated_features.view(concatenated_features.size(0), -1)   # Aplana las características a 2D (batch_size, feature_dim)
        size_dict[block_name] = [list(hook.in_size), list(hook.out_size)]                       # Almacena los tamaños de entrada y salida de la capa en el diccionario size_dict
        hook.close()                                                                            # Elimina el hook para liberar recursos

    # Guarda los resultados en un archivo .pth, incluyendo el nombre del modelo, los tamaños de las capas y las características extraídas
    results = {'model_name': model_name, 'size': size_dict, 'feat': feat_dict}
    output_path = Path(args.save_dir) / f"{model_name}.pth"                         # Define la ruta de salida para el archivo .pth, utilizando el nombre del modelo
    torch.save(results, output_path)                                                # Guarda el diccionario de resultados en un archivo .pth en la ruta especificada
    print(f"✅ Guardado en: {output_path}")                                         

def main():
    # Configuración de argumentos con argparse
    parser = argparse.ArgumentParser(description='Extracción de representaciones (Zoo completo)')
    
    # Permite especificar un modelo específico o procesar todo el Zoo si se deja como None
    parser.add_argument('--model', type=str, default=None, help='Modelo específico o None para todo el Zoo')
    
    # Configuración de dataset y parámetros de procesamiento
    parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'imagenet', 'imagenette'])
    # Configuración de ruta, tamaño de batch, directorio de guardado y tamaño de imagen
    parser.add_argument('--data_path', type=str, default='./data')     # Ruta base para los datos, se ajustará según el dataset seleccionado
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--save_dir', type=str, default='reps_folder')
    parser.add_argument('--img_size', type=int, default=160)            # Tamaño de imagen para modelos que requieren entrada de 224x224, se ajustará según el modelo seleccionado
    
    # Analiza los argumentos proporcionados por el usuario
    args = parser.parse_args()

    # Configura el dispositivo (GPU si está disponible, de lo contrario CPU)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Crea el directorio de guardado si no existe
    os.makedirs(args.save_dir, exist_ok=True)

    # Configura las transformaciones de imagen, ajustando el tamaño según el modelo seleccionado
    transform = transforms.Compose([
        transforms.Resize((args.img_size, args.img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),  # Normalización estándar para modelos preentrenados en ImageNet
    ])

    # Carga el dataset según la selección del usuario, ajustando la ruta y las transformaciones
    
    # Carga el dataset según la selección del usuario, ajustando la ruta y las transformaciones
    if args.dataset == 'cifar10':
        dataset = datasets.CIFAR10(root=args.data_path, train=False, download=True, transform=transform)
    elif args.dataset in ['imagenet', 'imagenette']:
        # Imagenette follows the standard ImageNet structure; features are extracted using the validation split.
        dataset = datasets.ImageFolder(root=os.path.join(args.data_path, 'val'), transform=transform)
    else:
        raise ValueError(f"Dataset configuration for '{args.dataset}' is undefined.")    

    # Configura el DataLoader para iterar sobre el dataset con el tamaño de batch especificado
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    # Lógica de selección de modelos
    models_to_process = [args.model] if args.model else list(MODEL_BLOCKS.keys()) # Si se especifica un modelo, procesa solo ese; de lo contrario, procesa todos los modelos definidos en MODEL_BLOCKS

    # Itera sobre los modelos seleccionados y procesa cada uno utilizando la función process_single_model
    for m_name in models_to_process:
        process_single_model(m_name, dataloader, device, args)

if __name__ == '__main__':
    main()