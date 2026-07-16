import json

"""
    Módulo que contiene metadata sobre los modelos soportados, sus estructuras de bloques, 
    frameworks asociados, y estadísticas de referencia.
    
    Aquí se definen:
    - MODEL_ZOO: Lista de modelos soportados.
    - MODEL_BLOCKS: Diccionario que mapea cada modelo a sus "atomic nodes".
    - MODEL_PRINT: Diccionario que mapea cada modelo a su nombre legible.
    - MODEL_STATS: Diccionario que contiene metadatos funcionales y estadísticas de referencia.
    
    Aquí se modifica para incluir nuevos modelos o actualizar los existentes.
    No olvidar: Lo modelos deben funcionar dentro de la infraestructura DeRy.

"""
# 1. LISTA PoC 
MODEL_ZOO = [
    'resnet18',
    'mobilenetv3_small_050',
    'efficientnet_b0',
    # Nuevos modelos añadidos para PoC
    #'swin_tiny_patch4_window7_224',
    'resnet50',
    'convnext_tiny',
]

# Diccionario que mapea cada modelo a sus "atomic nodes"
# Un bloque,en DeRy, es definido como un stack continuO de estos nodos atómicos
# Version
MODEL_BLOCKS = {
    'resnet18': ['layer1.0', 'layer1.1', 'layer2.0', 'layer2.1',
                 'layer3.0', 'layer3.1', 'layer4.0', 'layer4.1'],
    
    'mobilenetv3_small_050': [
        'blocks.0.0', 
        'blocks.1.0', 'blocks.1.1',
        'blocks.2.0', 'blocks.2.1', 'blocks.2.2',
        'blocks.3.0', 'blocks.3.1',
        'blocks.4.0', 'blocks.4.1', 'blocks.4.2',
        'blocks.5.0'
    ],
    
    'efficientnet_b0': [
        'blocks.0.0', 
        'blocks.1.0', 'blocks.1.1',
        'blocks.2.0', 'blocks.2.1',
        'blocks.3.0', 'blocks.3.1', 'blocks.3.2',
        'blocks.4.0', 'blocks.4.1', 'blocks.4.2',
        'blocks.5.0', 'blocks.5.1', 'blocks.5.2', 'blocks.5.3',
        'blocks.6.0'
    ],
    'resnet50': [
        'layer1.0', 'layer1.1', 'layer1.2',
        'layer2.0', 'layer2.1', 'layer2.2', 'layer2.3',
        'layer3.0', 'layer3.1', 'layer3.2', 'layer3.3', 'layer3.4', 'layer3.5',
        'layer4.0', 'layer4.1', 'layer4.2',
    ],

    'convnext_tiny': [
        'stages.0.blocks.0', 'stages.0.blocks.1', 'stages.0.blocks.2',
        'stages.1.blocks.0', 'stages.1.blocks.1', 'stages.1.blocks.2',
        'stages.2.blocks.0', 'stages.2.blocks.1', 'stages.2.blocks.2',
        'stages.2.blocks.3', 'stages.2.blocks.4', 'stages.2.blocks.5',
        'stages.2.blocks.6', 'stages.2.blocks.7', 'stages.2.blocks.8',
        'stages.3.blocks.0', 'stages.3.blocks.1', 'stages.3.blocks.2',  
    ],
    
}

# Diccionario que mapea cada modelo a su nombre legible
# Se usa simplemente para imprimir nombres en los logs y gráficas de manera más amigable
MODEL_PRINT = {
    'resnet18': 'ResNet-18',
    'mobilenetv3_small_050': 'MobileNetV3 Small (0.5x)',
    'efficientnet_b0': 'EfficientNet-B0',
    'convnext_tiny': 'ConvNeXt-Tiny',
    'resnet50': 'ResNet-50',
}

# Contiene metadatos funcionales que el sistema necesita para operar y 
# estadísticas de referencia para reportes
MODEL_STATS = {
    # --- Clásicos (Backend PyTorch) ---
    'resnet18': dict(
        arch='resnet18', 
        top1=69.76, 
        param=11.69, 
        backend='pytorch', 
        type='cnn'
    ),

    # --- Modernos (Backend TIMM) ---
    'mobilenetv3_small_050': dict(
        arch='mobilenetv3_small_050', 
        top1=57.9, 
        param=1.65, 
        backend='mytimm', 
        type='cnn'
    ),
    'efficientnet_b0': dict(
        arch='efficientnet_b0', 
        top1=77.1, 
        param=5.29, 
        backend='mytimm', 
        type='cnn'
    ),
    
    # Nuevos modelos añadidos para PoC
    'convnext_tiny': dict(
        arch='convnext_tiny.fb_in1k',   # tag timm verificado, entrenado solo en in1k
        top1=82.1, param=28.59, backend='mytimm', type='cnn'
    ),
    'resnet50': dict(
        arch='resnet50',
        top1=80.9,
        param=25.56,
        backend='pytorch',
        type='cnn'
    ),
}

try:
    with open('tools/MODEL_INOUT_SHAPE.json') as json_file:
        MODEL_INOUT_SHAPE = json.load(json_file)
except:
    MODEL_INOUT_SHAPE = None
    print('tools/MODEL_INOUT_SHAPE.json not loaded')
