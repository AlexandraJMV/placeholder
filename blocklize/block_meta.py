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
    'shufflenet_v2_x0_5',
    'efficientnet_b0'
]

# 2. LISTA ORIGINAL
"""
# Lista de modelos activos (presentes en la lista original) para la experimentación
MODEL_ZOO = [
            'resnet50', 
            'resnet101', 
            'resnet18',
            'swsl_resnext50_32x4d', 
            'mobilenetv3_large_100',
            ]
"""

# Diccionario que mapea cada modelo a sus "atomic nodes"
# Un bloque, en DeRy, es definido como un stack continuO de estos nodos atómicos
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

    'shufflenet_v2_x0_5': [
            'stage2.0', 'stage2.1', 'stage2.2', 'stage2.3',
            'stage3.0', 'stage3.1', 'stage3.2', 'stage3.3', 'stage3.4', 'stage3.5', 'stage3.6', 'stage3.7',
            'stage4.0', 'stage4.1', 'stage4.2', 'stage4.3'
        ],
    
    'efficientnet_b0': [
        'blocks.0.0', 
        'blocks.1.0', 'blocks.1.1',
        'blocks.2.0', 'blocks.2.1',
        'blocks.3.0', 'blocks.3.1', 'blocks.3.2',
        'blocks.4.0', 'blocks.4.1', 'blocks.4.2',
        'blocks.5.0', 'blocks.5.1', 'blocks.5.2', 'blocks.5.3',
        'blocks.6.0'
    ]
}

# Diccionario que mapea cada modelo a su nombre legible
# Se usa simplemente para imprimir nombres en los logs y gráficas de manera más amigable
MODEL_PRINT = {
    'resnet18': 'ResNet-18',
    'shufflenet_v2_x0_5': 'ShuffleNetV2 (0.5x)',
    'mobilenetv3_small_050': 'MobileNetV3 Small (0.5x)',
    'efficientnet_b0': 'EfficientNet-B0',
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
    'shufflenet_v2_x0_5': dict(
        arch='shufflenet_v2_x0_5', 
        top1=60.55, 
        param=1.36, 
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
}

try:
    with open('tools/MODEL_INOUT_SHAPE.json') as json_file:
        MODEL_INOUT_SHAPE = json.load(json_file)
except:
    MODEL_INOUT_SHAPE = None
    print('tools/MODEL_INOUT_SHAPE.json not loaded')
