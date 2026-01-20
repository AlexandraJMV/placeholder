# simple_poc/poc_config.py
import os
import torch

# --- 1. CONFIGURACIÓN DE HARDWARE (RTX 3050) ---
BATCH_SIZE = 32         # 32 es seguro para 4GB/8GB VRAM con imágenes de 224x224
NUM_WORKERS = 0         # En Windows, poner 0 evita errores de multiproceso
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

# --- 2. RUTAS ---
# Directorio raíz del proyecto
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Dónde guardar los features (.pth)
FEAT_DIR = os.path.join(ROOT_DIR, 'mis_features')

# Dónde guardar el archivo final de formas (.json)
JSON_OUTPUT_DIR = os.path.join(ROOT_DIR, 'tools')
JSON_OUTPUT_PATH = os.path.join(JSON_OUTPUT_DIR, 'MODEL_INOUT_SHAPE.json')

# Ruta de tus datos (CIFAR10)
# DeRy descarga CIFAR automáticamente si usas torchvision, así que definimos donde guardarlo
DATASET_ROOT = os.path.join(ROOT_DIR, 'data') 

# --- 3. PARAMETROS DE EXTACCIÓN ---
IMG_SIZE = 224          # CRÍTICO: DeRy necesita 224x224 para funcionar
MAX_BATCHES = 10        # Solo necesitamos unas pocas imágenes para medir formas y features