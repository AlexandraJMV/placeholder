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

FEAT_DIR = os.path.join(ROOT_DIR, 'mis_features')

JSON_OUTPUT_DIR = os.path.join(ROOT_DIR, 'tools')
JSON_OUTPUT_PATH = os.path.join(JSON_OUTPUT_DIR, 'MODEL_INOUT_SHAPE.json')

DATASET_ROOT = os.path.join(ROOT_DIR, 'data') 

IMG_SIZE = 224          
MAX_BATCHES = 10        