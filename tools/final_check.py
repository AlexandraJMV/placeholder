import sys
import os
import torch
import torch.nn as nn
import pickle

# --- SETUP ---
sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

# Importamos tu clase arreglada
from simple_poc.supernet import SuperNetwork

def check_supernet():
    print("🏥 INICIANDO DIAGNÓSTICO PROFUNDO DE LA SUPERRED (VERSIÓN CORREGIDA)...\n")
    
    pkl_path = "network_plan.pkl"
    if not os.path.exists(pkl_path):
        print("❌ Error: No existe network_plan.pkl")
        return

    # 1. CARGA
    try:
        model = SuperNetwork(pkl_path, num_classes=10)
        model.train()
        print("✅ SuperNetwork instanciada correctamente.")
    except Exception as e:
        print(f"❌ Falló la instanciación: {e}")
        return

    # 2. PRUEBA DE CAMINOS ALEATORIOS
    print("\n🎲 PRUEBA DE MULTI-PATH (5 Caminos Aleatorios):")
    dummy_input = torch.randn(2, 3, 224, 224)
    try:
        for i in range(5):
            out = model(dummy_input)
            print(f"   Intento {i+1}: Output Shape {out.shape} -> ✅ OK")
    except Exception as e:
        print(f"   ❌ Falló en caminos aleatorios: {e}")
        return

    # 3. PRUEBA DE VIDA (Backward Pass - CAMINO FORZADO)
    print("\n⚡ PRUEBA DE VIDA (Backward Pass - Camino Fijo [0,0,0,0]):")
    print("   (Forzando tráfico por la primera opción de cada etapa para verificar gradientes)")
    
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()
    
    inputs = torch.randn(4, 3, 224, 224)
    targets = torch.randint(0, 10, (4,))
    
    # FORZAMOS EL CAMINO 0
    path_zero = [0] * model.num_stages 
    
    try:
        optimizer.zero_grad()
        # Pasamos el path explícito para saber QUÉ revisar
        outputs = model(inputs, path=path_zero)
        loss = criterion(outputs, targets)
        loss.backward()
        
        # Revisamos explícitamente los parámetros del BLOQUE QUE USAMOS (Stage 0, Opción 0)
        # No usamos model.parameters() genérico porque podría darnos uno no usado.
        target_block = model.stages[0][0]
        found_grad = False
        
        for name, param in target_block.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                print(f"   ✅ ¡ÉXITO! Gradiente detectado en {name} (Norma: {grad_norm:.6f})")
                found_grad = True
                break # Con encontrar uno basta
        
        if found_grad:
            print("   🚀 CONCLUSIÓN: La red está VIVA y los cables transmiten aprendizaje.")
        else:
            print("   ❌ ERROR CRÍTICO REAL: Se forzó el paso por el bloque 0 pero no llegaron gradientes.")
            
        optimizer.step()
        
    except Exception as e:
        print(f"   ❌ Falló el Backward Pass: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_supernet()