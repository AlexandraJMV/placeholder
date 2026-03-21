import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse

sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

from simple_poc.supernet import SuperNetwork

def parse_args():
    parser = argparse.ArgumentParser(description="Entrenar SuperRed One-Shot")
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=0.01) 
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--init_mode', type=str, default='ls', choices=['ls', 'random'])
    parser.add_argument('--matrices_path', type=str, default=None)
    
    parser.add_argument('--resume', type=str, default=None, help="Ruta al archivo .pth para cargar pesos")
    
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Optimizaciones de CUDA
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"🚀 Iniciando entrenamiento en {DEVICE} | Init: {args.init_mode} | Batch: {args.batch_size}")

    # 1. DATASET
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=2)

    # 2. MODELO
    pkl_path = "network_plan.pkl"
    if not os.path.exists(pkl_path):
        raise FileNotFoundError("Falta network_plan.pkl")
    
    model = SuperNetwork(
        plan_path=pkl_path, 
        num_classes=10, 
        input_size=224,
        stitch_init_mode=args.init_mode,
        matrices_path=args.matrices_path
    )
    model = model.to(DEVICE)

    if args.resume:
        if os.path.isfile(args.resume):
            print(f"📥 Cargando pesos desde: {args.resume}")
            try:
                checkpoint = torch.load(args.resume, map_location=DEVICE)
                if 'state_dict' in checkpoint:
                    model.load_state_dict(checkpoint['state_dict'])
                else:
                    model.load_state_dict(checkpoint)
                print("✅ Pesos cargados exitosamente. Continuando entrenamiento...")
            except Exception as e:
                print(f"❌ Error cargando checkpoint: {e}")
                return
        else:
            print(f"⚠️ No se encontró el archivo: {args.resume}")
    # -------------------------------

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=3e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    model.train()
    print(f"🎯 Meta: {args.epochs} épocas adicionales.")
    
    for epoch in range(args.epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        
        loop = tqdm(trainloader, desc=f"Epoch +{epoch+1}")
        
        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            
            outputs = model(inputs)
            
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            
            loop.set_postfix(loss=loss.item(), acc=100.*correct/total)
        
        scheduler.step()
        print(f"🏁 Epoch +{epoch+1} -> Loss: {running_loss/len(trainloader):.4f} | Acc: {100.*correct/total:.2f}%")
        
        torch.save(model.state_dict(), f"simple_poc/supernet_checkpoint_resume.pth")

    print("\n💾 Entrenamiento extendido finalizado.")

if __name__ == "__main__":
    main()