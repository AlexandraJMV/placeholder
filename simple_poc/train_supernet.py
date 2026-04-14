import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import random
import numpy as np
import json

sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

from simple_poc.supernet import SuperNetwork

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False 
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train One-Shot SuperNet")

    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.005)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--init_mode', type=str, default='ls', choices=['ls', 'random'])
    parser.add_argument('--matrices_path', type=str, default=None)
    parser.add_argument('--output_name', type=str, default='supernet_experiment')
    parser.add_argument('--train_only_stitches', action='store_true')
    parser.add_argument('--val_paths', type=int, default=8)
    parser.add_argument('--resume', action='store_true', help="Reanudar desde latest.pth si existe")

    return parser.parse_args()

def generate_fixed_paths(model, num_paths=20, seed=42):
    random.seed(seed)
    paths = []
    for _ in range(num_paths):
        path = [random.randint(0, c - 1) for c in model.choices_per_stage]
        paths.append(path)
    return paths

def validate(model, loader, criterion, device, fixed_paths):
    model.eval()
    val_loss = 0.0
    acc_sum = 0.0

    with torch.no_grad():
        for inputs, targets in tqdm(loader, desc="🔍 Validating"):
            inputs, targets = inputs.to(device), targets.to(device)

            batch_loss = 0.0
            batch_acc = 0.0

            for path in fixed_paths:
                outputs = model(inputs, path=path)
                loss = criterion(outputs, targets)

                batch_loss += loss.item()
                _, predicted = outputs.max(1)
                batch_acc += predicted.eq(targets).sum().item() / targets.size(0)

            val_loss += batch_loss / len(fixed_paths)
            acc_sum += batch_acc / len(fixed_paths)

    return val_loss / len(loader), 100. * acc_sum / len(loader)

def sample_path(model):
    return [random.randint(0, c - 1) for c in model.choices_per_stage]

def main():
    args = parse_args()
    set_seed(42)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    out_dir = os.path.join('simple_poc', args.output_name)
    os.makedirs(out_dir, exist_ok=True)

    latest_ckpt = os.path.join(out_dir, 'latest.pth')
    best_ckpt = os.path.join(out_dir, 'best.pth')

    # ✅ NUEVO: ruta del JSON de métricas
    metrics_path = os.path.join(out_dir, f"{args.output_name}.json")

    # ✅ NUEVO: cargar métricas si existen (resume real)
    if args.resume and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)
    else:
        metrics = []

    print(f"🚀 Training on {DEVICE}")

    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform_val)

    trainloader = torch.utils.data.DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )

    valloader = torch.utils.data.DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    model = SuperNetwork(
        plan_path="network_plan.pkl",
        num_classes=10,
        input_size=32,
        stitch_init_mode=args.init_mode,
        matrices_path=args.matrices_path
    ).to(DEVICE)

    if args.train_only_stitches:
        for param in model.stages.parameters():
            param.requires_grad = False

    start_epoch = 0
    best_val_acc = 0.0

    optimizer = optim.SGD(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4
    )

    criterion = nn.CrossEntropyLoss()

    if args.resume and os.path.exists(latest_ckpt):
        print(f"📥 Cargando checkpoint desde: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=DEVICE)

        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])

        if 'epoch' in checkpoint:
            start_epoch = checkpoint['epoch']
        if 'best_val_acc' in checkpoint:
            best_val_acc = checkpoint['best_val_acc']

    total_epochs = start_epoch + args.epochs

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs
    )

    if args.resume and os.path.exists(latest_ckpt) and 'scheduler' in checkpoint:
        scheduler.load_state_dict(checkpoint['scheduler'])

    fixed_paths = generate_fixed_paths(model, args.val_paths)

    for epoch in range(start_epoch, total_epochs):

        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        loop = tqdm(trainloader, desc=f"Epoch {epoch+1}/{total_epochs}")

        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()

            path = sample_path(model)
            outputs = model(inputs, path=path)

            loss = criterion(outputs, targets)
            loss.backward()

            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            _, predicted = outputs.max(1)

            running_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loop.set_postfix(loss=loss.item(), acc=100. * correct / total)

        train_loss = running_loss / len(trainloader)
        train_acc = 100. * correct / total

        val_loss, val_acc = validate(model, valloader, criterion, DEVICE, fixed_paths)

        scheduler.step()

        current_lr = optimizer.param_groups[0]["lr"]

        print(f"\nEpoch {epoch+1}")
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}")

        # ✅ NUEVO: guardar métricas
        metrics.append({
            "epoch": epoch + 1,
            "train_loss": train_loss,
            "train_acc": train_acc,
            "val_loss": val_loss,
            "val_acc": val_acc,
            "lr": current_lr,
            "best_val_acc": max(best_val_acc, val_acc)
        })

        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        checkpoint = {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': max(best_val_acc, val_acc)
        }

        torch.save(checkpoint, latest_ckpt)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(checkpoint, best_ckpt)
            print("🌟 New best model saved!")

    print("✅ Training finished.")

if __name__ == "__main__":
    main()