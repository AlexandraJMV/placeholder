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

def get_imagenette(root='data/imagenette2-160', img_size=160):
    train_dir = os.path.join(root, 'train')
    val_dir   = os.path.join(root, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    return torchvision.datasets.ImageFolder(train_dir, transform_train), torchvision.datasets.ImageFolder(val_dir, transform_val)

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

def parse_args():
    parser = argparse.ArgumentParser(description="Train One-Shot SuperNet (SPOS) - Clean Run")
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=0.001)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--init_mode', type=str, default='ls', choices=['ls', 'random'])
    parser.add_argument('--matrices_path', type=str, default=None)
    parser.add_argument('--output_name', type=str, default='supernet_experiment')
    parser.add_argument('--train_only_stitches', action='store_true')
    parser.add_argument('--val_paths', type=int, default=8)
    parser.add_argument('--no_amp', action='store_true')
    parser.add_argument('--freeze_backbone', action='store_true')
    parser.add_argument('--eval_candidates', action='store_true')
    
    # Extra arg para pponer flag que realice validación solo luego de la época 200
    # Antes de eso, solo calcular validación cada 10 épocas
    parser.add_argument('--late_val_start', type=int, default=0, help="Epoch to start full validation. Before this, val is done every 10 epochs.")
    
    return parser.parse_args()

def generate_fixed_paths(choices_per_stage, num_paths=20, seed=42):
    rng = random.Random(seed)
    return [[rng.randint(0, c - 1) for c in choices_per_stage] for _ in range(num_paths)]

def validate_single_path(model, loader, criterion, device, path, use_amp=True):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(inputs, path=path)
                loss = criterion(outputs, targets)
            total_loss += loss.item()
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
    return total_loss / len(loader), 100.0 * correct / total

def validate_supernet(model, train_loader, val_loader, criterion, device, fixed_paths, use_amp=True, calibrate=True, calib_batches=100):
    path_accs, path_losses = [], []
    for path in fixed_paths:
        if calibrate:
            model.calibrate_bn(train_loader, path, n_batches=calib_batches, device=device)
        loss, acc = validate_single_path(model, val_loader, criterion, device, path, use_amp)
        path_accs.append(acc)
        path_losses.append(loss)
    return float(np.mean(path_losses)), float(np.mean(path_accs)), float(np.std(path_accs)), path_accs

def main():
    args = parse_args()
    set_seed(42)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_AMP = (not args.no_amp) and (DEVICE == 'cuda')

    out_dir = os.path.join('simple_poc', args.output_name)
    os.makedirs(out_dir, exist_ok=True)
    latest_ckpt  = os.path.join(out_dir, 'latest.pth')
    best_ckpt    = os.path.join(out_dir, 'best.pth')
    metrics_path = os.path.join(out_dir, f"{args.output_name}.json")

    metrics = []
    
    IMG_SIZE = 160
    trainset, valset = get_imagenette(img_size=IMG_SIZE)
    trainset = torch.utils.data.Subset(trainset, range(0, len(trainset), 2))
    
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    valloader = torch.utils.data.DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    model = SuperNetwork(plan_path="network_plan.pkl", num_classes=10, input_size=IMG_SIZE, stitch_init_mode=args.init_mode, matrices_path=args.matrices_path).to(DEVICE)

    if args.train_only_stitches or args.freeze_backbone:
        model.set_backbone_requires_grad(False)
    else:
        model.set_backbone_requires_grad(True)
        
    stitch_decay, stitch_no_decay = [], []
    backbone_decay, backbone_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        is_no_decay = len(param.shape) == 1 or name.endswith(".bias")
        
        if 'stages' in name:
            (backbone_no_decay if is_no_decay else backbone_decay).append(param)
        else:
            (stitch_no_decay if is_no_decay else stitch_decay).append(param)

    optimizer = optim.SGD([
        {'params': stitch_decay,      'lr': args.lr,       'weight_decay': 5e-4},
        {'params': stitch_no_decay,   'lr': args.lr,       'weight_decay': 0.0},
        {'params': backbone_decay,    'lr': args.lr * 0.1, 'weight_decay': 5e-4},
        {'params': backbone_no_decay, 'lr': args.lr * 0.1, 'weight_decay': 0.0}
    ], momentum=0.9)
    
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    criterion = nn.CrossEntropyLoss()

    best_val_acc = 0.0
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)

    fixed_paths = generate_fixed_paths(model.choices_per_stage, args.val_paths, seed=42)
    sampling_rng = random.Random(42)

    for epoch in range(args.epochs):
        model.train()
        model.set_bn_tracking(False)
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                path = model.sample_path(rng=sampling_rng)
                outputs = model(inputs, path=path)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_((p for p in model.parameters() if p.requires_grad), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            _, predicted = outputs.max(1)
            running_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()
            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

        train_loss, train_acc = running_loss / len(trainloader), 100.0 * correct / total
        val_loss, val_acc, val_std, per_path_accs = 0.0, 0.0, 0.0, []

        if args.late_val_start > 0 :
            if epoch >= args.late_val_start and args.eval_candidates:
                val_loss, val_acc, val_std, per_path_accs = validate_supernet(
                    model, trainloader, valloader, criterion, DEVICE, fixed_paths, use_amp=USE_AMP, calibrate=True, calib_batches=100)
            elif (epoch + 1) % 10 == 0 and args.eval_candidates:
                val_loss, val_acc, val_std, per_path_accs = validate_supernet(
                    model, trainloader, valloader, criterion, DEVICE, fixed_paths, use_amp=USE_AMP, calibrate=True, calib_batches=100)
        
        elif epoch == args.epochs - 1 or args.eval_candidates:
            val_loss, val_acc, val_std, per_path_accs = validate_supernet(
                model, trainloader, valloader, criterion, DEVICE, fixed_paths, use_amp=USE_AMP, calibrate=True, calib_batches=100)      
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1}/{args.epochs} | Train Loss: {train_loss:.4f} Acc: {train_acc:.2f}% | LR: {current_lr:.6f}")

        metrics.append({"epoch": epoch + 1, "train_loss": round(train_loss, 5), "train_acc": round(train_acc, 3), "val_loss": round(val_loss, 5), "val_acc": round(val_acc, 3), "val_std": round(val_std, 3), "lr": current_lr, "best_val_acc": round(max(best_val_acc, val_acc), 3), "per_path_accs": [round(a, 2) for a in per_path_accs]})
        with open(metrics_path, 'w') as f: json.dump(metrics, f, indent=4)

        ckpt = {'epoch': epoch + 1, 'state_dict': model.state_dict(), 'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(), 'scaler': scaler.state_dict(), 'val_acc': val_acc, 'best_val_acc': max(best_val_acc, val_acc)}
        torch.save(ckpt, latest_ckpt)

        if val_acc > best_val_acc and epoch > 0:
            best_val_acc = val_acc
            torch.save(ckpt, best_ckpt)
            print(f"  🌟 New best saved: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()