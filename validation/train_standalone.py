# =============================================================================
# validation/train_standalone.py — FINAL VERSION (FAIR COMPARISON)
#
# This script trains a single standalone path extracted from the SuperNetwork.
# Training protocol matches the supernet's --freeze_backbone run exactly:
#   - All backbone stages (0–3) frozen
#   - Only stitching layers and classifier heads trained
#   - BN tracking disabled during training (like supernet.set_bn_tracking(False))
#   - Same optimizer, LR schedule, AMP, gradient clipping
#   - Same dataset (ImageNette 160px, 50% subset)
# =============================================================================

import sys
import os
import argparse
import ast
import json
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
from tqdm import tqdm
import random
import numpy as np

# Ensure root directory is accessible
sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())
from simple_poc.supernet import SuperNetwork, OutputUnwrapper

# ------------------------------------------------------------------ #
# Data Loading (same as supernet)
# ------------------------------------------------------------------ #
def get_imagenette(root='data/imagenette2-160', img_size=160):
    train_dir = os.path.join(root, 'train')
    val_dir   = os.path.join(root, 'val')
    
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
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
    
    trainset = torchvision.datasets.ImageFolder(train_dir, transform_train)
    valset   = torchvision.datasets.ImageFolder(val_dir, transform_val)
    
    # 50% subset (same as supernet)
    trainset = torch.utils.data.Subset(trainset, range(0, len(trainset), 2))
    
    return trainset, valset

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# ------------------------------------------------------------------ #
# Standalone Wrapper
# ------------------------------------------------------------------ #
class StandaloneNetwork(nn.Module):
    """
    Assembles a standard PyTorch Sequential model by extracting the exact
    layers required for the specified path from an initialized SuperNetwork.
    """
    def __init__(self, supernet, path):
        super().__init__()
        self.path = path
        modules = []
        
        # Stage 0
        modules.append(supernet.stages[0][path[0]])
        
        # Subsequent stages + stitching
        for i in range(1, supernet.num_stages):
            prev_idx = path[i - 1]
            curr_idx = path[i]
            modules.append(supernet.stitches[i - 1][prev_idx][curr_idx])
            modules.append(supernet.stages[i][curr_idx])
            
        self.features = nn.Sequential(*modules)
        self.global_pool = supernet.global_pool
        self.head = supernet.heads[path[-1]]
        
    def forward(self, x):
        x = self.features(x)
        x = self.global_pool(x)
        return self.head(x)

# ------------------------------------------------------------------ #
# Main Execution
# ------------------------------------------------------------------ #
def main():
    parser = argparse.ArgumentParser(description="Train a Standalone Path")
    parser.add_argument('--path', type=str, required=True, help='List like "[0,1,2,3]"')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--batch_size', type=int, default=8)
    parser.add_argument('--output_file', type=str, required=True)
    args = parser.parse_args()

    set_seed(42)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_AMP = (DEVICE == 'cuda')
    
    # Safely parse the path argument
    try:
        target_path = ast.literal_eval(args.path)
        if not isinstance(target_path, list):
            raise ValueError
    except:
        raise ValueError("Path must be formatted as a list string, e.g., '[0,1,2,3]'")

    print(f"🚀 Initializing SuperNetwork to harvest path {target_path}...")
    
    # 1. Initialize SuperNetwork (resolves shapes silently)
    supernet = SuperNetwork(
        plan_path="network_plan.pkl",
        num_classes=10,
        input_size=160,
        stitch_init_mode='ls'
    )
    
    # 2. Extract the targeted path
    model = StandaloneNetwork(supernet, target_path).to(DEVICE)
    
    # Free memory held by supernet
    del supernet
    torch.cuda.empty_cache()

    print(f"✅ Standalone Network Assembled. Total Params: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # ------------------------------------------------------------------ #
    # Data Setup
    # ------------------------------------------------------------------ #
    trainset, valset = get_imagenette(img_size=160)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------ #
    # FAIRNESS PATCH 2: Freeze ALL backbone stages (matches supernet --freeze_backbone)
    # ------------------------------------------------------------------ #
    block_modules = [m for m in model.features.modules() if isinstance(m, OutputUnwrapper)]
    for m in block_modules:
        for p in m.parameters():
            p.requires_grad = False

    # ------------------------------------------------------------------ #
    # Separate parameters: stitches/head vs. backbone (backbone will be empty)
    # ------------------------------------------------------------------ #
    stitch_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'stitch' in name or 'head' in name or 'linear' in name:
            stitch_params.append(param)
        else:
            backbone_params.append(param)

    def no_weight_decay(module):
        return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm))

    optimizer = optim.SGD([
        {'params': stitch_params, 'lr': args.lr, 'momentum': 0.9, 'weight_decay': 5e-4},
        {'params': backbone_params, 'lr': args.lr * 0.1, 'momentum': 0.9, 'weight_decay': 5e-4}
    ])

    # Remove weight decay from BN layers
    for group in optimizer.param_groups:
        group['params'] = [p for p in group['params']
                           if not any(no_weight_decay(m) for m in model.modules()
                                      if hasattr(m, 'weight') and m.weight is p)]
    
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-5)
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------ #
    # Training Loop
    # ------------------------------------------------------------------ #
    metrics = []
    best_val_acc = 0.0

    model.train()   # training mode; BN uses batch stats because track_running_stats=False

    for epoch in range(args.epochs):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                outputs = model(inputs)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()
            
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            _, predicted = outputs.max(1)
            running_loss += loss.item()
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

            loop.set_postfix(loss=f"{loss.item():.4f}", acc=f"{100.*correct/total:.1f}%")

        train_loss = running_loss / len(trainloader)
        train_acc = 100.0 * correct / total

        # Validation
        model.eval()
        val_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    outputs = model(inputs)
                    loss = criterion(outputs, targets)
                
                val_loss += loss.item()
                _, predicted = outputs.max(1)
                val_total += targets.size(0)
                val_correct += predicted.eq(targets).sum().item()

        val_loss /= len(valloader)
        val_acc = 100.0 * val_correct / val_total
        
        scheduler.step()
        current_lr = optimizer.param_groups[0]['lr']

        print(f"  Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% | LR: {current_lr:.6f}")

        best_val_acc = max(best_val_acc, val_acc)

        metrics.append({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 5),
            "train_acc": round(train_acc, 3),
            "val_loss": round(val_loss, 5),
            "val_acc": round(val_acc, 3),
            "lr": current_lr
        })

        # Save metrics to file
        os.makedirs(os.path.dirname(args.output_file) or ".", exist_ok=True)
        with open(args.output_file, 'w') as f:
            json.dump(metrics, f, indent=4)

    print(f"\n✅ Standalone Training Complete. Best Val Acc: {best_val_acc:.2f}%")

if __name__ == "__main__":
    main()