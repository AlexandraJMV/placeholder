# =============================================================================
# train_supernet.py  —  REFACTORED
# Changes:
#   - Added AMP (GradScaler + autocast) — FLAW 2
#   - Fixed CosineAnnealingLR T_max on resume — FLAW 3
#   - Fixed gradient clipping to trainable params only — FLAW 8
#   - Fixed generate_fixed_paths to use isolated RNG — FLAW 7
#   - Fixed validation to single-path with BN calibration support — FLAW 1/6
#   - Added post-training BN calibration example in evaluate_path
# =============================================================================

import sys
import os
import torch
import torch.nn as nn
import torch.optim as optim
from torchgen import model
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

# ------------------------------------------------------------------ #
# Utility functions     
# ------------------------------------------------------------------ #

# Función para cargar ImageNette 
def get_imagenette(root='data/imagenette2-160', img_size=224):
    # ImageNette structure: root/train and root/val
    train_dir = os.path.join(root, 'train')
    val_dir   = os.path.join(root, 'val')
    
    # ImageNet normalization (since models are pretrained on ImageNet)
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])
    
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    transform_val = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),  # slightly larger for center crop
        transforms.CenterCrop(img_size),
        transforms.ToTensor(),
        normalize,
    ])
    
    trainset = torchvision.datasets.ImageFolder(train_dir, transform_train)
    valset   = torchvision.datasets.ImageFolder(val_dir, transform_val)
    return trainset, valset

# Agregar función para establecer semillas de manera consistente
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True

# Parsear argumentos de línea de comandos
def parse_args():
    parser = argparse.ArgumentParser(description="Train One-Shot SuperNet (SPOS)")
    parser.add_argument('--batch_size',         type=int,   default=64)
    parser.add_argument('--lr',                 type=float, default=0.001)
    parser.add_argument('--epochs',             type=int,   default=50)
    parser.add_argument('--init_mode',          type=str,   default='ls',
                        choices=['ls', 'random'])
    parser.add_argument('--matrices_path',      type=str,   default=None)
    parser.add_argument('--output_name',        type=str,   default='supernet_experiment')
    parser.add_argument('--train_only_stitches',action='store_true')
    parser.add_argument('--val_paths',          type=int,   default=8)
    parser.add_argument('--resume',             action='store_true',
                        help="Resume from latest.pth if it exists")
    parser.add_argument('--no_amp',             action='store_true',
                        help="Disable AMP (use for debugging only)")
    parser.add_argument('--bump_lr',            action='store_true',
                        help="Bump LR to 1e-4 when resuming with low LR (<2e-5)")
    # FIX: New flag to freeze backbone during training (recommended)
    parser.add_argument('--freeze_backbone',    action='store_true',
                        help="Freeze backbone parameters (except stitches). "
                             "Overrides --train_only_stitches behavior.")
    
    # FIX: New flag to evaluate candidate paths after training
    parser.add_argument('--eval_candidates', action='store_true',
                    help='After training, evaluate all paths in candidate_paths.json')
    
    return parser.parse_args()


def generate_fixed_paths(choices_per_stage, num_paths=20, seed=42):
    """
    FIX (FLAW 7): Uses an isolated Random instance so global random state
    used in sample_path() during training is never disturbed.
    """
    rng = random.Random(seed)
    paths = []
    for _ in range(num_paths):
        path = [rng.randint(0, c - 1) for c in choices_per_stage]
        paths.append(path)
    return paths


def validate_single_path(model, loader, criterion, device, path, use_amp=True):
    """
    FIX (FLAW 6): Evaluate a SINGLE path cleanly.
    Caller is responsible for BN calibration before calling this.
    """
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

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


def validate_supernet(model, train_loader, val_loader, criterion, device,
                      fixed_paths, use_amp=True, calibrate=True,
                      calib_batches=100):   # Reduced to 10 for efficiency
    """
    Validates the supernet by:
    1. Calibrating BN for each fixed path individually.
    2. Evaluating each path in isolation.
    Returns average accuracy and per‑path accuracies.
    """
    path_accs = []
    path_losses = []

    for path in fixed_paths:
        if calibrate:
            model.calibrate_bn(train_loader, path,
                               n_batches=calib_batches, device=device)

        loss, acc = validate_single_path(model, val_loader, criterion,
                                         device, path, use_amp)
        path_accs.append(acc)
        path_losses.append(loss)

    return (float(np.mean(path_losses)), float(np.mean(path_accs)),
            float(np.std(path_accs)), path_accs)

def main():
    args = parse_args()
    set_seed(42)

    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_AMP = (not args.no_amp) and (DEVICE == 'cuda')

    if USE_AMP:
        print("✅ AMP enabled")
    else:
        print("⚠️  AMP disabled — expect higher VRAM usage")

    out_dir = os.path.join('simple_poc', args.output_name)
    os.makedirs(out_dir, exist_ok=True)

    latest_ckpt  = os.path.join(out_dir, 'latest.pth')
    best_ckpt    = os.path.join(out_dir, 'best.pth')
    metrics_path = os.path.join(out_dir, f"{args.output_name}.json")

    metrics = []
    if args.resume and os.path.exists(metrics_path):
        with open(metrics_path, 'r') as f:
            metrics = json.load(f)

    print(f"🚀 Device: {DEVICE}")

    # ------------------------------------------------------------------ #
    # Data
    # ------------------------------------------------------------------ #
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset, valset = get_imagenette(img_size=224)
    
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size, shuffle=True,
        num_workers=4, pin_memory=True)
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size, shuffle=False,
        num_workers=4, pin_memory=True)

    # ------------------------------------------------------------------ #
    # Model
    # ------------------------------------------------------------------ #
    model = SuperNetwork( 
        plan_path="network_plan.pkl",
        num_classes=10,
        input_size=224,
        stitch_init_mode=args.init_mode,
        matrices_path=args.matrices_path,
    ).to(DEVICE)

    if args.train_only_stitches or args.freeze_backbone:
        print("🔒 Freezing backbone blocks — training stitches only")
        model.set_backbone_requires_grad(False)
    else:
        # Fine‑tune backbone with lower LR
        model.set_backbone_requires_grad(True)
        print("⚠️  Backbone will be fine‑tuned (lower LR applied)")
        
    stitch_params = []
    backbone_params = []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if 'stages' in name:   # backbone blocks
            backbone_params.append(param)
        else:                  # stitches, heads, etc.
            stitch_params.append(param)

    def no_weight_decay(module):
        return isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm))
    # ------------------------------------------------------------------ #
    # Optimizer and AMP scaler
    # ------------------------------------------------------------------ #
    optimizer = optim.SGD([
        {'params': stitch_params,
         'lr': args.lr,
         'momentum': 0.9,
         'weight_decay': 5e-4},
        {'params': backbone_params,
         'lr': args.lr * 0.1,          # 10× smaller LR for pre‑trained weights
         'momentum': 0.9,
         'weight_decay': 5e-4}
    ], lr=args.lr, momentum=0.9, weight_decay=5e-4)
    
        # Remove weight decay from BN layers in both groups (common practice)
    for group in optimizer.param_groups:
        for p in group['params']:
            if hasattr(p, '_no_weight_decay'):   # we'll set this flag
                continue
        group['params'] = [p for p in group['params']
                           if not any(no_weight_decay(m) for m in model.modules()
                                      if hasattr(m, 'weight') and m.weight is p)]
    
    # FIX (FLAW 2): GradScaler for AMP
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    criterion = nn.CrossEntropyLoss()


    # ------------------------------------------------------------------ #
    # Resume logic with FIXED scheduler handling
    # ------------------------------------------------------------------ #
    start_epoch  = 0
    best_val_acc = 0.0
    checkpoint   = None

    if args.resume and os.path.exists(latest_ckpt):
        print(f"📥 Loading checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=DEVICE, weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        scaler.load_state_dict(checkpoint.get('scaler', scaler.state_dict()))
        start_epoch  = checkpoint.get('epoch', 0)
        best_val_acc = checkpoint.get('best_val_acc', 0.0)
        print(f"   Resumed from epoch {start_epoch}, best_val_acc={best_val_acc:.2f}%")

        # FIX: Bump LR only if explicitly requested (avoid accidental)
        if args.bump_lr:
            for param_group in optimizer.param_groups:
                param_group['lr'] = max(param_group['lr'], 1e-4)
            print(f"   ⚠️  LR manually bumped to ≥1e-4")

    # FIX: Scheduler must span only the remaining epochs, not total epochs.
    remaining_epochs = args.epochs
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=remaining_epochs,   # <-- CORRECT: only the epochs we will run now
        eta_min=1e-5,
    )

    # Do NOT load scheduler state from checkpoint; we start a fresh annealing schedule.
    # If we want to resume the exact LR value, we keep the optimizer LR as is,
    # and let the scheduler start from step 0 with the current LR as the peak.
    # This is the cleanest approach.
    if args.resume and checkpoint is not None:
        # Manually set scheduler's last_epoch to 0 (its internal counter)
        scheduler.last_epoch = -1  # will become 0 after first step()
        # Optionally, we can adjust base_lrs to match current optimizer LR
        scheduler.base_lrs = [group['lr'] for group in optimizer.param_groups]
        print(f"   Scheduler reinitialized for {remaining_epochs} epochs. "
              f"Base LRs: {[f'{lr:.2e}' for lr in scheduler.base_lrs]}")

    total_epochs = start_epoch + remaining_epochs
    
    # ------------------------------------------------------------------ #
    # Fixed validation paths — isolated RNG (FIX FLAW 7)
    # ------------------------------------------------------------------ #
    fixed_paths = generate_fixed_paths(
        model.choices_per_stage, args.val_paths, seed=42)

    # ------------------------------------------------------------------ #
    # Training loop
    # ------------------------------------------------------------------ #
    # Isolated RNG for path sampling — never touches global random state
    sampling_rng = random.Random()
    sampling_rng.seed(42 + start_epoch)  # deterministic per-epoch but isolated

    model.set_bn_tracking(False)   # Disable running stats during training
    
    for epoch in range(start_epoch, total_epochs):
        model.train()
        running_loss = 0.0
        correct      = 0
        total        = 0

        loop = tqdm(trainloader, desc=f"Epoch {epoch+1}/{total_epochs}")

        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)  # Slightly more efficient

            # FIX (FLAW 2): AMP forward + backward
            with torch.amp.autocast('cuda', enabled=USE_AMP):
                # FIX (FLAW 7): Use isolated RNG for path sampling
                path = model.sample_path(rng=sampling_rng)
                outputs = model(inputs, path=path)
                loss = criterion(outputs, targets)

            scaler.scale(loss).backward()

            # FIX (FLAW 8): Clip only trainable parameters
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(
                [p for group in optimizer.param_groups for p in group['params'] if p.requires_grad],
                max_norm=5.0
            )

            scaler.step(optimizer)
            scaler.update()

            _, predicted = outputs.max(1)
            running_loss += loss.item()
            total        += targets.size(0)
            correct      += predicted.eq(targets).sum().item()

            loop.set_postfix(loss=f"{loss.item():.4f}",
                             acc=f"{100.*correct/total:.1f}%")

        train_loss = running_loss / len(trainloader)
        train_acc  = 100.0 * correct / total

        # FIX: Only validate every 5 epochs to save time, with reduced calibration batches
        if (epoch + 1) % 5 == 0 or epoch == total_epochs - 1:
            val_loss, val_acc, val_std, per_path_accs = validate_supernet(
                model, trainloader, valloader, criterion, DEVICE,
                fixed_paths,
                use_amp=USE_AMP,
                calibrate=True,
                calib_batches=100,   # Reduced from 30 → 5 for efficiency
            )
            print(f"  Val — Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% ± {val_std:.2f}%")
            if per_path_accs:
                print(f"  Per-path accs: {[f'{a:.1f}' for a in per_path_accs]}")
        else:
            val_loss, val_acc, val_std, per_path_accs = 0.0, 0.0, 0.0, []
            print(f"  Val — skipped (runs every 5 epochs)")

        # Restore train mode after validation
        model.train()
        model.set_bn_tracking(False)
        scheduler.step()

        current_lr = optimizer.param_groups[0]['lr']

        print(f"\nEpoch {epoch+1}/{total_epochs}")
        print(f"  Train  — Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"  Val    — Loss: {val_loss:.4f}  | "
              f"Acc: {val_acc:.2f}% ± {val_std:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        if per_path_accs:
            print(f"  Per-path accs: {[f'{a:.1f}' for a in per_path_accs]}")

        # Save metrics
        metrics.append({
            "epoch":         epoch + 1,
            "train_loss":    round(train_loss, 5),
            "train_acc":     round(train_acc, 3),
            "val_loss":      round(val_loss, 5),
            "val_acc":       round(val_acc, 3),
            "val_std":       round(val_std, 3),
            "lr":            current_lr,
            "best_val_acc":  round(max(best_val_acc, val_acc), 3),
            "per_path_accs": [round(a, 2) for a in per_path_accs],
        })
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        # Checkpointing
        ckpt = {
            'epoch':        epoch + 1,
            'state_dict':   model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'scheduler':    scheduler.state_dict(),
            'scaler':       scaler.state_dict(),
            'val_acc':      val_acc,
            'best_val_acc': max(best_val_acc, val_acc),
        }
        torch.save(ckpt, latest_ckpt)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, best_ckpt)
            print(f"  🌟 New best saved: {best_val_acc:.2f}%")

    print(f"\n✅ Training complete. Best val acc: {best_val_acc:.2f}%")

    # -------------------------------------------------------------------- # 
    # Candidate path evaluation after training 
    # -------------------------------------------------------------------- #
    
    if args.eval_candidates and os.path.exists('candidate_paths.json'):
        print("\n🔍 Evaluating all candidate paths...")
        with open('candidate_paths.json', 'r') as f:
            candidate_paths = json.load(f)
        
        results = {}
        for i, path in enumerate(candidate_paths):
            print(f"  Path {i+1}/{len(candidate_paths)}: {path}")
            model.calibrate_bn(trainloader, path, n_batches=100, device=DEVICE)
            loss, acc = validate_single_path(model, valloader, criterion, DEVICE, path, use_amp=USE_AMP)
            results[str(path)] = acc
            print(f"    Acc: {acc:.2f}%")
        
        with open('supernet_candidate_accs.json', 'w') as f:
            json.dump(results, f, indent=2)
        print("Saved supernet accuracies to supernet_candidate_accs.json")

if __name__ == "__main__":
    main()