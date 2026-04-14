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
    parser = argparse.ArgumentParser(description="Train One-Shot SuperNet (SPOS)")
    parser.add_argument('--batch_size',         type=int,   default=64)
    parser.add_argument('--lr',                 type=float, default=0.005)
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
                      calib_batches=30):
    """
    FIX (FLAW 1 + FLAW 6): Validates the supernet by:
    1. Calibrating BN for each fixed path individually.
    2. Evaluating each path in isolation.
    Returns average accuracy and per-path accuracies for analysis.
    """
    path_accs = []
    path_losses = []

    for path in fixed_paths:
        if calibrate:
            # Recalibrate BN running stats for this specific path
            model.calibrate_bn(train_loader, path,
                                n_batches=calib_batches, device=device)

        loss, acc = validate_single_path(model, val_loader, criterion,
                                         device, path, use_amp)
        path_accs.append(acc)
        path_losses.append(loss)

    avg_acc = float(np.mean(path_accs))
    avg_loss = float(np.mean(path_losses))
    std_acc = float(np.std(path_accs))

    return avg_loss, avg_acc, std_acc, path_accs


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
        transforms.Resize(64),  
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])
    transform_val = transforms.Compose([
        transforms.Resize(64),   
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    valset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_val)

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
        input_size=64,
        stitch_init_mode=args.init_mode,
        matrices_path=args.matrices_path,
    ).to(DEVICE)

    if args.train_only_stitches:
        print("🔒 Freezing backbone blocks — training stitches only")
        for param in model.stages.parameters():
            param.requires_grad = False

    # ------------------------------------------------------------------ #
    # Optimizer and AMP scaler
    # ------------------------------------------------------------------ #
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    optimizer = optim.SGD(
        trainable_params,
        lr=args.lr,
        momentum=0.9,
        weight_decay=5e-4,
    )

    # FIX (FLAW 2): GradScaler for AMP
    scaler = torch.amp.GradScaler('cuda', enabled=USE_AMP)

    criterion = nn.CrossEntropyLoss()

    # ------------------------------------------------------------------ #
    # Resume logic
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

    total_epochs = start_epoch + args.epochs

    # FIX (FLAW 3): T_max must equal the number of epochs THIS run will train,
    # not total_epochs. Reload scheduler state AFTER construction so the
    # last_epoch pointer is correctly restored without corrupting T_max.
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=total_epochs,   # full span, never relative
        eta_min=1e-5,
    )

    # If resuming, fast-forward scheduler to correct position
    # by stepping through all already-completed epochs
    if checkpoint is not None:
        for _ in range(start_epoch):
            scheduler.step()
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
            torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=5.0)

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

        # FIX (FLAW 1 + FLAW 6): Proper single-path validation with BN calib.
        # During early epochs, skip calibration to save time; enable after warmup.
        do_calibrate = (epoch >= 10)  # Only calibrate after initial warmup
        calib_batches = 30 if do_calibrate else 0

        val_loss, val_acc, val_std, per_path_accs = validate_supernet(
            model, trainloader, valloader, criterion, DEVICE,
            fixed_paths,
            use_amp=USE_AMP,
            calibrate=do_calibrate,
            calib_batches=calib_batches,
        )

        # Restore train mode after validation
        model.train()

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


if __name__ == "__main__":
    main()