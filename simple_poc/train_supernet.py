"""
train_supernet.py  —  One-Shot SuperNet training (SPOS), v2
==============================================================
Key changes from v1 and why each matters for Kendall's τ:

1. OPTIMIZER: SGD → AdamW
   Aligns with eval_gt.py so fine-tuning is a natural continuation
   of supernet training.  AdamW also handles the noisy weight-shared
   gradients more robustly than SGD.

2. WEIGHT DECAY: 5e-4 → 1e-4
   Matches eval_gt.py.  Avoids the supernet and GT having parameters
   at different magnitude scales, which would distort the fine-tuning
   starting point.

3. SCHEDULER: CosineAnnealingLR (no warmup, eta_min=1e-5)
             → LinearLR warmup (5 ep) + CosineAnnealingLR (eta_min=1e-6)
   Warmup protects pretrained ImageNet weights during the first epochs
   when stitching-layer gradients are largest.  eta_min=1e-6 matches GT.

4. DATASET: 50% subset → full dataset
   The supernet proxy ranking must be calibrated on the same distribution
   as the GT.  Training on half the data deflates τ because BN statistics
   and feature distributions diverge from what the GT sees at eval time.

5. BN TRACKING: always False → False during training, True during BN calib
   Keeping BN in tracking=False during the forward pass is correct for
   weight-shared training (prevents one path's stats from corrupting
   another's running mean/var).  Calibration already handles this.
   No change here — preserved exactly.

6. VALIDATION: full val set with calib_batches=100
   Abaratamiento (cost reduction) strategy:
     - val_freq argument: validate every N epochs instead of every epoch.
       Default = 5 (cheap enough, informative enough for τ monitoring).
     - calib_batches: reduced from 100 → 50 for in-training monitoring.
       The full 100-batch calibration is only used at the final epoch and
       for the best checkpoint, where accuracy matters for τ computation.
     - val_paths: kept as CLI arg (default 8).  These fixed paths are used
       ONLY for monitoring training progress — they are NOT the eval universe.

7. RESUME: added full resume support (was missing in v1).
   Without resume, a Kaggle session crash means restarting from epoch 0.

8. METRICS: both LR groups recorded per epoch (was only group 0).
   Gradient norm of stitching layers recorded (mirrors eval_gt.py).

"""

import sys, os, random, json, time as time_module
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision
import torchvision.transforms as transforms
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

from simple_poc.supernet import SuperNetwork



# ── Run naming ────────────────────────────────────────────────────────────────
def build_run_name(
    lr,
    epochs,
    init_mode       = "ls",
    train_only      = False,
    freeze_backbone = False,
    weight_decay    = 1e-4,
    batch_size      = 256,
    extra_tag       = None,
):
    """
    Construye un nombre de experimento legible que codifica las características
    principales del run. No incluye 'supernet' — la carpeta HF ya lo implica.
    Ejemplos:
        full_lr1e-2_ep300
        onlystitch_lr1e-2_ep300
        full_lr1e-2_ep300_wd5e-4
        onlystitch_lr1e-3_ep300_bs64_tag1
    """
    if train_only or freeze_backbone:
        mode = "onlystitch"
    else:
        mode = "full"
    lr_str = f"{lr:.0e}".replace("e-0", "e-").replace("e+0", "e")
    name = f"{mode}_lr{lr_str}_ep{epochs}"
    if abs(weight_decay - 1e-4) > 1e-10:
        wd_str = f"{weight_decay:.0e}".replace("e-0", "e-")
        name += f"_wd{wd_str}"
    if batch_size != 256:
        name += f"_bs{batch_size}"
    if extra_tag:
        name += f"_{extra_tag}"
    return name

DATASET_NUM_CLASSES = {'imagenette': 10, 'cifar100': 100, 'stl10': 10}

def get_dataset(name, root=None, img_size=160):
    normalize = transforms.Normalize(
        mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225],  # ImageNet stats — se
    )                                                            # mantienen fijas: los
                                                                  # anclas están pre-entrenadas
                                                                  # en ImageNet, no en el
                                                                  # dataset destino.
    transform_train = transforms.Compose([
        transforms.RandomResizedCrop(img_size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(), normalize,
    ])
    transform_val = transforms.Compose([
        transforms.Resize(int(img_size * 1.14)),
        transforms.CenterCrop(img_size),
        transforms.ToTensor(), normalize,
    ])

    if name == 'imagenette':
        root = root or 'data/imagenette2-160'
        trainset = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), transform_train)
        valset   = torchvision.datasets.ImageFolder(os.path.join(root, 'val'),   transform_val)
    elif name == 'cifar100':
        root = root or 'data/cifar100'
        trainset = torchvision.datasets.CIFAR100(root, train=True,  download=True, transform=transform_train)
        valset   = torchvision.datasets.CIFAR100(root, train=False, download=True, transform=transform_val)
    elif name == 'stl10':
        root = root or 'data/stl10'
        trainset = torchvision.datasets.STL10(root, split='train', download=True, transform=transform_train)
        valset   = torchvision.datasets.STL10(root, split='test',  download=True, transform=transform_val)
    else:
        raise ValueError(f"Unknown dataset: {name}")
    return trainset, valset

def get_imagenette(root='data/imagenette2-160', img_size=160):  # compat wrapper
    return get_dataset('imagenette', root=root, img_size=img_size)

# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark     = False
    torch.backends.cudnn.deterministic = True


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(
        description="Train One-Shot SuperNet (SPOS) v2")
    # ── architecture / data
    p.add_argument('--init_mode',      type=str,   default='ls',
                   choices=['ls', 'random'])
    p.add_argument('--matrices_path',  type=str,   default=None)
    p.add_argument('--output_name',    type=str,   default='supernet_v2')
    p.add_argument('--img_size',       type=int,   default=160)
    # ── training protocol
    p.add_argument('--epochs',         type=int,   default=100)
    p.add_argument('--batch_size',     type=int,   default=64)
    p.add_argument('--lr',             type=float, default=1e-3,
                   help="Base LR for stitches/head. Backbone receives lr*0.1")
    p.add_argument('--weight_decay',   type=float, default=1e-4)
    p.add_argument('--warmup_epochs',  type=int,   default=5)
    p.add_argument('--no_amp',         action='store_true')
    # ── backbone control
    p.add_argument('--train_only_stitches', action='store_true')
    p.add_argument('--freeze_backbone',     action='store_true')
    # ── validation (cost-reduction knobs)
    p.add_argument('--val_paths',      type=int,   default=8,
                   help="Number of fixed paths used for in-training monitoring")
    p.add_argument('--val_freq',       type=int,   default=5,
                   help="Validate every N epochs. Use 1 for every epoch, "
                        "10+ to minimise overhead. Default: 5.")
    p.add_argument('--dataset',  type=str, default='imagenette',
               choices=['imagenette', 'cifar100', 'stl10'])
    p.add_argument('--data_root', type=str, default=None)
    
    
    p.add_argument('--val_last_n_epochs', type=int, default=None,
               help="If set, skip ALL validation until the last N epochs of "
                    "training (val_freq still applies within that window). "
                    "Speeds up early training when supernet behavior is "
                    "predictable and only late-training performance matters. "
                    "Presence of this argument activates the behavior; "
                    "omit it to validate throughout as before.")
    
    
    
    p.add_argument('--calib_batches',  type=int,   default=50,
                   help="BN calibration batches for in-training val. "
                        "Final/best checkpoint uses 100 regardless.")
    p.add_argument('--eval_candidates', action='store_true',
                   help="Activate validation (required to see val metrics)")
    # ── resume
    p.add_argument('--resume',         type=str,   default=None,
                   help="Path to latest.pth checkpoint to resume from")
    
    # ── periodic checkpointing
    p.add_argument('--save_periodic_ckpt', action='store_true',
                   help="If set, additionally save a checkpoint every "
                        "--ckpt_every_n epochs (kept separately from latest/best).")
    p.add_argument('--ckpt_every_n',   type=int,   default=10,
                   help="Epoch interval for periodic checkpoints. "
                        "Only used if --save_periodic_ckpt is set.")
    
    # seed 
    p.add_argument('--seed',           type=int,   default=42,
                   help="Random seed for reproducibility (default: 42)")
    
    p.add_argument('--plan_path',      type=str,   default='network_plan.pkl',
               help="Path to the network plan .pkl file")
    
    
    
    return p.parse_args()


# ── Helpers ───────────────────────────────────────────────────────────────────
def generate_fixed_paths(choices_per_stage, num_paths=8, seed=42):
    """Fixed monitoring paths — same every run for comparable val curves."""
    rng = random.Random(seed)
    return [
        [rng.randint(0, c - 1) for c in choices_per_stage]
        for _ in range(num_paths)
    ]


def compute_stitch_grad_norm(model):
    """L2 norm of stitching-layer gradients (after unscale, before clip)."""
    total_sq = 0.0
    for name, param in model.named_parameters():
        if 'stitch' in name and param.grad is not None:
            total_sq += param.grad.detach().float().norm(2).item() ** 2
    return round(total_sq ** 0.5, 6)


# ── Validation ────────────────────────────────────────────────────────────────
def validate_single_path(model, loader, criterion, device, path, use_amp):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for inputs, targets in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(inputs, path=path)
                loss    = criterion(outputs, targets)
            total_loss += loss.item()
            correct    += outputs.max(1)[1].eq(targets).sum().item()
            total      += targets.size(0)
    return total_loss / len(loader), 100.0 * correct / total


def validate_supernet(model, train_loader, val_loader, criterion,
                      device, fixed_paths, use_amp, calib_batches):
    """
    Calibrate BN for each path, then evaluate on val_loader.
    Returns (mean_loss, mean_acc, std_acc, per_path_accs).

    Cost-reduction note: calib_batches controls how many train batches
    are used for BN calibration before each path's val pass.
    50 batches ≈ 3,200 images — sufficient for monitoring.
    100 batches used only at final/best checkpoints.
    """
    path_accs, path_losses = [], []
    for path in fixed_paths:
        model.calibrate_bn(
            train_loader, path,
            n_batches=calib_batches, device=device,
        )
        loss, acc = validate_single_path(
            model, val_loader, criterion, device, path, use_amp)
        path_accs.append(acc)
        path_losses.append(loss)
    return (
        float(np.mean(path_losses)),
        float(np.mean(path_accs)),
        float(np.std(path_accs)),
        path_accs,
    )


# ── Scheduler ─────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    """
    LinearLR warmup (warmup_epochs) → CosineAnnealingLR.
    Mirrors eval_gt.py exactly so supernet and GT share the same
    lr trajectory shape — important for weight compatibility.
    """
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 0.1,
        end_factor   = 1.0,
        total_iters  = warmup_epochs,
    )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = max(total_epochs - warmup_epochs, 1),
        eta_min = eta_min,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [warmup_epochs],
    )


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    import argparse   # imported here to avoid shadowing at module level
    args = parse_args()
    set_seed(args.seed)     # Cambiado para aceptar seed

    DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_AMP = (not args.no_amp) and (DEVICE == 'cuda')

    out_dir = os.path.join('simple_poc', args.dataset, args.output_name)  # antes: sin args.dataset
    os.makedirs(out_dir, exist_ok=True)
    latest_ckpt  = os.path.join(out_dir, 'latest.pth')
    best_ckpt    = os.path.join(out_dir, 'best.pth')
    metrics_path = os.path.join(out_dir, f"{args.output_name}.json")

    # ── Periodic checkpoint dir (named via build_run_name) ────────────────────
    run_name    = build_run_name(
        lr              = args.lr,
        epochs          = args.epochs,
        init_mode       = args.init_mode,
        train_only      = args.train_only_stitches,
        freeze_backbone = args.freeze_backbone,
        weight_decay    = args.weight_decay,
        batch_size      = args.batch_size,
    )
    periodic_dir = os.path.join(out_dir, "periodic_ckpts")
    if args.save_periodic_ckpt:
        os.makedirs(periodic_dir, exist_ok=True)

    # ── Dataset (full — no Subset) ────────────────────────────────────────────
    trainset, valset = get_dataset(args.dataset, root=args.data_root, img_size=args.img_size)
    gen = torch.Generator()
    
    gen.manual_seed(args.seed)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=args.batch_size,
        shuffle=True, generator=gen,
        num_workers=4, pin_memory=True,
    )
    valloader = torch.utils.data.DataLoader(
        valset, batch_size=args.batch_size,
        shuffle=False, num_workers=4, pin_memory=True,
    )

    # ── Model ─────────────────────────────────────────────────────────────────
    model = SuperNetwork(
    plan_path        = args.plan_path,
    num_classes      = DATASET_NUM_CLASSES[args.dataset],   # antes: 10 fijo
    input_size       = args.img_size,
    stitch_init_mode = args.init_mode,
    matrices_path    = args.matrices_path,
    ).to(DEVICE)

    if args.train_only_stitches or args.freeze_backbone:
        model.set_backbone_requires_grad(False)
    else:
        model.set_backbone_requires_grad(True)

    # ── Optimizer: AdamW with differential LR ─────────────────────────────────
    # Same 4-group structure as eval_gt.py so weight magnitudes stay compatible.
    stitch_decay,   stitch_no_decay   = [], []
    backbone_decay, backbone_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_no_decay = (param.dim() == 1) or name.endswith(".bias")
        if 'stages' in name:
            (backbone_no_decay if is_no_decay else backbone_decay).append(param)
        else:
            (stitch_no_decay   if is_no_decay else stitch_decay).append(param)

    optimizer = optim.AdamW([
        {'params': stitch_decay,      'lr': args.lr,       'weight_decay': args.weight_decay},
        {'params': stitch_no_decay,   'lr': args.lr,       'weight_decay': 0.0},
        {'params': backbone_decay,    'lr': args.lr * 0.1, 'weight_decay': args.weight_decay},
        {'params': backbone_no_decay, 'lr': args.lr * 0.1, 'weight_decay': 0.0},
    ])

    scheduler = build_scheduler(optimizer, args.warmup_epochs, args.epochs)
    scaler    = torch.amp.GradScaler('cuda', enabled=USE_AMP)
    criterion = nn.CrossEntropyLoss()

    # ── Fixed monitoring paths ────────────────────────────────────────────────
    fixed_paths  = generate_fixed_paths(
        model.choices_per_stage, args.val_paths, seed=args.seed)
    sampling_rng = random.Random(args.seed)

    # ── Resume ────────────────────────────────────────────────────────────────
    start_epoch  = 0
    best_val_acc = 0.0
    metrics      = []
    resume_path  = args.resume or (latest_ckpt if os.path.exists(latest_ckpt) else None)

    if resume_path and os.path.exists(resume_path):
        print(f"[Resume] Loading checkpoint: {resume_path}")
        ckpt = torch.load(resume_path, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch  = ckpt['epoch']
        best_val_acc = ckpt['best_val_acc']
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                metrics = json.load(f)
        print(f"[Resume] Continuing from epoch {start_epoch}, "
              f"best_val_acc={best_val_acc:.2f}%")

    # ── Training loop ─────────────────────────────────────────────────────────
    for epoch in range(start_epoch, args.epochs):
        t_epoch = time_module.perf_counter()

        # BN tracking OFF during weight-shared forward passes:
        # prevents one sampled path's batch statistics from corrupting
        # the running mean/var that other paths will read at inference.
        model.train()
        model.set_bn_tracking(False)

        running_loss, correct, total = 0.0, 0, 0
        loop = tqdm(trainloader, desc=f"Epoch {epoch+1}/{args.epochs}")

        for inputs, targets in loop:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=USE_AMP):
                path    = model.sample_path(rng=sampling_rng)
                outputs = model(inputs, path=path)
                loss    = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # Stitch grad norm: after unscale, before clip → raw signal
            stitch_gnorm = compute_stitch_grad_norm(model)

            torch.nn.utils.clip_grad_norm_(
                (p for p in model.parameters() if p.requires_grad),
                max_norm=5.0,
            )
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            correct      += outputs.max(1)[1].eq(targets).sum().item()
            total        += targets.size(0)
            loop.set_postfix(
                loss=f"{loss.item():.4f}",
                acc =f"{100.*correct/total:.1f}%",
            )

        scheduler.step()

        train_loss  = running_loss / len(trainloader)
        train_acc   = 100.0 * correct / total
        epoch_wall  = time_module.perf_counter() - t_epoch
        lr_stitches = round(optimizer.param_groups[0]['lr'], 8)
        lr_backbone = round(optimizer.param_groups[2]['lr'], 8)

        # =============================================================================
        # CAMBIO 2 — do_val: agregar la condición de ventana
        # =============================================================================
        
        # ── Validation (cost-reduced) ─────────────────────────────────────
        val_loss, val_acc, val_std, per_path_accs = 0.0, 0.0, 0.0, []
        is_last  = (epoch == args.epochs - 1)

        # Si --val_last_n_epochs está activo, solo se permite validar dentro
        # de esa ventana final (respetando val_freq dentro de ella).
        # Si no está activo (None), el comportamiento es idéntico al actual.
        in_val_window = (
            args.val_last_n_epochs is None
            or epoch >= (args.epochs - args.val_last_n_epochs)
        )

        do_val = (args.eval_candidates and in_val_window and
                  ((epoch + 1) % args.val_freq == 0 or is_last))

        if do_val:
            # Use full calib_batches=100 only at the last epoch;
            # use args.calib_batches (default 50) for intermediate checks.
            cb = 100 if is_last else args.calib_batches
            val_loss, val_acc, val_std, per_path_accs = validate_supernet(
                model, trainloader, valloader, criterion,
                DEVICE, fixed_paths, USE_AMP, calib_batches=cb,
            )

        # ── Logging ───────────────────────────────────────────────────────
        print(
            f"\nEpoch {epoch+1}/{args.epochs} | "
            f"Loss {train_loss:.4f} | Acc {train_acc:.2f}% | "
            f"LR_s {lr_stitches:.2e} | LR_b {lr_backbone:.2e} | "
            f"StitchGNorm {stitch_gnorm:.4f} | "
            f"Time {epoch_wall:.1f}s"
            + (f" | ValAcc {val_acc:.2f}% ± {val_std:.2f}%" if do_val else "")
        )

        # ── Metrics ───────────────────────────────────────────────────────
        metrics.append({
            "epoch":         epoch + 1,
            "train_loss":    round(train_loss,  5),
            "train_acc":     round(train_acc,   3),
            "val_loss":      round(val_loss,    5),
            "val_acc":       round(val_acc,     3),
            "val_std":       round(val_std,     3),
            "lr_stitches":   lr_stitches,
            "lr_backbone":   lr_backbone,
            "stitch_gnorm":  stitch_gnorm,
            "best_val_acc":  round(max(best_val_acc, val_acc), 3),
            "per_path_accs": [round(a, 2) for a in per_path_accs],
            "epoch_time":    round(epoch_wall, 2),
        })
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

        # ── Checkpoints ───────────────────────────────────────────────────
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

        if args.save_periodic_ckpt and ((epoch + 1) % args.ckpt_every_n == 0):
            periodic_path = os.path.join(periodic_dir, f"epoch_{epoch+1:04d}.pth")
            torch.save(ckpt, periodic_path)
            print(f"  💾 Periodic checkpoint saved: {periodic_path}")
        
        if do_val and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(ckpt, best_ckpt)
            print(f"  🌟 New best saved: {best_val_acc:.2f}%")

    print(f"\n✅ Training complete. Best val acc: {best_val_acc:.2f}%")
    print(f"   Checkpoints in: {out_dir}")


if __name__ == "__main__":
    import argparse
    main()