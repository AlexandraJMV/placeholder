# eval_ablation_a.py
"""
Ablation Study — Group A: Full standalone baselines (no stitching).

Trains resnet18, mobilenetv3_small_050, and efficientnet_b0 loaded directly
from timm with a fresh classification head for Imagenette (10 classes).
These are the reference upper bounds — same architecture, zero stitching overhead.

Protocol mirrors eval_gt.py v2:
- AdamW, single LR group (no differential — no stitching layers exist)
- Weight decay 1e-4 on decay params, 0.0 on bias/BN
- Linear warmup → CosineAnnealingLR
- Validate every epoch, val_acc + val_loss
- Early stopping with configurable patience
- Saves best.pth, latest.pth, loss_curve.json per model

Output schema per model (ablation_a_results.json):
    {
        "model_name":    str,
        "group":         "A",
        "best_acc":      float,
        "epochs":        int,
        "lr":            float,
        "n_params":      int,
        "ckpt_dir":      str,
        "time":          float,
        "epoch_time":    float,
        "epoch_std":     float,
        "stopped_early": bool,
        "stopped_epoch": int,
    }
"""
import os, sys, json, argparse, random, gc, time as time_module
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'third_package'))

import timm
from simple_poc.train_supernet import get_imagenette


MODELS_GROUP_A = ['resnet18', 'mobilenetv3_small_050', 'efficientnet_b0']
NUM_CLASSES    = 10


# ── Reproducibility ───────────────────────────────────────────────────────────
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark     = False


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Ablation Group A: full timm baselines")
    p.add_argument('--ablation_dir',   type=str, default="dery/ablation",
                   help="Root ablation directory")
    p.add_argument('--models',         type=str, default=None,
                   help="Comma-separated subset of models to run, "
                        "e.g. 'resnet18,efficientnet_b0'. Default: all 3.")
    p.add_argument('--epochs',         type=int,   default=60)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--weight_decay',   type=float, default=1e-4)
    p.add_argument('--batch_size',     type=int,   default=256)
    p.add_argument('--warmup_epochs',  type=int,   default=5)
    p.add_argument('--patience',       type=int,   default=15)
    p.add_argument('--img_size',       type=int,   default=160)
    p.add_argument('--no_amp',         action='store_true')
    return p.parse_args()


# ── Parameter groups (differential LR — head at lr, backbone at lr*0.1) ──────
# Head module names by architecture (timm naming convention):
#   resnet18               → model.fc
#   efficientnet_b0        → model.classifier
#   mobilenetv3_small_050  → model.classifier
HEAD_PARAM_PREFIXES = ('fc.', 'classifier.')

def build_optimizer(model, lr, weight_decay):
    """
    Differential LR mirroring eval_gt.py protocol:
        head params  → lr        (same as stitches/head in GT)
        backbone     → lr * 0.1  (same as backbone blocks in GT)
    Within each group, bias/BN params get weight_decay=0.0.

    This is required for a fair comparison with stitched subnets (Groups B/C)
    which use the same differential LR scheme. Using a single LR for Group A
    caused catastrophic forgetting (val_acc collapsed after epoch 1).
    """
    head_decay,     head_no_decay     = [], []
    backbone_decay, backbone_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        is_head    = any(name.startswith(pfx) for pfx in HEAD_PARAM_PREFIXES)
        no_decay   = (param.dim() == 1) or name.endswith('.bias')
        if is_head:
            (head_no_decay     if no_decay else head_decay).append(param)
        else:
            (backbone_no_decay if no_decay else backbone_decay).append(param)

    return optim.AdamW([
        {'params': head_decay,       'lr': lr,       'weight_decay': weight_decay},
        {'params': head_no_decay,    'lr': lr,       'weight_decay': 0.0},
        {'params': backbone_decay,   'lr': lr * 0.1, 'weight_decay': weight_decay},
        {'params': backbone_no_decay,'lr': lr * 0.1, 'weight_decay': 0.0},
    ])


# ── Scheduler ─────────────────────────────────────────────────────────────────
def build_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    warmup = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=eta_min)
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])


# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, device, use_amp):
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct, total, running_loss = 0, 0, 0.0
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
        running_loss += loss.item()
        correct      += outputs.max(1)[1].eq(targets).sum().item()
        total        += targets.size(0)
    model.train()
    return round(100.0 * correct / total, 4), round(running_loss / len(val_loader), 5)


# ── Training loop ─────────────────────────────────────────────────────────────
def train_model(
    model, model_name,
    train_loader, val_loader,
    epochs, lr, weight_decay,
    warmup_epochs, patience,
    device, use_amp, ckpt_dir,
    resume=True,
):
    os.makedirs(ckpt_dir, exist_ok=True)
    best_ckpt    = os.path.join(ckpt_dir, 'best.pth')
    latest_ckpt  = os.path.join(ckpt_dir, 'latest.pth')
    metrics_path = os.path.join(ckpt_dir, 'metrics.json')
    curve_path   = os.path.join(ckpt_dir, 'loss_curve.json')

    optimizer = build_optimizer(model, lr, weight_decay)
    scheduler = build_scheduler(optimizer, warmup_epochs, epochs)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    start_epoch    = 0
    best_val_acc   = 0.0
    no_improve_cnt = 0
    existing_metrics = []

    if resume and os.path.exists(latest_ckpt):
        print(f"    [Resume] Loading: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch    = ckpt['epoch']
        best_val_acc   = ckpt['best_val_acc']
        no_improve_cnt = ckpt.get('no_improve_cnt', 0)
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                existing_metrics = json.load(f)
        print(f"    [Resume] Epoch {start_epoch}, best={best_val_acc:.2f}%, "
              f"no_improve={no_improve_cnt}")

    if start_epoch >= epochs:
        print(f"    [Skip] Already completed {start_epoch}/{epochs} epochs.")
        epoch_times = [m.get('epoch_time', 0.0) for m in existing_metrics]
        return best_val_acc, epoch_times, start_epoch

    metrics     = list(existing_metrics)
    epoch_times = [m.get('epoch_time', 0.0) for m in existing_metrics]
    stopped_epoch = epochs
    model.train()

    for epoch in range(start_epoch, epochs):
        t0 = time_module.perf_counter()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                loss    = criterion(outputs, targets)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()
            running_loss += loss.item()
            correct      += outputs.max(1)[1].eq(targets).sum().item()
            total        += targets.size(0)

        scheduler.step()
        epoch_wall = time_module.perf_counter() - t0
        epoch_times.append(epoch_wall)

        train_loss  = round(running_loss / len(train_loader), 5)
        train_acc   = round(100.0 * correct / total, 3)
        lr_head     = round(optimizer.param_groups[0]['lr'], 8)  # head_decay group
        lr_backbone = round(optimizer.param_groups[2]['lr'], 8)  # backbone_decay group
        val_acc, val_loss = evaluate(model, val_loader, device, use_amp)

        improved = val_acc > best_val_acc
        if improved:
            best_val_acc   = val_acc
            no_improve_cnt = 0
            torch.save({
                'epoch': epoch + 1, 'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'scaler': scaler.state_dict(),
                'val_acc': val_acc, 'best_val_acc': best_val_acc,
                'no_improve_cnt': no_improve_cnt,
                'model_name': model_name,
            }, best_ckpt)
            print(f"    🌟 Epoch {epoch+1:3d} | Loss {train_loss:.4f} | "
                  f"TrainAcc {train_acc:.1f}% | ValAcc {val_acc:.2f}% | "
                  f"ValLoss {val_loss:.4f} ← best")
        else:
            no_improve_cnt += 1
            print(f"    Epoch {epoch+1:3d} | Loss {train_loss:.4f} | "
                  f"TrainAcc {train_acc:.1f}% | ValAcc {val_acc:.2f}% | "
                  f"ValLoss {val_loss:.4f} | NoImprove {no_improve_cnt}/{patience}")

        torch.save({
            'epoch': epoch + 1, 'state_dict': model.state_dict(),
            'optimizer': optimizer.state_dict(),
            'scheduler': scheduler.state_dict(),
            'scaler': scaler.state_dict(),
            'val_acc': val_acc, 'best_val_acc': best_val_acc,
            'no_improve_cnt': no_improve_cnt,
            'model_name': model_name,
        }, latest_ckpt)

        metrics.append({
            "epoch":        epoch + 1,
            "train_loss":   train_loss,
            "train_acc":    train_acc,
            "val_acc":      val_acc,
            "val_loss":     val_loss,
            "lr_head":      lr_head,
            "lr_backbone":  lr_backbone,
            "epoch_time":   round(epoch_wall, 3),
            "best_val_acc": round(best_val_acc, 4),
        })

        for p in (metrics_path, curve_path):
            with open(p, 'w') as f:
                json.dump(metrics, f, indent=4)

        if no_improve_cnt >= patience:
            stopped_epoch = epoch + 1
            print(f"    ⏹  Early stop at epoch {stopped_epoch} "
                  f"(patience={patience}). Best: {best_val_acc:.2f}%")
            break
    else:
        stopped_epoch = epochs

    return best_val_acc, epoch_times, stopped_epoch


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(42)

    group_a_dir = os.path.join(args.ablation_dir, "group_a")
    models_dir  = os.path.join(group_a_dir, "models")
    os.makedirs(group_a_dir, exist_ok=True)
    os.makedirs(models_dir,  exist_ok=True)

    DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_AMP = (not args.no_amp) and (DEVICE == 'cuda')

    output_path = os.path.join(group_a_dir, "ablation_a_results.json")

    # Which models to run
    targets = ([m.strip() for m in args.models.split(',')]
               if args.models else MODELS_GROUP_A)
    invalid = [m for m in targets if m not in MODELS_GROUP_A]
    if invalid:
        raise ValueError(f"Unknown model(s): {invalid}. "
                         f"Valid: {MODELS_GROUP_A}")

    print(f"[AblationA] Models     : {targets}")
    print(f"[AblationA] Epochs     : {args.epochs}  |  LR: {args.lr}")
    print(f"[AblationA] Warmup     : {args.warmup_epochs}  |  Patience: {args.patience}")
    print(f"[AblationA] Output     : {output_path}\n")

    # Resume summary
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for entry in json.load(f):
                existing[entry['model_name']] = entry
        print(f"[AblationA] {len(existing)} model(s) already in summary")

    gen = torch.Generator()
    gen.manual_seed(42)
    trainset, valset = get_imagenette(img_size=args.img_size)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, generator=gen,
                             num_workers=4, pin_memory=True)
    valloader   = DataLoader(valset,   batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    summary = dict(existing)

    for model_name in targets:
        ckpt_dir = os.path.join(models_dir, model_name)

        # Skip if fully trained
        if model_name in existing:
            latest = os.path.join(ckpt_dir, 'latest.pth')
            if os.path.exists(latest):
                ckpt = torch.load(latest, map_location='cpu', weights_only=False)
                if (ckpt['epoch'] >= args.epochs or
                        ckpt.get('no_improve_cnt', 0) >= args.patience):
                    print(f"  [skip] {model_name} fully trained "
                          f"({ckpt['epoch']} epochs)")
                    continue

        print(f"\n[AblationA] ── {model_name} ──────────────────────────────")

        # Load full pretrained model from timm, replace head for 10 classes
        model = timm.create_model(
            model_name, pretrained=True, num_classes=NUM_CLASSES)
        model = model.to(DEVICE)
        n_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {n_params:,}")

        t_start = time_module.perf_counter()
        best_acc, epoch_times, stopped_epoch = train_model(
            model         = model,
            model_name    = model_name,
            train_loader  = trainloader,
            val_loader    = valloader,
            epochs        = args.epochs,
            lr            = args.lr,
            weight_decay  = args.weight_decay,
            warmup_epochs = args.warmup_epochs,
            patience      = args.patience,
            device        = DEVICE,
            use_amp       = USE_AMP,
            ckpt_dir      = ckpt_dir,
            resume        = True,
        )

        total_time    = round(time_module.perf_counter() - t_start, 2)
        epoch_arr     = np.array(epoch_times) if epoch_times else np.array([0.0])
        stopped_early = stopped_epoch < args.epochs

        print(f"  ✅ {model_name} done.")
        print(f"     Best val acc  : {best_acc:.2f}%")
        print(f"     Stopped epoch : {stopped_epoch}/{args.epochs} "
              f"({'early stop' if stopped_early else 'full run'})")
        print(f"     Total time    : {total_time:.1f}s  |  "
              f"Epoch avg {epoch_arr.mean():.1f}s ± {epoch_arr.std():.1f}s")

        del model
        torch.cuda.empty_cache()
        gc.collect()

        summary[model_name] = {
            "model_name":    model_name,
            "group":         "A",
            "best_acc":      best_acc,
            "epochs":        args.epochs,
            "lr":            args.lr,
            "n_params":      n_params,
            "ckpt_dir":      ckpt_dir,
            "time":          total_time,
            "epoch_time":    round(float(epoch_arr.mean()), 3),
            "epoch_std":     round(float(epoch_arr.std()),  3),
            "stopped_early": stopped_early,
            "stopped_epoch": stopped_epoch,
        }

        with open(output_path, 'w') as f:
            json.dump(list(summary.values()), f, indent=4)
        print(f"  📄 Summary updated: {output_path}")

    print(f"\n✅ Group A complete. {len(summary)} model(s) in {output_path}")


if __name__ == "__main__":
    main()