# eval_gt.py
"""
Phase 2: Ground truth training for a selected subset of paths.
Each path gets its own checkpoint directory and metrics JSON,
mirroring supernet training structure.

Supports resume: if latest.pth exists for a path, training continues
from that checkpoint rather than restarting.

Metrics recorded per epoch enable retroactive convergence analysis
(Experiment 2) without retraining.

Index control (mutually exclusive):
  --start_idx 0 --end_idx 10     → paths 0..9
  --indices "0,3,7,12"           → specific global indices
"""
import os, sys, json, argparse, random, copy, gc, glob
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'third_package'))

from simple_poc.supernet import SuperNetwork
from simple_poc.train_supernet import get_imagenette

def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def parse_args():
    p = argparse.ArgumentParser(description="Phase 2: GT training for selected paths")
    p.add_argument('--plan_path',    type=str, default="network_plan.pkl")
    p.add_argument('--paths_file',   type=str, default="eval_paths_universe.json")
    p.add_argument('--gt_dir',       type=str, default="dery/validation/gt",
                   help="Root GT directory. Summaries saved here, models in gt_dir/models/")
    # Index control
    p.add_argument('--start_idx',    type=int, default=None)
    p.add_argument('--end_idx',      type=int, default=None)
    p.add_argument('--indices',      type=str, default=None,
                   help="Comma-separated global indices, e.g. '0,3,7,12'")
    # Training protocol — matched to supernet
    p.add_argument('--epochs_gt',    type=int,   default=150)
    p.add_argument('--lr',           type=float, default=0.001)
    p.add_argument('--batch_size',   type=int,   default=64)
    p.add_argument('--val_freq',     type=int,   default=10,
                   help="Validate every N epochs. Always validates on final epoch.")
    p.add_argument('--no_amp',       action='store_true')
    return p.parse_args()


# ── SubNetworkExtractor ────────────────────────────────────────────────────────
class SubNetworkExtractor(nn.Module):
    def __init__(self, blueprint: SuperNetwork, path: list):
        super().__init__()
        self.path     = path
        self.stages   = nn.ModuleList([copy.deepcopy(blueprint.stages[0][path[0]])])
        self.stitches = nn.ModuleList()
        for i in range(1, blueprint.num_stages):
            self.stitches.append(copy.deepcopy(blueprint.stitches[i-1][path[i-1]][path[i]]))
            self.stages.append(copy.deepcopy(blueprint.stages[i][path[i]]))
        self.global_pool = copy.deepcopy(blueprint.global_pool)
        self.head        = copy.deepcopy(blueprint.heads[path[-1]])

    def forward(self, x):
        out = self.stages[0](x)
        for i in range(len(self.stitches)):
            out = self.stitches[i](out)
            out = self.stages[i+1](out)
        return self.head(self.global_pool(out))


# ── Validation ────────────────────────────────────────────────────────────────
@torch.no_grad()
def evaluate(model, val_loader, device, use_amp):
    model.eval()
    correct, total = 0, 0
    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
        correct += outputs.max(1)[1].eq(targets).sum().item()
        total   += targets.size(0)
    model.train()
    return round(100.0 * correct / total, 4)


# ── GT Training ───────────────────────────────────────────────────────────────
def train_gt_subnet(model, train_loader, val_loader,
                    epochs, lr, device, use_amp,
                    val_freq, ckpt_dir, global_idx, path,
                    resume=True):
    """
    Full training loop mirroring supernet protocol:
    - SGD + momentum 0.9
    - Differential LR: stitches/head at lr, backbone stages at lr*0.1
    - Gradient clipping max_norm=5.0
    - CosineAnnealingLR with eta_min=1e-5
    - AMP
    - Per-epoch metrics saved to metrics.json
    - best.pth and latest.pth checkpoints
    - Resume from latest.pth if it exists and resume=True

    Returns: best_val_acc (float)
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt  = os.path.join(ckpt_dir, 'latest.pth')
    best_ckpt    = os.path.join(ckpt_dir, 'best.pth')
    metrics_path = os.path.join(ckpt_dir, 'metrics.json')

    # ── Parameter groups (mirrors supernet) ──────────────────────────────────
    stitch_decay,   stitch_no_decay   = [], []
    backbone_decay, backbone_no_decay = [], []

    for name, param in model.named_parameters():
        is_no_decay = len(param.shape) == 1 or name.endswith(".bias")
        if 'stages' in name:
            (backbone_no_decay if is_no_decay else backbone_decay).append(param)
        else:
            (stitch_no_decay   if is_no_decay else stitch_decay).append(param)

    optimizer = optim.SGD([
        {'params': stitch_decay,      'lr': lr,       'weight_decay': 5e-4},
        {'params': stitch_no_decay,   'lr': lr,       'weight_decay': 0.0},
        {'params': backbone_decay,    'lr': lr * 0.1, 'weight_decay': 5e-4},
        {'params': backbone_no_decay, 'lr': lr * 0.1, 'weight_decay': 0.0},
    ], momentum=0.9)

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=epochs, eta_min=1e-5)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch  = 0
    best_val_acc = 0.0
    existing_metrics = []

    if resume and os.path.exists(latest_ckpt):
        print(f"    [Resume] Loading checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch  = ckpt['epoch']
        best_val_acc = ckpt['best_val_acc']
        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                existing_metrics = json.load(f)
        print(f"    [Resume] Continuing from epoch {start_epoch}, "
              f"best_val_acc={best_val_acc:.2f}%")

    if start_epoch >= epochs:
        print(f"    [Skip] Already completed {start_epoch}/{epochs} epochs.")
        return best_val_acc

    metrics = existing_metrics
    model.train()

    for epoch in range(start_epoch, epochs):
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
            _, predicted  = outputs.max(1)
            correct      += predicted.eq(targets).sum().item()
            total        += targets.size(0)

        scheduler.step()

        train_loss = running_loss / len(train_loader)
        train_acc  = round(100.0 * correct / total, 3)
        current_lr = optimizer.param_groups[0]['lr']

        # Validate on val_freq cadence and always on last epoch
        is_last   = (epoch == epochs - 1)
        do_val    = is_last or ((epoch + 1) % val_freq == 0)
        val_acc   = evaluate(model, val_loader, device, use_amp) if do_val else None

        if val_acc is not None and val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch':        epoch + 1,
                'state_dict':   model.state_dict(),
                'optimizer':    optimizer.state_dict(),
                'scheduler':    scheduler.state_dict(),
                'scaler':       scaler.state_dict(),
                'val_acc':      val_acc,
                'best_val_acc': best_val_acc,
                'global_idx':   global_idx,
                'path':         path,
            }, best_ckpt)
            print(f"    🌟 Epoch {epoch+1:3d} | "
                  f"Loss {train_loss:.4f} | Train {train_acc:.1f}% | "
                  f"Val {val_acc:.2f}% ← best")
        else:
            val_str = f"Val {val_acc:.2f}%" if val_acc is not None else "Val —"
            print(f"    Epoch {epoch+1:3d} | "
                  f"Loss {train_loss:.4f} | Train {train_acc:.1f}% | {val_str}")

        # Always save latest
        torch.save({
            'epoch':        epoch + 1,
            'state_dict':   model.state_dict(),
            'optimizer':    optimizer.state_dict(),
            'scheduler':    scheduler.state_dict(),
            'scaler':       scaler.state_dict(),
            'val_acc':      val_acc,
            'best_val_acc': best_val_acc,
            'global_idx':   global_idx,
            'path':         path,
        }, latest_ckpt)

        # Record metrics — val_acc is None on skipped epochs, stored as 0.0
        # (matches supernet JSON convention; plotting script already handles this)
        metrics.append({
            "epoch":      epoch + 1,
            "train_loss": round(train_loss, 5),
            "train_acc":  train_acc,
            "val_acc":    round(val_acc, 3) if val_acc is not None else 0.0,
            "val_std":    0.0,   # not applicable for standalone; kept for schema compat
            "lr":         current_lr,
            "best_val_acc": round(best_val_acc, 3),
        })
        with open(metrics_path, 'w') as f:
            json.dump(metrics, f, indent=4)

    return best_val_acc


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(42)

    models_dir = os.path.join(args.gt_dir, "models")
    os.makedirs(args.gt_dir,  exist_ok=True)
    os.makedirs(models_dir,   exist_ok=True)

    DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_AMP = (not args.no_amp) and (DEVICE == 'cuda')

    with open(args.paths_file) as f:
        all_paths = json.load(f)

    # ── Index resolution ─────────────────────────────────────────────────────
    if args.indices is not None:
        selected_indices = [int(x) for x in args.indices.split(',')]
        label = f"custom_{min(selected_indices)}_{max(selected_indices)}"
    elif args.start_idx is not None and args.end_idx is not None:
        selected_indices = list(range(args.start_idx, args.end_idx))
        label = f"{args.start_idx}_{args.end_idx}"
    else:
        raise ValueError("Provide either --start_idx/--end_idx or --indices")

    output_path = os.path.join(args.gt_dir, f"gt_results_{label}.json")

    print(f"[GT] Paths to evaluate: {selected_indices}")
    print(f"[GT] Epochs: {args.epochs_gt} | LR: {args.lr} "
          f"(backbone: {args.lr*0.1}) | Val every: {args.val_freq} epochs")
    print(f"[GT] Checkpoints: {models_dir}/path_{{idx}}/")
    print(f"[GT] Summary output: {output_path}")

    # ── Resume summary ───────────────────────────────────────────────────────
    existing_summary = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for entry in json.load(f):
                existing_summary[entry['global_idx']] = entry
        print(f"[GT] {len(existing_summary)} paths already in summary")

    gen = torch.Generator(); gen.manual_seed(42)
    trainset, valset = get_imagenette(img_size=160)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, generator=gen,
                             num_workers=4, pin_memory=True)
    valloader   = DataLoader(valset, batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    print("[GT] Loading blueprint SuperNetwork on CPU...")
    blueprint = SuperNetwork(args.plan_path, input_size=160).to('cpu')

    summary = dict(existing_summary)

    for global_idx in selected_indices:
        if global_idx >= len(all_paths):
            print(f"  [skip] Index {global_idx} out of range")
            continue

        path     = all_paths[global_idx]
        ckpt_dir = os.path.join(models_dir, f"path_{global_idx}")

        # Only skip if summary entry exists AND training fully completed
        if global_idx in summary:
            latest = os.path.join(ckpt_dir, 'latest.pth')
            if os.path.exists(latest):
                ckpt = torch.load(latest, map_location='cpu', weights_only=False)
                if ckpt['epoch'] >= args.epochs_gt:
                    print(f"  [skip] Path {global_idx} fully trained "
                          f"({ckpt['epoch']} epochs)")
                    continue

        print(f"\n[GT] ── Path {global_idx}: {path} ──────────────────────────")
        subnet = SubNetworkExtractor(blueprint, path).to(DEVICE)

        best_val_acc = train_gt_subnet(
            model        = subnet,
            train_loader = trainloader,
            val_loader   = valloader,
            epochs       = args.epochs_gt,
            lr           = args.lr,
            device       = DEVICE,
            use_amp      = USE_AMP,
            val_freq     = args.val_freq,
            ckpt_dir     = ckpt_dir,
            global_idx   = global_idx,
            path         = path,
            resume       = True,
        )

        print(f"  ✅ Path {global_idx} done. Best val acc: {best_val_acc:.2f}%")

        del subnet; torch.cuda.empty_cache(); gc.collect()

        summary[global_idx] = {
            "global_idx":  global_idx,
            "path":        path,
            "gt_acc":      best_val_acc,   # best checkpoint, not last epoch
            "epochs_gt":   args.epochs_gt,
            "lr":          args.lr,
            "ckpt_dir":    ckpt_dir,
        }

        with open(output_path, 'w') as f:
            json.dump(
                sorted(summary.values(), key=lambda x: x['global_idx']),
                f, indent=4)

    print(f"\n✅ Complete. {len(summary)} paths in {output_path}")

if __name__ == "__main__":
    main()