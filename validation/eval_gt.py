# eval_gt.py
"""
Phase 2: Ground truth training for a selected subset of paths.

Protocol (v3 - Robust NAS Baseline):
- Moving Average (Smoothed) Checkpointing to eliminate stochastic noise.
- AdamW optimizer with differential LR (stitches/head at lr, backbone at lr*0.1)
- Weight decay 1e-4 on decay groups; 0.0 on bias/BN params
- Linear warmup followed by CosineAnnealingLR
- val_loss and val_acc are smoothed over a sliding window (default 5 epochs)
- best.pth is saved based on the MAXIMUM SMOOTHED validation accuracy.
- Early stopping is triggered by SMOOTHED validation loss.
"""
import os, sys, json, argparse, random, copy, gc, time as time_module
from collections import deque
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'third_package'))

from simple_poc.supernet import SuperNetwork
from simple_poc.train_supernet import get_imagenette

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
    p = argparse.ArgumentParser(description="Phase 2: GT training (Smoothed Protocol)")
    p.add_argument('--plan_path',      type=str,   default="network_plan.pkl")
    p.add_argument('--paths_file',     type=str,   default="eval_paths_universe.json")
    p.add_argument('--gt_dir',         type=str,   default="dery/validation/gt",
                   help="Root GT directory.")
    p.add_argument('--start_idx',      type=int,   default=None)
    p.add_argument('--end_idx',        type=int,   default=None)
    p.add_argument('--indices',        type=str,   default=None)
    
    # Training protocol
    p.add_argument('--epochs_gt',      type=int,   default=200)
    p.add_argument('--lr',             type=float, default=1e-3)
    p.add_argument('--weight_decay',   type=float, default=1e-4)
    p.add_argument('--batch_size',     type=int,   default=64)
    p.add_argument('--warmup_epochs',  type=int,   default=5)
    p.add_argument('--patience',       type=int,   default=15)
    
    # Robustness parameter
    p.add_argument('--smooth_k',       type=int,   default=5,
                   help="Number of epochs to average for stable accuracy/loss tracking")
    
    p.add_argument('--no_amp',         action='store_true')
    p.add_argument('--seed',           type=int,   default=42)
    return p.parse_args()

# ── SubNetworkExtractor ───────────────────────────────────────────────────────
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
            out = self.stages[i + 1](out)
        return self.head(self.global_pool(out))

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

def compute_stitch_grad_norm(model):
    total_sq = 0.0
    for param in model.stitches.parameters():
        if param.grad is not None:
            total_sq += param.grad.detach().float().norm(2).item() ** 2
    return round(total_sq ** 0.5, 6)

def build_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=0.1, total_iters=warmup_epochs)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(total_epochs - warmup_epochs, 1), eta_min=eta_min)
    return optim.lr_scheduler.SequentialLR(optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs])

# ── GT Training ───────────────────────────────────────────────────────────────
def train_gt_subnet(
    model, train_loader, val_loader, epochs, lr, weight_decay,
    warmup_epochs, patience, smooth_k, device, use_amp,
    ckpt_dir, global_idx, path
):
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt  = os.path.join(ckpt_dir, 'latest.pth')
    best_ckpt    = os.path.join(ckpt_dir, 'best.pth')
    metrics_path = os.path.join(ckpt_dir, 'metrics.json')

    stitch_decay, stitch_no_decay, backbone_decay, backbone_no_decay = [], [], [], []
    for name, param in model.named_parameters():
        if not param.requires_grad: continue
        no_decay = (param.dim() == 1) or name.endswith(".bias")
        if 'stages' in name:
            (backbone_no_decay if no_decay else backbone_decay).append(param)
        else:
            (stitch_no_decay if no_decay else stitch_decay).append(param)

    optimizer = optim.AdamW([
        {'params': stitch_decay,      'lr': lr,       'weight_decay': weight_decay},
        {'params': stitch_no_decay,   'lr': lr,       'weight_decay': 0.0},
        {'params': backbone_decay,    'lr': lr * 0.1, 'weight_decay': weight_decay},
        {'params': backbone_no_decay, 'lr': lr * 0.1, 'weight_decay': 0.0},
    ])

    scheduler = build_scheduler(optimizer, warmup_epochs, epochs)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    # Moving Average Tracking
    val_acc_window  = deque(maxlen=smooth_k)
    val_loss_window = deque(maxlen=smooth_k)
    
    best_smoothed_acc  = 0.0
    best_smoothed_loss = float('inf')
    no_improve_cnt     = 0
    metrics, epoch_times = [], []

    model.train()
    stopped_epoch = epochs
    stopped_by_es = False

    for epoch in range(epochs):
        epoch_start = time_module.perf_counter()
        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                loss    = criterion(outputs, targets)

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            stitch_gnorm = compute_stitch_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            correct      += outputs.max(1)[1].eq(targets).sum().item()
            total        += targets.size(0)

        scheduler.step()
        epoch_times.append(time_module.perf_counter() - epoch_start)

        train_loss = running_loss / len(train_loader)
        train_acc  = 100.0 * correct / total
        
        # Validation & Smoothing
        val_acc, val_loss = evaluate(model, val_loader, device, use_amp)
        val_acc_window.append(val_acc)
        val_loss_window.append(val_loss)
        
        # Only start tracking bests once the window is full
        smoothed_acc  = sum(val_acc_window) / len(val_acc_window)
        smoothed_loss = sum(val_loss_window) / len(val_loss_window)
        
        improved_acc = False
        improved_loss = False
        
        if len(val_acc_window) == smooth_k:
            improved_acc  = smoothed_acc > best_smoothed_acc
            improved_loss = smoothed_loss < best_smoothed_loss
            
            if improved_acc:
                best_smoothed_acc = smoothed_acc
                torch.save({
                    'epoch': epoch + 1,
                    'state_dict': model.state_dict(),
                    'val_acc_raw': val_acc,
                    'smoothed_acc': smoothed_acc,
                    'global_idx': global_idx,
                    'path': path,
                }, best_ckpt)

            if improved_loss:
                best_smoothed_loss = smoothed_loss
                no_improve_cnt = 0
            else:
                no_improve_cnt += 1

        acc_m  = " ← BEST SMOOTHED" if improved_acc else ""
        print(f"    Epoch {epoch+1:3d} | TrainAcc {train_acc:.1f}% | Raw ValAcc {val_acc:.2f}% | "
              f"SmoothAcc {smoothed_acc:.2f}%{acc_m} | SmoothLoss {smoothed_loss:.4f} | NoImp {no_improve_cnt}/{patience}")

        metrics.append({
            "epoch": epoch + 1, "train_acc": round(train_acc, 2), "val_acc": val_acc,
            "smoothed_acc": round(smoothed_acc, 3), "smoothed_loss": round(smoothed_loss, 4),
            "best_smoothed_acc": round(best_smoothed_acc, 3)
        })

        if no_improve_cnt >= patience and len(val_acc_window) == smooth_k:
            stopped_epoch = epoch + 1
            stopped_by_es = True
            print(f"    ⏹  Early stop at epoch {stopped_epoch}. Best Smoothed Acc: {best_smoothed_acc:.2f}%")
            break

    return best_smoothed_acc, epoch_times, stopped_epoch, stopped_by_es

# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    models_dir = os.path.join(args.gt_dir, "models")
    os.makedirs(models_dir, exist_ok=True)
    DEVICE, USE_AMP = ('cuda' if torch.cuda.is_available() else 'cpu'), not args.no_amp

    with open(args.paths_file) as f: all_paths = json.load(f)
    selected_indices = list(range(args.start_idx, args.end_idx)) if args.indices is None else [int(x) for x in args.indices.split(',')]
    output_path = os.path.join(args.gt_dir, f"gt_results_{args.seed}.json")

    summary = {}
    trainset, valset = get_imagenette(img_size=160)
    gen = torch.Generator(); gen.manual_seed(args.seed)
    
    def seed_worker(worker_id):
        s = torch.initial_seed() % 2**32
        np.random.seed(s); random.seed(s)

    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, generator=gen, worker_init_fn=seed_worker, num_workers=4)
    valloader   = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4)

    blueprint = SuperNetwork(args.plan_path, input_size=160).to('cpu')

    for global_idx in selected_indices:
        print(f"\n[GT] ── Path {global_idx}: {all_paths[global_idx]} (Seed {args.seed}) ──────────")
        subnet = SubNetworkExtractor(blueprint, all_paths[global_idx]).to(DEVICE)
        
        t_start = time_module.perf_counter()
        best_smoothed_acc, epoch_times, stopped_epoch, stopped_by_es = train_gt_subnet(
            subnet, trainloader, valloader, args.epochs_gt, args.lr, args.weight_decay,
            args.warmup_epochs, args.patience, args.smooth_k, DEVICE, USE_AMP,
            os.path.join(models_dir, f"path_{global_idx}"), global_idx, all_paths[global_idx]
        )

        del subnet; torch.cuda.empty_cache(); gc.collect()

        summary[global_idx] = {
            "global_idx": global_idx, "path": all_paths[global_idx],
            "gt_acc": best_smoothed_acc, # Now mathematically stable!
            "epochs_gt": args.epochs_gt, "smooth_k": args.smooth_k,
            "seed": args.seed, "n_params": sum(p.numel() for p in SubNetworkExtractor(blueprint, all_paths[global_idx]).parameters()),
            "time": round(time_module.perf_counter() - t_start, 2),
            "stopped_early": stopped_by_es, "stopped_epoch": stopped_epoch,
        }

        with open(output_path, 'w') as f:
            json.dump(sorted(summary.values(), key=lambda x: x['global_idx']), f, indent=4)

if __name__ == "__main__":
    main()