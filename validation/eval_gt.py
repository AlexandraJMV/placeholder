# eval_gt.py
"""
Phase 2: Ground truth training for a selected subset of paths.

Protocol (v2):
- AdamW optimizer with differential LR (stitches/head at lr, backbone at lr*0.1)
- Weight decay 1e-4 on decay groups; 0.0 on bias/BN params
- Linear warmup (--warmup_epochs, default 5) followed by CosineAnnealingLR
- val_freq=1: validate every epoch for accurate best-checkpoint tracking
- val_loss recorded every epoch alongside val_acc
- Stitch gradient norm recorded every epoch (after unscale, before clip)
- Both LR groups (stitches, backbone) recorded per epoch
- Early stopping with configurable patience (--patience, default 10)
- n_params, total time, epoch_time mean/std recorded in summary
- loss_curve.json saved inside ckpt_dir alongside best.pth / latest.pth
- Summary schema includes: global_idx, path, gt_acc, epochs_gt, lr,
  n_params, ckpt_dir, time, epoch_time, epoch_std, stopped_early, stopped_epoch

Index control (mutually exclusive):
  --start_idx 0 --end_idx 10     → paths 0..9
  --indices "0,3,7,12"           → specific global indices
"""
import os, sys, json, argparse, random, copy, gc, time as time_module
import torch, torch.nn as nn, torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'third_package'))

from simple_poc.supernet import SuperNetwork
from simple_poc.train_supernet import get_dataset, DATASET_NUM_CLASSES  # antes: get_imagenette


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
    p = argparse.ArgumentParser(description="Phase 2: GT training for selected paths")
    p.add_argument('--plan_path',      type=str,   default="network_plan.pkl")
    p.add_argument('--paths_file',     type=str,   default="eval_paths_universe.json")
    
    # Index control
    p.add_argument('--start_idx',      type=int,   default=None)
    p.add_argument('--end_idx',        type=int,   default=None)
    p.add_argument('--indices',        type=str,   default=None,
                   help="Comma-separated global indices, e.g. '0,3,7,12'")
    # Training protocol
    p.add_argument('--epochs_gt',      type=int,   default=15,
                   help="Hard ceiling on training epochs (early stopping may exit earlier)")
    p.add_argument('--lr',             type=float, default=1e-3,
                   help="Base LR for stitches and head. Backbone receives lr*0.1")
    p.add_argument('--weight_decay',   type=float, default=1e-4,
                   help="AdamW weight decay applied to decay parameter groups")
    p.add_argument('--batch_size',     type=int,   default=64)
    p.add_argument('--warmup_epochs',  type=int,   default=5,
                   help="Linear warmup epochs before cosine annealing begins")
    p.add_argument('--patience',       type=int,   default=10,
                   help="Early stopping patience (epochs without val_ACC improvement). "
                        "Ignored when --no_early_stop is set.")
    p.add_argument('--no_early_stop',  action='store_true',
                   help="Disable early stopping entirely. Trains for exactly --epochs_gt "
                        "epochs. Recommended for GT reproducibility across seeds.")
    p.add_argument('--no_amp',         action='store_true')

    p.add_argument('--seed',           type=int,   default=42,
                   help="Random seed for weight init (stitches, head). Default: 42.")
    p.add_argument('--data_seed',      type=int,   default=0,
                   help="Separate seed for DataLoader shuffle. Fixed at 0 by default so "
                        "all GT runs see the same data order regardless of --seed. "
                        "Set explicitly only if you want to study data-order variance.")
    # img size arg
    p.add_argument('--img_size', type=int, default=160)
    
    # Dataset
    p.add_argument('--dataset',  type=str, default='imagenette',
               choices=['imagenette', 'cifar100', 'stl10'])
    p.add_argument('--data_root', type=str, default=None)
    p.add_argument('--gt_dir',         type=str,   default=None,
                   help="Root GT directory. Default: dery/validation/gt/{dataset}. "
                        "Summaries saved here, models in gt_dir/models/")
    
    return p.parse_args()


# ── SubNetworkExtractor ───────────────────────────────────────────────────────
class SubNetworkExtractor(nn.Module):
    """
    Extracts a standalone subnetwork from a SuperNetwork blueprint by
    deep-copying the selected blocks, stitching layers, and head.
    The extracted model is fully independent from the supernet.
    """
    def __init__(self, blueprint: SuperNetwork, path: list):
        super().__init__()
        self.path     = path
        self.stages   = nn.ModuleList([copy.deepcopy(blueprint.stages[0][path[0]])])
        self.stitches = nn.ModuleList()
        for i in range(1, blueprint.num_stages):
            self.stitches.append(
                copy.deepcopy(blueprint.stitches[i-1][path[i-1]][path[i]])
            )
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
    """
    Returns (val_acc: float, val_loss: float).
    val_loss is the mean CrossEntropyLoss over the validation set.
    """
    criterion = nn.CrossEntropyLoss()
    model.eval()
    correct, total = 0, 0
    running_loss   = 0.0

    for inputs, targets in val_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        with torch.amp.autocast('cuda', enabled=use_amp):
            outputs = model(inputs)
            loss    = criterion(outputs, targets)
        # Weight by actual batch size so the last (possibly smaller) batch
        # doesn't get equal weight — gives the true per-sample mean loss.
        n = targets.size(0)
        running_loss += loss.item() * n
        correct      += outputs.max(1)[1].eq(targets).sum().item()
        total        += n

    model.train()
    val_acc  = round(100.0 * correct / total, 4)
    val_loss = round(running_loss / total, 5)  # per-sample mean
    return val_acc, val_loss


# ── Stitch gradient norm ──────────────────────────────────────────────────────
def compute_stitch_grad_norm(model):
    """
    Computes the L2 norm of gradients for all stitching-layer parameters.
    Must be called after scaler.unscale_() and before clip_grad_norm_()
    so the raw (unclipped) norm is measured.
    Returns 0.0 if no stitch gradients exist (e.g. no stitching layers).
    """
    total_sq = 0.0
    for param in model.stitches.parameters():
        if param.grad is not None:
            total_sq += param.grad.detach().float().norm(2).item() ** 2
    return round(total_sq ** 0.5, 6)


# ── Scheduler: warmup + cosine ────────────────────────────────────────────────
def build_scheduler(optimizer, warmup_epochs, total_epochs, eta_min=1e-6):
    """
    LinearLR warmup from lr*0.1 → lr over `warmup_epochs` steps,
    then CosineAnnealingLR for the remaining epochs.

    SequentialLR chains the two schedulers at the milestone.
    eta_min is the floor LR at the end of cosine decay.
    """
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor = 0.1,
        end_factor   = 1.0,
        total_iters  = warmup_epochs,
    )

    cosine_epochs = max(total_epochs - warmup_epochs, 1)
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max   = cosine_epochs,
        eta_min = eta_min,
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers = [warmup, cosine],
        milestones = [warmup_epochs],
    )
    return scheduler


# ── GT Training ───────────────────────────────────────────────────────────────
def train_gt_subnet(
    model, train_loader, val_loader,
    epochs, lr, weight_decay,
    warmup_epochs, patience,
    no_early_stop,
    device, use_amp,
    ckpt_dir, global_idx, path,
    resume=True,
):
    """
    Full fine-tuning loop for a standalone subnetwork.

    Optimizer  : AdamW with differential LR
                   stitches + head  → lr          (weight_decay for decay group)
                   backbone stages  → lr * 0.1    (weight_decay for decay group)
                   bias / BN params → respective lr, weight_decay=0.0
    Scheduler  : Linear warmup (warmup_epochs) → CosineAnnealingLR
    Regularise : weight_decay=1e-4 on decay groups; gradient clip max_norm=5.0
    Validation : every epoch (val_freq=1); both val_acc and val_loss recorded
    Stitch norm: measured after unscale, before clip → raw signal

    Early stopping (when no_early_stop=False):
      Criterion : val_ACC does not improve for `patience` epochs.
      Rationale : val_acc is the ranking metric; stopping when acc plateaus
                  ensures gt_acc is always the true maximum of a completed
                  plateau, not an artifact of when val_loss happened to stop
                  improving (which is independent of accuracy).

    When no_early_stop=True:
      Training runs for exactly `epochs` epochs regardless of val_acc trend.
      This is the recommended mode for GT runs that need to be compared across
      seeds, because every path is evaluated under identical compute budgets.

    best.pth   : saved whenever val_acc reaches a new maximum
                 gt_acc reported in the summary is the true best accuracy
                 seen during training. This is correct for NAS path ranking.

    Checkpoints: best.pth (highest val_acc), latest.pth (most recent epoch)
    Curves     : loss_curve.json / metrics.json saved inside ckpt_dir

    Returns
    -------
    best_val_acc  : float        — maximum val_acc seen during training
    epoch_times   : list[float]  — wall-clock seconds per epoch
    stopped_epoch : int          — epoch at which training ended (1-indexed)
    stopped_by_es : bool         — True iff training ended via early stopping
    """
    os.makedirs(ckpt_dir, exist_ok=True)
    latest_ckpt  = os.path.join(ckpt_dir, 'latest.pth')
    best_ckpt    = os.path.join(ckpt_dir, 'best.pth')
    metrics_path = os.path.join(ckpt_dir, 'metrics.json')
    curve_path   = os.path.join(ckpt_dir, 'loss_curve.json')

    # ── Parameter groups ─────────────────────────────────────────────────────
    # Separate decay / no-decay for stitches+head and backbone stages.
    # No-decay: 1-D tensors (BN scale/bias, standalone bias terms).
    stitch_decay,   stitch_no_decay   = [], []
    backbone_decay, backbone_no_decay = [], []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        no_decay = (param.dim() == 1) or name.endswith(".bias")
        if 'stages' in name:
            (backbone_no_decay if no_decay else backbone_decay).append(param)
        else:
            # stitches + global_pool + head
            (stitch_no_decay if no_decay else stitch_decay).append(param)

    optimizer = optim.AdamW([
        {'params': stitch_decay,      'lr': lr,       'weight_decay': weight_decay},
        {'params': stitch_no_decay,   'lr': lr,       'weight_decay': 0.0},
        {'params': backbone_decay,    'lr': lr * 0.1, 'weight_decay': weight_decay},
        {'params': backbone_no_decay, 'lr': lr * 0.1, 'weight_decay': 0.0},
    ])

    scheduler = build_scheduler(optimizer, warmup_epochs, epochs, eta_min=1e-6)
    criterion = nn.CrossEntropyLoss()
    scaler    = torch.amp.GradScaler('cuda', enabled=use_amp)

    # ── Resume ───────────────────────────────────────────────────────────────
    start_epoch    = 0
    best_val_loss  = float('inf')  # tracked for logging only, not for ES
    best_val_acc   = 0.0           # running maximum val_acc → gt_acc / best.pth
    no_improve_cnt = 0             # counts epochs without val_acc improvement
    existing_metrics = []

    if resume and os.path.exists(latest_ckpt):
        print(f"    [Resume] Loading checkpoint: {latest_ckpt}")
        ckpt = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['state_dict'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        scaler.load_state_dict(ckpt['scaler'])
        start_epoch    = ckpt['epoch']
        best_val_loss  = ckpt.get('best_val_loss', float('inf'))
        # Prefer acc-based counter; fall back to old loss-based counter only for
        # backward compat with existing checkpoints (will reset naturally after
        # one epoch anyway since we now count acc stagnation).
        no_improve_cnt = ckpt.get('no_improve_cnt_acc',
                                  ckpt.get('no_improve_cnt', 0))

        # Backward compat: old checkpoints stored acc-at-best-loss in best_val_acc.
        # best_val_acc_running (if present) is the true running max — prefer it.
        best_val_acc = ckpt.get('best_val_acc_running',
                                ckpt.get('best_val_acc', 0.0))

        if os.path.exists(metrics_path):
            with open(metrics_path) as f:
                existing_metrics = json.load(f)
            # If checkpoint lacked best_val_acc_running, reconstruct from metrics.
            if 'best_val_acc_running' not in ckpt:
                best_val_acc = max(
                    m.get('val_acc', 0.0) for m in existing_metrics
                )

        print(f"    [Resume] Continuing from epoch {start_epoch}, "
              f"best_val_loss={best_val_loss:.4f}, "
              f"best_val_acc={best_val_acc:.2f}%, "
              f"no_improve_acc_cnt={no_improve_cnt}")

    if start_epoch >= epochs:
        print(f"    [Skip] Already completed {start_epoch}/{epochs} epochs.")
        epoch_times    = [m.get('epoch_time', 0.0) for m in existing_metrics]
        best_acc_epoch = max(
            (m['epoch'] for m in existing_metrics if m.get('val_acc', 0) == best_val_acc),
            default=start_epoch,
        )
        return best_val_acc, epoch_times, start_epoch, False, best_acc_epoch

    metrics      = list(existing_metrics)
    epoch_times  = [m.get('epoch_time', 0.0) for m in existing_metrics]
    model.train()
    stopped_epoch  = epochs  # overwritten if ES fires
    stopped_by_es  = False   # explicit flag — avoids false negative on last epoch
    best_acc_epoch = 0       # epoch (1-indexed) where best_val_acc was achieved

    for epoch in range(start_epoch, epochs):
        epoch_start = time_module.perf_counter()

        running_loss, correct, total = 0.0, 0, 0

        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)

            with torch.amp.autocast('cuda', enabled=use_amp):
                outputs = model(inputs)
                loss    = criterion(outputs, targets)

            scaler.scale(loss).backward()

            # Unscale before measuring stitch grad norm and before clipping
            scaler.unscale_(optimizer)
            stitch_gnorm = compute_stitch_grad_norm(model)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)

            scaler.step(optimizer)
            scaler.update()

            running_loss += loss.item()
            correct      += outputs.max(1)[1].eq(targets).sum().item()
            total        += targets.size(0)

        scheduler.step()

        epoch_wall = time_module.perf_counter() - epoch_start
        epoch_times.append(epoch_wall)

        train_loss = round(running_loss / len(train_loader), 5)
        train_acc  = round(100.0 * correct / total, 3)

        # Both LR groups recorded explicitly
        lr_stitches = round(optimizer.param_groups[0]['lr'], 8)
        lr_backbone = round(optimizer.param_groups[2]['lr'], 8)

        # Validate every epoch
        val_acc, val_loss = evaluate(model, val_loader, device, use_amp)

        # ── Criterion A: best.pth — driven by val_acc ─────────────────────
        # Saves the model with the highest accuracy seen so far.
        # This is what gt_acc reports, and what NAS path ranking uses.
        improved_acc = val_acc > best_val_acc
        if improved_acc:
            best_val_acc   = val_acc
            best_acc_epoch = epoch + 1
            torch.save({
                'epoch':                epoch + 1,
                'state_dict':           model.state_dict(),
                'optimizer':            optimizer.state_dict(),
                'scheduler':            scheduler.state_dict(),
                'scaler':               scaler.state_dict(),
                'val_acc':              val_acc,
                'val_loss':             val_loss,
                'best_val_loss':        best_val_loss,
                'best_val_acc':         best_val_acc,
                'best_val_acc_running': best_val_acc,  # same value, kept for compat
                'best_acc_epoch':       best_acc_epoch,
                'no_improve_cnt_acc':   0,
                'global_idx':           global_idx,
                'path':                 path,
            }, best_ckpt)

        # ── Criterion B: val_loss tracking (logging only) ─────────────────
        # val_loss is no longer the early stopping criterion.
        # It is still tracked and logged for analysis purposes.
        improved_loss = val_loss < best_val_loss
        if improved_loss:
            best_val_loss = val_loss

        # ── Early stopping — driven by val_ACC ────────────────────────────
        # Counts epochs where val_acc did not improve. Stopping when accuracy
        # plateaus guarantees that gt_acc is the true maximum reachable under
        # this budget, and that the stopping epoch is consistent with the
        # ranking metric. This makes gt_acc comparable across seeds.
        if not improved_acc:
            no_improve_cnt += 1
        else:
            no_improve_cnt = 0

        # ── Logging ───────────────────────────────────────────────────────
        acc_marker  = " ← best acc" if improved_acc  else ""
        loss_marker = " ← best loss" if improved_loss else ""
        es_status   = f"NoImprove(acc) {no_improve_cnt}/{patience}" if not no_early_stop else "ES disabled"
        if improved_acc or improved_loss:
            print(f"    🌟 Epoch {epoch+1:3d} | "
                  f"Loss {train_loss:.4f} | TrainAcc {train_acc:.1f}% | "
                  f"ValAcc {val_acc:.2f}%{acc_marker} | "
                  f"ValLoss {val_loss:.4f}{loss_marker} | "
                  f"StitchGNorm {stitch_gnorm:.4f}")
        else:
            print(f"    Epoch {epoch+1:3d} | "
                  f"Loss {train_loss:.4f} | TrainAcc {train_acc:.1f}% | "
                  f"ValAcc {val_acc:.2f}% | ValLoss {val_loss:.4f} | "
                  f"StitchGNorm {stitch_gnorm:.4f} | "
                  f"{es_status}")

        # Latest checkpoint (always saved — enables resume)
        torch.save({
            'epoch':                epoch + 1,
            'state_dict':           model.state_dict(),
            'optimizer':            optimizer.state_dict(),
            'scheduler':            scheduler.state_dict(),
            'scaler':               scaler.state_dict(),
            'val_acc':              val_acc,
            'val_loss':             val_loss,
            'best_val_loss':        best_val_loss,
            'best_val_acc':         best_val_acc,
            'best_val_acc_running': best_val_acc,  # kept for compat
            'best_acc_epoch':       best_acc_epoch,
            'no_improve_cnt_acc':   no_improve_cnt,
            'no_improve_cnt':       no_improve_cnt,  # kept for compat reads
            'global_idx':           global_idx,
            'path':                 path,
        }, latest_ckpt)

        # Per-epoch metrics entry
        metrics.append({
            "epoch":                epoch + 1,
            "train_loss":           train_loss,
            "train_acc":            train_acc,
            "val_acc":              val_acc,
            "val_loss":             val_loss,
            "lr_stitches":          lr_stitches,
            "lr_backbone":          lr_backbone,
            "stitch_gnorm":         stitch_gnorm,
            "epoch_time":           round(epoch_wall, 3),
            # running max val_acc — always non-decreasing, used as gt_acc
            "best_val_acc":         round(best_val_acc, 3),
            # best val_loss seen so far — for plotting only
            "best_val_loss":        round(best_val_loss, 5),
        })

        # Write metrics.json and loss_curve.json after every epoch
        for path_ in (metrics_path, curve_path):
            with open(path_, 'w') as f:
                json.dump(metrics, f, indent=4)

        # Early stopping check (val_ACC criterion) — skipped if no_early_stop
        if not no_early_stop and no_improve_cnt >= patience:
            stopped_epoch = epoch + 1
            stopped_by_es = True
            print(f"    ⏹  Early stopping triggered at epoch {stopped_epoch} "
                  f"(no improvement in val_ACC for {patience} epochs). "
                  f"Best val_acc: {best_val_acc:.2f}% at epoch {best_acc_epoch}")
            break
    else:
        stopped_epoch = epochs

    return best_val_acc, epoch_times, stopped_epoch, stopped_by_es, best_acc_epoch


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args = parse_args()
    set_seed(args.seed)

    if args.gt_dir is None:
        args.gt_dir = f"dery/validation/gt/{args.dataset}"
    
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

    print(f"[GT] Paths to evaluate : {selected_indices}")
    print(f"[GT] Epochs (max)      : {args.epochs_gt}")
    print(f"[GT] LR (stitches)     : {args.lr}  |  LR (backbone): {args.lr * 0.1}")
    print(f"[GT] Weight decay      : {args.weight_decay}")
    print(f"[GT] Warmup epochs     : {args.warmup_epochs}")
    if args.no_early_stop:
        print(f"[GT] Early stopping    : DISABLED — fixed {args.epochs_gt} epochs per path")
    else:
        print(f"[GT] Early stop pat.   : {args.patience}  (criterion: val_ACC)")
    print(f"[GT] Best ckpt crit.   : val_acc (for gt_acc / NAS ranking)")
    print(f"[GT] Init seed         : {args.seed}  |  Data seed: {args.data_seed}")
    print(f"[GT] Checkpoints dir   : {models_dir}/path_{{idx}}/")
    print(f"[GT] Summary output    : {output_path}")

    # ── Resume summary ───────────────────────────────────────────────────────
    existing_summary = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for entry in json.load(f):
                existing_summary[entry['global_idx']] = entry
        print(f"[GT] {len(existing_summary)} paths already in summary")

    # data_seed is intentionally separate from args.seed (init seed).
    # Keeping data_seed fixed (default=0) across all GT runs ensures every
    # run sees identical data order, isolating weight-init as the only
    # seed-dependent variable. Pass --data_seed N to study data-order variance.
    data_gen = torch.Generator()
    data_gen.manual_seed(args.data_seed)
    trainset, valset = get_dataset(args.dataset, root=args.data_root, img_size=args.img_size)

    print(f"[GT] Dataset       : {args.dataset}")
    print(f"[GT] Data root     : {args.data_root or '(default)'}")
    print(f"[GT] Train samples : {len(trainset)}")
    print(f"[GT] Val samples   : {len(valset)}")
    print(f"[GT] Num classes   : {DATASET_NUM_CLASSES[args.dataset]}")
    
    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2**32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    trainloader = DataLoader(
        trainset,
        batch_size=args.batch_size,
        shuffle=True,
        generator=data_gen,
        worker_init_fn=seed_worker,
        num_workers=4,
        pin_memory=True,
    )

    valloader = DataLoader(
        valset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    # ── Sanity check: el rango real de labels debe coincidir con el dataset ─────
    sample_inputs, sample_targets = next(iter(trainloader))
    label_min = sample_targets.min().item()
    label_max = sample_targets.max().item()
    expected_max = DATASET_NUM_CLASSES[args.dataset] - 1

    print(f"[GT] Sanity check  : batch shape={tuple(sample_inputs.shape)} | "
        f"label range=[{label_min}, {label_max}] | expected_max={expected_max}")

    if label_max > expected_max:
        raise ValueError(
            f"[GT] Label {label_max} excede el máximo esperado ({expected_max}) "
            f"para dataset='{args.dataset}'. El DataLoader está entregando labels "
            f"que no corresponden a este dataset — abortando antes de entrenar."
        )

    
    print("[GT] Loading blueprint SuperNetwork on CPU...")
    blueprint = SuperNetwork(args.plan_path,
                              num_classes=DATASET_NUM_CLASSES[args.dataset],
                              input_size=args.img_size).to('cpu')

    summary = dict(existing_summary)

    for global_idx in selected_indices:
        if global_idx >= len(all_paths):
            print(f"  [skip] Index {global_idx} out of range")
            continue

        path     = all_paths[global_idx]
        ckpt_dir = os.path.join(models_dir, f"path_{global_idx}")

        # Skip only if summary exists AND training fully completed
        if global_idx in existing_summary:
            latest = os.path.join(ckpt_dir, 'latest.pth')
            if os.path.exists(latest):
                ckpt = torch.load(latest, map_location='cpu', weights_only=False)
                if ckpt['epoch'] >= args.epochs_gt:
                    print(f"  [skip] Path {global_idx} fully trained "
                          f"({ckpt['epoch']} epochs)")
                    continue

        print(f"\n[GT] ── Path {global_idx}: {path} ──────────────────────────")

        subnet   = SubNetworkExtractor(blueprint, path).to(DEVICE)
        n_params = sum(p.numel() for p in subnet.parameters())
        print(f"  Parameters: {n_params:,}")

        t_start = time_module.perf_counter()

        best_val_acc, epoch_times, stopped_epoch, stopped_by_es, best_acc_epoch = train_gt_subnet(
            model          = subnet,
            train_loader   = trainloader,
            val_loader     = valloader,
            epochs         = args.epochs_gt,
            lr             = args.lr,
            weight_decay   = args.weight_decay,
            warmup_epochs  = args.warmup_epochs,
            patience       = args.patience,
            no_early_stop  = args.no_early_stop,
            device         = DEVICE,
            use_amp        = USE_AMP,
            ckpt_dir       = ckpt_dir,
            global_idx     = global_idx,
            path           = path,
            resume         = True,
        )

        total_time  = round(time_module.perf_counter() - t_start, 2)
        epoch_arr   = np.array(epoch_times) if epoch_times else np.array([0.0])
        epoch_mean  = round(float(epoch_arr.mean()), 3)
        epoch_std   = round(float(epoch_arr.std()),  3)
        stopped_early = stopped_by_es

        print(f"  ✅ Path {global_idx} done.")
        print(f"     Best val acc  : {best_val_acc:.2f}%")
        print(f"     Stopped epoch : {stopped_epoch}/{args.epochs_gt} "
              f"({'early stop' if stopped_early else 'full run'})")
        print(f"     Total time    : {total_time:.1f}s  |  "
              f"Epoch avg {epoch_mean:.1f}s ± {epoch_std:.1f}s")

        del subnet
        torch.cuda.empty_cache()
        gc.collect()

        # ── Summary entry (matches required output schema) ────────────────
        summary[global_idx] = {
            "global_idx":      global_idx,
            "path":            path,
            "gt_acc":          best_val_acc,   # true max val_acc across all epochs
            "best_acc_epoch":  best_acc_epoch, # epoch where gt_acc was achieved
            "epochs_gt":       args.epochs_gt,
            "lr":              args.lr,
            "n_params":        n_params,
            "ckpt_dir":        ckpt_dir,
            "time":            total_time,
            "epoch_time":      epoch_mean,
            "epoch_std":       epoch_std,
            "stopped_early":   stopped_early,
            "stopped_epoch":   stopped_epoch,
            "init_seed":       args.seed,
            "data_seed":       args.data_seed,
        }

        with open(output_path, 'w') as f:
            json.dump(
                sorted(summary.values(), key=lambda x: x['global_idx']),
                f, indent=4,
            )
        print(f"  📄 Summary updated: {output_path}")

    print(f"\n✅ Complete. {len(summary)} paths in {output_path}")


if __name__ == "__main__":
    main()