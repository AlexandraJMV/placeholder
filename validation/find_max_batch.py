# find_max_batch.py
"""
Batch size probe for GT subnetworks.

For every path in eval_paths_universe.json, finds the largest power-of-2
batch size that fits in GPU VRAM for a forward + backward pass (training
memory footprint, not inference). Reports per-path results and a global
safe batch size recommendation (the minimum across all paths).

Strategy: binary search between --bs_min and --bs_max (both must be
powers of 2). For each candidate batch size, runs --n_warmup forward+
backward passes. If any OOM is raised, the size is marked as too large.
Each path is tested independently with its own SubNetworkExtractor so
the result reflects actual per-path memory cost.

Output
------
Prints a table to stdout and saves find_max_batch_results.json to
--out_dir with the following schema per path:
    {
        "global_idx":   int,
        "path":         list[int],
        "n_params":     int,
        "max_batch":    int | null,   # null = even bs_min OOMed
        "vram_mb":      float,        # peak VRAM at max_batch (MB)
        "oom_at":       int | null    # first batch size that OOMed
    }

Usage (standalone):
    python dery/validation/find_max_batch.py \
        --plan_path    network_plan.pkl \
        --paths_file   eval_paths_universe.json \
        --out_dir      dery/validation \
        --bs_min       16 \
        --bs_max       256 \
        --n_warmup     3 \
        --img_size     160
"""
import os, sys, json, argparse, copy, gc
import torch
import torch.nn as nn

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'third_package'))

from simple_poc.supernet import SuperNetwork


# ── CLI ───────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser(description="Find max safe batch size per path")
    p.add_argument('--plan_path',  type=str, default="network_plan.pkl")
    p.add_argument('--paths_file', type=str, default="eval_paths_universe.json")
    p.add_argument('--out_dir',    type=str, default="dery/validation",
                   help="Directory where find_max_batch_results.json is saved")
    p.add_argument('--bs_min',     type=int, default=16,
                   help="Smallest batch size to try (power of 2)")
    p.add_argument('--bs_max',     type=int, default=256,
                   help="Largest batch size to try (power of 2)")
    p.add_argument('--n_warmup',   type=int, default=3,
                   help="Forward+backward passes per batch size candidate")
    p.add_argument('--img_size',   type=int, default=160,
                   help="Input image size (must match supernet plan)")
    return p.parse_args()


# ── SubNetworkExtractor (mirrors eval_gt.py) ──────────────────────────────────
class SubNetworkExtractor(nn.Module):
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


# ── Powers of 2 between min and max ──────────────────────────────────────────
def powers_of_two(bs_min: int, bs_max: int) -> list:
    """Returns sorted list of powers of 2 in [bs_min, bs_max]."""
    sizes = []
    bs = bs_min
    while bs <= bs_max:
        sizes.append(bs)
        bs *= 2
    return sizes


# ── Single probe: forward + backward at a given batch size ───────────────────
def probe(model, batch_size: int, img_size: int, device: str,
          n_warmup: int, use_amp: bool) -> bool:
    """
    Attempts n_warmup forward+backward passes with a synthetic batch.
    Returns True if all passes succeeded, False on OOM.
    Cleans up GPU memory before returning regardless of outcome.
    """
    criterion = nn.CrossEntropyLoss()
    try:
        for _ in range(n_warmup):
            x = torch.randn(batch_size, 3, img_size, img_size,
                            device=device)
            y = torch.randint(0, 10, (batch_size,), device=device)

            with torch.amp.autocast('cuda', enabled=use_amp):
                out  = model(x)
                loss = criterion(out, y)
            loss.backward()
            model.zero_grad(set_to_none=True)

            del x, y, out, loss
        return True

    except torch.cuda.OutOfMemoryError:
        return False

    finally:
        # Always flush regardless of success/failure
        torch.cuda.empty_cache()
        gc.collect()


# ── Binary search for max batch size ─────────────────────────────────────────
def find_max_batch(model, candidates: list, img_size: int,
                   device: str, n_warmup: int, use_amp: bool):
    """
    Binary search over `candidates` (sorted ascending powers of 2).
    Returns (max_batch, oom_at) where:
        max_batch : largest size that passed, or None if even candidates[0] OOMed
        oom_at    : first size that OOMed, or None if all passed
    """
    lo, hi   = 0, len(candidates) - 1
    max_ok   = None
    oom_at   = None

    # First check the minimum — if this OOMs nothing will work
    if not probe(model, candidates[0], img_size, device, n_warmup, use_amp):
        return None, candidates[0]

    # Binary search
    while lo <= hi:
        mid = (lo + hi) // 2
        bs  = candidates[mid]
        ok  = probe(model, bs, img_size, device, n_warmup, use_amp)
        if ok:
            max_ok = bs
            lo     = mid + 1
        else:
            oom_at = bs
            hi     = mid - 1

    return max_ok, oom_at


# ── VRAM reading ──────────────────────────────────────────────────────────────
def peak_vram_mb(model, batch_size: int, img_size: int,
                 device: str, use_amp: bool) -> float:
    """
    Runs one forward+backward and returns peak VRAM allocated (MB).
    Called only after max_batch is confirmed to not OOM.
    """
    criterion = nn.CrossEntropyLoss()
    torch.cuda.reset_peak_memory_stats(device)
    x = torch.randn(batch_size, 3, img_size, img_size, device=device)
    y = torch.randint(0, 10, (batch_size,), device=device)
    with torch.amp.autocast('cuda', enabled=use_amp):
        out  = model(x)
        loss = criterion(out, y)
    loss.backward()
    model.zero_grad(set_to_none=True)
    peak = torch.cuda.max_memory_allocated(device) / 1024 / 1024
    del x, y, out, loss
    torch.cuda.empty_cache()
    gc.collect()
    return round(peak, 1)


# ── Main ──────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    use_amp = device == 'cuda'

    if device == 'cpu':
        print("⚠️  No GPU detected. Results will reflect CPU memory, not VRAM.")

    os.makedirs(args.out_dir, exist_ok=True)
    out_path   = os.path.join(args.out_dir, "find_max_batch_results.json")
    candidates = powers_of_two(args.bs_min, args.bs_max)

    with open(args.paths_file) as f:
        all_paths = json.load(f)

    print(f"[BatchProbe] {len(all_paths)} paths | "
          f"candidates: {candidates} | "
          f"warmup passes: {args.n_warmup}")
    if device == 'cuda':
        total_vram = torch.cuda.get_device_properties(0).total_memory / 1024**2
        print(f"[BatchProbe] GPU: {torch.cuda.get_device_name(0)} | "
              f"Total VRAM: {total_vram:.0f} MB\n")

    print(f"[BatchProbe] Loading blueprint on CPU...")
    blueprint = SuperNetwork(args.plan_path, input_size=args.img_size).to('cpu')

    results  = []
    col_w    = 10

    # Table header
    print(f"{'Idx':<6} {'Path':<20} {'Params':>10} "
          f"{'MaxBatch':>{col_w}} {'VRAM(MB)':>{col_w}} {'OOM@':>{col_w}}")
    print("─" * 62)

    for global_idx, path in enumerate(all_paths):
        subnet   = SubNetworkExtractor(blueprint, path).to(device)
        subnet.train()   # train mode — measures training memory footprint
        n_params = sum(p.numel() for p in subnet.parameters())

        max_batch, oom_at = find_max_batch(
            subnet, candidates, args.img_size, device, args.n_warmup, use_amp
        )

        vram_mb = None
        if max_batch is not None:
            vram_mb = peak_vram_mb(
                subnet, max_batch, args.img_size, device, use_amp
            )

        path_str  = str(path)
        oom_str   = str(oom_at)  if oom_at   is not None else "—"
        vram_str  = f"{vram_mb:.1f}" if vram_mb is not None else "—"
        batch_str = str(max_batch)   if max_batch is not None else "OOM(min)"

        print(f"{global_idx:<6} {path_str:<20} {n_params:>10,} "
              f"{batch_str:>{col_w}} {vram_str:>{col_w}} {oom_str:>{col_w}}")

        results.append({
            "global_idx": global_idx,
            "path":       path,
            "n_params":   n_params,
            "max_batch":  max_batch,
            "vram_mb":    vram_mb,
            "oom_at":     oom_at,
        })

        # Save after every path — crash-safe
        with open(out_path, 'w') as f:
            json.dump(results, f, indent=4)

        del subnet
        torch.cuda.empty_cache()
        gc.collect()

    # ── Summary ──────────────────────────────────────────────────────────────
    valid   = [r for r in results if r['max_batch'] is not None]
    failed  = [r for r in results if r['max_batch'] is None]

    print("\n" + "═" * 62)
    print("SUMMARY")
    print("═" * 62)

    if valid:
        safe_batch = min(r['max_batch'] for r in valid)
        max_vram   = max(r['vram_mb']   for r in valid)
        print(f"  Paths probed         : {len(results)}")
        print(f"  Successful           : {len(valid)}")
        print(f"  OOM at bs_min ({args.bs_min:>3}) : {len(failed)}")
        print(f"  ✅ Safe batch size   : {safe_batch}  "
              f"(min across all paths)")
        print(f"  Peak VRAM at safe bs : {max_vram:.1f} MB")
        if failed:
            print(f"  ⚠️  Paths that OOM at bs={args.bs_min}: "
                  f"{[r['global_idx'] for r in failed]}")
            print(f"     These paths cannot be trained — "
                  f"investigate architecture size.")
    else:
        print(f"  ❌ ALL paths OOM at bs_min={args.bs_min}. "
              f"Lower --bs_min or reduce --img_size.")

    print(f"\n  Results saved to: {out_path}")
    print("═" * 62)


if __name__ == "__main__":
    main()