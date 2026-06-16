# eval_proxy.py
"""
Phase 1: Proxy evaluation over a fixed set of paths.
Run once per supernet checkpoint after training. No gradient computation.

Output: {proxy_dir}/proxy_results_{run_name}.json

Fixes applied (v2):
  P1 — Output filename derived from weights_path (run_name), so 20 supernets
       produce 20 separate files and never overwrite each other.
  P2 — Every result entry records 'run_name' and 'weights_path' so files are
       self-identifying even if moved or merged.
  P3 — Resume key is the path tuple itself (str(path)), not global_idx, so
       the resume survives regeneration or reordering of eval_paths_universe.json.
  P4 — weights_only=False: checkpoints contain optimizer/scheduler/scaler;
       weights_only=True raises in PyTorch >= 2.0 for those objects.
  P5 — Per-path wall-clock time recorded in each result entry; total and mean
       reported at the end.
"""
import os, sys, json, argparse, random, time, torch, torch.nn as nn
import numpy as np
from pathlib import Path
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


def infer_run_name(weights_path: str) -> str:
    """
    Derive a stable run identifier from the checkpoint path.

    Examples
    --------
    .../supernets/ls/full_lr1e-2_ep500/best.pth   -> ls_full_lr1e-2_ep500
    .../supernets/random/onlystitch_lr1e-3_ep500_seed123/latest.pth
                                                  -> random_onlystitch_lr1e-3_ep500_seed123

    Strategy: take the two path components above the filename
    (init_mode / run_folder) and join them with '_'.
    Falls back to the stem of the filename if the directory layout is different.
    """
    parts = Path(weights_path).parts          # e.g. (..., 'ls', 'full_lr1e-2_ep500', 'best.pth')
    filename_stem = Path(weights_path).stem   # 'best' or 'latest'

    # Typical layout: .../supernets/<init_mode>/<run_folder>/<file>.pth
    if len(parts) >= 3:
        run_folder = parts[-2]   # e.g. 'full_lr1e-2_ep500'
        init_mode  = parts[-3]   # e.g. 'ls' or 'random'
        if init_mode in ('ls', 'random'):
            return f"{init_mode}_{run_folder}"
        # init_mode not recognised — use just the run folder
        return run_folder

    # Fallback: use directory name + filename stem
    parent = Path(weights_path).parent.name
    return f"{parent}_{filename_stem}" if parent else filename_stem


def parse_args():
    p = argparse.ArgumentParser(description="Phase 1: Proxy evaluation for all paths")
    p.add_argument('--weights_path',  type=str, required=True,
                   help="Path to supernet checkpoint (.pth)")
    p.add_argument('--run_name',      type=str, default=None,
                   help="Identifier for this supernet run. "
                        "Auto-derived from weights_path if not given. "
                        "Used as the output filename suffix: "
                        "proxy_results_{run_name}.json")
    p.add_argument('--plan_path',     type=str, default="network_plan.pkl")
    p.add_argument('--paths_file',    type=str, default="eval_paths_universe.json")
    p.add_argument('--proxy_dir',     type=str, default="dery/validation/proxy",
                   help="Directory where proxy_results_{run_name}.json will be saved")
    p.add_argument('--calib_batches', type=int, default=100)
    p.add_argument('--batch_size',    type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(42)

    # ── P1: run-specific output filename ──────────────────────────────────────
    run_name = args.run_name or infer_run_name(args.weights_path)
    os.makedirs(args.proxy_dir, exist_ok=True)
    output_path = os.path.join(args.proxy_dir, f"proxy_results_{run_name}.json")

    DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_AMP = DEVICE == 'cuda'

    print(f"[Proxy] run_name    : {run_name}")
    print(f"[Proxy] weights_path: {args.weights_path}")
    print(f"[Proxy] output_path : {output_path}")

    with open(args.paths_file) as f:
        all_paths = json.load(f)
    print(f"[Proxy] Loaded {len(all_paths)} paths from {args.paths_file}")

    gen = torch.Generator(); gen.manual_seed(42)
    trainset, valset = get_imagenette(img_size=160)
    trainloader = DataLoader(trainset, batch_size=args.batch_size,
                             shuffle=True, generator=gen, num_workers=4, pin_memory=True)
    valloader   = DataLoader(valset,   batch_size=args.batch_size,
                             shuffle=False, num_workers=4, pin_memory=True)

    supernet = SuperNetwork(args.plan_path, input_size=160).to(DEVICE)

    # P4: weights_only=False — checkpoints include optimizer/scheduler/scaler
    ckpt = torch.load(args.weights_path, map_location=DEVICE, weights_only=False)
    supernet.load_state_dict(ckpt['state_dict'])
    supernet.eval()
    print(f"[Proxy] Supernet loaded (epoch {ckpt.get('epoch', '?')}, "
          f"best_val_acc={ckpt.get('best_val_acc', '?')}%)")

    # ── P3: resume keyed on path tuple, not global_idx ────────────────────────
    # Existing entries are indexed by str(path) so they survive file reordering.
    existing: dict[str, dict] = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for entry in json.load(f):
                existing[str(entry['path'])] = entry
        print(f"[Proxy] Resuming — {len(existing)} paths already evaluated")

    results: dict[str, dict] = dict(existing)
    t_total_start = time.perf_counter()

    for global_idx, path in enumerate(all_paths):
        path_key = str(path)   # e.g. "[0, 1, 2, 0]"

        if path_key in results:
            print(f"  [skip] Path {global_idx} {path} already evaluated")
            continue

        print(f"\n[Proxy] Evaluating path {global_idx}/{len(all_paths)-1}: {path}")
        t_path_start = time.perf_counter()

        # 1. Calibración BN: Recalcula μ y σ² usando el trainloader sin alterar gradientes.
        supernet.calibrate_bn(trainloader, path,
                              n_batches=args.calib_batches, device=DEVICE)
        
        # 2. Inferencia Determinista: Congela el tracking de BN para validación.
        supernet.eval()

        correct, total = 0, 0
        
        # 4. Prevención de Fugas: Garantizamos que no exista retención en el grafo.
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    outputs = supernet(inputs, path=path)
                
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total   += targets.size(0)
                
                # Liberación iterativa para evitar picos de VRAM
                del inputs, targets, outputs, predicted

        proxy_acc   = round(100.0 * correct / total, 4)
        path_time_s = round(time.perf_counter() - t_path_start, 2)
        print(f"  -> Proxy Acc: {proxy_acc:.2f}%  ({path_time_s}s)")

        # P2 + P3 + P5: self-identifying entry with path as key and timing
        results[path_key] = {
            "global_idx":   global_idx,
            "path":         path,
            "proxy_acc":    proxy_acc,
            "eval_time_s":  path_time_s,
            "run_name":     run_name,
            "weights_path": args.weights_path,
        }

        # Write after every path — crash-safe
        sorted_entries = sorted(results.values(), key=lambda x: x['global_idx'])
        with open(output_path, 'w') as f:
            json.dump(sorted_entries, f, indent=4)
            
        # Limpieza estricta de VRAM al terminar cada ruta
        if DEVICE == 'cuda':
            torch.cuda.empty_cache()



    # ── Summary ───────────────────────────────────────────────────────────────
    total_wall = time.perf_counter() - t_total_start
    evaluated  = [v for v in results.values() if v.get('eval_time_s') is not None]
    mean_time  = round(sum(v['eval_time_s'] for v in evaluated) / len(evaluated), 2) if evaluated else 0.0

    print(f"\n✅ Proxy evaluation complete.")
    print(f"   Run        : {run_name}")
    print(f"   Paths saved: {len(results)}")
    print(f"   Output     : {output_path}")
    print(f"   Wall time  : {total_wall:.1f}s  |  Mean per path: {mean_time}s")


if __name__ == "__main__":
    main()