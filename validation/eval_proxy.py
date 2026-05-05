# eval_proxy.py
"""
Phase 1: Proxy evaluation for ALL paths in the universe.
Run once after supernet training. Fast — no gradient computation.
Output: results/proxy_results.json
"""
import os, sys, json, argparse, random, torch, torch.nn as nn
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
    p = argparse.ArgumentParser(description="Phase 1: Proxy evaluation for all paths")
    p.add_argument('--weights_path',  type=str, required=True)
    p.add_argument('--plan_path',     type=str, default="network_plan.pkl")
    p.add_argument('--paths_file',    type=str, default="eval_paths_universe.json")
    p.add_argument('--proxy_dir',     type=str, default="dery/validation/proxy",
                   help="Directory where proxy_results.json will be saved")
    p.add_argument('--calib_batches', type=int, default=100)
    p.add_argument('--batch_size',    type=int, default=64)
    return p.parse_args()


def main():
    args = parse_args()
    set_seed(42)
    os.makedirs(args.proxy_dir, exist_ok=True)
    output_path = os.path.join(args.proxy_dir, "proxy_results.json")
    DEVICE  = 'cuda' if torch.cuda.is_available() else 'cpu'
    USE_AMP = DEVICE == 'cuda'

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
    ckpt = torch.load(args.weights_path, map_location=DEVICE, weights_only=True)
    supernet.load_state_dict(ckpt['state_dict'])
    supernet.eval()
    print(f"[Proxy] Supernet loaded from {args.weights_path}")

    # Resume support: skip already evaluated paths
    existing = {}
    if os.path.exists(output_path):
        with open(output_path) as f:
            for entry in json.load(f):
                existing[entry['global_idx']] = entry
        print(f"[Proxy] Resuming — {len(existing)} paths already evaluated")

    results = dict(existing)

    for global_idx, path in enumerate(all_paths):
        if global_idx in results:
            print(f"  [skip] Path {global_idx} already evaluated")
            continue

        print(f"\n[Proxy] Evaluating path {global_idx}/{len(all_paths)-1}: {path}")
        supernet.calibrate_bn(trainloader, path,
                              n_batches=args.calib_batches, device=DEVICE)
        supernet.eval()

        correct, total = 0, 0
        with torch.no_grad():
            for inputs, targets in valloader:
                inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
                with torch.amp.autocast('cuda', enabled=USE_AMP):
                    outputs = supernet(inputs, path=path)
                _, predicted = outputs.max(1)
                correct += predicted.eq(targets).sum().item()
                total   += targets.size(0)

        proxy_acc = round(100.0 * correct / total, 4)
        print(f"  -> Proxy Acc: {proxy_acc:.2f}%")

        results[global_idx] = {
            "global_idx": global_idx,
            "path":       path,
            "proxy_acc":  proxy_acc
        }

        # Write after every path — crash-safe
        with open(output_path, 'w') as f:
            json.dump(sorted(results.values(), key=lambda x: x['global_idx']), f, indent=4)

    print(f"\n✅ Proxy evaluation complete. {len(results)} paths saved to {output_path}")

if __name__ == "__main__":
    main()