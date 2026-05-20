# classify_ablation.py
"""
Ablation classifier for Group B and Group C.

Reads eval_paths_universe.json and all gt_results_*.json from the v2 GT
directory. Classifies each path as:
    Group B — homogeneous (all stages use the same family index)
    Group C — heterogeneous (at least one stage uses a different family)

Does NOT duplicate checkpoint data — it only references the existing
gt_results entries by global_idx. Output is a single
ablation_bc_summary.json saved under --ablation_dir.

Schema:
    {
        "group_b": [
            {
                "global_idx": int,
                "path":       list[int],
                "family":     str,        # e.g. "resnet18"
                "gt_acc":     float | null,
                "gt_ref":     str,        # source gt_results file
                ... (all other GT fields if available)
            },
            ...
        ],
        "group_c": [
            {
                "global_idx": int,
                "path":       list[int],
                "families":   list[str],  # family at each stage
                "gt_acc":     float | null,
                "gt_ref":     str,
                ... (all other GT fields if available)
            },
            ...
        ],
        "meta": {
            "total_paths":   int,
            "n_group_b":     int,
            "n_group_c":     int,
            "n_trained":     int,
            "n_untrained":   int,
            "family_map":    {str: int},  # model_name → stage index
        }
    }

Usage:
    python dery/ablation/classify_ablation.py \
        --paths_file   eval_paths_universe.json \
        --gt_dir       dery/validation/gt_v2 \
        --ablation_dir dery/ablation \
        --plan_path    network_plan.pkl
"""
import os, sys, json, argparse, glob

CURRENT_DIR  = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path: sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.join(PROJECT_ROOT, 'third_package'))


# Family index → model name mapping
# Must match the order in which SuperNetwork builds its stages
FAMILY_NAMES = {
    0: 'resnet18',
    1: 'mobilenetv3_small_050',
    2: 'efficientnet_b0',
}


def parse_args():
    p = argparse.ArgumentParser(description="Classify paths into Group B / C")
    p.add_argument('--paths_file',   type=str, default="eval_paths_universe.json")
    p.add_argument('--gt_dir',       type=str, default="dery/validation/gt_v2",
                   help="Directory containing gt_results_*_v2.json files")
    p.add_argument('--ablation_dir', type=str, default="dery/ablation")
    return p.parse_args()


def load_gt_results(gt_dir: str) -> dict:
    """
    Loads all gt_results_*.json files from gt_dir.
    Returns a dict mapping global_idx → GT entry.
    """
    gt_map = {}
    pattern = os.path.join(gt_dir, "gt_results_*.json")
    files   = sorted(glob.glob(pattern))
    if not files:
        print(f"  ⚠️  No gt_results_*.json found in {gt_dir}")
        return gt_map
    for fpath in files:
        fname = os.path.basename(fpath)
        with open(fpath) as f:
            entries = json.load(f)
        for entry in entries:
            idx = entry['global_idx']
            if idx not in gt_map:
                gt_map[idx] = {**entry, 'gt_ref': fname}
            else:
                print(f"  ⚠️  Duplicate global_idx {idx} — keeping first occurrence")
    print(f"  Loaded {len(gt_map)} GT entries from {len(files)} file(s)")
    return gt_map


def classify_path(path: list) -> str:
    """Returns 'B' if all stages use the same family, 'C' otherwise."""
    return 'B' if len(set(path)) == 1 else 'C'


def main():
    args = parse_args()
    os.makedirs(args.ablation_dir, exist_ok=True)
    output_path = os.path.join(args.ablation_dir, "ablation_bc_summary.json")

    with open(args.paths_file) as f:
        all_paths = json.load(f)
    print(f"[Classify] {len(all_paths)} paths loaded from {args.paths_file}")

    gt_map = load_gt_results(args.gt_dir)

    group_b, group_c = [], []

    for global_idx, path in enumerate(all_paths):
        group = classify_path(path)

        # Base entry — always present
        families = [FAMILY_NAMES.get(s, f"unknown_{s}") for s in path]
        entry = {
            "global_idx": global_idx,
            "path":       path,
            "families":   families,
            "gt_acc":     None,
            "gt_ref":     None,
        }

        # Annotate with family name for Group B
        if group == 'B':
            entry["family"] = families[0]  # all same

        # Merge GT data if available (no duplication — just references fields)
        if global_idx in gt_map:
            gt = gt_map[global_idx]
            entry.update({
                "gt_acc":      gt.get('gt_acc'),
                "gt_ref":      gt.get('gt_ref'),
                "n_params":    gt.get('n_params'),
                "ckpt_dir":    gt.get('ckpt_dir'),
                "epochs_gt":   gt.get('epochs_gt'),
                "stopped_early": gt.get('stopped_early'),
                "stopped_epoch": gt.get('stopped_epoch'),
                "time":        gt.get('time'),
                "epoch_time":  gt.get('epoch_time'),
                "epoch_std":   gt.get('epoch_std'),
            })

        if group == 'B':
            group_b.append(entry)
        else:
            group_c.append(entry)

    # ── Stats ─────────────────────────────────────────────────────────────────
    n_trained_b = sum(1 for e in group_b if e['gt_acc'] is not None)
    n_trained_c = sum(1 for e in group_c if e['gt_acc'] is not None)
    n_trained   = n_trained_b + n_trained_c
    n_untrained = len(all_paths) - n_trained

    meta = {
        "total_paths":  len(all_paths),
        "n_group_b":    len(group_b),
        "n_group_c":    len(group_c),
        "n_trained":    n_trained,
        "n_untrained":  n_untrained,
        "family_map":   FAMILY_NAMES,
    }

    output = {"group_b": group_b, "group_c": group_c, "meta": meta}

    with open(output_path, 'w') as f:
        json.dump(output, f, indent=4)

    # ── Report ────────────────────────────────────────────────────────────────
    print(f"\n{'═'*55}")
    print(f"  ABLATION CLASSIFICATION REPORT")
    print(f"{'═'*55}")
    print(f"  Total paths   : {len(all_paths)}")
    print(f"  Group B (homo): {len(group_b)}")
    print(f"  Group C (hetero): {len(group_c)}")
    print(f"  Trained (GT available): {n_trained}")
    print(f"    └─ B trained: {n_trained_b}/{len(group_b)}")
    print(f"    └─ C trained: {n_trained_c}/{len(group_c)}")
    print(f"  Untrained (GT pending): {n_untrained}")
    print(f"{'─'*55}")

    if group_b:
        print(f"\n  Group B paths:")
        for e in group_b:
            acc_str = f"{e['gt_acc']:.2f}%" if e['gt_acc'] is not None else "not trained"
            print(f"    idx={e['global_idx']}  path={e['path']}  "
                  f"family={e['family']}  acc={acc_str}")

    print(f"\n  Group C paths ({len(group_c)} total):")
    for e in group_c:
        acc_str = f"{e['gt_acc']:.2f}%" if e['gt_acc'] is not None else "not trained"
        print(f"    idx={e['global_idx']}  path={e['path']}  acc={acc_str}")

    print(f"\n✅ Saved to {output_path}")

    # Warn if Group B paths are missing from the 30-path pool
    if len(group_b) == 0:
        print("\n  ⚠️  No homogeneous paths found in the pool.")
        print("     Paths [0,0,0,0], [1,1,1,1], [2,2,2,2] are not in "
              "eval_paths_universe.json.")
        print("     Run eval_gt.py with --indices for these paths separately "
              "or add them to the universe.")


if __name__ == "__main__":
    main()