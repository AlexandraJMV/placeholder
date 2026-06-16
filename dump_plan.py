import pickle
import sys
import os
import argparse

# Ensure the blocklize/simlarity modules can be found so the pickle can deserialize the objects
sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

from simlarity.utils import Block_Assign, Block


def inspect_plan(pkl_path="network_plan.pkl", img_size=160):
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return

    with open(pkl_path, 'rb') as f:
        plan = pickle.load(f)

    choices = [len(stage) for stage in plan.center2block]
    total_paths = 1
    for c in choices:
        total_paths *= c

    print(f"╔══════════════════════════════════════════╗")
    print(f"  NETWORK PLAN DUMP")
    print(f"  Input resolution : {img_size}×{img_size} px  (3 channels)")
    print(f"  Total Stages     : {len(plan.center2block)}")
    print(f"  Choices per Stage: {choices}")
    print(f"  Total paths      : {total_paths:,}")
    print(f"╚══════════════════════════════════════════╝")

    for i, stage in enumerate(plan.center2block):
        print(f"\n┌─ STAGE {i}  ({len(stage)} candidate blocks)")
        print("│")
        for j, block in enumerate(stage):
            model_name = getattr(block, 'model_name', 'UNKNOWN')
            nodes      = getattr(block, 'node_list', [])
            in_size    = getattr(block, 'in_size',  None)
            out_size   = getattr(block, 'out_size', None)

            print(f"│  Choice {j}: {model_name}")
            print(f"│    Nodes : {nodes}")

            if in_size is not None and out_size is not None:
                print(f"│    Shapes: IN {in_size}  →  OUT {out_size}")
            else:
                # Shapes are resolved dynamically at runtime from the {img_size}px input.
                # Run dump_stitches.py --img_size {img_size} to see the resolved channel/res info.
                print(f"│    Shapes: (resolved at runtime from {img_size}×{img_size} input "
                      f"— see dump_stitches.py)")
            print("│")
        print("└" + "─" * 50)

    print(f"\n✅ Plan loaded from: {pkl_path}")
    print(f"   Run `python dump_stitches.py --img_size {img_size}` "
          f"to see full channel/resolution flow.\n")


def parse_args():
    p = argparse.ArgumentParser(description="Inspect a network_plan.pkl file")
    p.add_argument('--plan',     type=str, default='network_plan.pkl',
                   help="Path to the plan pickle (default: network_plan.pkl)")
    p.add_argument('--img_size', type=int, default=160,
                   help="Input spatial resolution used for this plan (default: 160)")
    return p.parse_args()


if __name__ == '__main__':
    args = parse_args()
    inspect_plan(args.plan, args.img_size)