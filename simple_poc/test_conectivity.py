# test_connectivity.py
"""
Sanity-check script: verifies that EVERY possible path through the
SuperNetwork's stitching space produces a valid forward pass, BEFORE
spending any time on SPOS training.

This does NOT train anything. It only:
  1. Builds the SuperNetwork once (random init is enough — we're testing
     shapes/wiring, not learned representations).
  2. Enumerates all paths (cartesian product of choices_per_stage).
  3. Runs model.eval() + a dummy forward pass for each path.
  4. Reports which paths succeed, which fail, and why.

Usage:
    python -m simple_poc.test_connectivity --plan_path network_plan.pkl
"""

import sys, os, argparse, itertools, traceback, time
import torch

sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

from simple_poc.supernet import SuperNetwork


def parse_args():
    p = argparse.ArgumentParser(description="Test SuperNetwork path connectivity")
    p.add_argument('--plan_path',   type=str, default="network_plan.pkl")
    p.add_argument('--matrices_path', type=str, default=None)
    p.add_argument('--init_mode',  type=str, default='random',
                    choices=['ls', 'random'],
                    help="'random' is faster since we're only testing wiring")
    p.add_argument('--img_size',   type=int, default=160)
    p.add_argument('--num_classes', type=int, default=10)
    p.add_argument('--batch_size', type=int, default=2,
                    help="Use >=2 so BatchNorm doesn't choke on batch=1")
    p.add_argument('--device',     type=str, default=None)
    p.add_argument('--max_paths',  type=int, default=None,
                    help="Optional cap for quick smoke-testing a subset")
    p.add_argument('--verbose',    action='store_true',
                    help="Print every path's result, not just failures")
    return p.parse_args()


def main():
    args = parse_args()
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')

    print(f">>> Building SuperNetwork for connectivity test (device={device})...")
    model = SuperNetwork(
        plan_path        = args.plan_path,
        num_classes      = args.num_classes,
        input_size       = args.img_size,
        stitch_init_mode = args.init_mode,
        matrices_path    = args.matrices_path,
    ).to(device)
    model.eval()

    all_paths = list(itertools.product(*[range(c) for c in model.choices_per_stage]))
    if args.max_paths:
        all_paths = all_paths[:args.max_paths]

    total = len(all_paths)
    print(f">>> Total candidate paths in search space: {total}")

    dummy_input = torch.randn(args.batch_size, 3, args.img_size, args.img_size, device=device)

    results = {'ok': [], 'fail': []}
    t0 = time.perf_counter()

    with torch.no_grad():
        for i, path in enumerate(all_paths):
            try:
                out = model(dummy_input, path=list(path))

                if out.shape != (args.batch_size, args.num_classes):
                    raise RuntimeError(
                        f"Unexpected output shape {tuple(out.shape)}, "
                        f"expected {(args.batch_size, args.num_classes)}"
                    )
                if torch.isnan(out).any():
                    raise RuntimeError("Output contains NaNs")
                if torch.isinf(out).any():
                    raise RuntimeError("Output contains Infs")

                results['ok'].append(path)
                if args.verbose:
                    print(f"  [OK]   {i+1}/{total} path={path} -> shape={tuple(out.shape)}")

            except Exception as e:
                results['fail'].append((path, str(e), traceback.format_exc()))
                print(f"  [FAIL] {i+1}/{total} path={path} -> {e}")

    elapsed = time.perf_counter() - t0

    # ── Summary ──────────────────────────────────────────────────────────
    n_ok, n_fail = len(results['ok']), len(results['fail'])
    print("\n" + "=" * 60)
    print(f"Connectivity test complete in {elapsed:.1f}s")
    print(f"  Total paths tested : {total}")
    print(f"  Passed             : {n_ok}")
    print(f"  Failed             : {n_fail}")
    print("=" * 60)

    if n_fail > 0:
        print("\nFailed paths (path -> error):")
        for path, err, tb in results['fail']:
            print(f"  {path} -> {err}")
        print("\nFull traceback of first failure for debugging:")
        print(results['fail'][0][2])
        sys.exit(1)  # non-zero exit so this can gate a CI/script pipeline
    else:
        print("\n✅ All paths in the search space are structurally valid.")
        sys.exit(0)


if __name__ == "__main__":
    main()