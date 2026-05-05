# compute_tau.py
"""
Phase 3: Merge proxy_results.json + all gt_results_*.json files.
Computes Kendall's tau on the intersection (paths with both scores).
Reports coverage and warns on partial evaluation.
"""
import json, os, glob, argparse
import scipy.stats as stats

def parse_args():
    p = argparse.ArgumentParser(description="Phase 3: Compute Kendall's Tau")
    p.add_argument('--proxy_dir',   type=str, default="dery/validation/proxy",
                   help="Directory containing proxy_results.json")
    p.add_argument('--gt_dir',      type=str, default="dery/validation/gt",
                   help="Directory containing gt_results_*.json files")
    p.add_argument('--min_n',       type=int, default=30)
    return p.parse_args()
def main():
    args = parse_args()

    # ── Load proxy results ────────────────────────────────────────────────────
    proxy_path = os.path.join(args.proxy_dir, "proxy_results.json")
    if not os.path.exists(proxy_path):
        print(f"❌ Missing: {proxy_path}. Run eval_proxy.py first.")
        return

    with open(proxy_path) as f:
        proxy_data = json.load(f)
    proxy_map = {entry['global_idx']: entry['proxy_acc'] for entry in proxy_data}
    print(f"[Tau] Proxy results loaded: {len(proxy_map)} paths")

    # ── Load all GT result files ──────────────────────────────────────────────
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "gt_results_*.json")))
    if not gt_files:
        print(f"❌ No gt_results_*.json files found in {args.results_dir}. Run eval_gt.py first.")
        return

    gt_map = {}
    for gt_file in gt_files:
        with open(gt_file) as f:
            for entry in json.load(f):
                idx = entry['global_idx']
                if idx in gt_map:
                    print(f"⚠️  Duplicate global_idx {idx} across GT files — keeping first occurrence")
                else:
                    gt_map[idx] = entry['gt_acc']

    print(f"[Tau] GT results loaded: {len(gt_map)} paths across {len(gt_files)} file(s)")

    # ── Compute intersection ──────────────────────────────────────────────────
    common_indices = sorted(set(proxy_map.keys()) & set(gt_map.keys()))
    n = len(common_indices)

    print(f"[Tau] Paths with both proxy and GT: {n}")
    print(f"      Proxy-only: {len(proxy_map) - n} | GT-only: {len(gt_map) - n}")

    if n < args.min_n:
        print(f"⚠️  Warning: N={n} is below the recommended minimum of {args.min_n}. "
              f"Results may lack statistical power.")

    if n < 3:
        print("❌ Fewer than 3 matched paths. Cannot compute tau.")
        return

    vector_proxy = [proxy_map[i] for i in common_indices]
    vector_gt    = [gt_map[i]    for i in common_indices]

    # ── Statistical computation ───────────────────────────────────────────────
    tau, p_value = stats.kendalltau(vector_proxy, vector_gt)

    # Approximate 95% CI for tau (large-sample normal approximation)
    se_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1))
    se_tau = (se_tau) ** 0.5
    ci_low  = tau - 1.96 * se_tau
    ci_high = tau + 1.96 * se_tau

    print("\n" + "=" * 55)
    print("FINAL RANKING ANALYSIS — KENDALL'S TAU")
    print("=" * 55)
    print(f"  Paths evaluated (N):    {n}")
    print(f"  Kendall's Tau (τ):      {tau:.4f}")
    print(f"  95% CI:                 [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  P-Value:                {p_value:.4e}")
    print("=" * 55)

    if tau > 0.4 and p_value < 0.05:
        print("✅ SUCCESS: Strong predictive correlation detected.")
    elif tau > 0.2 and p_value < 0.05:
        print("⚠️  CAUTION: Moderate correlation. Weight sharing partially effective.")
    elif p_value >= 0.05:
        print("❌ NOT SIGNIFICANT: Cannot reject null hypothesis of random ranking.")
    else:
        print("❌ FAILURE: Weak correlation. Weight sharing ineffective.")

if __name__ == "__main__":
    main()