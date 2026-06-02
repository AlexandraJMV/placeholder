# compute_tau.py
"""
Phase 3: Merge proxy_results_{run_name}.json + gt_results_*_v2.json files.
Computes Kendall's tau on the intersection (paths with both scores).
Persists result to tau_results/{run_name}_tau.json for cross-supernet comparison.

Fixes applied (v2):
  B1 — proxy_path ahora incluye run_name: proxy_results_{run_name}.json
  B2 — gt_dir default corregido a dery/validation/gt_v2
  B3 — NameError corregido: args.results_dir → args.gt_dir
  B4 — Resultado escrito a disco en tau_results/{run_name}_tau.json
  B5 — Argumento --run_name añadido para seleccionar qué proxy cargar
  M1 — min_n default actualizado a 60 (GT v2)
"""
import json, os, glob, argparse, time
import scipy.stats as stats


def parse_args():
    p = argparse.ArgumentParser(description="Phase 3: Compute Kendall's Tau")
    p.add_argument('--run_name',    type=str, required=True,
                   help="Supernet run identifier — selects "
                        "proxy_results_{run_name}.json from proxy_dir. "
                        "Same value used in eval_proxy.py. "
                        "Example: ls_full_lr1e-2_ep500_run2fix")
    p.add_argument('--proxy_dir',  type=str, default="dery/validation/proxy",
                   help="Directory containing proxy_results_{run_name}.json")
    p.add_argument('--gt_dir',     type=str, default="dery/validation/gt_v2",
                   help="Directory containing gt_results_*_v2.json files")
    p.add_argument('--output_dir', type=str, default="dery/validation/tau_results",
                   help="Directory where {run_name}_tau.json will be written")
    p.add_argument('--min_n',      type=int, default=60,
                   help="Minimum N for reliable tau estimate (default: 60)")
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    # ── Load proxy results (B1, B5) ───────────────────────────────────────────
    proxy_path = os.path.join(args.proxy_dir,
                              f"proxy_results_{args.run_name}.json")
    if not os.path.exists(proxy_path):
        print(f"❌ Missing: {proxy_path}")
        print(f"   Run eval_proxy.py --run_name {args.run_name} first.")
        return

    with open(proxy_path) as f:
        proxy_data = json.load(f)
    proxy_map = {entry['global_idx']: entry['proxy_acc'] for entry in proxy_data}
    print(f"[Tau] Proxy results loaded: {len(proxy_map)} paths  ({proxy_path})")

    # ── Load GT results (B2, B3) ──────────────────────────────────────────────
    gt_files = sorted(glob.glob(os.path.join(args.gt_dir, "gt_results_*.json")))
    if not gt_files:
        # B3: era args.results_dir, corregido a args.gt_dir
        print(f"❌ No gt_results_*.json files found in {args.gt_dir}.")
        print(f"   Check --gt_dir or download GT files first.")
        return

    gt_map = {}
    for gt_file in gt_files:
        with open(gt_file) as f:
            entries = json.load(f)
        for entry in entries:
            idx = entry['global_idx']
            if idx in gt_map:
                print(f"⚠️  Duplicate global_idx {idx} — keeping first occurrence")
            else:
                gt_map[idx] = entry['gt_acc']

    print(f"[Tau] GT results loaded: {len(gt_map)} paths "
          f"across {len(gt_files)} file(s)")

    # ── Intersection ──────────────────────────────────────────────────────────
    common_indices = sorted(set(proxy_map.keys()) & set(gt_map.keys()))
    n = len(common_indices)

    proxy_only = len(proxy_map) - n
    gt_only    = len(gt_map)    - n
    print(f"[Tau] Matched paths: {n}  |  proxy-only: {proxy_only}  |  GT-only: {gt_only}")

    if n < args.min_n:
        print(f"⚠️  N={n} is below min_n={args.min_n}. "
              f"CI will be wide (~±{1.96 * ((2*(2*n+5))/(9*n*(n-1)))**0.5:.2f}).")

    if n < 3:
        print("❌ Fewer than 3 matched paths. Cannot compute tau.")
        return

    vector_proxy = [proxy_map[i] for i in common_indices]
    vector_gt    = [gt_map[i]    for i in common_indices]

    # ── Kendall's tau ─────────────────────────────────────────────────────────
    tau, p_value = stats.kendalltau(vector_proxy, vector_gt)

    # 95% CI — large-sample normal approximation (Kendall 1948)
    # Var(tau) = 2(2n+5) / (9n(n-1))
    var_tau = (2 * (2 * n + 5)) / (9 * n * (n - 1))
    se_tau  = var_tau ** 0.5
    ci_low  = tau - 1.96 * se_tau
    ci_high = tau + 1.96 * se_tau

    print("\n" + "=" * 58)
    print("KENDALL'S TAU — PROXY vs GT RANKING")
    print("=" * 58)
    print(f"  Run              : {args.run_name}")
    print(f"  Paths matched (N): {n}")
    print(f"  Kendall's τ      : {tau:.4f}")
    print(f"  95% CI           : [{ci_low:.4f}, {ci_high:.4f}]")
    print(f"  P-value          : {p_value:.4e}")
    print("=" * 58)

    if tau > 0.4 and p_value < 0.05:
        verdict = "SUCCESS: Strong predictive correlation."
        print(f"✅ {verdict}")
    elif tau > 0.2 and p_value < 0.05:
        verdict = "CAUTION: Moderate correlation. Weight sharing partially effective."
        print(f"⚠️  {verdict}")
    elif p_value >= 0.05:
        verdict = "NOT SIGNIFICANT: Cannot reject null hypothesis of random ranking."
        print(f"❌ {verdict}")
    else:
        verdict = "FAILURE: Weak or negative correlation. Weight sharing ineffective."
        print(f"❌ {verdict}")

    # ── Persist result (B4) ───────────────────────────────────────────────────
    result = {
        "run_name":      args.run_name,
        "n":             n,
        "tau":           round(tau,     6),
        "p_value":       round(p_value, 8),
        "ci_low":        round(ci_low,  6),
        "ci_high":       round(ci_high, 6),
        "se_tau":        round(se_tau,  6),
        "proxy_paths":   len(proxy_map),
        "gt_paths":      len(gt_map),
        "proxy_only":    proxy_only,
        "gt_only":       gt_only,
        "verdict":       verdict,
        "proxy_dir":     args.proxy_dir,
        "gt_dir":        args.gt_dir,
        "timestamp":     time.strftime("%Y-%m-%dT%H:%M:%S"),
    }

    output_path = os.path.join(args.output_dir, f"{args.run_name}_tau.json")
    with open(output_path, 'w') as f:
        json.dump(result, f, indent=4)
    print(f"\n💾 Result saved → {output_path}")


if __name__ == "__main__":
    main()