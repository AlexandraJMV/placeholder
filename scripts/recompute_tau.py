"""
Recomputa Kendall-tau usando gt_acc alternativo derivado de metrics.json ya existentes.
NO requiere reentrenar nada.

Uso:
    python scripts/recompute_tau.py \
        --checkpoints_dir dery/validation/gt_v2/checkpoints \
        --proxy_results   dery/validation/proxy/proxy_results.json \
        --K 10 \
        --fixed_epoch 50
"""

import os, json, zipfile, argparse
import numpy as np
from scipy.stats import kendalltau

# ─────────────────────────────────────────────
# 1. Carga de métricas desde ZIPs en HF o local
# ─────────────────────────────────────────────
def load_metrics_from_zip(zip_path: str) -> list[dict]:
    """Extrae metrics.json del zip y lo devuelve como lista de dicts."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        names = zf.namelist()
        metrics_file = next(n for n in names if n.endswith("metrics.json"))
        with zf.open(metrics_file) as f:
            return json.load(f)

def compute_gt_acc_variants(metrics: list[dict], K: int = 10, fixed_epoch: int = 50):
    """Devuelve dict con distintas variantes de gt_acc."""
    val_accs = [e["val_acc"] for e in metrics]
    epochs   = [e["epoch"]   for e in metrics]
    
    variants = {
        "peak":          max(val_accs),                        # método actual
        "last_1":        val_accs[-1],                         # última época
        f"last_{K}":     float(np.mean(val_accs[-K:])),        # promedio últimas K
    }
    
    # val_acc en época fija (si existe)
    epoch_map = {e["epoch"]: e["val_acc"] for e in metrics}
    if fixed_epoch in epoch_map:
        variants[f"epoch_{fixed_epoch}"] = epoch_map[fixed_epoch]
    else:
        # Buscar la más cercana
        closest = min(epoch_map.keys(), key=lambda e: abs(e - fixed_epoch))
        variants[f"epoch_{fixed_epoch}_approx"] = epoch_map[closest]
        print(f"  ⚠️  Época {fixed_epoch} no encontrada, usando {closest}")
    
    return variants

# ─────────────────────────────────────────────
# 2. Carga de proxy scores
# ─────────────────────────────────────────────
def load_proxy_scores(proxy_path: str) -> dict[str, float]:
    """
    Espera un JSON con estructura:
    { "path_X": {"proxy_score": 0.42, ...}, ... }
    O una lista de { "arch_id": "path_X", "proxy_score": 0.42 }
    Ajusta según tu formato real.
    """
    with open(proxy_path) as f:
        data = json.load(f)
    
    if isinstance(data, list):
        return {d["arch_id"]: d["proxy_score"] for d in data}
    elif isinstance(data, dict):
        return {k: v["proxy_score"] for k, v in data.items()}
    else:
        raise ValueError("Formato de proxy_results.json no reconocido")

# ─────────────────────────────────────────────
# 3. Main
# ─────────────────────────────────────────────
def main(args):
    ckpt_dir   = args.checkpoints_dir
    proxy_path = args.proxy_results
    K          = args.K
    fixed_epoch = args.fixed_epoch

    # Listar zips disponibles
    zip_files = {
        os.path.splitext(f)[0]: os.path.join(ckpt_dir, f)
        for f in os.listdir(ckpt_dir) if f.endswith(".zip")
    }
    print(f"📦 Encontrados {len(zip_files)} checkpoints")

    # Cargar proxy scores
    proxy_scores = load_proxy_scores(proxy_path)
    print(f"🔍 Proxy scores cargados para {len(proxy_scores)} arqs")

    # Intersección: solo arqs con ambos datos
    common_ids = sorted(set(zip_files) & set(proxy_scores))
    print(f"✅ Arqs en común: {len(common_ids)}")

    if len(common_ids) < 5:
        print("⚠️  Muy pocas arqs en común — revisa nombres de archivo vs keys del proxy JSON")
        return

    # Acumular variantes de gt_acc para cada arq
    variants_all = []
    proxy_vec    = []

    for arch_id in common_ids:
        metrics = load_metrics_from_zip(zip_files[arch_id])
        variants = compute_gt_acc_variants(metrics, K=K, fixed_epoch=fixed_epoch)
        variants_all.append(variants)
        proxy_vec.append(proxy_scores[arch_id])
        print(f"  {arch_id}: peak={variants['peak']:.2f}  "
              f"last_{K}={variants[f'last_{K}']:.2f}  "
              f"last_1={variants['last_1']:.2f}")

    # Calcular τ para cada variante
    print("\n" + "="*60)
    print(f"{'Variante gt_acc':<25} {'τ':>8}  {'p-value':>10}")
    print("="*60)

    variant_keys = list(variants_all[0].keys())
    results = {}
    for key in variant_keys:
        gt_vec = [v[key] for v in variants_all]
        tau, pval = kendalltau(proxy_vec, gt_vec)
        results[key] = {"tau": tau, "pval": pval}
        flag = " ← ACTUAL" if key == "peak" else ""
        sig  = " ✅" if pval < 0.05 else " ❌(no sig)"
        print(f"  {key:<23} {tau:>8.4f}  {pval:>10.4f}{sig}{flag}")

    print("="*60)

    # Diagnóstico
    tau_peak  = results["peak"]["tau"]
    tau_lastk = results[f"last_{K}"]["tau"]
    delta     = tau_lastk - tau_peak

    print(f"\n📊 DIAGNÓSTICO:")
    print(f"   Δτ (last_{K} vs peak) = {delta:+.4f}")
    if delta > 0.15:
        print("   ✅ CONFIRMADO: el problema era el pico/early-stop.")
        print("   → Proceder con PASO 2: reentrenar GT con --no_early_stop")
    elif abs(delta) < 0.05:
        print("   ⚠️  Cambio marginal — el problema podría ser otro.")
        print("   → Revisar varianza de proxy, número de arqs, o semillas")
    else:
        print("   🔶 Mejora moderada. Evaluar si justifica reentrenamiento.")

    # Guardar resultados
    out_path = os.path.join(ckpt_dir, "..", "tau_recomputed.json")
    with open(out_path, "w") as f:
        json.dump({
            "n_archs":  len(common_ids),
            "arch_ids": common_ids,
            "variants": results,
        }, f, indent=2)
    print(f"\n💾 Resultados guardados en: {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoints_dir", default="dery/validation/gt_v2/checkpoints")
    parser.add_argument("--proxy_results",   default="dery/validation/proxy/proxy_results.json")
    parser.add_argument("--K",              type=int, default=10)
    parser.add_argument("--fixed_epoch",    type=int, default=50)
    main(parser.parse_args())