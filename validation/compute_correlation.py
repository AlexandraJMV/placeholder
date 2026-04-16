import json, glob
from scipy.stats import kendalltau

supernet_accs = [35.62, 30.98, 30.98, 29.07, 20.38, 16.87, 14.27, 11.31, 11.13, 4.41]
standalone_accs = []

for f in sorted(glob.glob("standalone_results/path_*.json")):
    with open(f) as fp:
        data = json.load(fp)
        standalone_accs.append(data['accuracy'])

tau, p = kendalltau(supernet_accs, standalone_accs)
print(f"Kendall's Tau: {tau:.3f} (p={p:.4f})")