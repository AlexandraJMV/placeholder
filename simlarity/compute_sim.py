import argparse
import os
from itertools import combinations
import torch
import pickle
from tqdm import tqdm

from simlarity.compare_functions import SIM_FUNC


def parse_args():
    parser = argparse.ArgumentParser(description='mmcls test model')
    parser.add_argument('--feat_path', type=str, help='path containing feature .pth files')
    parser.add_argument('--out', default='', help='output directory for similarity .pkl files')
    parser.add_argument('--sim_func', default='cka', choices=['cka', 'rbf_cka', 'lr'],
                        help='metric function')
    return parser.parse_args()


def main():
    args = parse_args()

    if args.out:
        os.makedirs(args.out, exist_ok=True)

    files = [f for f in os.listdir(args.feat_path) if f.endswith('.pth')]
    full_paths = [os.path.join(args.feat_path, p) for p in files]

    pkls_comb = list(combinations(full_paths, 2))
    pkls_comb += [(p, p) for p in full_paths]

    print(f"Procesando {len(pkls_comb)} pares de modelos...")

    for path1, path2 in tqdm(reversed(pkls_comb), total=len(pkls_comb)):
        try:
            # FIX 1: Explicit weights_only=False to silence FutureWarning and
            # document intent clearly. Avoids unexpected behaviour when default flips.
            data1 = torch.load(path1,
                               map_location='cuda' if torch.cuda.is_available() else 'cpu',
                               weights_only=False)
            data2 = torch.load(path2,
                               map_location='cuda' if torch.cuda.is_available() else 'cpu',
                               weights_only=False)
        except Exception as e:
            print(f"Error cargando {path1} o {path2}: {e}")
            continue

        name1 = data1['model_name']
        name2 = data2['model_name']

        if 'train_strategy' in data1.keys():
            name1 += data1['train_strategy']
        if 'train_strategy' in data2.keys():
            name2 += data2['train_strategy']

        save_path = os.path.join(args.out, f'{name1}.{name2}.pkl')

        if os.path.exists(save_path):
            continue

        sim = SIM_FUNC[args.sim_func](data1, data2, bs=2048)

        # FIX 2: Normalise sim to numpy immediately at save time.
        # This ensures that downstream consumers (Block_Sim.get_sim) always
        # receive a numpy.ndarray regardless of whether the similarity function
        # returned a torch.Tensor or numpy array. Prevents np.int64 indexing
        # failures on torch.Tensor objects in PyTorch >= 2.0.
        if hasattr(sim, 'detach'):
            sim = sim.detach().cpu().numpy()

        results = dict(
            sim=sim,
            model1=dict(arch=name1, model_name=name1),
            model2=dict(arch=name2, model_name=name2)
        )

        with open(save_path, 'wb') as f:
            pickle.dump(results, f)

        del data1
        del data2
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()