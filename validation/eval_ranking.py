import os
import sys
import argparse
import random
import copy
import numpy as np
import scipy.stats as stats
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm


### PATH RESOLUTION FOR IMPORTS (Ensuring Robustness Across Environments) ###

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

third_package_dir = os.path.join(PROJECT_ROOT, 'third_package')
if third_package_dir not in sys.path:
    sys.path.insert(0, third_package_dir)
    
######



# Ensure absolute path resolution for imports
sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

# Contextual Imports
from simple_poc.supernet import SuperNetwork
from simple_poc.train_supernet import get_imagenette

# ==========================================
# 1. Reproducibility & Environment
# ==========================================
def set_deterministic_seed(seed: int = 42):
    """Strict enforcement of determinism across all compute layers."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"[Core] Deterministic execution enforced (Seed: {seed})")

# ==========================================
# 2. Path Sampling Logic
# ==========================================
class PathSampler:
    @staticmethod
    def sample_n_unique_paths(choices_per_stage: list, n: int, seed: int = 42) -> list:
        """Samples N statistically unique architectural paths."""
        rng = random.Random(seed)
        unique_paths = set()
        
        # Calculate theoretical maximum permutations to prevent infinite loops
        max_perms = 1
        for c in choices_per_stage: max_perms *= c
        if n > max_perms: n = max_perms

        while len(unique_paths) < n:
            path = tuple(rng.randint(0, c - 1) for c in choices_per_stage)
            unique_paths.add(path)
            
        return [list(p) for p in unique_paths]

# ==========================================
# 3. Sub-Network Extraction (Memory Safe)
# ==========================================
class SubNetworkExtractor(nn.Module):
    """
    A pure structural container that physically detaches a path from the SuperNetwork.
    Prevents unselected module VRAM leakage during isolated fine-tuning.
    """
    def __init__(self, blueprint: SuperNetwork, path: list):
        super().__init__()
        self.path = path
        
        # Deepcopy to guarantee structural/weight isolation from the blueprint
        self.stages = nn.ModuleList([copy.deepcopy(blueprint.stages[0][path[0]])])
        self.stitches = nn.ModuleList()
        
        for i in range(1, blueprint.num_stages):
            self.stitches.append(copy.deepcopy(blueprint.stitches[i-1][path[i-1]][path[i]]))
            self.stages.append(copy.deepcopy(blueprint.stages[i][path[i]]))
            
        self.global_pool = copy.deepcopy(blueprint.global_pool)
        self.head = copy.deepcopy(blueprint.heads[path[-1]])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stages[0](x)
        for i in range(len(self.stitches)):
            out = self.stitches[i](out)
            out = self.stages[i+1](out)
        out = self.global_pool(out)
        return self.head(out)

# ==========================================
# 4. Evaluation Engine
# ==========================================
class Evaluator:
    def __init__(self, device: str, val_loader: DataLoader, criterion: nn.Module):
        self.device = device
        self.val_loader = val_loader
        self.criterion = criterion

    @torch.no_grad()
    def evaluate_model(self, model: nn.Module, path: list = None) -> float:
        """Executes a standard forward pass validation."""
        model.eval()
        correct, total = 0, 0
        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            # If path is provided, it's the SuperNetwork. If None, it's an isolated SubNet.
            outputs = model(inputs, path=path) if path else model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        return 100.0 * correct / total

    def train_subnet_to_convergence(self, model: nn.Module, train_loader: DataLoader, epochs: int) -> float:
        """Fully fine-tunes an isolated SubNet for Ground Truth (Vector Y)."""
        model.train()
        optimizer = optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        for epoch in range(epochs):
            for inputs, targets in train_loader:
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                optimizer.zero_grad(set_to_none=True)
                outputs = model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                optimizer.step()
            scheduler.step()
            
        return self.evaluate_model(model)

# ==========================================
# 5. Main Execution Protocol
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="DeRy Kendall's Tau Ranking Validator")
    parser.add_argument('--weights_path', type=str, required=True, help="Path to trained SPOS SuperNet (best.pth)")
    parser.add_argument('--plan_path', type=str, default="network_plan.pkl")
    parser.add_argument('--num_paths', type=int, default=30, help="N >= 30 required for statistical significance")
    parser.add_argument('--epochs_gt', type=int, default=20, help="Epochs for GT fine-tuning per extracted subnet")
    parser.add_argument('--batch_size', type=int, default=64)
    args = parser.parse_args()

    # 1. Environment & Determinism
    set_deterministic_seed(42)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    IMG_SIZE = 160

    # 2. Data Loaders (with isolated Generator for exact shuffling)
    gen = torch.Generator()
    gen.manual_seed(42)
    trainset, valset = get_imagenette(img_size=IMG_SIZE)
    trainloader = DataLoader(trainset, batch_size=args.batch_size, shuffle=True, generator=gen, num_workers=4, pin_memory=True)
    valloader = DataLoader(valset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # 3. Model Orchestration
    print("[Engine] Booting Untrained Blueprint SuperNetwork (for GT extraction)...")
    blueprint_supernet = SuperNetwork(args.plan_path, input_size=IMG_SIZE).to('cpu')
    
    print(f"[Engine] Booting SPOS SuperNetwork from {args.weights_path}...")
    spos_supernet = SuperNetwork(args.plan_path, input_size=IMG_SIZE).to(DEVICE)
    ckpt = torch.load(args.weights_path, map_location=DEVICE, weights_only=True)
    spos_supernet.load_state_dict(ckpt['state_dict'])
    
    # 4. Sampling
    paths = PathSampler.sample_n_unique_paths(spos_supernet.choices_per_stage, args.num_paths)
    print(f"[Engine] Sampled {len(paths)} unique paths for evaluation.")

    evaluator = Evaluator(DEVICE, valloader, nn.CrossEntropyLoss())
    
    vector_x_proxy = []
    vector_y_gt = []

    for idx, path in enumerate(paths):
        print(f"\n--- Evaluating Path {idx+1}/{len(paths)}: {path} ---")
        
        # ==========================================
        # PHASE B: Proxy Evaluation (Vector X)
        # ==========================================
        print("  -> Executing BN Calibration on SPOS SuperNetwork...")
        spos_supernet.calibrate_bn(trainloader, path, n_batches=100, device=DEVICE)
        acc_proxy = evaluator.evaluate_model(spos_supernet, path=path)
        vector_x_proxy.append(acc_proxy)
        print(f"  -> Proxy Acc (Vector X): {acc_proxy:.2f}%")

        # ==========================================
        # PHASE A: Ground Truth Evaluation (Vector Y)
        # ==========================================
        print("  -> Extracting isolated standalone SubNetwork...")
        # Extract from CPU blueprint to prevent VRAM spiking, then migrate
        subnet = SubNetworkExtractor(blueprint_supernet, path).to(DEVICE)
        
        print(f"  -> Fine-tuning standalone SubNetwork for {args.epochs_gt} epochs...")
        acc_gt = evaluator.train_subnet_to_convergence(subnet, trainloader, args.epochs_gt)
        vector_y_gt.append(acc_gt)
        print(f"  -> Ground Truth Acc (Vector Y): {acc_gt:.2f}%")
        
        # Surgical Memory Deallocation
        del subnet
        torch.cuda.empty_cache()

    # ==========================================
    # PHASE C: Statistical Correlation
    # ==========================================
    tau, p_value = stats.kendalltau(vector_x_proxy, vector_y_gt)
    print("\n" + "="*50)
    print("FINAL RANKING ANALYSIS (KENDALL'S TAU)")
    print("="*50)
    print(f"Sample Size (N): {len(paths)}")
    print(f"Kendall's Tau (τ): {tau:.4f}")
    print(f"P-Value: {p_value:.4e}")
    if tau > 0.4 and p_value < 0.05:
        print("[Status] SUCCESS: High confidence ranking correlation detected.")
    else:
        print("[Status] WARNING: Poor correlation or insufficient statistical significance.")

if __name__ == '__main__':
    main()