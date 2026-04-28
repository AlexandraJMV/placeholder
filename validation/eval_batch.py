import os
import sys
import argparse
import json
import random
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np

### PATH RESOLUTION FOR IMPORTS (Ensuring Robustness Across Environments) ###
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

third_package_dir = os.path.join(PROJECT_ROOT, 'third_package')
if third_package_dir not in sys.path:
    sys.path.insert(0, third_package_dir)
######

from simple_poc.supernet import SuperNetwork
from simple_poc.train_supernet import get_imagenette

# ==========================================
# 1. Reproducibility & Environment
# ==========================================
def set_deterministic_seed(seed: int = 42):
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# ==========================================
# 2. Sub-Network Extraction (Direct Assignment)
# ==========================================
class SubNetworkExtractor(nn.Module):
    """
    Physically detaches a path from a FRESH SuperNetwork blueprint.
    Uses direct assignment instead of copy.deepcopy to prevent 
    torch.fx.GraphModule attribute corruption (e.g., missing eval_graph).
    """
    def __init__(self, blueprint: SuperNetwork, path: list):
        super().__init__()
        self.path = path
        
        # Direct assignment transfers PyTorch module ownership.
        # No deepcopy corruption.
        self.stages = nn.ModuleList([blueprint.stages[0][path[0]]])
        self.stitches = nn.ModuleList()
        
        for i in range(1, blueprint.num_stages):
            self.stitches.append(blueprint.stitches[i-1][path[i-1]][path[i]])
            self.stages.append(blueprint.stages[i][path[i]])
            
        self.global_pool = blueprint.global_pool
        self.head = blueprint.heads[path[-1]]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.stages[0](x)
        for i in range(len(self.stitches)):
            out = self.stitches[i](out)
            out = self.stages[i+1](out)
        out = self.global_pool(out)
        return self.head(out)

# ==========================================
# 3. Evaluation Engine
# ==========================================
class Evaluator:
    def __init__(self, device: str, val_loader: DataLoader, criterion: nn.Module):
        self.device = device
        self.val_loader = val_loader
        self.criterion = criterion

    @torch.no_grad()
    def evaluate_model(self, model: nn.Module, path: list = None) -> float:
        model.eval()
        correct, total = 0, 0
        for inputs, targets in self.val_loader:
            inputs, targets = inputs.to(self.device), targets.to(self.device)
            outputs = model(inputs, path=path) if path else model(inputs)
            _, predicted = outputs.max(1)
            correct += predicted.eq(targets).sum().item()
            total += targets.size(0)
        return 100.0 * correct / total

    def train_subnet_to_convergence(self, model: nn.Module, train_loader: DataLoader, epochs: int) -> float:
        model.train()
        import torch.optim as optim
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
# 4. Main Execution Loop
# ==========================================
def main():
    parser = argparse.ArgumentParser(description="DeRy Batch Evaluator")
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--plan_path', type=str, default="network_plan.pkl")
    parser.add_argument('--paths_file', type=str, default="eval_paths_universe.json")
    parser.add_argument('--batch_idx', type=int, required=True)
    parser.add_argument('--batch_size_paths', type=int, default=5)
    parser.add_argument('--epochs_gt', type=int, default=15)
    args = parser.parse_args()

    set_deterministic_seed(42)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    if not os.path.exists(args.paths_file):
        raise FileNotFoundError(f"Missing {args.paths_file}.")
        
    with open(args.paths_file, "r") as f:
        all_paths = json.load(f)
        
    start_idx = args.batch_idx * args.batch_size_paths
    end_idx = start_idx + args.batch_size_paths
    batch_paths = all_paths[start_idx:end_idx]

    print(f"🚀 Executing Batch {args.batch_idx} (Paths {start_idx} to {end_idx-1})")

    gen = torch.Generator(); gen.manual_seed(42)
    trainset, valset = get_imagenette(img_size=160)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, generator=gen, num_workers=4)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)

    # Boot the SPOS SuperNetwork once for proxy evaluation
    spos_supernet = SuperNetwork(args.plan_path, input_size=160).to(DEVICE)
    ckpt = torch.load(args.weights_path, map_location=DEVICE, weights_only=True)
    spos_supernet.load_state_dict(ckpt['state_dict'])
    
    evaluator = Evaluator(DEVICE, valloader, nn.CrossEntropyLoss())
    batch_results = []

    for i, path in enumerate(batch_paths):
        global_idx = start_idx + i
        print(f"\n--- Processing Path Global ID {global_idx}: {path} ---")
        
        # ----------------------------------------------------
        # Proxy Evaluation (Vector X)
        # ----------------------------------------------------
        spos_supernet.calibrate_bn(trainloader, path, n_batches=100, device=DEVICE)
        acc_proxy = evaluator.evaluate_model(spos_supernet, path=path)
        print(f"  -> Proxy Acc (Vector X): {acc_proxy:.2f}%")
        
        # ----------------------------------------------------
        # Ground Truth Evaluation (Vector Y)
        # ----------------------------------------------------
        # Instantiate a FRESH blueprint per path to avoid deepcopy graph corruption
        blueprint_supernet = SuperNetwork(args.plan_path, input_size=160).to('cpu')
        subnet = SubNetworkExtractor(blueprint_supernet, path).to(DEVICE)
        
        acc_gt = evaluator.train_subnet_to_convergence(subnet, trainloader, args.epochs_gt)
        print(f"  -> Ground Truth Acc (Vector Y): {acc_gt:.2f}%")
        
        # Free memory and trigger garbage collection for the dangling blueprint parts
        del subnet
        del blueprint_supernet
        torch.cuda.empty_cache()
        import gc; gc.collect()

        batch_results.append({
            "global_idx": global_idx,
            "path": path,
            "proxy_acc": round(acc_proxy, 4),
            "gt_acc": round(acc_gt, 4)
        })

    # Save Isolated Artifact
    output_filename = f"batch_{args.batch_idx}_results.json"
    with open(output_filename, "w") as f:
        json.dump(batch_results, f, indent=4)
    print(f"\n✅ Batch {args.batch_idx} completed. Saved to {output_filename}")

if __name__ == "__main__":
    main()