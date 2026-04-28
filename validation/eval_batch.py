# eval_batch.py
import argparse
import json
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# (Include all previous imports, set_deterministic_seed, SubNetworkExtractor, and Evaluator classes here)
from eval_ranking import set_deterministic_seed, SubNetworkExtractor, Evaluator
from simple_poc.supernet import SuperNetwork
from simple_poc.train_supernet import get_imagenette

def main():
    parser = argparse.ArgumentParser(description="DeRy Batch Evaluator")
    parser.add_argument('--weights_path', type=str, required=True)
    parser.add_argument('--paths_file', type=str, default="eval_paths_universe.json")
    parser.add_argument('--batch_idx', type=int, required=True, help="0 to 5 for 6 batches")
    parser.add_argument('--batch_size_paths', type=int, default=5, help="Number of paths per batch")
    parser.add_argument('--epochs_gt', type=int, default=15) # Reduced to 15 to ensure safety
    args = parser.parse_args()

    set_deterministic_seed(42)
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Load Universe and Slice
    if not os.path.exists(args.paths_file):
        raise FileNotFoundError(f"Missing {args.paths_file}. Run generate_paths.py first.")
        
    with open(args.paths_file, "r") as f:
        all_paths = json.load(f)
        
    start_idx = args.batch_idx * args.batch_size_paths
    end_idx = start_idx + args.batch_size_paths
    batch_paths = all_paths[start_idx:end_idx]
    
    if not batch_paths:
        raise ValueError(f"Batch index {args.batch_idx} is out of bounds for paths length {len(all_paths)}")

    print(f"🚀 Executing Batch {args.batch_idx} (Paths {start_idx} to {end_idx-1})")

    # Orchestration & Dataloaders (Same as before)
    gen = torch.Generator(); gen.manual_seed(42)
    trainset, valset = get_imagenette(img_size=160)
    trainloader = DataLoader(trainset, batch_size=64, shuffle=True, generator=gen, num_workers=4)
    valloader = DataLoader(valset, batch_size=64, shuffle=False, num_workers=4)

    blueprint_supernet = SuperNetwork("network_plan.pkl", input_size=160).to('cpu')
    spos_supernet = SuperNetwork("network_plan.pkl", input_size=160).to(DEVICE)
    spos_supernet.load_state_dict(torch.load(args.weights_path, map_location=DEVICE, weights_only=True)['state_dict'])
    
    evaluator = Evaluator(DEVICE, valloader, nn.CrossEntropyLoss())
    
    batch_results = []

    for i, path in enumerate(batch_paths):
        global_idx = start_idx + i
        print(f"\n--- Processing Path Global ID {global_idx}: {path} ---")
        
        # Proxy (X)
        spos_supernet.calibrate_bn(trainloader, path, n_batches=100, device=DEVICE)
        acc_proxy = evaluator.evaluate_model(spos_supernet, path=path)
        
        # Ground Truth (Y)
        subnet = SubNetworkExtractor(blueprint_supernet, path).to(DEVICE)
        acc_gt = evaluator.train_subnet_to_convergence(subnet, trainloader, args.epochs_gt)
        del subnet; torch.cuda.empty_cache()

        # Record mapping
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