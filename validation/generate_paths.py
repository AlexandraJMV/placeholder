# generate_paths.py
import json
import random
import os, sys

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

def main():
    # 1. Boot blueprint to get choice dimensions
    blueprint = SuperNetwork("network_plan.pkl", input_size=160).to('cpu')
    choices = blueprint.choices_per_stage
    
    # 2. Enforce Strict Determinism
    rng = random.Random(42)
    unique_paths = set()
    
    # 3. Sample 30 unique paths
    while len(unique_paths) < 30:
        path = tuple(rng.randint(0, c - 1) for c in choices)
        unique_paths.add(path)
        
    paths_list = [list(p) for p in unique_paths]
    
    # 4. Serialize to disk
    with open("eval_paths_universe.json", "w") as f:
        json.dump(paths_list, f, indent=4)
        
    print(f"✅ Successfully serialized {len(paths_list)} paths to eval_paths_universe.json")

if __name__ == "__main__":
    main()