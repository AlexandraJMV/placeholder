# generate_candidate_paths.py
import json
import random
import os

# From your supernet log: 5 choices per stage (4 stages)
choices_per_stage = [5, 5, 5, 5]

def generate_paths(choices_per_stage, num_random=20, seed=42):
    rng = random.Random(seed)
    paths = []
    
    # 1. Random paths
    for _ in range(num_random):
        path = [rng.randint(0, c-1) for c in choices_per_stage]
        paths.append(path)
    
    # 2. Add homogeneous baselines (all blocks from the same model index)
    for idx in range(choices_per_stage[0]):
        # Check if this index exists in every stage
        if all(idx < c for c in choices_per_stage):
            paths.append([idx, idx, idx, idx])
    
    # 3. Remove duplicates (just in case)
    unique_paths = []
    seen = set()
    for p in paths:
        t = tuple(p)
        if t not in seen:
            seen.add(t)
            unique_paths.append(p)
    
    return unique_paths

if __name__ == "__main__":
    paths = generate_paths(choices_per_stage, num_random=25)
    print(f"Generated {len(paths)} unique paths.")
    with open('candidate_paths.json', 'w') as f:
        json.dump(paths, f, indent=2)
    print("Saved to candidate_paths.json")