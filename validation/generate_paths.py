# generate_paths.py
import json
import random
import os, sys
from collections import Counter

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

def stratified_sample_paths(choices_per_stage: list, n_paths: int, seed: int = 42) -> list:
    """
    Stratified sampling: each model appears exactly (or as evenly as possible)
    n_paths // n_choices times at every stage, independently shuffled.
    
    Guarantees stage-level marginal balance regardless of joint combinations.
    Falls back to appending random extras if n_paths is not evenly divisible
    by the number of choices at a given stage.
    """
    rng = random.Random(seed)

    # Build a balanced pool for each stage
    # Each choice index appears floor(n_paths / n_choices) times,
    # with the remainder distributed randomly to avoid systematic bias
    stage_pools = []
    for n_choices in choices_per_stage:
        base_reps  = n_paths // n_choices
        remainder  = n_paths  % n_choices

        pool = [choice for choice in range(n_choices) for _ in range(base_reps)]

        # Distribute remainder randomly among choices (no choice gets 2 extras)
        extra_choices = rng.sample(range(n_choices), remainder)
        pool += extra_choices

        rng.shuffle(pool)
        stage_pools.append(pool)

    # Zip across stages: path i = [stage_pools[s][i] for each stage s]
    paths = [list(stage_pools[s][i] for s in range(len(choices_per_stage)))
             for i in range(n_paths)]

    # Uniqueness check — duplicates are unlikely but possible with small spaces
    seen = set()
    unique_paths = []
    duplicates   = 0
    for path in paths:
        key = tuple(path)
        if key not in seen:
            seen.add(key)
            unique_paths.append(path)
        else:
            duplicates += 1

    if duplicates > 0:
        print(f"⚠️  {duplicates} duplicate path(s) detected after stratified sampling.")
        print(f"   Filling with additional random unique paths...")
        max_possible = 1
        for c in choices_per_stage:
            max_possible *= c

        attempts = 0
        while len(unique_paths) < n_paths and attempts < 10_000:
            candidate = tuple(rng.randint(0, c - 1) for c in choices_per_stage)
            if candidate not in seen:
                seen.add(candidate)
                unique_paths.append(list(candidate))
            attempts += 1

        if len(unique_paths) < n_paths:
            print(f"⚠️  Search space exhausted. Only {len(unique_paths)} unique paths available.")

    return unique_paths


def print_balance_report(paths: list, choices_per_stage: list):
    """Prints how many times each model appears at each stage."""
    print("\n📊 Sampling Balance Report:")
    print(f"   {'Stage':<8} " + " ".join(f"Model {m:<4}" for m in range(max(choices_per_stage))))
    for s, n_choices in enumerate(choices_per_stage):
        counts = Counter(p[s] for p in paths)
        row = " ".join(f"{counts.get(m, 0):<10}" for m in range(n_choices))
        print(f"   Stage {s}:  {row}")
    print()


def main():
    N_PATHS = 30
    SEED    = 42

    # 1. Boot blueprint to get choice dimensions
    blueprint = SuperNetwork("network_plan.pkl", input_size=160).to('cpu')
    choices   = blueprint.choices_per_stage
    print(f"Search space: {choices} → {eval('*'.join(map(str, choices)))} total paths")

    # 2. Stratified sampling
    paths_list = stratified_sample_paths(choices, n_paths=N_PATHS, seed=SEED)

    # 3. Balance diagnostic
    print_balance_report(paths_list, choices)

    # 4. Serialize to disk
    with open("eval_paths_universe.json", "w") as f:
        json.dump(paths_list, f, indent=4)

    print(f"✅ Successfully serialized {len(paths_list)} stratified paths to eval_paths_universe.json")

if __name__ == "__main__":
    main()