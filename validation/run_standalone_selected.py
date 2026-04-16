import json, subprocess, os

selected_paths = [
    [0,0,0,0], [1,1,1,1], [2,2,2,2], [4,4,4,4],
    [1,1,0,4], [4,4,0,4], [2,0,1,4], [3,1,2,1],
    [3,4,0,3], [1,0,2,3]
]

os.makedirs("standalone_results", exist_ok=True)

for i, path in enumerate(selected_paths):
    path_str = str(path).replace(" ", "")
    out_file = f"standalone_results/path_{i}_{path_str}.json"
    if os.path.exists(out_file):
        print(f"Skipping {path_str}, already done.")
        continue
    print(f"Training {i+1}/10: {path}")
    subprocess.run([
        "python", "validation/train_standalone.py",
        "--path", path_str,
        "--epochs", "50",
        "--lr", "0.01",
        "--batch_size", "8",
        "--output_file", out_file
    ])