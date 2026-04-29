import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def plot_dery_metrics(json_path):
    # 1. Path Management
    # Logic: Root is 'dery', curves go into 'dery/graphs'
    base_dir = "dery"
    graph_dir = os.path.join(base_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)
    
    filename = os.path.basename(json_path).replace('.json', '')
    output_path = os.path.join(graph_dir, f"{filename}_curves.png")

    # 2. Data Extraction
    with open(json_path, 'r') as f:
        data = json.load(f)

    epochs = [d['epoch'] for d in data]
    train_loss = [d['train_loss'] for d in data]
    val_loss = [d['val_loss'] for d in data]
    train_acc = [d['train_acc'] for d in data]
    val_acc = [d['val_acc'] for d in data]
    val_std = [d['val_std'] for d in data]

    # 3. Visualization Architecture
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"DeRy Training Analysis: {filename}", fontsize=16, fontweight='bold')

    # --- Plot A: Accuracy & Loss Convergence ---
    ax1.set_title("Global Convergence (Acc & Loss)")
    ax1.set_xlabel("Epoch")
    
    # Loss Axis (Left)
    ax1.set_ylabel("Cross Entropy Loss", color='tab:red')
    ax1.plot(epochs, train_loss, label='Train Loss', color='tab:red', linestyle='--', alpha=0.6)
    ax1.plot(epochs, val_loss, label='Val Loss', color='tab:red', linewidth=2)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    # Accuracy Axis (Right)
    ax1_acc = ax1.twinx()
    ax1_acc.set_ylabel("Top-1 Accuracy (%)", color='tab:blue')
    ax1_acc.plot(epochs, train_acc, label='Train Acc', color='tab:blue', linestyle='--', alpha=0.6)
    ax1_acc.plot(epochs, val_acc, label='Val Acc (Mean)', color='tab:blue', linewidth=2)
    ax1_acc.tick_params(axis='y', labelcolor='tab:blue')

    # Merge Legends
    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_acc.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    # --- Plot B: Path Ranking Stability ---
    # Logic: High STD means the SuperNet is biased toward specific paths.
    ax2.set_title("Sub-Network Structural Stability")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Standard Deviation (Acc %)")
    
    # Fill between Mean +/- STD to show path variance
    val_acc_np = np.array(val_acc)
    val_std_np = np.array(val_std)
    ax2.fill_between(epochs, val_acc_np - val_std_np, val_acc_np + val_std_np, 
                     color='tab:green', alpha=0.2, label='Path Variance (±1 STD)')
    ax2.plot(epochs, val_acc, color='tab:green', linewidth=1.5, label='Mean Val Acc')
    
    # Standard Deviation Line
    ax2_std = ax2.twinx()
    ax2_std.plot(epochs, val_std, color='black', alpha=0.8, linestyle=':', label='Raw STD')
    ax2_std.set_ylabel("STD Magnitude")

    ax2.legend(loc='upper left')
    ax2.grid(True, alpha=0.3)

    # 4. Export
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(output_path, dpi=150)
    print(f"✅ Graphs successfully rendered to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate DeRy Architectural Curves")
    parser.add_argument("file", type=str, help="Path to the JSON metrics file")
    args = parser.parse_args()
    
    if os.path.exists(args.file):
        plot_dery_metrics(args.file)
    else:
        print(f"❌ Error: File {args.file} not found.")