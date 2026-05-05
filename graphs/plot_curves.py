import json
import os
import argparse
import matplotlib.pyplot as plt
import numpy as np

def plot_dery_metrics(json_path):
    base_dir = "dery"
    graph_dir = os.path.join(base_dir, "graphs")
    os.makedirs(graph_dir, exist_ok=True)
    
    filename = os.path.basename(json_path).replace('.json', '')
    output_path = os.path.join(graph_dir, f"{filename}_curves.png")

    with open(json_path, 'r') as f:
        data = json.load(f)

    # Training data — always present every epoch
    epochs      = [d['epoch'] for d in data]
    train_loss  = [d['train_loss'] for d in data]
    train_acc   = [d['train_acc'] for d in data]

    # Validation data — only epochs where it was actually recorded
    val_data    = [d for d in data if d.get('val_acc', 0) != 0]
    val_epochs  = [d['epoch'] for d in val_data]
    val_loss    = [d['val_loss'] for d in val_data]
    val_acc     = [d['val_acc'] for d in val_data]
    val_std     = [d['val_std'] for d in val_data]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(f"DeRy Training Analysis: {filename}", fontsize=16, fontweight='bold')

    # --- Plot A: Accuracy & Loss Convergence ---
    ax1.set_title("Global Convergence (Acc & Loss)")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Cross Entropy Loss", color='tab:red')
    ax1.plot(epochs, train_loss, label='Train Loss', color='tab:red', linestyle='--', alpha=0.6)
    ax1.tick_params(axis='y', labelcolor='tab:red')
    ax1.grid(True, which='both', linestyle='--', alpha=0.5)

    ax1_acc = ax1.twinx()
    ax1_acc.set_ylabel("Top-1 Accuracy (%)", color='tab:blue')
    ax1_acc.plot(epochs, train_acc, label='Train Acc', color='tab:blue', linestyle='--', alpha=0.6)
    ax1_acc.tick_params(axis='y', labelcolor='tab:blue')

    if val_epochs:
        ax1.plot(val_epochs, val_loss, label='Val Loss', color='tab:red', linewidth=2)
        ax1_acc.plot(val_epochs, val_acc, label='Val Acc (Mean)', color='tab:blue', linewidth=2)

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1_acc.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, loc='center right')

    # --- Plot B: Path Ranking Stability ---
    ax2.set_title("Sub-Network Structural Stability")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Standard Deviation (Acc %)")
    ax2.grid(True, alpha=0.3)

    if val_epochs:
        val_acc_np = np.array(val_acc)
        val_std_np = np.array(val_std)
        ax2.fill_between(val_epochs, val_acc_np - val_std_np, val_acc_np + val_std_np,
                         color='tab:green', alpha=0.2, label='Path Variance (±1 STD)')
        ax2.plot(val_epochs, val_acc, color='tab:green', linewidth=1.5, label='Mean Val Acc')

        ax2_std = ax2.twinx()
        ax2_std.plot(val_epochs, val_std, color='black', alpha=0.8, linestyle=':', label='Raw STD')
        ax2_std.set_ylabel("STD Magnitude")
    else:
        ax2.text(0.5, 0.5, 'No validation data recorded', 
                 transform=ax2.transAxes, ha='center', va='center', color='gray')

    ax2.legend(loc='upper left')

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