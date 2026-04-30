# compute_tau.py
import json
import os
import scipy.stats as stats
import sys

def main():
    expected_batches = 6
    global_results = []
    
    # SYSTEM ARCHITECT FIX: 
    # Check if files exist in Current Working Directory (Root) 
    # or in the script's parent directory.
    cwd = os.getcwd()
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # We prioritize the CWD (Kaggle /working/)
    search_path = cwd 

    print(f"[Internal] Execution CWD: {cwd}")
    print(f"[Internal] Script Directory: {script_dir}")
    
    for idx in range(expected_batches):
        file_name = f"batch_{idx}_results.json"
        full_path = os.path.join(search_path, file_name)
        
        # Fallback check
        if not os.path.exists(full_path):
            # Try looking one level up if the script was called from inside validation/
            full_path = os.path.join(os.path.dirname(search_path), file_name)
            
        if not os.path.exists(full_path):
            print(f"❌ Missing artifact: {file_name} at {search_path}. Cannot compute global Tau.")
            return
            
        with open(full_path, "r") as f:
            batch_data = json.load(f)
            global_results.extend(batch_data)
            
    # Sort by global_idx to ensure consistent vector alignment
    global_results = sorted(global_results, key=lambda k: k['global_idx'])
    
    vector_x_proxy = [res['proxy_acc'] for res in global_results]
    vector_y_gt = [res['gt_acc'] for res in global_results]
    
    if len(vector_x_proxy) < 30:
        print(f"⚠️ Warning: Found {len(vector_x_proxy)} paths. N=30 required for statistical significance.")

    # Statistical computation
    tau, p_value = stats.kendalltau(vector_x_proxy, vector_y_gt)
    
    print("\n" + "="*50)
    print("FINAL RANKING ANALYSIS (KENDALL'S TAU)")
    print("="*50)
    print(f"Total Paths Evaluated: {len(vector_x_proxy)}")
    print(f"Kendall's Tau (τ):     {tau:.4f}")
    print(f"P-Value:               {p_value:.4e}")
    print("="*50)
    
    if tau > 0.4 and p_value < 0.05:
        print("✅ SUCCESS: Strong predictive correlation detected.")
    elif tau > 0.2:
        print("⚠️ CAUTION: Moderate correlation. Weight sharing is partially effective.")
    else:
        print("❌ FAILURE: Weak correlation. Sub-networks are not sharing weights effectively.")

if __name__ == "__main__":
    main()