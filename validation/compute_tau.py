# compute_tau.py
import json
import os
import scipy.stats as stats

def main():
    expected_batches = 6
    global_results = []
    
    for idx in range(expected_batches):
        file_name = f"batch_{idx}_results.json"
        if not os.path.exists(file_name):
            print(f"❌ Missing artifact: {file_name}. Cannot compute global Tau.")
            return
            
        with open(file_name, "r") as f:
            batch_data = json.load(f)
            global_results.extend(batch_data)
            
    # Sort by global_idx to ensure consistent vector alignment
    global_results = sorted(global_results, key=lambda k: k['global_idx'])
    
    vector_x_proxy = [res['proxy_acc'] for res in global_results]
    vector_y_gt = [res['gt_acc'] for res in global_results]
    
    if len(vector_x_proxy) < 30:
        print(f"⚠️ Warning: Found {len(vector_x_proxy)} paths. N=30 required for statistical significance.")

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
    else:
        print("⚠️ WARNING: Weak correlation. Sub-networks are not sharing weights effectively.")

if __name__ == "__main__":
    main()