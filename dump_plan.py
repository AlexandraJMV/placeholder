import pickle
import sys
import os

# Ensure the blocklize/simlarity modules can be found so the pickle can deserialize the objects
sys.path.insert(0, os.path.join(os.getcwd(), 'third_package'))
sys.path.append(os.getcwd())

from simlarity.utils import Block_Assign, Block

def inspect_plan(pkl_path="network_plan.pkl"):
    if not os.path.exists(pkl_path):
        print(f"Error: {pkl_path} not found.")
        return

    with open(pkl_path, 'rb') as f:
        plan = pickle.load(f)
        
    print(f"========== NETWORK PLAN DUMP ==========")
    print(f"Total Stages: {len(plan.center2block)}")
    print(f"Choices per Stage: {[len(stage) for stage in plan.center2block]}")
    print("=======================================")
    
    for i, stage in enumerate(plan.center2block):
        print(f"\n[ STAGE {i} ] - {len(stage)} Candidate Blocks:")
        for j, block in enumerate(stage):
            # Extract basic attributes
            model_name = getattr(block, 'model_name', 'UNKNOWN')
            nodes = getattr(block, 'node_list', [])
            
            # Extract input/output shapes if available in the block
            in_size = getattr(block, 'in_size', 'Not explicitly defined')
            out_size = getattr(block, 'out_size', 'Not explicitly defined')
            
            print(f"  -> Choice {j}: {model_name}")
            print(f"       Nodes:    {nodes}")
            if in_size != 'Not explicitly defined' or out_size != 'Not explicitly defined':
                print(f"       Shapes:   IN: {in_size} -> OUT: {out_size}")

if __name__ == '__main__':
    inspect_plan()