#!/usr/bin/env python3
"""
Simple test script to evaluate L2D models on available datasets.
This version avoids the argparse conflict by not importing Params directly.
"""

import sys
import os
import numpy as np
import torch
import time

# Test on a small subset first
def test_evaluation():
    # Import modules
    from models.actor_critic import ActorCritic
    from JSSP_Env import SJSSP
    from agent_utils import greedy_select_action
    from mb_agg import g_pool_cal
    
    # Configuration
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Model parameters (from Params.py defaults)
    model_params = {
        'num_layers': 3,
        'neighbor_pooling_type': 'sum',
        'input_dim': 2,
        'hidden_dim': 64,
        'num_mlp_layers_feature_extract': 2,
        'num_mlp_layers_actor': 2,
        'hidden_dim_actor': 32,
        'num_mlp_layers_critic': 2,
        'hidden_dim_critic': 32,
        'learn_eps': False,
        'device': device
    }
    
    # Test on 6x6 generated data
    n_j, n_m = 6, 6
    print(f"\nTesting on {n_j}x{n_m} generated data...")
    
    # Load model
    model = ActorCritic(n_j=n_j, n_m=n_m, **model_params)
    model_path = f'SavedNetwork/{n_j}_{n_m}_1_99.pth'
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print("Using random initialization (no trained model found)")
    model.eval()
    
    # Load dataset
    dataset_path = f'DataGen/Vali/generatedData{n_j}_{n_m}_Seed200.npy'
    if not os.path.exists(dataset_path):
        dataset_path = f'DataGen/generatedData{n_j}_{n_m}_Seed200.npy'
    
    if not os.path.exists(dataset_path):
        print(f"Dataset not found: {dataset_path}")
        return
    
    data = np.load(dataset_path)
    print(f"Loaded dataset with shape: {data.shape}")
    
    # Create environment
    env = SJSSP(n_j=n_j, n_m=n_m)
    
    # Calculate g_pool
    g_pool = g_pool_cal(
        graph_pool_type='average',
        batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
        n_nodes=n_j * n_m,
        device=device
    )
    
    # Evaluate first 3 instances
    makespans = []
    for i in range(min(3, data.shape[0])):
        print(f"\nEvaluating instance {i+1}...")
        instance = (data[i, 0], data[i, 1])
        
        # Reset environment
        adj, fea, candidate, mask = env.reset(instance)
        
        rewards = 0
        steps = 0
        while not env.done():
            # Prepare tensors
            fea_tensor = torch.from_numpy(fea).to(device)
            adj_tensor = torch.from_numpy(adj).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(candidate).to(device)
            mask_tensor = torch.from_numpy(mask).to(device)
            
            # Get action from model
            with torch.no_grad():
                pi, _ = model(
                    x=fea_tensor,
                    graph_pool=g_pool,
                    padded_nei=None,
                    adj=adj_tensor,
                    candidate=candidate_tensor.unsqueeze(0),
                    mask=mask_tensor.unsqueeze(0)
                )
                action = greedy_select_action(pi, candidate)
            
            # Step environment
            adj, fea, reward, _, candidate, mask = env.step(action.item())
            rewards += reward
            steps += 1
        
        makespan = -rewards
        makespans.append(makespan)
        print(f"  Makespan: {makespan}, Steps: {steps}")
    
    print(f"\nResults for {n_j}x{n_m}:")
    print(f"Mean makespan: {np.mean(makespans):.2f}")
    print(f"Std makespan: {np.std(makespans):.2f}")

if __name__ == '__main__':
    test_evaluation()