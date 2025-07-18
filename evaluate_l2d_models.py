#!/usr/bin/env python3
"""
Comprehensive evaluation script for L2D models on all available datasets.
Presents results in a clean table format.

Usage:
    python3 evaluate_l2d_models.py --dataset_type all
    python3 evaluate_l2d_models.py --dataset_type generated --max_instances 10
    python3 evaluate_l2d_models.py --dataset_type benchmark
"""

import sys
import os
import numpy as np
import torch
import time
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

def main():
    # Parse command line arguments manually to avoid conflicts
    import sys
    dataset_type = 'all'
    max_instances = None
    
    for i, arg in enumerate(sys.argv):
        if arg == '--dataset_type' and i + 1 < len(sys.argv):
            dataset_type = sys.argv[i + 1]
        elif arg == '--max_instances' and i + 1 < len(sys.argv):
            max_instances = int(sys.argv[i + 1])
        elif arg in ['-h', '--help']:
            print(__doc__)
            sys.exit(0)
    
    # Import modules after argument parsing
    from tabulate import tabulate
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
    
    def load_model(n_j, n_m):
        """Load a trained model for given problem size"""
        model = ActorCritic(n_j=n_j, n_m=n_m, **model_params)
        
        # Try different model paths
        model_paths = [
            f'SavedNetwork/{n_j}_{n_m}_1_99.pth',
            f'SavedNetwork/{n_j}_{n_m}_1_199.pth',
            f'{n_j}_{n_m}_1_99.pth'  # For newly trained models
        ]
        
        model_loaded = False
        for path in model_paths:
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=device))
                model_loaded = True
                break
        
        if not model_loaded:
            print(f"  Warning: No trained model found for {n_j}x{n_m}")
        
        model.eval()
        return model
    
    def evaluate_instance(env, model, instance_data, g_pool):
        """Evaluate a single instance"""
        adj, fea, candidate, mask = env.reset(instance_data)
        
        rewards = 0
        while not env.done():
            fea_tensor = torch.from_numpy(fea).to(device)
            adj_tensor = torch.from_numpy(adj).to(device).to_sparse()
            candidate_tensor = torch.from_numpy(candidate).to(device)
            mask_tensor = torch.from_numpy(mask).to(device)
            
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
            
            adj, fea, reward, _, candidate, mask = env.step(action.item())
            rewards += reward
        
        return -rewards
    
    def evaluate_dataset(dataset_path, model, n_j, n_m, dataset_name, limit=None):
        """Evaluate model on a dataset"""
        env = SJSSP(n_j=n_j, n_m=n_m)
        
        # Calculate g_pool
        g_pool = g_pool_cal(
            graph_pool_type='average',
            batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
            n_nodes=n_j * n_m,
            device=device
        )
        
        # Load dataset
        if not os.path.exists(dataset_path):
            return None
        
        data = np.load(dataset_path)
        if len(data.shape) == 4:  # Generated data format
            instances = [(data[i, 0], data[i, 1]) for i in range(data.shape[0])]
        elif len(data.shape) == 3:  # Single benchmark instance
            instances = [(data[0], data[1])]
        else:  # Multiple benchmark instances
            instances = [data]
        
        if limit:
            instances = instances[:limit]
        
        makespans = []
        start_time = time.time()
        
        for i, instance in enumerate(instances):
            makespan = evaluate_instance(env, model, instance, g_pool)
            makespans.append(makespan)
        
        eval_time = time.time() - start_time
        
        return {
            'dataset': dataset_name,
            'size': f"{n_j}x{n_m}",
            'instances': len(instances),
            'mean': np.mean(makespans),
            'std': np.std(makespans),
            'min': np.min(makespans),
            'max': np.max(makespans),
            'time': eval_time
        }
    
    # Collect all results
    all_results = []
    
    # Evaluate benchmark datasets
    if dataset_type in ['all', 'benchmark']:
        print("\n=== Evaluating Benchmark Datasets ===")
        
        # Taillard instances
        tai_datasets = [
            ('tai15x15.npy', 15, 15),
            ('tai20x15.npy', 20, 15),
            ('tai20x20.npy', 20, 20),
            ('tai30x15.npy', 30, 15),
            ('tai30x20.npy', 30, 20),
        ]
        
        for filename, n_j, n_m in tai_datasets:
            print(f"Evaluating Taillard {n_j}x{n_m}...")
            model = load_model(n_j, n_m)
            path = f'BenchDataNmpy/{filename}'
            result = evaluate_dataset(path, model, n_j, n_m, f'Taillard_{n_j}x{n_m}', max_instances)
            if result:
                all_results.append(result)
        
        # DMU instances
        dmu_datasets = [
            ('dmu20x15.npy', 20, 15),
            ('dmu20x20.npy', 20, 20),
            ('dmu30x15.npy', 30, 15),
            ('dmu30x20.npy', 30, 20),
        ]
        
        for filename, n_j, n_m in dmu_datasets:
            print(f"Evaluating DMU {n_j}x{n_m}...")
            model = load_model(n_j, n_m)
            path = f'BenchDataNmpy/{filename}'
            result = evaluate_dataset(path, model, n_j, n_m, f'DMU_{n_j}x{n_m}', max_instances)
            if result:
                all_results.append(result)
    
    # Evaluate generated datasets
    if dataset_type in ['all', 'generated']:
        print("\n=== Evaluating Generated Datasets ===")
        
        generated_sizes = [
            (6, 6), (8, 8), (10, 10), (15, 15),
            (20, 15), (20, 20), (30, 15), (30, 20)
        ]
        
        for n_j, n_m in generated_sizes:
            print(f"Evaluating Generated {n_j}x{n_m}...")
            model = load_model(n_j, n_m)
            
            # Try different paths
            paths = [
                f'DataGen/Vali/generatedData{n_j}_{n_m}_Seed200.npy',
                f'DataGen/generatedData{n_j}_{n_m}_Seed200.npy'
            ]
            
            for path in paths:
                if os.path.exists(path):
                    result = evaluate_dataset(path, model, n_j, n_m, f'Generated_{n_j}x{n_m}', max_instances)
                    if result:
                        all_results.append(result)
                    break
    
    # Display results in table format
    if all_results:
        print("\n" + "="*90)
        print("L2D MODEL EVALUATION RESULTS")
        print("="*90)
        
        # Prepare table data
        table_data = []
        for r in all_results:
            table_data.append([
                r['dataset'],
                r['size'],
                r['instances'],
                f"{r['mean']:.1f}",
                f"{r['std']:.1f}",
                f"{r['min']:.0f}",
                f"{r['max']:.0f}",
                f"{r['time']:.1f}s"
            ])
        
        headers = ['Dataset', 'Size', 'Inst', 'Mean', 'Std', 'Min', 'Max', 'Time']
        print(tabulate(table_data, headers=headers, tablefmt='simple'))
        
        # Summary
        total_time = sum(r['time'] for r in all_results)
        print(f"\nTotal datasets evaluated: {len(all_results)}")
        print(f"Total evaluation time: {total_time:.1f} seconds")
        
        # Save to file
        with open('evaluation_results.txt', 'w') as f:
            f.write("L2D MODEL EVALUATION RESULTS\n")
            f.write("="*90 + "\n\n")
            f.write(tabulate(table_data, headers=headers, tablefmt='simple'))
            f.write(f"\n\nTotal datasets evaluated: {len(all_results)}\n")
            f.write(f"Total evaluation time: {total_time:.1f} seconds\n")
        
        print("\nResults saved to: evaluation_results.txt")
    else:
        print("\nNo results to display.")

if __name__ == '__main__':
    main()