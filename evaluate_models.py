#!/usr/bin/env python3
"""
Standalone evaluation script for L2D models on all available datasets.
This script evaluates trained models and presents results in a table format.
"""

import sys
import os
import numpy as np
import torch
import time
import argparse

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

def main():
    parser = argparse.ArgumentParser(description='Evaluate L2D models on all datasets')
    parser.add_argument('--max_instances', type=int, default=None, help='Maximum instances per dataset')
    parser.add_argument('--dataset_type', choices=['all', 'benchmark', 'generated'], default='all',
                        help='Which datasets to evaluate')
    args = parser.parse_args()
    
    # Import after parsing args to avoid conflicts
    from tabulate import tabulate
    from models.actor_critic import ActorCritic
    from JSSP_Env import SJSSP
    from agent_utils import greedy_select_action
    from mb_agg import g_pool_cal
    
    # Default configurations
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_layers = 3
    neighbor_pooling_type = 'sum'
    graph_pool_type = 'average'
    input_dim = 2
    hidden_dim = 64
    num_mlp_layers_feature_extract = 2
    num_mlp_layers_actor = 2
    hidden_dim_actor = 32
    num_mlp_layers_critic = 2
    hidden_dim_critic = 32
    
    def load_model(n_j, n_m, model_path=None):
        """Load a trained model for given problem size"""
        model = ActorCritic(
            n_j=n_j,
            n_m=n_m,
            num_layers=num_layers,
            learn_eps=False,
            neighbor_pooling_type=neighbor_pooling_type,
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_mlp_layers_feature_extract=num_mlp_layers_feature_extract,
            num_mlp_layers_actor=num_mlp_layers_actor,
            hidden_dim_actor=hidden_dim_actor,
            num_mlp_layers_critic=num_mlp_layers_critic,
            hidden_dim_critic=hidden_dim_critic,
            device=device
        )
        
        if model_path and os.path.exists(model_path):
            model.load_state_dict(torch.load(model_path, map_location=device))
            model.eval()
            return model
        else:
            # Try to find a pre-trained model
            for suffix in ['_1_99.pth', '_1_199.pth']:
                path = f'SavedNetwork/{n_j}_{n_m}{suffix}'
                if os.path.exists(path):
                    model.load_state_dict(torch.load(path, map_location=device))
                    model.eval()
                    return model
            # Try the model we just trained
            path = f'{n_j}_{n_m}_1_99.pth'
            if os.path.exists(path):
                model.load_state_dict(torch.load(path, map_location=device))
                model.eval()
                return model
            print(f"Warning: No model found for {n_j}x{n_m}, using random initialization")
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
    
    def evaluate_dataset(dataset_path, model, n_j, n_m, dataset_name, max_instances=None):
        """Evaluate model on a dataset"""
        env = SJSSP(n_j=n_j, n_m=n_m)
        
        # Calculate g_pool
        g_pool = g_pool_cal(
            graph_pool_type=graph_pool_type,
            batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
            n_nodes=n_j * n_m,
            device=device
        )
        
        # Load dataset
        if os.path.exists(dataset_path):
            data = np.load(dataset_path)
            if len(data.shape) == 4:  # Generated data format
                instances = [(data[i, 0], data[i, 1]) for i in range(data.shape[0])]
            else:  # Benchmark format
                instances = [data]
        else:
            print(f"Dataset not found: {dataset_path}")
            return None
        
        if max_instances:
            instances = instances[:max_instances]
        
        makespans = []
        start_time = time.time()
        
        for i, instance in enumerate(instances):
            makespan = evaluate_instance(env, model, instance, g_pool)
            makespans.append(makespan)
            if (i + 1) % 10 == 0:
                print(f"  Evaluated {i + 1}/{len(instances)} instances...")
        
        eval_time = time.time() - start_time
        
        results = {
            'dataset': dataset_name,
            'size': f"{n_j}x{n_m}",
            'instances': len(instances),
            'mean_makespan': np.mean(makespans),
            'std_makespan': np.std(makespans),
            'min_makespan': np.min(makespans),
            'max_makespan': np.max(makespans),
            'eval_time': eval_time
        }
        
        return results
    
    all_results = []
    
    # Benchmark datasets
    if args.dataset_type in ['all', 'benchmark']:
        print("\n=== Evaluating on Benchmark Datasets ===")
        
        # Taillard instances
        tai_datasets = [
            ('tai15x15.npy', 15, 15),
            ('tai20x15.npy', 20, 15),
            ('tai20x20.npy', 20, 20),
            ('tai30x15.npy', 30, 15),
            ('tai30x20.npy', 30, 20),
            ('tai50x15.npy', 50, 15),
            ('tai50x20.npy', 50, 20),
            ('tai100x20.npy', 100, 20),
        ]
        
        for filename, n_j, n_m in tai_datasets:
            print(f"\nEvaluating Taillard {n_j}x{n_m}...")
            model = load_model(n_j, n_m)
            dataset_path = f'BenchDataNmpy/{filename}'
            results = evaluate_dataset(dataset_path, model, n_j, n_m, f'Taillard_{n_j}x{n_m}', args.max_instances)
            if results:
                all_results.append(results)
        
        # DMU instances
        dmu_datasets = [
            ('dmu20x15.npy', 20, 15),
            ('dmu20x20.npy', 20, 20),
            ('dmu30x15.npy', 30, 15),
            ('dmu30x20.npy', 30, 20),
            ('dmu40x15.npy', 40, 15),
            ('dmu40x20.npy', 40, 20),
            ('dmu50x15.npy', 50, 15),
            ('dmu50x20.npy', 50, 20),
        ]
        
        for filename, n_j, n_m in dmu_datasets:
            print(f"\nEvaluating DMU {n_j}x{n_m}...")
            model = load_model(n_j, n_m)
            dataset_path = f'BenchDataNmpy/{filename}'
            results = evaluate_dataset(dataset_path, model, n_j, n_m, f'DMU_{n_j}x{n_m}', args.max_instances)
            if results:
                all_results.append(results)
    
    # Generated validation datasets
    if args.dataset_type in ['all', 'generated']:
        print("\n=== Evaluating on Generated Validation Datasets ===")
        
        generated_datasets = [
            (6, 6),
            (8, 8),  # Include our newly trained model
            (10, 10),
            (15, 15),
            (20, 15),
            (20, 20),
            (30, 15),
            (30, 20),
            (50, 20),
            (100, 20),
            (200, 50),
        ]
        
        for n_j, n_m in generated_datasets:
            print(f"\nEvaluating Generated {n_j}x{n_m}...")
            model = load_model(n_j, n_m)
            dataset_path = f'DataGen/Vali/generatedData{n_j}_{n_m}_Seed200.npy'
            if not os.path.exists(dataset_path):
                dataset_path = f'DataGen/generatedData{n_j}_{n_m}_Seed200.npy'
            if os.path.exists(dataset_path):
                results = evaluate_dataset(dataset_path, model, n_j, n_m, f'Generated_{n_j}x{n_m}', args.max_instances)
                if results:
                    all_results.append(results)
            else:
                print(f"  Skipping - dataset not found")
    
    # Print results table
    print("\n" + "="*100)
    print("EVALUATION RESULTS")
    print("="*100)
    
    # Prepare table data
    table_data = []
    for r in all_results:
        table_data.append([
            r['dataset'],
            r['size'],
            r['instances'],
            f"{r['mean_makespan']:.2f}",
            f"{r['std_makespan']:.2f}",
            f"{r['min_makespan']:.0f}",
            f"{r['max_makespan']:.0f}",
            f"{r['eval_time']:.1f}s"
        ])
    
    headers = ['Dataset', 'Size', 'Instances', 'Mean', 'Std', 'Min', 'Max', 'Time']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    # Summary statistics
    print(f"\nTotal datasets evaluated: {len(all_results)}")
    print(f"Total evaluation time: {sum(r['eval_time'] for r in all_results):.1f} seconds")
    
    # Save results to file
    output_file = 'evaluation_results.txt'
    with open(output_file, 'w') as f:
        f.write("L2D Model Evaluation Results\n")
        f.write("="*100 + "\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
        f.write(f"\n\nTotal datasets evaluated: {len(all_results)}\n")
        f.write(f"Total evaluation time: {sum(r['eval_time'] for r in all_results):.1f} seconds\n")
    
    print(f"\nResults saved to: {output_file}")

if __name__ == '__main__':
    main()