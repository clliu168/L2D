#!/bin/bash
# Wrapper script to run evaluation avoiding argparse conflicts

cd "$(dirname "$0")"

# Create a temporary Python script
cat > temp_eval.py << 'EOF'
import os
import sys
import numpy as np
import torch
import time
import warnings
warnings.filterwarnings('ignore')

# Get arguments
dataset_type = sys.argv[1] if len(sys.argv) > 1 else 'all'
max_instances = int(sys.argv[2]) if len(sys.argv) > 2 else None

from tabulate import tabulate
from models.actor_critic import ActorCritic
from JSSP_Env import SJSSP
from agent_utils import greedy_select_action
from mb_agg import g_pool_cal

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

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
    model = ActorCritic(n_j=n_j, n_m=n_m, **model_params)
    model_paths = [
        f'SavedNetwork/{n_j}_{n_m}_1_99.pth',
        f'SavedNetwork/{n_j}_{n_m}_1_199.pth',
        f'{n_j}_{n_m}_1_99.pth'
    ]
    
    for path in model_paths:
        if os.path.exists(path):
            model.load_state_dict(torch.load(path, map_location=device))
            break
    
    model.eval()
    return model

def evaluate_instance(env, model, instance_data, g_pool):
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
    if not os.path.exists(dataset_path):
        return None
    
    env = SJSSP(n_j=n_j, n_m=n_m)
    
    g_pool = g_pool_cal(
        graph_pool_type='average',
        batch_size=torch.Size([1, n_j * n_m, n_j * n_m]),
        n_nodes=n_j * n_m,
        device=device
    )
    
    data = np.load(dataset_path)
    if len(data.shape) == 4:
        instances = [(data[i, 0], data[i, 1]) for i in range(data.shape[0])]
    elif len(data.shape) == 3:
        instances = [(data[0], data[1])]
    else:
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

all_results = []

if dataset_type in ['all', 'benchmark']:
    print("\n=== Evaluating Benchmark Datasets ===")
    
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

if dataset_type in ['all', 'generated']:
    print("\n=== Evaluating Generated Datasets ===")
    
    generated_sizes = [
        (6, 6), (8, 8), (10, 10), (15, 15),
        (20, 15), (20, 20), (30, 15), (30, 20)
    ]
    
    for n_j, n_m in generated_sizes:
        print(f"Evaluating Generated {n_j}x{n_m}...")
        model = load_model(n_j, n_m)
        
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

if all_results:
    print("\n" + "="*90)
    print("L2D MODEL EVALUATION RESULTS")
    print("="*90 + "\n")
    
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
    
    headers = ['Dataset', 'Size', 'Instances', 'Mean', 'Std', 'Min', 'Max', 'Time']
    print(tabulate(table_data, headers=headers, tablefmt='grid'))
    
    total_time = sum(r['time'] for r in all_results)
    print(f"\nTotal datasets evaluated: {len(all_results)}")
    print(f"Total evaluation time: {total_time:.1f} seconds")
    
    with open('evaluation_results.txt', 'w') as f:
        f.write("L2D MODEL EVALUATION RESULTS\n")
        f.write("="*90 + "\n\n")
        f.write(tabulate(table_data, headers=headers, tablefmt='grid'))
        f.write(f"\n\nTotal datasets evaluated: {len(all_results)}\n")
        f.write(f"Total evaluation time: {total_time:.1f} seconds\n")
    
    print("\nResults saved to: evaluation_results.txt")
EOF

# Run the temporary script with clean environment
python3 temp_eval.py "$@"

# Clean up
rm -f temp_eval.py