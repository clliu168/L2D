#!/usr/bin/env python3
"""
Comprehensive evaluation script for dispatching rules on JSSP datasets.

This script evaluates all implemented dispatching rules on the same datasets
used for L2D evaluation, allowing direct comparison of performance.
"""

import os
import sys
import numpy as np
import time
from tabulate import tabulate
from JSSP_Env import SJSSP
from dispatching_rules import DispatchingRules, apply_dispatching_rule


def evaluate_rule_on_instance(env, rule_name, instance_data):
    """
    Evaluate a single dispatching rule on one instance.
    
    Args:
        env: JSSP environment
        rule_name: Name of the dispatching rule
        instance_data: Tuple of (durations, machines)
        
    Returns:
        Makespan value
    """
    env.reset(instance_data)
    makespan = apply_dispatching_rule(env, rule_name)
    return makespan


def evaluate_rule_on_dataset(dataset_path, rule_name, n_j, n_m, limit=None):
    """
    Evaluate a dispatching rule on a complete dataset.
    
    Args:
        dataset_path: Path to dataset file
        rule_name: Name of the dispatching rule
        n_j: Number of jobs
        n_m: Number of machines
        limit: Maximum number of instances to evaluate
        
    Returns:
        Dictionary with evaluation metrics
    """
    if not os.path.exists(dataset_path):
        return None
    
    env = SJSSP(n_j=n_j, n_m=n_m)
    
    # Load dataset
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
    
    for instance in instances:
        makespan = evaluate_rule_on_instance(env, rule_name, instance)
        makespans.append(makespan)
    
    eval_time = time.time() - start_time
    
    return {
        'mean': np.mean(makespans),
        'std': np.std(makespans),
        'min': np.min(makespans),
        'max': np.max(makespans),
        'time': eval_time,
        'instances': len(instances)
    }


def main():
    # Parse command line arguments
    max_instances = None
    dataset_type = 'all'
    
    for i, arg in enumerate(sys.argv):
        if arg == '--max_instances' and i + 1 < len(sys.argv):
            max_instances = int(sys.argv[i + 1])
        elif arg == '--dataset_type' and i + 1 < len(sys.argv):
            dataset_type = sys.argv[i + 1]
        elif arg in ['-h', '--help']:
            print("Usage: python3 evaluate_dispatching_rules.py [--max_instances N] [--dataset_type all|benchmark|generated]")
            sys.exit(0)
    
    # Get all dispatching rules
    rules = DispatchingRules.get_all_rules()
    rule_names = sorted(rules.keys())
    
    print("Evaluating Dispatching Rules on JSSP Datasets")
    print("=" * 80)
    print(f"Rules to evaluate: {', '.join(rule_names)}")
    print(f"Dataset type: {dataset_type}")
    if max_instances:
        print(f"Max instances per dataset: {max_instances}")
    print()
    
    # Collect results for all rules and datasets
    all_results = []
    
    # Benchmark datasets
    if dataset_type in ['all', 'benchmark']:
        print("\n=== Evaluating on Benchmark Datasets ===")
        
        # Taillard instances
        tai_datasets = [
            ('tai15x15.npy', 15, 15),
            ('tai20x15.npy', 20, 15),
            ('tai20x20.npy', 20, 20),
            ('tai30x15.npy', 30, 15),
            ('tai30x20.npy', 30, 20),
        ]
        
        for filename, n_j, n_m in tai_datasets:
            print(f"\nEvaluating Taillard {n_j}x{n_m}...")
            dataset_path = f'BenchDataNmpy/{filename}'
            
            for rule_name in rule_names:
                print(f"  {rule_name}...", end='', flush=True)
                result = evaluate_rule_on_dataset(dataset_path, rule_name, n_j, n_m, max_instances)
                if result:
                    result['dataset'] = f'Taillard_{n_j}x{n_m}'
                    result['rule'] = rule_name
                    result['size'] = f'{n_j}x{n_m}'
                    all_results.append(result)
                    print(f" done (mean: {result['mean']:.1f})")
        
        # DMU instances
        dmu_datasets = [
            ('dmu20x15.npy', 20, 15),
            ('dmu20x20.npy', 20, 20),
            ('dmu30x15.npy', 30, 15),
            ('dmu30x20.npy', 30, 20),
        ]
        
        for filename, n_j, n_m in dmu_datasets:
            print(f"\nEvaluating DMU {n_j}x{n_m}...")
            dataset_path = f'BenchDataNmpy/{filename}'
            
            for rule_name in rule_names:
                print(f"  {rule_name}...", end='', flush=True)
                result = evaluate_rule_on_dataset(dataset_path, rule_name, n_j, n_m, max_instances)
                if result:
                    result['dataset'] = f'DMU_{n_j}x{n_m}'
                    result['rule'] = rule_name
                    result['size'] = f'{n_j}x{n_m}'
                    all_results.append(result)
                    print(f" done (mean: {result['mean']:.1f})")
    
    # Generated datasets
    if dataset_type in ['all', 'generated']:
        print("\n=== Evaluating on Generated Datasets ===")
        
        generated_sizes = [
            (6, 6), (8, 8), (10, 10), (15, 15),
            (20, 15), (20, 20), (30, 15), (30, 20)
        ]
        
        for n_j, n_m in generated_sizes:
            print(f"\nEvaluating Generated {n_j}x{n_m}...")
            
            # Try different paths
            paths = [
                f'DataGen/Vali/generatedData{n_j}_{n_m}_Seed200.npy',
                f'DataGen/generatedData{n_j}_{n_m}_Seed200.npy'
            ]
            
            dataset_path = None
            for path in paths:
                if os.path.exists(path):
                    dataset_path = path
                    break
            
            if dataset_path:
                for rule_name in rule_names:
                    print(f"  {rule_name}...", end='', flush=True)
                    result = evaluate_rule_on_dataset(dataset_path, rule_name, n_j, n_m, max_instances)
                    if result:
                        result['dataset'] = f'Generated_{n_j}x{n_m}'
                        result['rule'] = rule_name
                        result['size'] = f'{n_j}x{n_m}'
                        all_results.append(result)
                        print(f" done (mean: {result['mean']:.1f})")
            else:
                print(f"  Dataset not found, skipping")
    
    # Display results
    print("\n" + "="*120)
    print("DISPATCHING RULES EVALUATION RESULTS")
    print("="*120)
    
    # Group results by dataset for comparison
    datasets = sorted(set(r['dataset'] for r in all_results))
    
    for dataset in datasets:
        dataset_results = [r for r in all_results if r['dataset'] == dataset]
        if not dataset_results:
            continue
        
        print(f"\n{dataset} ({dataset_results[0]['size']}):")
        print("-" * 80)
        
        # Prepare table data
        table_data = []
        for r in sorted(dataset_results, key=lambda x: x['mean']):
            table_data.append([
                r['rule'],
                f"{r['mean']:.1f}",
                f"{r['std']:.1f}",
                f"{r['min']:.0f}",
                f"{r['max']:.0f}",
                f"{r['time']:.2f}s"
            ])
        
        headers = ['Rule', 'Mean', 'Std', 'Min', 'Max', 'Time']
        print(tabulate(table_data, headers=headers, tablefmt='simple'))
    
    # Overall summary
    print("\n" + "="*120)
    print("SUMMARY BY RULE (Average across all datasets)")
    print("="*120)
    
    rule_summary = {}
    for rule_name in rule_names:
        rule_results = [r for r in all_results if r['rule'] == rule_name]
        if rule_results:
            rule_summary[rule_name] = {
                'avg_mean': np.mean([r['mean'] for r in rule_results]),
                'avg_time': np.mean([r['time'] for r in rule_results]),
                'datasets': len(rule_results)
            }
    
    summary_data = []
    for rule, stats in sorted(rule_summary.items(), key=lambda x: x[1]['avg_mean']):
        summary_data.append([
            rule,
            f"{stats['avg_mean']:.1f}",
            f"{stats['avg_time']:.3f}s",
            stats['datasets']
        ])
    
    headers = ['Rule', 'Avg Mean Makespan', 'Avg Time', 'Datasets']
    print(tabulate(summary_data, headers=headers, tablefmt='simple'))
    
    # Save detailed results
    output_file = 'dispatching_rules_results.txt'
    with open(output_file, 'w') as f:
        f.write("DISPATCHING RULES EVALUATION RESULTS\n")
        f.write("="*120 + "\n\n")
        
        for dataset in datasets:
            dataset_results = [r for r in all_results if r['dataset'] == dataset]
            if not dataset_results:
                continue
            
            f.write(f"\n{dataset} ({dataset_results[0]['size']}):\n")
            f.write("-" * 80 + "\n")
            
            table_data = []
            for r in sorted(dataset_results, key=lambda x: x['mean']):
                table_data.append([
                    r['rule'],
                    f"{r['mean']:.1f}",
                    f"{r['std']:.1f}",
                    f"{r['min']:.0f}",
                    f"{r['max']:.0f}",
                    f"{r['time']:.2f}s"
                ])
            
            f.write(tabulate(table_data, headers=['Rule', 'Mean', 'Std', 'Min', 'Max', 'Time'], tablefmt='simple'))
            f.write("\n")
    
    print(f"\nDetailed results saved to: {output_file}")
    
    # Create comparison with L2D if needed
    print("\n" + "="*120)
    print("For comparison with L2D results, see CLAUDE.md")


if __name__ == '__main__':
    main()