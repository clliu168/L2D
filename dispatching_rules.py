"""
Implementation of well-known dispatching rules for Job Shop Scheduling Problem (JSSP).

This module implements classic dispatching rules that can be used as baselines
for comparison with learned policies like L2D.
"""

import numpy as np
import random


class DispatchingRules:
    """
    Collection of dispatching rules for JSSP.
    
    Each rule is a static method that takes the current state and returns
    the selected operation from available candidates.
    """
    
    @staticmethod
    def get_operation_info(env, candidates, mask):
        """
        Extract useful information about candidate operations.
        
        Args:
            env: JSSP environment instance
            candidates: Array of candidate operation IDs (omega)
            mask: Boolean mask indicating completed jobs
            
        Returns:
            Dictionary with operation information
        """
        # Get valid candidates (operations from uncompleted jobs)
        valid_candidates = candidates[mask == 0]
        
        # Extract information for each valid candidate
        info = {
            'candidates': valid_candidates,
            'processing_times': [],
            'job_ids': [],
            'operation_ids': [],
            'remaining_work': [],
            'remaining_ops': [],
            'machine_ids': []
        }
        
        for op in valid_candidates:
            job_id = op // env.number_of_machines
            op_id_in_job = op % env.number_of_machines
            
            info['job_ids'].append(job_id)
            info['operation_ids'].append(op_id_in_job)
            info['processing_times'].append(env.dur[job_id, op_id_in_job])
            info['machine_ids'].append(env.m[job_id, op_id_in_job] - 1)
            
            # Calculate remaining work for this job
            remaining_work = np.sum(env.dur[job_id, op_id_in_job:])
            info['remaining_work'].append(remaining_work)
            
            # Calculate remaining operations for this job
            remaining_ops = env.number_of_machines - op_id_in_job
            info['remaining_ops'].append(remaining_ops)
        
        # Convert lists to numpy arrays
        for key in ['processing_times', 'job_ids', 'operation_ids', 
                    'remaining_work', 'remaining_ops', 'machine_ids']:
            info[key] = np.array(info[key])
        
        return info
    
    @staticmethod
    def SPT(env, candidates, mask):
        """
        Shortest Processing Time (SPT) rule.
        Select the operation with minimum processing time.
        """
        info = DispatchingRules.get_operation_info(env, candidates, mask)
        if len(info['candidates']) == 0:
            return None
        
        min_idx = np.argmin(info['processing_times'])
        return info['candidates'][min_idx]
    
    @staticmethod
    def LPT(env, candidates, mask):
        """
        Longest Processing Time (LPT) rule.
        Select the operation with maximum processing time.
        """
        info = DispatchingRules.get_operation_info(env, candidates, mask)
        if len(info['candidates']) == 0:
            return None
        
        max_idx = np.argmax(info['processing_times'])
        return info['candidates'][max_idx]
    
    @staticmethod
    def FIFO(env, candidates, mask):
        """
        First In First Out (FIFO) rule.
        Select operation from the job with smallest ID (earliest job).
        """
        info = DispatchingRules.get_operation_info(env, candidates, mask)
        if len(info['candidates']) == 0:
            return None
        
        min_idx = np.argmin(info['job_ids'])
        return info['candidates'][min_idx]
    
    @staticmethod
    def LIFO(env, candidates, mask):
        """
        Last In First Out (LIFO) rule.
        Select operation from the job with largest ID (latest job).
        """
        info = DispatchingRules.get_operation_info(env, candidates, mask)
        if len(info['candidates']) == 0:
            return None
        
        max_idx = np.argmax(info['job_ids'])
        return info['candidates'][max_idx]
    
    @staticmethod
    def MWR(env, candidates, mask):
        """
        Most Work Remaining (MWR) rule.
        Select operation from the job with most total remaining processing time.
        """
        info = DispatchingRules.get_operation_info(env, candidates, mask)
        if len(info['candidates']) == 0:
            return None
        
        max_idx = np.argmax(info['remaining_work'])
        return info['candidates'][max_idx]
    
    @staticmethod
    def LWR(env, candidates, mask):
        """
        Least Work Remaining (LWR) rule.
        Select operation from the job with least total remaining processing time.
        """
        info = DispatchingRules.get_operation_info(env, candidates, mask)
        if len(info['candidates']) == 0:
            return None
        
        min_idx = np.argmin(info['remaining_work'])
        return info['candidates'][min_idx]
    
    @staticmethod
    def MOPNR(env, candidates, mask):
        """
        Most Operations Remaining (MOPNR) rule.
        Select operation from the job with most remaining operations.
        """
        info = DispatchingRules.get_operation_info(env, candidates, mask)
        if len(info['candidates']) == 0:
            return None
        
        max_idx = np.argmax(info['remaining_ops'])
        return info['candidates'][max_idx]
    
    @staticmethod
    def LOPNR(env, candidates, mask):
        """
        Least Operations Remaining (LOPNR) rule.
        Select operation from the job with least remaining operations.
        """
        info = DispatchingRules.get_operation_info(env, candidates, mask)
        if len(info['candidates']) == 0:
            return None
        
        min_idx = np.argmin(info['remaining_ops'])
        return info['candidates'][min_idx]
    
    @staticmethod
    def RANDOM(env, candidates, mask):
        """
        Random selection rule.
        Randomly select from available operations.
        """
        info = DispatchingRules.get_operation_info(env, candidates, mask)
        if len(info['candidates']) == 0:
            return None
        
        return np.random.choice(info['candidates'])
    
    @staticmethod
    def MTWR(env, candidates, mask):
        """
        Modified Total Work Remaining (MTWR) rule.
        Select operation that minimizes total work remaining on its machine.
        This considers machine workload balance.
        """
        info = DispatchingRules.get_operation_info(env, candidates, mask)
        if len(info['candidates']) == 0:
            return None
        
        # Calculate total remaining work on each machine
        machine_workloads = np.zeros(env.number_of_machines)
        
        for job_id in range(env.number_of_jobs):
            for op_id in range(env.number_of_machines):
                op_idx = job_id * env.number_of_machines + op_id
                if op_idx not in env.partial_sol_sequeence:
                    machine_id = env.m[job_id, op_id] - 1
                    machine_workloads[machine_id] += env.dur[job_id, op_id]
        
        # Find operation that goes to least loaded machine
        candidate_machine_loads = []
        for i, op in enumerate(info['candidates']):
            machine_id = info['machine_ids'][i]
            candidate_machine_loads.append(machine_workloads[machine_id])
        
        min_idx = np.argmin(candidate_machine_loads)
        return info['candidates'][min_idx]
    
    @staticmethod
    def get_all_rules():
        """Return dictionary of all available dispatching rules."""
        return {
            'SPT': DispatchingRules.SPT,
            'LPT': DispatchingRules.LPT,
            'FIFO': DispatchingRules.FIFO,
            'LIFO': DispatchingRules.LIFO,
            'MWR': DispatchingRules.MWR,
            'LWR': DispatchingRules.LWR,
            'MOPNR': DispatchingRules.MOPNR,
            'LOPNR': DispatchingRules.LOPNR,
            'RANDOM': DispatchingRules.RANDOM,
            'MTWR': DispatchingRules.MTWR
        }


def apply_dispatching_rule(env, rule_name='SPT'):
    """
    Apply a dispatching rule to solve a JSSP instance.
    
    Args:
        env: JSSP environment instance
        rule_name: Name of the dispatching rule to use
        
    Returns:
        Final makespan (total completion time)
    """
    rules = DispatchingRules.get_all_rules()
    if rule_name not in rules:
        raise ValueError(f"Unknown rule: {rule_name}. Available rules: {list(rules.keys())}")
    
    rule_func = rules[rule_name]
    
    # Get initial state
    _, _, candidates, mask = env.omega, env.mask, env.omega, env.mask
    
    total_reward = 0
    steps = 0
    
    while not env.done():
        # Select action using dispatching rule
        action = rule_func(env, candidates, mask)
        
        if action is None:
            raise RuntimeError("No valid action found")
        
        # Take action
        _, _, reward, done, candidates, mask = env.step(action)
        total_reward += reward
        steps += 1
    
    # Return final makespan from environment
    return env.LBs.max()


if __name__ == "__main__":
    # Test the dispatching rules
    from JSSP_Env import SJSSP
    from uniform_instance_gen import uni_instance_gen
    
    # Create a small test instance
    n_j, n_m = 3, 3
    env = SJSSP(n_j=n_j, n_m=n_m)
    
    # Generate random instance
    np.random.seed(42)
    data = uni_instance_gen(n_j=n_j, n_m=n_m, low=1, high=99)
    
    print("Test instance:")
    print("Durations:\n", data[0])
    print("Machines:\n", data[1])
    print()
    
    # Test each rule
    rules = DispatchingRules.get_all_rules()
    results = {}
    
    for rule_name in rules:
        env.reset(data)
        makespan = apply_dispatching_rule(env, rule_name)
        results[rule_name] = makespan
        print(f"{rule_name}: Makespan = {makespan}")
    
    print("\nBest rule:", min(results, key=results.get))
    print("Worst rule:", max(results, key=results.get))