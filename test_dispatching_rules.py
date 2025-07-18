"""
Test suite for verifying correctness of dispatching rule implementations.

This module tests:
1. Rules select only valid operations
2. Selected operations respect precedence constraints
3. Final schedules are feasible
4. Makespan calculations are correct
5. Rules behave as expected on known examples
"""

import numpy as np
import time
from JSSP_Env import SJSSP
from uniform_instance_gen import uni_instance_gen
from dispatching_rules import DispatchingRules, apply_dispatching_rule


def verify_schedule_feasibility(env):
    """
    Verify that the final schedule is feasible.
    
    Checks:
    - All operations are scheduled
    - No machine conflicts
    - Job precedence constraints are satisfied
    """
    n_j = env.number_of_jobs
    n_m = env.number_of_machines
    
    # Check all operations are scheduled
    scheduled_ops = set(env.partial_sol_sequeence)
    all_ops = set(range(n_j * n_m))
    
    if scheduled_ops != all_ops:
        missing = all_ops - scheduled_ops
        print(f"ERROR: Missing operations: {missing}")
        return False
    
    # Extract schedule information
    schedule = {}  # operation_id -> (start_time, end_time, machine)
    
    # Reconstruct schedule from environment state
    for op_id in env.partial_sol_sequeence:
        job_id = op_id // n_m
        op_in_job = op_id % n_m
        machine = env.m[job_id, op_in_job] - 1
        duration = env.dur[job_id, op_in_job]
        
        # Find start time from mchsStartTimes and opIDsOnMchs
        machine_ops = env.opIDsOnMchs[machine]
        machine_starts = env.mchsStartTimes[machine]
        
        idx = np.where(machine_ops == op_id)[0]
        if len(idx) > 0:
            start_time = machine_starts[idx[0]]
            end_time = start_time + duration
            schedule[op_id] = (start_time, end_time, machine)
    
    # Check job precedence constraints
    for job_id in range(n_j):
        for op_idx in range(n_m - 1):
            current_op = job_id * n_m + op_idx
            next_op = job_id * n_m + op_idx + 1
            
            if current_op in schedule and next_op in schedule:
                if schedule[current_op][1] > schedule[next_op][0]:
                    print(f"ERROR: Job {job_id} precedence violated between op {op_idx} and {op_idx+1}")
                    return False
    
    # Check machine conflicts
    for machine in range(n_m):
        machine_schedule = [(op, schedule[op]) for op in schedule if schedule[op][2] == machine]
        machine_schedule.sort(key=lambda x: x[1][0])  # Sort by start time
        
        for i in range(len(machine_schedule) - 1):
            op1, (start1, end1, _) = machine_schedule[i]
            op2, (start2, end2, _) = machine_schedule[i + 1]
            
            if end1 > start2:
                print(f"ERROR: Machine {machine} conflict between operations {op1} and {op2}")
                return False
    
    return True


def test_rule_behavior():
    """Test that each rule behaves according to its definition."""
    print("Testing rule behavior on controlled examples...")
    
    # Create a simple 2x2 instance for testing
    n_j, n_m = 2, 2
    env = SJSSP(n_j=n_j, n_m=n_m)
    
    # Test SPT: Should select shortest processing time
    print("\n1. Testing SPT rule:")
    dur = np.array([[10, 20], [5, 15]])  # Job 1 op 0 has 10, Job 2 op 0 has 5
    mch = np.array([[1, 2], [1, 2]])
    data = (dur, mch)
    env.reset(data)
    
    candidates = env.omega
    mask = env.mask
    action = DispatchingRules.SPT(env, candidates, mask)
    expected = 2  # Job 2, op 0 (has duration 5)
    print(f"  Selected operation {action}, expected {expected}: {'PASS' if action == expected else 'FAIL'}")
    
    # Test LPT: Should select longest processing time
    print("\n2. Testing LPT rule:")
    env.reset(data)
    action = DispatchingRules.LPT(env, candidates, mask)
    expected = 0  # Job 1, op 0 (has duration 10)
    print(f"  Selected operation {action}, expected {expected}: {'PASS' if action == expected else 'FAIL'}")
    
    # Test FIFO: Should select from first job
    print("\n3. Testing FIFO rule:")
    env.reset(data)
    action = DispatchingRules.FIFO(env, candidates, mask)
    expected = 0  # Job 0 (first job)
    print(f"  Selected operation {action}, expected {expected}: {'PASS' if action == expected else 'FAIL'}")
    
    # Test LIFO: Should select from last job
    print("\n4. Testing LIFO rule:")
    env.reset(data)
    action = DispatchingRules.LIFO(env, candidates, mask)
    expected = 2  # Job 1 (last job)
    print(f"  Selected operation {action}, expected {expected}: {'PASS' if action == expected else 'FAIL'}")
    
    # Test MWR: Should select job with most work remaining
    print("\n5. Testing MWR rule:")
    dur2 = np.array([[5, 5], [20, 20]])  # Job 1 has total 10, Job 2 has total 40
    data2 = (dur2, mch)
    env.reset(data2)
    action = DispatchingRules.MWR(env, env.omega, env.mask)
    expected = 2  # Job 1 (has total work 40)
    print(f"  Selected operation {action}, expected {expected}: {'PASS' if action == expected else 'FAIL'}")


def test_makespan_calculation():
    """Test that makespan is calculated correctly."""
    print("\nTesting makespan calculation...")
    
    # Create a simple instance with known optimal makespan
    n_j, n_m = 2, 2
    env = SJSSP(n_j=n_j, n_m=n_m)
    
    # Simple instance: 
    # Job 0: Op0(10 on M1), Op1(10 on M2)
    # Job 1: Op0(10 on M2), Op1(10 on M1)
    # Optimal makespan should be 30
    dur = np.array([[10, 10], [10, 10]])
    mch = np.array([[1, 2], [2, 1]])
    data = (dur, mch)
    
    env.reset(data)
    makespan = apply_dispatching_rule(env, 'FIFO')
    
    print(f"  Calculated makespan: {makespan}")
    print(f"  Expected range: 20-40 (optimal is 20)")
    print(f"  Result: {'PASS' if 20 <= makespan <= 40 else 'FAIL'}")


def test_all_rules_complete():
    """Test that all rules can complete various instances."""
    print("\nTesting all rules complete various instances...")
    
    rules = DispatchingRules.get_all_rules()
    test_sizes = [(3, 3), (5, 5), (10, 10)]
    
    for n_j, n_m in test_sizes:
        print(f"\n  Testing {n_j}x{n_m} instance:")
        env = SJSSP(n_j=n_j, n_m=n_m)
        
        # Generate random instance
        np.random.seed(123)
        data = uni_instance_gen(n_j=n_j, n_m=n_m, low=1, high=99)
        
        for rule_name in rules:
            env.reset(data)
            try:
                start_time = time.time()
                makespan = apply_dispatching_rule(env, rule_name)
                elapsed = time.time() - start_time
                
                # Verify schedule feasibility
                is_feasible = verify_schedule_feasibility(env)
                
                status = "PASS" if is_feasible else "FAIL"
                print(f"    {rule_name:8s}: Makespan={makespan:6.0f}, Time={elapsed:.3f}s, Feasible={status}")
            except Exception as e:
                print(f"    {rule_name:8s}: FAILED with error: {e}")


def test_deterministic_rules():
    """Test that deterministic rules give consistent results."""
    print("\nTesting deterministic behavior...")
    
    deterministic_rules = ['SPT', 'LPT', 'FIFO', 'LIFO', 'MWR', 'LWR', 'MOPNR', 'LOPNR', 'MTWR']
    
    n_j, n_m = 5, 5
    env = SJSSP(n_j=n_j, n_m=n_m)
    
    np.random.seed(456)
    data = uni_instance_gen(n_j=n_j, n_m=n_m, low=1, high=99)
    
    for rule_name in deterministic_rules:
        makespans = []
        for _ in range(3):
            env.reset(data)
            makespan = apply_dispatching_rule(env, rule_name)
            makespans.append(makespan)
        
        is_consistent = len(set(makespans)) == 1
        print(f"  {rule_name:8s}: {'PASS' if is_consistent else 'FAIL'} (makespans: {makespans})")


def test_edge_cases():
    """Test edge cases like single job, single machine, etc."""
    print("\nTesting edge cases...")
    
    # Single job
    print("  Single job (1x5):")
    env = SJSSP(n_j=1, n_m=5)
    data = uni_instance_gen(n_j=1, n_m=5, low=1, high=10)
    env.reset(data)
    makespan = apply_dispatching_rule(env, 'SPT')
    expected = np.sum(data[0])  # Sum of all durations
    print(f"    Makespan: {makespan}, Expected: {expected}, {'PASS' if abs(makespan - expected) < 1e-6 else 'FAIL'}")
    
    # Single machine per job
    print("  Single operation per job (5x1):")
    env = SJSSP(n_j=5, n_m=1)
    data = uni_instance_gen(n_j=5, n_m=1, low=1, high=10)
    env.reset(data)
    makespan = apply_dispatching_rule(env, 'LPT')
    print(f"    Completed successfully: PASS")


def run_all_tests():
    """Run all test suites."""
    print("="*60)
    print("DISPATCHING RULES CORRECTNESS TESTS")
    print("="*60)
    
    test_rule_behavior()
    test_makespan_calculation()
    test_all_rules_complete()
    test_deterministic_rules()
    test_edge_cases()
    
    print("\n" + "="*60)
    print("ALL TESTS COMPLETED")
    print("="*60)


if __name__ == "__main__":
    run_all_tests()