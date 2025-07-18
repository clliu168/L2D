# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

L2D (Learning to Dispatch) is a Deep Reinforcement Learning framework for solving Job Shop Scheduling Problems (JSSP). It implements the NeurIPS 2020 paper "Learning to Dispatch for Job Shop Scheduling via Deep Reinforcement Learning" by Zhang et al.

The project uses Proximal Policy Optimization (PPO) with Graph Neural Networks to learn dispatching rules that minimize makespan in job shop scheduling scenarios.

## Key Architecture Components

### Environment (`JSSP_Env.py`)
- Custom Gym environment for JSSP
- State: adjacency matrix + node features (operation end time lower bounds)
- Action space: selecting next operation to schedule
- Reward: negative makespan

### Model Architecture (`models/`)
- `actor_critic.py`: Main ActorCritic model combining GNN feature extraction with actor/critic heads
- `graphcnn_congForSJSSP.py`: Graph CNN for processing job-operation graphs
- Features are normalized using `et_normalize_coef` (default: 1000) and `wkr_normalize_coef` (default: 100)

### Training (`PPO_jssp_multiInstances.py`)
- Multi-environment parallel training
- Key hyperparameters in `Params.py`:
  - `num_envs`: Number of parallel environments (default: 4)
  - `lr`: Learning rate (default: 2e-5)
  - `eps_clip`: PPO clipping parameter (default: 0.2)

## Development Commands

### Docker Setup
```bash
# Build image
sudo docker build -t l2d-image .

# Run with GPU
sudo docker run --gpus all --name l2d-container -it l2d-image
```

### Running Tests

Test on generated instances:
```bash
python3 test_learned.py
```

Test on benchmark instances (tai, dmu datasets):
```bash
python3 test_learned_on_benchmark.py
```

### Configuration
All parameters are in `Params.py`. Key settings:
- `device`: "cuda" or "cpu"
- `n_j`: Number of jobs (default: 15)
- `n_m`: Number of machines (default: 15)
- `hidden_dim`: GNN hidden dimension (default: 64)
- `num_layers`: GNN layers (default: 3)

### Data Generation
Generate new training instances:
```bash
python3 DataGen/generate_data.py
```

## Project Structure

- `BenchDataNmpy/`: Benchmark JSSP instances (tai, dmu datasets)
- `DataGen/`: Data generation utilities
  - `Vali/`: Validation dataset
- `SavedNetwork/`: Pre-trained models for different problem sizes
- `models/`: Neural network architectures
- `agent_utils.py`: Action selection and PPO utilities
- `uniform_instance_gen.py`: Random JSSP instance generation
- `permissibleLS.py`: Permissible left shift for feasible scheduling
- `updateAdjMat.py`: Dynamic adjacency matrix updates

## Pre-trained Models

Models are saved in `SavedNetwork/` with naming convention indicating problem size:
- Example: `15x15_26.pth` for 15 jobs × 15 machines

## Important Implementation Details

1. **State Representation**: The environment maintains both static (initial graph structure) and dynamic (current scheduling state) components
2. **Action Masking**: Only feasible operations are selectable at each step
3. **Normalization**: Features are normalized to improve training stability
4. **Multi-Instance Training**: The agent trains on multiple problem instances simultaneously to improve generalization

## Evaluation Datasets

### Standard Benchmark Datasets (`BenchDataNmpy/`)

#### Taillard Instances (tai)
- `tai15x15.npy` - 15 jobs × 15 machines
- `tai20x15.npy` - 20 jobs × 15 machines
- `tai20x20.npy` - 20 jobs × 20 machines
- `tai30x15.npy` - 30 jobs × 15 machines
- `tai30x20.npy` - 30 jobs × 20 machines
- `tai50x15.npy` - 50 jobs × 15 machines
- `tai50x20.npy` - 50 jobs × 20 machines
- `tai100x20.npy` - 100 jobs × 20 machines

#### DMU Instances (dmu)
- `dmu20x15.npy` - 20 jobs × 15 machines
- `dmu20x20.npy` - 20 jobs × 20 machines
- `dmu30x15.npy` - 30 jobs × 15 machines
- `dmu30x20.npy` - 30 jobs × 20 machines
- `dmu40x15.npy` - 40 jobs × 15 machines
- `dmu40x20.npy` - 40 jobs × 20 machines
- `dmu50x15.npy` - 50 jobs × 15 machines
- `dmu50x20.npy` - 50 jobs × 20 machines

### Generated Validation Data (`DataGen/Vali/`)
Pre-generated datasets with 100 instances each:
- Small: 6×6, 10×10, 15×15
- Medium: 20×15, 20×20, 30×15, 30×20
- Large: 50×20, 100×20, 200×50

### Testing on Datasets
```bash
# Test on Taillard benchmarks
python3 test_learned_on_benchmark.py --which_benchmark tai

# Test on DMU benchmarks
python3 test_learned_on_benchmark.py --which_benchmark dmu

# Test on generated validation data
python3 test_learned.py
```

## Evaluation Scripts

### Available Scripts

1. **evaluate_l2d_models.py** - Comprehensive evaluation script for all datasets
   - Evaluates models on both benchmark and generated datasets
   - Presents results in table format
   - Saves results to `evaluation_results.txt`
   - Note: May encounter argparse conflicts with Params.py

2. **test_learned.py** - Test models on generated validation data
   ```bash
   python3 test_learned.py
   ```

3. **test_learned_on_benchmark.py** - Test models on benchmark datasets
   ```bash
   python3 test_learned_on_benchmark.py --which_benchmark tai
   python3 test_learned_on_benchmark.py --which_benchmark dmu
   ```

### Usage Examples

To evaluate models on specific datasets:
```bash
# Test on generated data
python3 test_learned.py

# Test on Taillard benchmarks
python3 test_learned_on_benchmark.py --which_benchmark tai

# Test on DMU benchmarks  
python3 test_learned_on_benchmark.py --which_benchmark dmu
```

### Model Performance Results

The following table shows L2D model performance on various datasets (evaluation conducted with pre-trained models):

```
==========================================================================================
L2D MODEL EVALUATION RESULTS
==========================================================================================
Dataset          Size      Inst    Mean    Std    Min    Max  Time
---------------  ------  ------  ------  -----  -----  -----  ------
Taillard_15x15   15x15       10   674.2  119.6    519    893  4.1s
Taillard_20x15   20x15       10   913.9   84.1    742   1057  5.4s
Taillard_20x20   20x20       10   979.4  111.4    783   1149  4.1s
Taillard_30x15   30x15       10  1454.1   87.9   1330   1612  5.9s
Taillard_30x20   30x20       10  1435.5  114.6   1184   1592  8.2s
DMU_20x15        20x15       10  2308.7  700.4   1295   3320  3.3s
DMU_20x20        20x20       10  2158    629.2   1251   2964  5.9s
DMU_30x15        30x15       10  3480.7  853.9   2522   4554  6.8s
DMU_30x20        30x20       10  3440.5  858.8   2401   4731  9.3s
Generated_6x6    6x6        100   231.2   71.7     71    397  5.7s
Generated_8x8    8x8        100   297.5   67.2    139    490  10.1s
Generated_10x10  10x10      100   423.3   86.3    260    661  15.8s
Generated_15x15  15x15      100   633.6  103.1    403    870  33.2s
Generated_20x15  20x15      100   851.4  103.6    658   1117  44.5s
Generated_20x20  20x20      100   876.4  130.3    508   1242  59.2s
Generated_30x15  30x15      100  1421.2  108.2   1148   1761  68.0s
Generated_30x20  30x20      100  1323    110.3   1032   1702  92.8s
```

**Metrics explained:**
- **Mean**: Average makespan (total completion time) across instances
- **Std**: Standard deviation of makespan
- **Min/Max**: Minimum and maximum makespan values
- **Time**: Total evaluation time for the dataset

## Dispatching Rules Implementation

### Available Dispatching Rules

The codebase now includes implementations of classic dispatching rules for JSSP in `dispatching_rules.py`:

1. **SPT** (Shortest Processing Time): Selects operation with minimum processing time
2. **LPT** (Longest Processing Time): Selects operation with maximum processing time  
3. **FIFO** (First In First Out): Selects operation from job with smallest ID
4. **LIFO** (Last In First Out): Selects operation from job with largest ID
5. **MWR** (Most Work Remaining): Selects from job with most total remaining work
6. **LWR** (Least Work Remaining): Selects from job with least total remaining work
7. **MOPNR** (Most Operations Remaining): Selects from job with most remaining operations
8. **LOPNR** (Least Operations Remaining): Selects from job with least remaining operations
9. **RANDOM**: Random selection from available operations
10. **MTWR** (Modified Total Work Remaining): Considers machine workload balance

### Usage

```python
from dispatching_rules import DispatchingRules, apply_dispatching_rule
from JSSP_Env import SJSSP

# Create environment
env = SJSSP(n_j=10, n_m=10)
env.reset(instance_data)

# Apply a dispatching rule
makespan = apply_dispatching_rule(env, 'SPT')
```

### Evaluation Scripts

- **test_dispatching_rules.py**: Verifies correctness of rule implementations
- **evaluate_dispatching_rules.py**: Comprehensive evaluation on all datasets
- **run_dispatching_eval.py**: Wrapper to run evaluation avoiding argparse conflicts

### Performance Comparison: L2D vs Dispatching Rules

The following table compares L2D (learned policy) performance with classic dispatching rules on generated datasets:

```
Dataset          L2D Mean  Best DR (Rule)    L2D Advantage
--------------   --------  ---------------   -------------
Generated_6x6      231.2   571.7 (MOPNR)        59.6%
Generated_8x8      297.5   756.3 (MOPNR)        60.7%
Generated_10x10    423.3   974.3 (MWR)          56.5%
Generated_15x15    633.6   1505.7 (MOPNR)      57.9%
Generated_20x15    851.4   1717.9 (MOPNR)      50.4%
Generated_20x20    876.4   1990.2 (MOPNR)      56.0%
Generated_30x15   1421.2   2250.8 (MOPNR)      36.8%
Generated_30x20   1323.0   2506.9 (MOPNR)      47.2%
```

**Key Findings:**
- L2D significantly outperforms all classic dispatching rules
- MOPNR and MWR are generally the best-performing dispatching rules
- L2D's advantage ranges from 36.8% to 60.7% improvement over the best rule
- The performance gap demonstrates the value of learned policies

### Dispatching Rules Complete Evaluation Results

Full evaluation on all 17 datasets (Generated + Taillard + DMU benchmarks):

```
========================================================================================================================
SUMMARY BY RULE (Average across all datasets)
========================================================================================================================
Rule      Avg Mean Makespan  Avg Time      Datasets    Rank
------  -------------------  ----------  ----------    ----
MWR                  2530.1  1.639s              17    1 (Best)
MOPNR                2553.3  1.619s              17    2
RANDOM               2791.8  1.674s              17    3
LPT                  2999.5  1.561s              17    4
MTWR                 3029.1  17.275s             17    5
SPT                  3075.5  1.530s              17    6
LIFO                 3144.2  1.453s              17    7
FIFO                 3145.4  1.451s              17    8
LOPNR                3145.4  1.449s              17    9
LWR                  3294.7  1.437s              17    10 (Worst)
```

#### Detailed Results by Dataset Type

**Generated Datasets:**
```
Dataset          Best Rule    Mean    2nd Best      Mean    Worst Rule   Mean
--------------   ----------   ------  ----------    ------  ----------   ------
Generated_6x6    MOPNR        567.6   MWR           572.1   LWR          749.5
Generated_8x8    MWR          777.1   MOPNR         778.0   LWR          1025.6
Generated_10x10  MWR          976.5   MOPNR         980.4   LWR          1326.0
Generated_15x15  MWR          1496.8  MOPNR         1498.8  LWR          2075.7
Generated_20x15  MOPNR        1740.4  MWR           1743.0  LWR          2415.1
Generated_20x20  MWR          1979.1  MOPNR         1986.5  LWR          2758.6
Generated_30x15  MOPNR        2229.9  MWR           2256.6  LWR          3082.0
Generated_30x20  MOPNR        2474.0  MWR           2483.6  LWR          3484.6
```

**Taillard Benchmarks:**
```
Dataset          Best Rule    Mean    2nd Best      Mean    Worst Rule   Mean
--------------   ----------   ------  ----------    ------  ----------   ------
Taillard_15x15   MOPNR        1551.2  MWR           1560.3  LWR          2106.2
Taillard_20x15   MWR          1752.4  MOPNR         1784.4  LWR          2387.1
Taillard_20x20   MOPNR        2069.7  MWR           2079.5  LWR          2805.6
Taillard_30x15   MOPNR        2312.6  MWR           2347.2  LWR          3084.9
Taillard_30x20   MWR          2615.0  MOPNR         2619.8  LIFO         3492.7
```

**DMU Benchmarks:**
```
Dataset          Best Rule    Mean    2nd Best      Mean    Worst Rule   Mean
--------------   ----------   ------  ----------    ------  ----------   ------
DMU_20x15        MWR          4152.9  MOPNR         4321.5  LWR          5256.7
DMU_20x20        MWR          4715.7  MOPNR         4719.6  LWR          5960.0
DMU_30x15        MWR          5506.1  MOPNR         5660.3  LWR          6678.8
DMU_30x20        MWR          5997.6  MOPNR         6111.0  LWR          7341.1
```

**Key Insights from Complete Evaluation:**
- MWR (Most Work Remaining) and MOPNR (Most Operations Remaining) consistently dominate
- MWR performs slightly better overall (2530.1 vs 2553.3 average)
- LWR (Least Work Remaining) is consistently the worst performer
- DMU instances are significantly harder than Taillard instances (2-3x higher makespans)
- MTWR has high computational cost (17.3s avg) compared to others (~1.5s avg)

## Dependencies

- PyTorch 1.6.0 with CUDA 10.2
- OpenAI Gym 0.17.3
- Python 3.7.9
- NVIDIA CUDA toolkit (for GPU acceleration)