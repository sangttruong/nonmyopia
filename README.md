# Non-myopic H-Entropy Search

This repo support Bayesian optimization experiments with non-myopic H-Entropy Search. There are two main experiments:
1. Synthetic experiments: We consider the synthetic environment with the following settings:
    - 2D environment: Ackley, Alpine, Beale, Branin, EggHolder, Griewank, HolderTable, Levy, SixHumpCamel,  StyblinskiTang, and SynGP
    - 4D environment: Powell
    - 6D environment: Hartmann
    - 8D environment: Cosine8
2. Real-world experiments: We consider the real-world environment with protein sequence optimization.

## How to reproduce
1. Install the requirements
```bash
 pip install -r requirements.txt
 ```
2. Run the experiments by bash script `scripts.sh`
```bash
python _0_main.py [-h] [--seeds SEEDS [SEEDS ...]] [--task TASK] [--env_names ENV_NAMES [ENV_NAMES ...]] [--env_noise ENV_NOISE] [--env_discretized] [--algos ALGOS [ALGOS ...]]
                  [--algo_ts] [--algo_n_iterations ALGO_N_ITERATIONS] [--algo_lookahead_steps ALGO_LOOKAHEAD_STEPS] [--cost_spotlight_k COST_SPOTLIGHT_K] [--cost_p_norm COST_P_NORM]
                  [--cost_max_noise COST_MAX_NOISE] [--cost_discount COST_DISCOUNT] [--cost_discount_threshold COST_DISCOUNT_THRESHOLD] [--gpu_id GPU_ID [GPU_ID ...]]
                  [--continue_once CONTINUE_ONCE] [--test_only]

options:
  -h, --help            show this help message and exit
  --seeds SEEDS [SEEDS ...]
  --task TASK
  --env_names ENV_NAMES [ENV_NAMES ...]
  --env_noise ENV_NOISE
  --env_discretized
  --algos ALGOS [ALGOS ...]
  --algo_ts
  --algo_n_iterations ALGO_N_ITERATIONS
  --algo_lookahead_steps ALGO_LOOKAHEAD_STEPS
  --cost_spotlight_k COST_SPOTLIGHT_K
  --cost_p_norm COST_P_NORM
  --cost_max_noise COST_MAX_NOISE
  --cost_discount COST_DISCOUNT
  --cost_discount_threshold COST_DISCOUNT_THRESHOLD
  --gpu_id GPU_ID [GPU_ID ...]
  --continue_once CONTINUE_ONCE
  --test_only
```
3. Plot the results by command
```bash
python draw_regrets.py [ENV_NAMES]
```

## Analyzing world models
```bash
python _0_main_gp.py [-h] [--seeds SEEDS [SEEDS ...]] [--env_names ENV_NAMES [ENV_NAMES ...]] [--env_noise ENV_NOISE] [--env_discretized] [--gpu_id GPU_ID]

options:
  -h, --help            show this help message and exit
  --seeds SEEDS [SEEDS ...]
  --env_names ENV_NAMES [ENV_NAMES ...]
  --env_noise ENV_NOISE
  --env_discretized
  --gpu_id GPU_ID
```