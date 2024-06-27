# Non-myopic H-Entropy Search

This repo support Bayesian optimization experiments with non-myopic H-Entropy Search. Bayesian optimization is a widely used approach for making optimal decisions in uncertain scenarios by acquiring information through costly experiments. Many real-world applications can be cast as instances of this problem, ranging from designing biological sequences to conducting ground surveys. In these contexts, the cost associated with each experiment can be dynamic and non-uniform. For instance, in cases where each experiment corresponds to a location, there exists a variable travel cost contingent on the distances between successive experiments. Conventional Bayesian optimization techniques, often reliant on myopic acquisition functions and assuming a fixed cost structure, yield suboptimal results in dynamic cost environments. To address these limitations, we introduce a scalable nonmyopic acquisition function grounded in a decision-theoretic extension of mutual information. Our empirical evaluations demonstrate that our method outperforms numerous baseline approaches across a range of global optimization tasks.

There are two main experiments:
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
2. Run the experiments by bash script
```bash
python main.py [-h] [--seed SEED] [--task TASK] [--env_name ENV_NAME] [--env_noise ENV_NOISE] [--env_discretized ENV_DISCRETIZED] [--algo ALGO]
               [--cost_fn COST_FN] [--plot PLOT] [--gpu_id GPU_ID] [--cont CONT]

options:
  --seed SEED
  --task TASK
  --env_name ENV_NAME
  --env_noise ENV_NOISE
  --env_discretized ENV_DISCRETIZED
  --algo ALGO
  --cost_fn COST_FN
  --plot PLOT
  --gpu_id GPU_ID
  --cont CONT
```
3. Compute metrics
```bash
python compute_metrics.py [-h] [--seed SEED] [--task TASK] [--env_name ENV_NAME] [--env_noise ENV_NOISE] [--env_discretized ENV_DISCRETIZED]
                          [--algo ALGO] [--cost_fn COST_FN] [--plot PLOT] [--gpu_id GPU_ID] [--cont CONT]

options:
  --seed SEED
  --task TASK
  --env_name ENV_NAME
  --env_noise ENV_NOISE
  --env_discretized ENV_DISCRETIZED
  --algo ALGO
  --cost_fn COST_FN
  --plot PLOT
  --gpu_id GPU_ID
  --cont CONT
```
4. Draw regrets
```bash
python draw_metrics.py
```

## Running mass experiments with WandB Sweep
1. Firstly, run below command to get the command to start sweep agent(s). 
```bash
wandb sweep wnb_configs/full.yaml
```
The result will look like "wandb agent your_name/nonmyopia/some_text".

2. Start a single sweep agent.
```bash
CUDA_VISIBLE_DEVICES=0 wandb agent your_name/nonmyopia/some_text &
```
If you want to start more agents, simply rerun above command of different terminals/servers/... You can start as many sweep agents as your server can handle.

## Analyzing world models
```bash
python test_surrogate_convergence.py [-h] [--seeds SEEDS [SEEDS ...]] [--env_names ENV_NAMES [ENV_NAMES ...]] [--env_noise ENV_NOISE]
                                     [--env_discretized] [--gpu_id GPU_ID]

options:
  --seeds SEEDS [SEEDS ...]
  --env_names ENV_NAMES [ENV_NAMES ...]
  --env_noise ENV_NOISE
  --env_discretized
  --gpu_id GPU_ID
```

## Running the real-world experiments
1. Train the oracle model
```bash
accelerate launch --main_process_port 29505 src/train_bash.py \
    --stage oracle \
    --do_train \
    --template default \
    --model_name_or_path facebook/esm2_t36_3B_UR50D \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn False \
    --dataset proteinea/fluorescence \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10.0 \
    --bf16 False \
    --tf32 False \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle2_test \
    --save_total_limit 5 \
    --report_to none \
    --plot_loss True
```
