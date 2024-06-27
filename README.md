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

This source code is used to deisgn sequence(s) to maximize/minimize a property. It includes three main models:
- **Oracle** is used as groundtruth to replace wet-lab experiments
- **WorldModel** is the reward model trained with current observed data
- **Policy** is the amortized network with ability to generate better sequence(s)

## Building Oracle
### Training
This is a two-step process.
1. Data preprocessing and embedding. Example with ESM2 model and Proteina Fluorescence dataset.
```bash
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file single_config.yaml \
    extract_emb_dataset.py \
    --oracle_model_name_or_path "" \
    --wm_model_name_or_path facebook/esm2_t33_650M_UR50D \
    --policy_model_name_or_path "" \
    --template default \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --wm_export_hub_model_id <saving_hf_repo> \
    --wm_hf_hub_token <hf_token> \
    --output_dir ckpts/embedding
```
2. In this step, we simply train linear models using sklearn. Currently, three models are suppported: linear, ridge, bayesridge
```bash
python src/train_bash.py \
    --seed 2 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path bayesridge \
    --dataset <saving_hf_repo> \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.0 \
    --preprocessing_num_workers 32 \
    --output_dir ckpts/oracle_bayesridge-seed2
```

### Embedding model selection
To selecct the most suited embedding model for next steps, we tested various well-known LLMs. To do this, firstly, we embedd our dataset by various models, then run below script. Note: This script is designed to work with bayesridge and upto 7 embedding models, feel free to edit it by `hf_embedding_names` variable. 
```bash
python test_oracle.py
```

## Runing full pipeline code
Currently, this code is only support HES-TS-AM acquision function. Some notes are: 
- The world model should has the same embedding model with oracle.
- Policy model can be different models with the above two models.
- We would better finetuning Policy with LoRA to minimzing the catastrophic knowledge loss.
  
To run full pipeline, please use script in [Pipeline Script](scripts/run_exp.sh)

## Next steps
- Re-adding the histories of sequence when optimizing with lookahead. See `configs.py` for designed prompt.
- Verify the correctness of modified PPO pipeline.
- Implement more acquisition functions.
