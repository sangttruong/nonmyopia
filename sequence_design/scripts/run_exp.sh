export PATH=/usr/local/cuda-12.1/bin:/dfs/user/sttruong/miniconda3/envs/train_llm/bin:$PATH
export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
export TRITON_CACHE_DIR="/lfs/local/0/sttruong/.triton_1"

export LIBRARY_PATH=/dfs/user/sttruong/miniconda3/envs/train_llm/lib/python3.10/site-packages/torch/lib:/dfs/user/sttruong/miniconda3/envs/train_llm/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/dfs/user/sttruong/miniconda3/envs/train_llm/lib/python3.10/site-packages/torch/lib:/dfs/user/sttruong/miniconda3/envs/train_llm/lib:$LD_LIBRARY_PATH
    
CUDA_VISIBLE_DEVICES=4 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    main.py ...

# Sample command
CUDA_VISIBLE_DEVICES=1 python main.py \
    --seed 0 \
    --oracle_model_name_or_path google/gemma-7b \
    --oracle_linear_head_path ckpts/oracle_bayesridge-seed2 \
    --surr_model_name_or_path google/gemma-7b \
    --policy_model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct \
    --policy_adapter_name_or_path ckpts/policy \
    --num_train_epochs 1 \
    --do_train \
    --template llama3 \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 8 \
    --gradient_accumulation_steps 4 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --output_dir ckpts \
    --overwrite_output_dir True \
    --report_to none \
    --temperature 0.6 \
    --do_sample True \
    --top_k 10 \
    --top_p 1.0 \
    --max_new_tokens 512 \
    --algo SR \
    --algo_n_iterations 20 \
    --algo_lookahead_steps 20 \
    --initinal_sequences 10 \
    --n_sequences 10 \
    --n_restarts 1 \
    --rollout_sequences 15
