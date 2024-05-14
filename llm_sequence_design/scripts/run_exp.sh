export PATH=/usr/local/cuda-12.1/bin:/dfs/user/sttruong/miniconda3/envs/train_llm/bin:$PATH
export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
export TRITON_CACHE_DIR="/lfs/local/0/sttruong/.triton_1"

export LIBRARY_PATH=/dfs/user/sttruong/miniconda3/envs/train_llm/lib/python3.10/site-packages/torch/lib:/dfs/user/sttruong/miniconda3/envs/train_llm/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/dfs/user/sttruong/miniconda3/envs/train_llm/lib/python3.10/site-packages/torch/lib:/dfs/user/sttruong/miniconda3/envs/train_llm/lib:$LD_LIBRARY_PATH
    
CUDA_VISIBLE_DEVICES=4 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    main.py ...


CUDA_VISIBLE_DEVICES=1 python main.py \
    --oracle_model_name_or_path google/gemma-7b \
    --oracle_linear_head_path ckpts/oracle_bayesridge-seed2 \
    --wm_model_name_or_path google/gemma-7b \
    --policy_model_name_or_path meta-llama/Llama-2-7b-chat-hf \
    --policy_adapter_name_or_path ckpts/policy \
    --policy_finetuning_type lora \
    --policy_lora_alpha 16 \
    --policy_lora_rank 8 \
    --policy_lora_dropout 0.1 \
    --policy_lora_target q_proj,k_proj,v_proj,o_proj \
    --policy_vllm_gpu_util 0.5 \
    --num_train_epochs 1 \
    --do_train \
    --template llama2 \
    --dataset proteinea/fluorescence \
    --max_new_tokens 256 \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --output_dir ckpts \
    --overwrite_output_dir True \
    --report_to none \
    --algo HES-TS-AM \
    --algo_n_iterations 20 \
    --algo_lookahead_steps 20 \
    --initinal_sequences 1000 \
    --n_sequences 10 \
    --n_restarts 4 \
    --rollout_sequences 100
    
    # --algo_n_iterations 20 \
    # --algo_lookahead_steps 2 \
    # --initinal_sequences 10 \
    # --n_sequences 5 \
    # --n_restarts 3 \
    # --rollout_sequences 16