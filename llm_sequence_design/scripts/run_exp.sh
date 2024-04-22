export HF_HOME="/lfs/local/0/sttruong/env/.huggingface"
export HF_DATASETS_CACHE="/lfs/local/0/sttruong/env/.huggingface/datasets"
export TRITON_CACHE_DIR="/lfs/local/0/sttruong/.triton_1"

export LIBRARY_PATH=/dfs/user/sttruong/miniconda3/envs/train_llm/lib/python3.10/site-packages/torch/lib:/dfs/user/sttruong/miniconda3/envs/train_llm/lib:$LIBRARY_PATH
export LD_LIBRARY_PATH=/dfs/user/sttruong/miniconda3/envs/train_llm/lib/python3.10/site-packages/torch/lib:/dfs/user/sttruong/miniconda3/envs/train_llm/lib:$LD_LIBRARY_PATH


accelerate launch main.py \
    --oracle_model_name_or_path facebook/esm2_t6_8M_UR50D \
    --wm_model_name_or_path facebook/esm2_t6_8M_UR50D \
    --policy_model_name_or_path meta-ai/Llama-7B-hf-chat \
    --template llama2 \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 30.0 \
    --output_dir ckpts/policy