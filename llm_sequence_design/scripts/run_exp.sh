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