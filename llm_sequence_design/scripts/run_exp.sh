accelerate launch main.py \
    --oracle_model_name_or_path facebook/esm2_t6_8M_UR50D \
    --wm_model_name_or_path facebook/esm2_t6_8M_UR50D \
    --policy_model_name_or_path meta-ai/Llama-7B-hf-chat \
    --template llama \
    --dataset proteinea/fluorescence \
    --output_dir ckpts/policy