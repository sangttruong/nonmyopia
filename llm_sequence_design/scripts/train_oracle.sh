# Extract dataset embedding 
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    extract_emb_dataset.py \
    --oracle_model_name_or_path "" \
    --wm_model_name_or_path facebook/esm2_t33_650M_UR50D \
    --policy_model_name_or_path "" \
    --template default \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --wm_export_hub_model_id ura-hcmut/proteinea_fluorescence-esm2_t33_650M_UR50D-embedding \
    --wm_hf_hub_token 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 \
    --output_dir ckpts/embedding
    
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29501 \
    --config_file examples/accelerate/single_config.yaml \
    extract_emb_dataset.py \
    --oracle_model_name_or_path "" \
    --wm_model_name_or_path facebook/esm2_t36_3B_UR50D \
    --policy_model_name_or_path "" \
    --template default \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --wm_export_hub_model_id ura-hcmut/proteinea_fluorescence-esm2_t36_3B_UR50D-embedding \
    --wm_hf_hub_token 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 \
    --output_dir ckpts/embedding

CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29502 \
    --config_file examples/accelerate/single_config.yaml \
    extract_emb_dataset.py \
    --oracle_model_name_or_path "" \
    --wm_model_name_or_path meta-llama/Llama-2-7b-hf \
    --policy_model_name_or_path "" \
    --template llama2 \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --wm_export_hub_model_id ura-hcmut/proteinea_fluorescence-Llama-2-7b-hf-embedding \
    --wm_hf_hub_token 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 \
    --output_dir ckpts/embedding

CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port 29503 \
    --config_file examples/accelerate/single_config.yaml \
    extract_emb_dataset.py \
    --oracle_model_name_or_path "" \
    --wm_model_name_or_path meta-llama/Meta-Llama-3-8B \
    --policy_model_name_or_path "" \
    --template llama2 \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --wm_export_hub_model_id ura-hcmut/proteinea_fluorescence-Meta-Llama-3-8B-embedding \
    --wm_hf_hub_token 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 \
    --output_dir ckpts/embedding

CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port 29504 \
    --config_file examples/accelerate/single_config.yaml \
    extract_emb_dataset.py \
    --oracle_model_name_or_path "" \
    --wm_model_name_or_path mistralai/Mistral-7B-v0.1 \
    --policy_model_name_or_path "" \
    --template llama2 \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --wm_export_hub_model_id ura-hcmut/proteinea_fluorescence-Mistral-7B-v0.1-embedding \
    --wm_hf_hub_token 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 \
    --output_dir ckpts/embedding

CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port 29505 \
    --config_file examples/accelerate/single_config.yaml \
    extract_emb_dataset.py \
    --oracle_model_name_or_path "" \
    --wm_model_name_or_path google/gemma-7b \
    --policy_model_name_or_path "" \
    --template default \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --wm_export_hub_model_id ura-hcmut/proteinea_fluorescence-gemma-7b-embedding \
    --wm_hf_hub_token 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 \
    --output_dir ckpts/embedding
    
# Training with embedded dataset
accelerate launch --main_process_port 29505 src/train_bash.py \
    --config_file examples/accelerate/single_config.yaml \
    --stage oracle \
    --do_train \
    --template default \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn False \
    --dataset ura-hcmut/proteinea_fluorescence-Llama-2-7b-hf-embedding \
    --emb_enabled True \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10.0 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 8 \
    --learning_rate 5e-05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_llama_emb \
    --save_total_limit 5 \
    --report_to none \
    --plot_loss True
    
# Normal training with raw dataset
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
    --bf16 True \
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

accelerate launch --main_process_port 29501 src/train_bash.py \
    --stage oracle \
    --do_train \
    --template default \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn False \
    --dataset proteinea/fluorescence \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10.0 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 4 \
    --gradient_accumulation_steps 32 \
    --learning_rate 5e-05 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_llama \
    --save_total_limit 5 \
    --report_to neptune \
    --plot_loss True