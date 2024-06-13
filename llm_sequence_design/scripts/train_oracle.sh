# Extract dataset embedding 
CUDA_VISIBLE_DEVICES=0 accelerate launch \
    --config_file examples/accelerate/single_config.yaml \
    extract_emb_dataset.py \
    --model_name_or_path facebook/esm2_t33_650M_UR50D \
    --template default \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --export_hub_model_id ura-hcmut/proteinea_fluorescence-esm2_t33_650M_UR50D-embedding \
    --hf_hub_token <hf_token> \
    --output_dir ckpts/embedding
    
CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29501 \
    --config_file examples/accelerate/single_config.yaml \
    extract_emb_dataset.py \
    --model_name_or_path facebook/esm2_t36_3B_UR50D \
    --template default \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --export_hub_model_id ura-hcmut/proteinea_fluorescence-esm2_t36_3B_UR50D-embedding \
    --hf_hub_token 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 \
    --output_dir ckpts/embedding

CUDA_VISIBLE_DEVICES=0 accelerate launch --main_process_port 29502 \
    --config_file examples/accelerate/single_config.yaml \
    extract_emb_dataset.py \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --template llama2 \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --export_hub_model_id ura-hcmut/proteinea_fluorescence-Llama-2-7b-hf-embedding \
    --hf_hub_token <hf_token> \
    --output_dir ckpts/embedding

CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port 29503 \
    --config_file examples/accelerate/single_config.yaml \
    extract_emb_dataset.py \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --template llama2 \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --export_hub_model_id ura-hcmut/proteinea_fluorescence-Meta-Llama-3-8B-embedding \
    --hf_hub_token <hf_token> \
    --output_dir ckpts/embedding

CUDA_VISIBLE_DEVICES=1 accelerate launch --main_process_port 29504 \
    --config_file examples/accelerate/single_config.yaml \
    extract_emb_dataset.py \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --template llama2 \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --export_hub_model_id ura-hcmut/proteinea_fluorescence-Mistral-7B-v0.1-embedding \
    --hf_hub_token <hf_token> \
    --output_dir ckpts/embedding

CUDA_VISIBLE_DEVICES=3 accelerate launch --main_process_port 29505 \
    --config_file examples/accelerate/single_config.yaml \
    extract_emb_dataset.py \
    --model_name_or_path google/gemma-7b \
    --template default \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --export_hub_model_id ura-hcmut/proteinea_fluorescence-gemma-7b-embedding \
    --hf_hub_token <hf_token> \
    --output_dir ckpts/embedding


CUDA_VISIBLE_DEVICES=7 accelerate launch --main_process_port 29505 \
    --config_file examples/accelerate/single_config.yaml \
    extract_emb_dataset.py \
    --model_name_or_path zjunlp/llama-molinst-protein-7b \
    --template default \
    --dataset proteinea/fluorescence \
    --overwrite_cache False \
    --preprocessing_num_workers 8 \
    --num_train_epochs 0 \
    --export_hub_model_id ura-hcmut/proteinea_fluorescence-llama-molinst-protein-7b-embedding \
    --hf_hub_token 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 \
    --output_dir ckpts/embedding
    

# Train sklearn model
python src/train_bash.py \
    --seed 2 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path bayesridge \
    --dataset ura-hcmut/proteinea_fluorescence-gemma-7b-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.0 \
    --preprocessing_num_workers 32 \
    --output_dir ckpts/oracle_bayesridge-seed2

    
# Training with embedded dataset
CUDA_VISIBLE_DEVICES=9 python src/train_bash.py \
    --seed 2 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path facebook/esm2_t33_650M_UR50D \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-esm2_t33_650M_UR50D-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_esm2_t33_650M_UR50D-seed2 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=9 python src/train_bash.py \
    --seed 3 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path facebook/esm2_t33_650M_UR50D \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-esm2_t33_650M_UR50D-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_esm2_t33_650M_UR50D-seed3 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=9 python src/train_bash.py \
    --seed 5 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path facebook/esm2_t33_650M_UR50D \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-esm2_t33_650M_UR50D-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_esm2_t33_650M_UR50D-seed5 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=9 python src/train_bash.py \
    --seed 7 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path facebook/esm2_t33_650M_UR50D \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-esm2_t33_650M_UR50D-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_esm2_t33_650M_UR50D-seed7 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &


CUDA_VISIBLE_DEVICES=9 python src/train_bash.py \
    --seed 11 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path facebook/esm2_t33_650M_UR50D \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-esm2_t33_650M_UR50D-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_esm2_t33_650M_UR50D-seed11 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True





    
CUDA_VISIBLE_DEVICES=9 python src/train_bash.py \
    --seed 2 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path facebook/esm2_t36_3B_UR50D \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-esm2_t36_3B_UR50D-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_esm2_t36_3B_UR50D-seed2 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=9 python src/train_bash.py \
    --seed 3 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path facebook/esm2_t36_3B_UR50D \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-esm2_t36_3B_UR50D-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_esm2_t36_3B_UR50D-seed3 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=9 python src/train_bash.py \
    --seed 5 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path facebook/esm2_t36_3B_UR50D \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-esm2_t36_3B_UR50D-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_esm2_t36_3B_UR50D-seed5 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=9 python src/train_bash.py \
    --seed 7 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path facebook/esm2_t36_3B_UR50D \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-esm2_t36_3B_UR50D-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_esm2_t36_3B_UR50D-seed7 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &


CUDA_VISIBLE_DEVICES=9 python src/train_bash.py \
    --seed 11 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path facebook/esm2_t36_3B_UR50D \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-esm2_t36_3B_UR50D-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_esm2_t36_3B_UR50D-seed11 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True





    
CUDA_VISIBLE_DEVICES=8 python src/train_bash.py \
    --seed 2 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Llama-2-7b-hf-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Llama-2-7b-hf-seed2 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=8 python src/train_bash.py \
    --seed 3 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Llama-2-7b-hf-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Llama-2-7b-hf-seed3 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=8 python src/train_bash.py \
    --seed 5 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Llama-2-7b-hf-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Llama-2-7b-hf-seed5 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=8 python src/train_bash.py \
    --seed 7 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Llama-2-7b-hf-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Llama-2-7b-hf-seed7 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &


CUDA_VISIBLE_DEVICES=8 python src/train_bash.py \
    --seed 11 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path meta-llama/Llama-2-7b-hf \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Llama-2-7b-hf-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Llama-2-7b-hf-seed11 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &





    
CUDA_VISIBLE_DEVICES=8 python src/train_bash.py \
    --seed 2 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Meta-Llama-3-8B-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Meta-Llama-3-8B-seed2 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=8 python src/train_bash.py \
    --seed 3 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Meta-Llama-3-8B-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Meta-Llama-3-8B-seed3 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=8 python src/train_bash.py \
    --seed 5 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Meta-Llama-3-8B-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Meta-Llama-3-8B-seed5 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=8 python src/train_bash.py \
    --seed 7 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Meta-Llama-3-8B-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Meta-Llama-3-8B-seed7 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &


CUDA_VISIBLE_DEVICES=8 python src/train_bash.py \
    --seed 11 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path meta-llama/Meta-Llama-3-8B \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Meta-Llama-3-8B-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Meta-Llama-3-8B-seed11 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True





    
CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 2 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Mistral-7B-v0.1-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Mistral-7B-v0.1-seed2 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 3 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Mistral-7B-v0.1-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Mistral-7B-v0.1-seed3 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 5 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Mistral-7B-v0.1-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Mistral-7B-v0.1-seed5 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 7 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Mistral-7B-v0.1-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Mistral-7B-v0.1-seed7 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &


CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 11 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path mistralai/Mistral-7B-v0.1 \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-Mistral-7B-v0.1-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_Mistral-7B-v0.1-seed11 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &







CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 2 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path google/gemma-7b \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-gemma-7b-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 5 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_gemma-7b-seed2 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 3 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path google/gemma-7b \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-gemma-7b-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 5 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_gemma-7b-seed3 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 5 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path google/gemma-7b \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-gemma-7b-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 5 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_gemma-7b-seed5 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 7 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path google/gemma-7b \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-gemma-7b-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 5 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_gemma-7b-seed7 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &


CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 11 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path google/gemma-7b \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-gemma-7b-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 5 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_gemma-7b-seed11 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True










CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 2 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path zjunlp/llama-molinst-protein-7b \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-llama-molinst-protein-7b-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_llama-molinst-protein-7b-seed2 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 3 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path zjunlp/llama-molinst-protein-7b \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-llama-molinst-protein-7b-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_llama-molinst-protein-7b-seed3 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 5 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path zjunlp/llama-molinst-protein-7b \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-llama-molinst-protein-7b-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_llama-molinst-protein-7b-seed5 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &

CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 7 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path zjunlp/llama-molinst-protein-7b \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-llama-molinst-protein-7b-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_llama-molinst-protein-7b-seed7 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True &


CUDA_VISIBLE_DEVICES=7 python src/train_bash.py \
    --seed 11 \
    --stage oracle \
    --do_train \
    --do_eval \
    --template default \
    --model_name_or_path zjunlp/llama-molinst-protein-7b \
    --use_fast_tokenizer True \
    --finetuning_type freeze \
    --flash_attn True \
    --dataset ura-hcmut/proteinea_fluorescence-llama-molinst-protein-7b-embedding \
    --emb_enabled True \
    --label_names rewards \
    --val_size 0.1 \
    --preprocessing_num_workers 32 \
    --num_train_epochs 10 \
    --bf16 True \
    --tf32 False \
    --per_device_train_batch_size 32 \
    --gradient_accumulation_steps 1 \
    --learning_rate 1e-4 \
    --lr_scheduler_type cosine \
    --max_grad_norm 1.0 \
    --logging_steps 1 \
    --warmup_ratio 0.01 \
    --save_steps 1000 \
    --output_dir ckpts/oracle_llama-molinst-protein-7b-seed11 \
    --save_total_limit 5 \
    --report_to wandb \
    --overwrite_output_dir True \
    --plot_loss True
    
    
# Normal training with raw dataset
# accelerate launch --main_process_port 29505 src/train_bash.py \
#     --stage oracle \
#     --do_train \
#     --template default \
#     --model_name_or_path facebook/esm2_t36_3B_UR50D \
#     --use_fast_tokenizer True \
#     --finetuning_type freeze \
#     --flash_attn True \
#     --dataset proteinea/fluorescence \
#     --preprocessing_num_workers 32 \
#     --num_train_epochs 10.0 \
#     --bf16 True \
#     --tf32 False \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 8 \
#     --learning_rate 1e-4 \
#     --lr_scheduler_type cosine \
#     --max_grad_norm 1.0 \
#     --logging_steps 1 \
#     --warmup_ratio 0.01 \
#     --save_steps 1000 \
#     --output_dir ckpts/oracle2_test \
#     --save_total_limit 5 \
#     --report_to none \
#     --plot_loss True

# accelerate launch --main_process_port 29501 src/train_bash.py \
#     --stage oracle \
#     --do_train \
#     --template default \
#     --model_name_or_path meta-llama/Llama-2-7b-hf \
#     --use_fast_tokenizer True \
#     --finetuning_type freeze \
#     --flash_attn True \
#     --dataset proteinea/fluorescence \
#     --preprocessing_num_workers 32 \
#     --num_train_epochs 10.0 \
#     --bf16 True \
#     --tf32 False \
#     --per_device_train_batch_size 4 \
#     --gradient_accumulation_steps 32 \
#     --learning_rate 1e-4 \
#     --lr_scheduler_type cosine \
#     --max_grad_norm 1.0 \
#     --logging_steps 1 \
#     --warmup_ratio 0.01 \
#     --save_steps 1000 \
#     --output_dir ckpts/oracle_llama \
#     --save_total_limit 5 \
#     --report_to neptune \
#     --plot_loss True