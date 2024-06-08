# LLM Sequence Design

## Description
This source code is used to deisgn sequence(s) to maximize/minimize a property. It includes three main models:
- **Oracle** is used as groundtruth to replace wet-lab experiments
- **WorldModel** is the reward model trained with current observed data
- **Policy** is the amortized network with ability to generate better sequence(s)

## Installization
First of all, you might need a suitable envorinment to run the code.
Pease refre to pakagess in [LLaLa-Factory](https://github.com/hiyouga/LLaMA-Factory).

You can also refer the file [here](llm_sequence_design/requirements.txt).
```bash
pip install -r requirements.txt
```

## Building Oracle
### Training
This is a two-step process.
1. Data preprocessing and embedding. Example with ESM2 model and Proteina Fluorescence dataset.
```bash
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
  
To run full pipeline, please use script in [Pipeline Script](lllm_sequence_design/scripts/run_exp.sh)

## Next steps
- Re-adding the histories of sequence when optimizing with lookahead. See `configs.py` for designed prompt.
- Verify the correctness of modified PPO pipeline.
- Implement more acquisition functions.