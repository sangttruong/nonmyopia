import random
import torch
from tqdm import tqdm
from functools import partial
from torch.utils.data import DataLoader
from datasets import Dataset


def get_dataset_embedding(dataset, model, tokenizer, data_args):
    def tokenize_dataset(
        examples,
        tokenizer,
        data_args
    ):
        # build inputs with format `<bos> X1 Y1 <eos> <bos> X2 Y2 <eos>`
        # and labels with format `<ignore> ... <ignore> Y1 <eos> <ignore> ... <ignore> Y2 <eos>`
        model_inputs = {
            "input_ids": [],
            "attention_mask": [],
            "rewards": []
        }

        for i in range(len(examples["text"])):
            input_ids = tokenizer.encode(
                examples["text"][i], add_special_tokens=False,
                # padding='max_length', truncation=True,
                # max_length=data_args.cutoff_len
            )
            attention_mask = [1] * len(input_ids)
            labels = examples["reward"][i]

            model_inputs["input_ids"].append(input_ids)
            model_inputs["attention_mask"].append(attention_mask)
            model_inputs["rewards"].append(labels)

        return model_inputs

    preprocess_func = partial(
        tokenize_dataset, tokenizer=tokenizer, data_args=data_args
    )
    kwargs = dict(
        num_proc=data_args.preprocessing_num_workers,
        load_from_cache_file=(not data_args.overwrite_cache),
        desc="Running tokenizer on dataset",
    )
    column_names = list(next(iter(dataset)).keys())
    tokenized_dataset = dataset.map(preprocess_func, batched=True,
                                    remove_columns=column_names, **kwargs)
    
    model_inputs = {
        "inputs_embeds": [],
        "rewards": []
    }
    
    for example in tqdm(tokenized_dataset):
        embeds = model.pretrained_model.model(
            input_ids=torch.tensor([example['input_ids']], device=model.pretrained_model.device).long(),
            attention_mask=torch.tensor([example['attention_mask']], device=model.pretrained_model.device).long()
        )
        model_inputs['inputs_embeds'].append(
            embeds.last_hidden_state[0][-1].detach().cpu().tolist())
        model_inputs['rewards'].append(example['rewards'])

    embeded_dataset = Dataset.from_dict(model_inputs)
    return embeded_dataset


def random_sampling(dataset, num_samples, *args, **kwargs):
    total_samples = len(dataset)
    indices = random.sample(range(total_samples), num_samples)
    return dataset.select(indices)


def fix_oracle_model_args(model_args):
    model_args.compute_dtype = model_args.oracle_compute_dtype
    model_args.device_map = model_args.oracle_device_map
    model_args.model_max_length = model_args.oracle_model_max_length

    model_args.model_name_or_path = model_args.oracle_model_name_or_path
    model_args.adapter_name_or_path = model_args.oracle_adapter_name_or_path
    model_args.cache_dir = model_args.oracle_cache_dir
    model_args.use_fast_tokenizer = model_args.oracle_use_fast_tokenizer
    model_args.resize_vocab = model_args.oracle_resize_vocab
    model_args.split_special_tokens = model_args.oracle_split_special_tokens
    model_args.model_revision = model_args.oracle_model_revision
    model_args.low_cpu_mem_usage = model_args.oracle_low_cpu_mem_usage
    model_args.quantization_bit = model_args.oracle_quantization_bit
    model_args.quantization_type = model_args.oracle_quantization_type
    model_args.double_quantization = model_args.oracle_double_quantization
    model_args.quantization_device_map = model_args.oracle_quantization_device_map
    model_args.rope_scaling = model_args.oracle_rope_scaling
    model_args.flash_attn = model_args.oracle_flash_attn
    model_args.shift_attn = model_args.oracle_shift_attn
    model_args.use_unsloth = model_args.oracle_use_unsloth
    model_args.moe_aux_loss_coef = model_args.oracle_moe_aux_loss_coef
    model_args.disable_gradient_checkpointing = model_args.oracle_disable_gradient_checkpointing
    model_args.upcast_layernorm = model_args.oracle_upcast_layernorm
    model_args.upcast_lmhead_output = model_args.oracle_upcast_lmhead_output
    model_args.infer_backend = model_args.oracle_infer_backend
    model_args.vllm_maxlen = model_args.oracle_vllm_maxlen
    model_args.vllm_gpu_util = model_args.oracle_vllm_gpu_util
    model_args.vllm_enforce_eager = model_args.oracle_vllm_enforce_eager
    model_args.offload_folder = model_args.oracle_offload_folder
    model_args.use_cache = model_args.oracle_use_cache
    model_args.hf_hub_token = model_args.oracle_hf_hub_token
    model_args.ms_hub_token = model_args.oracle_ms_hub_token
    model_args.export_dir = model_args.oracle_export_dir
    model_args.export_size = model_args.oracle_export_size
    model_args.export_device = model_args.oracle_export_device
    model_args.export_quantization_bit = model_args.oracle_export_quantization_bit
    model_args.export_quantization_dataset = model_args.oracle_export_quantization_dataset
    model_args.export_quantization_nsamples = model_args.oracle_export_quantization_nsamples
    model_args.export_quantization_maxlen = model_args.oracle_export_quantization_maxlen
    model_args.export_legacy_format = model_args.oracle_export_legacy_format
    model_args.export_hub_model_id = model_args.oracle_export_hub_model_id
    model_args.print_param_status = model_args.oracle_print_param_status


def fix_wm_model_args(model_args):
    model_args.compute_dtype = model_args.wm_compute_dtype
    model_args.device_map = model_args.wm_device_map
    model_args.model_max_length = model_args.wm_model_max_length

    model_args.model_name_or_path = model_args.wm_model_name_or_path
    model_args.adapter_name_or_path = model_args.wm_adapter_name_or_path
    model_args.cache_dir = model_args.wm_cache_dir
    model_args.use_fast_tokenizer = model_args.wm_use_fast_tokenizer
    model_args.resize_vocab = model_args.wm_resize_vocab
    model_args.split_special_tokens = model_args.wm_split_special_tokens
    model_args.model_revision = model_args.wm_model_revision
    model_args.low_cpu_mem_usage = model_args.wm_low_cpu_mem_usage
    model_args.quantization_bit = model_args.wm_quantization_bit
    model_args.quantization_type = model_args.wm_quantization_type
    model_args.double_quantization = model_args.wm_double_quantization
    model_args.quantization_device_map = model_args.wm_quantization_device_map
    model_args.rope_scaling = model_args.wm_rope_scaling
    model_args.flash_attn = model_args.wm_flash_attn
    model_args.shift_attn = model_args.wm_shift_attn
    model_args.use_unsloth = model_args.wm_use_unsloth
    model_args.moe_aux_loss_coef = model_args.wm_moe_aux_loss_coef
    model_args.disable_gradient_checkpointing = model_args.wm_disable_gradient_checkpointing
    model_args.upcast_layernorm = model_args.wm_upcast_layernorm
    model_args.upcast_lmhead_output = model_args.wm_upcast_lmhead_output
    model_args.infer_backend = model_args.wm_infer_backend
    model_args.vllm_maxlen = model_args.wm_vllm_maxlen
    model_args.vllm_gpu_util = model_args.wm_vllm_gpu_util
    model_args.vllm_enforce_eager = model_args.wm_vllm_enforce_eager
    model_args.offload_folder = model_args.wm_offload_folder
    model_args.use_cache = model_args.wm_use_cache
    model_args.hf_hub_token = model_args.wm_hf_hub_token
    model_args.ms_hub_token = model_args.wm_ms_hub_token
    model_args.export_dir = model_args.wm_export_dir
    model_args.export_size = model_args.wm_export_size
    model_args.export_device = model_args.wm_export_device
    model_args.export_quantization_bit = model_args.wm_export_quantization_bit
    model_args.export_quantization_dataset = model_args.wm_export_quantization_dataset
    model_args.export_quantization_nsamples = model_args.wm_export_quantization_nsamples
    model_args.export_quantization_maxlen = model_args.wm_export_quantization_maxlen
    model_args.export_legacy_format = model_args.wm_export_legacy_format
    model_args.export_hub_model_id = model_args.wm_export_hub_model_id
    model_args.print_param_status = model_args.wm_print_param_status


def fix_policy_model_args(model_args):
    model_args.compute_dtype = model_args.policy_compute_dtype
    model_args.device_map = model_args.policy_device_map
    model_args.model_max_length = model_args.policy_model_max_length

    model_args.model_name_or_path = model_args.policy_model_name_or_path
    model_args.adapter_name_or_path = model_args.policy_adapter_name_or_path
    model_args.cache_dir = model_args.policy_cache_dir
    model_args.use_fast_tokenizer = model_args.policy_use_fast_tokenizer
    model_args.resize_vocab = model_args.policy_resize_vocab
    model_args.split_special_tokens = model_args.policy_split_special_tokens
    model_args.model_revision = model_args.policy_model_revision
    model_args.low_cpu_mem_usage = model_args.policy_low_cpu_mem_usage
    model_args.quantization_bit = model_args.policy_quantization_bit
    model_args.quantization_type = model_args.policy_quantization_type
    model_args.double_quantization = model_args.policy_double_quantization
    model_args.quantization_device_map = model_args.policy_quantization_device_map
    model_args.rope_scaling = model_args.policy_rope_scaling
    model_args.flash_attn = model_args.policy_flash_attn
    model_args.shift_attn = model_args.policy_shift_attn
    model_args.use_unsloth = model_args.policy_use_unsloth
    model_args.moe_aux_loss_coef = model_args.policy_moe_aux_loss_coef
    model_args.disable_gradient_checkpointing = model_args.policy_disable_gradient_checkpointing
    model_args.upcast_layernorm = model_args.policy_upcast_layernorm
    model_args.upcast_lmhead_output = model_args.policy_upcast_lmhead_output
    model_args.infer_backend = model_args.policy_infer_backend
    model_args.vllm_maxlen = model_args.policy_vllm_maxlen
    model_args.vllm_gpu_util = model_args.policy_vllm_gpu_util
    model_args.vllm_enforce_eager = model_args.policy_vllm_enforce_eager
    model_args.offload_folder = model_args.policy_offload_folder
    model_args.use_cache = model_args.policy_use_cache
    model_args.hf_hub_token = model_args.policy_hf_hub_token
    model_args.ms_hub_token = model_args.policy_ms_hub_token
    model_args.export_dir = model_args.policy_export_dir
    model_args.export_size = model_args.policy_export_size
    model_args.export_device = model_args.policy_export_device
    model_args.export_quantization_bit = model_args.policy_export_quantization_bit
    model_args.export_quantization_dataset = model_args.policy_export_quantization_dataset
    model_args.export_quantization_nsamples = model_args.policy_export_quantization_nsamples
    model_args.export_quantization_maxlen = model_args.policy_export_quantization_maxlen
    model_args.export_legacy_format = model_args.policy_export_legacy_format
    model_args.export_hub_model_id = model_args.policy_export_hub_model_id
    model_args.print_param_status = model_args.policy_print_param_status


def fix_finetuning_wm_args(finetuning_args):
    finetuning_args.name_module_trainable = finetuning_args.wm_name_module_trainable
    finetuning_args.num_layer_trainable = finetuning_args.wm_num_layer_trainable
    finetuning_args.additional_target = finetuning_args.wm_additional_target
    finetuning_args.lora_alpha = finetuning_args.wm_lora_alpha
    finetuning_args.lora_dropout = finetuning_args.wm_lora_dropout
    finetuning_args.lora_rank = finetuning_args.wm_lora_rank
    finetuning_args.lora_target = finetuning_args.wm_lora_target
    finetuning_args.loraplus_lr_ratio = finetuning_args.wm_loraplus_lr_ratio
    finetuning_args.loraplus_lr_embedding = finetuning_args.wm_loraplus_lr_embedding
    finetuning_args.use_galore = finetuning_args.wm_use_galore
    finetuning_args.use_rslora = finetuning_args.wm_use_rslora
    finetuning_args.use_dora = finetuning_args.wm_use_dora
    finetuning_args.create_new_adapter = finetuning_args.wm_create_new_adapter
    finetuning_args.dpo_beta = finetuning_args.wm_dpo_beta
    finetuning_args.dpo_loss = finetuning_args.wm_dpo_loss
    finetuning_args.dpo_label_smoothing = finetuning_args.wm_dpo_label_smoothing
    finetuning_args.dpo_ftx = finetuning_args.wm_dpo_ftx
    finetuning_args.orpo_beta = finetuning_args.wm_orpo_beta
    finetuning_args.ppo_buffer_size = finetuning_args.wm_ppo_buffer_size
    finetuning_args.ppo_epochs = finetuning_args.wm_ppo_epochs
    finetuning_args.ppo_score_norm = finetuning_args.wm_ppo_score_norm
    finetuning_args.ppo_target = finetuning_args.wm_ppo_target
    finetuning_args.ppo_whiten_rewards = finetuning_args.wm_ppo_whiten_rewards
    finetuning_args.ref_model = finetuning_args.wm_ref_model
    finetuning_args.ref_model_adapters = finetuning_args.wm_ref_model_adapters
    finetuning_args.ref_model_quantization_bit = finetuning_args.wm_ref_model_quantization_bit
    finetuning_args.reward_model = finetuning_args.wm_reward_model
    finetuning_args.reward_model_adapters = finetuning_args.wm_reward_model_adapters
    finetuning_args.reward_model_quantization_bit = finetuning_args.wm_reward_model_quantization_bit
    finetuning_args.reward_model_type = finetuning_args.wm_reward_model_type
    finetuning_args.pure_bf16 = finetuning_args.wm_pure_bf16
    finetuning_args.stage = finetuning_args.wm_stage
    finetuning_args.finetuning_type = finetuning_args.wm_finetuning_type
    finetuning_args.use_llama_pro = finetuning_args.wm_use_llama_pro
    finetuning_args.plot_loss = finetuning_args.wm_plot_loss


def fix_finetuning_oracle_args(finetuning_args):
    finetuning_args.name_module_trainable = finetuning_args.oracle_name_module_trainable
    finetuning_args.num_layer_trainable = finetuning_args.oracle_num_layer_trainable
    finetuning_args.additional_target = finetuning_args.oracle_additional_target
    finetuning_args.lora_alpha = finetuning_args.oracle_lora_alpha
    finetuning_args.lora_dropout = finetuning_args.oracle_lora_dropout
    finetuning_args.lora_rank = finetuning_args.oracle_lora_rank
    finetuning_args.lora_target = finetuning_args.oracle_lora_target
    finetuning_args.loraplus_lr_ratio = finetuning_args.oracle_loraplus_lr_ratio
    finetuning_args.loraplus_lr_embedding = finetuning_args.oracle_loraplus_lr_embedding
    finetuning_args.use_galore = finetuning_args.oracle_use_galore
    finetuning_args.use_rslora = finetuning_args.oracle_use_rslora
    finetuning_args.use_dora = finetuning_args.oracle_use_dora
    finetuning_args.create_new_adapter = finetuning_args.oracle_create_new_adapter
    finetuning_args.dpo_beta = finetuning_args.oracle_dpo_beta
    finetuning_args.dpo_loss = finetuning_args.oracle_dpo_loss
    finetuning_args.dpo_label_smoothing = finetuning_args.oracle_dpo_label_smoothing
    finetuning_args.dpo_ftx = finetuning_args.oracle_dpo_ftx
    finetuning_args.orpo_beta = finetuning_args.oracle_orpo_beta
    finetuning_args.ppo_buffer_size = finetuning_args.oracle_ppo_buffer_size
    finetuning_args.ppo_epochs = finetuning_args.oracle_ppo_epochs
    finetuning_args.ppo_score_norm = finetuning_args.oracle_ppo_score_norm
    finetuning_args.ppo_target = finetuning_args.oracle_ppo_target
    finetuning_args.ppo_whiten_rewards = finetuning_args.oracle_ppo_whiten_rewards
    finetuning_args.ref_model = finetuning_args.oracle_ref_model
    finetuning_args.ref_model_adapters = finetuning_args.oracle_ref_model_adapters
    finetuning_args.ref_model_quantization_bit = finetuning_args.oracle_ref_model_quantization_bit
    finetuning_args.reward_model = finetuning_args.oracle_reward_model
    finetuning_args.reward_model_adapters = finetuning_args.oracle_reward_model_adapters
    finetuning_args.reward_model_quantization_bit = finetuning_args.oracle_reward_model_quantization_bit
    finetuning_args.reward_model_type = finetuning_args.oracle_reward_model_type
    finetuning_args.pure_bf16 = finetuning_args.oracle_pure_bf16
    finetuning_args.stage = finetuning_args.oracle_stage
    finetuning_args.finetuning_type = finetuning_args.oracle_finetuning_type
    finetuning_args.use_llama_pro = finetuning_args.oracle_use_llama_pro
    finetuning_args.plot_loss = finetuning_args.oracle_plot_loss


def fix_finetuning_policy_args(finetuning_args):
    finetuning_args.name_module_trainable = finetuning_args.policy_name_module_trainable
    finetuning_args.num_layer_trainable = finetuning_args.policy_num_layer_trainable
    finetuning_args.additional_target = finetuning_args.policy_additional_target
    finetuning_args.lora_alpha = finetuning_args.policy_lora_alpha
    finetuning_args.lora_dropout = finetuning_args.policy_lora_dropout
    finetuning_args.lora_rank = finetuning_args.policy_lora_rank
    finetuning_args.lora_target = finetuning_args.policy_lora_target
    finetuning_args.loraplus_lr_ratio = finetuning_args.policy_loraplus_lr_ratio
    finetuning_args.loraplus_lr_embedding = finetuning_args.policy_loraplus_lr_embedding
    finetuning_args.use_galore = finetuning_args.policy_use_galore
    finetuning_args.use_rslora = finetuning_args.policy_use_rslora
    finetuning_args.use_dora = finetuning_args.policy_use_dora
    finetuning_args.create_new_adapter = finetuning_args.policy_create_new_adapter
    finetuning_args.dpo_beta = finetuning_args.policy_dpo_beta
    finetuning_args.dpo_loss = finetuning_args.policy_dpo_loss
    finetuning_args.dpo_label_smoothing = finetuning_args.policy_dpo_label_smoothing
    finetuning_args.dpo_ftx = finetuning_args.policy_dpo_ftx
    finetuning_args.orpo_beta = finetuning_args.policy_orpo_beta
    finetuning_args.ppo_buffer_size = finetuning_args.policy_ppo_buffer_size
    finetuning_args.ppo_epochs = finetuning_args.policy_ppo_epochs
    finetuning_args.ppo_score_norm = finetuning_args.policy_ppo_score_norm
    finetuning_args.ppo_target = finetuning_args.policy_ppo_target
    finetuning_args.ppo_whiten_rewards = finetuning_args.policy_ppo_whiten_rewards
    finetuning_args.ref_model = finetuning_args.policy_ref_model
    finetuning_args.ref_model_adapters = finetuning_args.policy_ref_model_adapters
    finetuning_args.ref_model_quantization_bit = finetuning_args.policy_ref_model_quantization_bit
    finetuning_args.reward_model = finetuning_args.policy_reward_model
    finetuning_args.reward_model_adapters = finetuning_args.policy_reward_model_adapters
    finetuning_args.reward_model_quantization_bit = finetuning_args.policy_reward_model_quantization_bit
    finetuning_args.reward_model_type = finetuning_args.policy_reward_model_type
    finetuning_args.pure_bf16 = finetuning_args.policy_pure_bf16
    finetuning_args.stage = finetuning_args.policy_stage
    finetuning_args.finetuning_type = finetuning_args.policy_finetuning_type
    finetuning_args.use_llama_pro = finetuning_args.policy_use_llama_pro
    finetuning_args.plot_loss = finetuning_args.policy_plot_loss
