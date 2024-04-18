from dataclasses import asdict, dataclass, field
from typing import Any, Dict, Literal, Optional


@dataclass
class WMArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    wm_model_name_or_path: str = field(
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    wm_adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the adapter weight or identifier from huggingface.co/models."},
    )
    wm_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    wm_use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    wm_resize_vocab: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    wm_split_special_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    wm_model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    wm_low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    wm_quantization_bit: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of bits to quantize the model using bitsandbytes."},
    )
    wm_quantization_type: Literal["fp4", "nf4"] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."},
    )
    wm_double_quantization: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use double quantization in int4 training."},
    )
    wm_quantization_device_map: Optional[Literal["auto"]] = field(
        default=None,
        metadata={
            "help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."},
    )
    wm_rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={
            "help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )
    wm_flash_attn: bool = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."},
    )
    wm_shift_attn: bool = field(
        default=False,
        metadata={
            "help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."},
    )
    wm_use_unsloth: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    wm_moe_aux_loss_coef: Optional[float] = field(
        default=None,
        metadata={
            "help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    )
    wm_disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    wm_upcast_layernorm: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    wm_upcast_lmhead_output: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to upcast the output of lm_head in fp32."},
    )
    wm_infer_backend: Literal["huggingface", "vllm"] = field(
        default="huggingface",
        metadata={"help": "Backend engine used at inference."},
    )
    wm_vllm_maxlen: int = field(
        default=2048,
        metadata={"help": "Maximum input length of the vLLM engine."},
    )
    wm_vllm_gpu_util: float = field(
        default=0.9,
        metadata={
            "help": "The fraction of GPU memory in (0,1) to be used for the vLLM engine."},
    )
    wm_vllm_enforce_eager: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to disable CUDA graph in the vLLM engine."},
    )
    wm_offload_folder: str = field(
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    wm_use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    wm_hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    wm_ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    wm_export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    wm_export_size: int = field(
        default=1,
        metadata={
            "help": "The file shard size (in GB) of the exported model."},
    )
    wm_export_device: str = field(
        default="cpu",
        metadata={"help": "The device used in model export."},
    )
    wm_export_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the exported model."},
    )
    wm_export_quantization_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the dataset or dataset name to use in quantizing the exported model."},
    )
    wm_export_quantization_nsamples: int = field(
        default=128,
        metadata={"help": "The number of samples used for quantization."},
    )
    wm_export_quantization_maxlen: int = field(
        default=1024,
        metadata={
            "help": "The maximum length of the model inputs used for quantization."},
    )
    wm_export_legacy_format: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to save the `.bin` files instead of `.safetensors`."},
    )
    wm_export_hub_model_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the repository if push the model to the Hugging Face hub."},
    )
    wm_print_param_status: bool = field(
        default=False,
        metadata={
            "help": "For debugging purposes, print the status of the parameters in the model."},
    )

    def __post_init__(self):
        self.wm_compute_dtype = None
        self.wm_device_map = None
        self.wm_model_max_length = None

        if self.wm_split_special_tokens and self.wm_use_fast_tokenizer:
            raise ValueError(
                "`split_special_tokens` is only supported for slow tokenizers.")

        if self.wm_adapter_name_or_path is not None:  # support merging multiple lora weights
            self.wm_adapter_name_or_path = [
                path.strip() for path in self.wm_adapter_name_or_path.split(",")]

        assert self.wm_quantization_bit in [
            None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.wm_export_quantization_bit in [
            None, 8, 4, 3, 2], "We only accept 2/3/4/8-bit quantization."

        if self.wm_export_quantization_bit is not None and self.wm_export_quantization_dataset is None:
            raise ValueError(
                "Quantization dataset is necessary for exporting.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class OracleArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    oracle_model_name_or_path: str = field(
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    oracle_adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the adapter weight or identifier from huggingface.co/models."},
    )
    oracle_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    oracle_use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    oracle_resize_vocab: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    oracle_split_special_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    oracle_model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    oracle_low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    oracle_quantization_bit: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of bits to quantize the model using bitsandbytes."},
    )
    oracle_quantization_type: Literal["fp4", "nf4"] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."},
    )
    oracle_double_quantization: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use double quantization in int4 training."},
    )
    oracle_quantization_device_map: Optional[Literal["auto"]] = field(
        default=None,
        metadata={
            "help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."},
    )
    oracle_rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={
            "help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )
    oracle_flash_attn: bool = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."},
    )
    oracle_shift_attn: bool = field(
        default=False,
        metadata={
            "help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."},
    )
    oracle_use_unsloth: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    oracle_moe_aux_loss_coef: Optional[float] = field(
        default=None,
        metadata={
            "help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    )
    oracle_disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    oracle_upcast_layernorm: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    oracle_upcast_lmhead_output: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to upcast the output of lm_head in fp32."},
    )
    oracle_infer_backend: Literal["huggingface", "vllm"] = field(
        default="huggingface",
        metadata={"help": "Backend engine used at inference."},
    )
    oracle_vllm_maxlen: int = field(
        default=2048,
        metadata={"help": "Maximum input length of the vLLM engine."},
    )
    oracle_vllm_gpu_util: float = field(
        default=0.9,
        metadata={
            "help": "The fraction of GPU memory in (0,1) to be used for the vLLM engine."},
    )
    oracle_vllm_enforce_eager: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to disable CUDA graph in the vLLM engine."},
    )
    oracle_offload_folder: str = field(
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    oracle_use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    oracle_hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    oracle_ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    oracle_export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    oracle_export_size: int = field(
        default=1,
        metadata={
            "help": "The file shard size (in GB) of the exported model."},
    )
    oracle_export_device: str = field(
        default="cpu",
        metadata={"help": "The device used in model export."},
    )
    oracle_export_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the exported model."},
    )
    oracle_export_quantization_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the dataset or dataset name to use in quantizing the exported model."},
    )
    oracle_export_quantization_nsamples: int = field(
        default=128,
        metadata={"help": "The number of samples used for quantization."},
    )
    oracle_export_quantization_maxlen: int = field(
        default=1024,
        metadata={
            "help": "The maximum length of the model inputs used for quantization."},
    )
    oracle_export_legacy_format: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to save the `.bin` files instead of `.safetensors`."},
    )
    oracle_export_hub_model_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the repository if push the model to the Hugging Face hub."},
    )
    oracle_print_param_status: bool = field(
        default=False,
        metadata={
            "help": "For debugging purposes, print the status of the parameters in the model."},
    )

    def __post_init__(self):
        self.oracle_compute_dtype = None
        self.oracle_device_map = None
        self.oracle_model_max_length = None

        if self.oracle_split_special_tokens and self.oracle_use_fast_tokenizer:
            raise ValueError(
                "`split_special_tokens` is only supported for slow tokenizers.")

        if self.oracle_adapter_name_or_path is not None:  # support merging multiple lora weights
            self.oracle_adapter_name_or_path = [
                path.strip() for path in self.oracle_adapter_name_or_path.split(",")]

        assert self.oracle_quantization_bit in [
            None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.oracle_export_quantization_bit in [
            None, 8, 4, 3, 2], "We only accept 2/3/4/8-bit quantization."

        if self.oracle_export_quantization_bit is not None and self.oracle_export_quantization_dataset is None:
            raise ValueError(
                "Quantization dataset is necessary for exporting.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class PolicyArguments:
    r"""
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune or infer.
    """

    policy_model_name_or_path: str = field(
        metadata={
            "help": "Path to the model weight or identifier from huggingface.co/models or modelscope.cn/models."
        },
    )
    policy_adapter_name_or_path: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the adapter weight or identifier from huggingface.co/models."},
    )
    policy_cache_dir: Optional[str] = field(
        default=None,
        metadata={
            "help": "Where to store the pre-trained models downloaded from huggingface.co or modelscope.cn."},
    )
    policy_use_fast_tokenizer: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use one of the fast tokenizer (backed by the tokenizers library)."},
    )
    policy_resize_vocab: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to resize the tokenizer vocab and the embedding layers."},
    )
    policy_split_special_tokens: bool = field(
        default=False,
        metadata={
            "help": "Whether or not the special tokens should be split during the tokenization process."},
    )
    policy_model_revision: str = field(
        default="main",
        metadata={
            "help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    policy_low_cpu_mem_usage: bool = field(
        default=True,
        metadata={"help": "Whether or not to use memory-efficient model loading."},
    )
    policy_quantization_bit: Optional[int] = field(
        default=None,
        metadata={
            "help": "The number of bits to quantize the model using bitsandbytes."},
    )
    policy_quantization_type: Literal["fp4", "nf4"] = field(
        default="nf4",
        metadata={"help": "Quantization data type to use in int4 training."},
    )
    policy_double_quantization: bool = field(
        default=True,
        metadata={
            "help": "Whether or not to use double quantization in int4 training."},
    )
    policy_quantization_device_map: Optional[Literal["auto"]] = field(
        default=None,
        metadata={
            "help": "Device map used to infer the 4-bit quantized model, needs bitsandbytes>=0.43.0."},
    )
    policy_rope_scaling: Optional[Literal["linear", "dynamic"]] = field(
        default=None,
        metadata={
            "help": "Which scaling strategy should be adopted for the RoPE embeddings."},
    )
    policy_flash_attn: bool = field(
        default=False,
        metadata={"help": "Enable FlashAttention-2 for faster training."},
    )
    policy_shift_attn: bool = field(
        default=False,
        metadata={
            "help": "Enable shift short attention (S^2-Attn) proposed by LongLoRA."},
    )
    policy_use_unsloth: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use unsloth's optimization for the LoRA training."},
    )
    policy_moe_aux_loss_coef: Optional[float] = field(
        default=None,
        metadata={
            "help": "Coefficient of the auxiliary router loss in mixture-of-experts model."},
    )
    policy_disable_gradient_checkpointing: bool = field(
        default=False,
        metadata={"help": "Whether or not to disable gradient checkpointing."},
    )
    policy_upcast_layernorm: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to upcast the layernorm weights in fp32."},
    )
    policy_upcast_lmhead_output: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to upcast the output of lm_head in fp32."},
    )
    policy_infer_backend: Literal["huggingface", "vllm"] = field(
        default="huggingface",
        metadata={"help": "Backend engine used at inference."},
    )
    policy_vllm_maxlen: int = field(
        default=2048,
        metadata={"help": "Maximum input length of the vLLM engine."},
    )
    policy_vllm_gpu_util: float = field(
        default=0.9,
        metadata={
            "help": "The fraction of GPU memory in (0,1) to be used for the vLLM engine."},
    )
    policy_vllm_enforce_eager: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to disable CUDA graph in the vLLM engine."},
    )
    policy_offload_folder: str = field(
        default="offload",
        metadata={"help": "Path to offload model weights."},
    )
    policy_use_cache: bool = field(
        default=True,
        metadata={"help": "Whether or not to use KV cache in generation."},
    )
    policy_hf_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with Hugging Face Hub."},
    )
    policy_ms_hub_token: Optional[str] = field(
        default=None,
        metadata={"help": "Auth token to log in with ModelScope Hub."},
    )
    policy_export_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the directory to save the exported model."},
    )
    policy_export_size: int = field(
        default=1,
        metadata={
            "help": "The file shard size (in GB) of the exported model."},
    )
    policy_export_device: str = field(
        default="cpu",
        metadata={"help": "The device used in model export."},
    )
    policy_export_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the exported model."},
    )
    policy_export_quantization_dataset: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the dataset or dataset name to use in quantizing the exported model."},
    )
    policy_export_quantization_nsamples: int = field(
        default=128,
        metadata={"help": "The number of samples used for quantization."},
    )
    policy_export_quantization_maxlen: int = field(
        default=1024,
        metadata={
            "help": "The maximum length of the model inputs used for quantization."},
    )
    policy_export_legacy_format: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to save the `.bin` files instead of `.safetensors`."},
    )
    policy_export_hub_model_id: Optional[str] = field(
        default=None,
        metadata={
            "help": "The name of the repository if push the model to the Hugging Face hub."},
    )
    policy_print_param_status: bool = field(
        default=False,
        metadata={
            "help": "For debugging purposes, print the status of the parameters in the model."},
    )

    def __post_init__(self):
        self.policy_compute_dtype = None
        self.policy_device_map = None
        self.policy_model_max_length = None

        if self.policy_split_special_tokens and self.policy_use_fast_tokenizer:
            raise ValueError(
                "`split_special_tokens` is only supported for slow tokenizers.")

        if self.policy_adapter_name_or_path is not None:  # support merging multiple lora weights
            self.policy_adapter_name_or_path = [
                path.strip() for path in self.policy_adapter_name_or_path.split(",")]

        assert self.policy_quantization_bit in [
            None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.policy_export_quantization_bit in [
            None, 8, 4, 3, 2], "We only accept 2/3/4/8-bit quantization."

        if self.policy_export_quantization_bit is not None and self.policy_export_quantization_dataset is None:
            raise ValueError(
                "Quantization dataset is necessary for exporting.")

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
