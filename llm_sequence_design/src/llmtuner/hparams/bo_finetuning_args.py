import json
from dataclasses import asdict, dataclass, field
from typing import Literal, Optional


@dataclass
class WMFreezeArguments:
    r"""
    Arguments pertaining to the freeze (partial-parameter) training.
    """

    wm_name_module_trainable: str = field(
        default="all",
        metadata={
            "help": """Name of trainable modules for partial-parameter (freeze) fine-tuning. \
                    Use commas to separate multiple modules. \
                    Use "all" to specify all the available modules. \
                    LLaMA choices: ["mlp", "self_attn"], \
                    BLOOM & Falcon & ChatGLM choices: ["mlp", "self_attention"], \
                    Qwen choices: ["mlp", "attn"], \
                    InternLM2 choices: ["feed_forward", "attention"], \
                    Others choices: the same as LLaMA."""
        },
    )
    wm_num_layer_trainable: int = field(
        default=2,
        metadata={
            "help": "The number of trainable layers for partial-parameter (freeze) fine-tuning."},
    )


@dataclass
class WMLoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    wm_additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    wm_lora_alpha: Optional[int] = field(
        default=None,
        metadata={
            "help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    wm_lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    wm_lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    wm_lora_target: str = field(
        default="all",
        metadata={
            "help": """Name(s) of target modules to apply LoRA. \
                    Use commas to separate multiple modules. \
                    Use "all" to specify all the linear modules. \
                    LLaMA choices: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                    BLOOM & Falcon & ChatGLM choices: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], \
                    Baichuan choices: ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                    Qwen choices: ["c_attn", "attn.c_proj", "w1", "w2", "mlp.c_proj"], \
                    InternLM2 choices: ["wqkv", "wo", "w1", "w2", "w3"], \
                    Others choices: the same as LLaMA."""
        },
    )
    wm_loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    wm_loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    wm_use_rslora: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    wm_use_dora: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    wm_create_new_adapter: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class WMRLHFArguments:
    r"""
    Arguments pertaining to the PPO and DPO training.
    """

    wm_dpo_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter for the DPO loss."},
    )
    wm_dpo_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    wm_dpo_label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    wm_dpo_ftx: float = field(
        default=0.0,
        metadata={
            "help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    wm_orpo_beta: float = field(
        default=0.1,
        metadata={
            "help": "The beta (lambda) parameter in ORPO loss representing the weight of the SFT loss."},
    )
    wm_ppo_buffer_size: int = field(
        default=1,
        metadata={
            "help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    wm_ppo_epochs: int = field(
        default=4,
        metadata={
            "help": "The number of epochs to perform in a PPO optimization step."},
    )
    wm_ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    wm_ppo_target: float = field(
        default=6.0,
        metadata={
            "help": "Target KL value for adaptive KL control in PPO training."},
    )
    wm_ppo_whiten_rewards: bool = field(
        default=False,
        metadata={
            "help": "Whiten the rewards before compute advantages in PPO training."},
    )
    wm_ref_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the reference model used for the PPO or DPO training."},
    )
    wm_ref_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    wm_ref_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    wm_reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    wm_reward_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    wm_reward_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    wm_reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={
            "help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )


@dataclass
class WMGaloreArguments:
    r"""
    Arguments pertaining to the GaLore algorithm.
    """

    wm_use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use gradient low-Rank projection."},
    )
    wm_galore_target: str = field(
        default="all",
        metadata={
            "help": """Name(s) of modules to apply GaLore. Use commas to separate multiple modules. \
                    Use "all" to specify all the linear modules."""
        },
    )
    wm_galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients."},
    )
    wm_galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection."},
    )
    wm_galore_scale: float = field(
        default=0.25,
        metadata={"help": "GaLore scaling coefficient."},
    )
    wm_galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection."},
    )
    wm_galore_layerwise: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to enable layer-wise update to further save memory."},
    )


@dataclass
class WMFinetuningArguments(WMFreezeArguments, WMLoraArguments, WMRLHFArguments, WMGaloreArguments):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """

    wm_pure_bf16: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    wm_stage: Literal["pt", "sft", "rm", "ppo", "dpo", "orpo", "oracle"] = field(
        default="oracle",
        metadata={"help": "Which stage will be performed in training."},
    )
    wm_finetuning_type: Literal["lora", "freeze", "full"] = field(
        default="freeze",
        metadata={"help": "Which fine-tuning method to use."},
    )
    wm_use_llama_pro: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    wm_plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.wm_name_module_trainable = split_arg(
            self.wm_name_module_trainable)
        self.wm_lora_alpha = self.wm_lora_alpha or self.wm_lora_rank * 2
        self.wm_lora_target = split_arg(self.wm_lora_target)
        self.wm_additional_target = split_arg(self.wm_additional_target)
        self.wm_galore_target = split_arg(self.wm_galore_target)

        assert self.wm_finetuning_type in [
            "lora", "freeze", "full"], "Invalid fine-tuning method."
        assert self.wm_ref_model_quantization_bit in [
            None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.wm_reward_model_quantization_bit in [
            None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.wm_stage == "ppo" and self.wm_reward_model is None:
            raise ValueError("`reward_model` is necessary for PPO training.")

        if self.wm_stage == "ppo" and self.wm_reward_model_type == "lora" and self.wm_finetuning_type != "lora":
            raise ValueError(
                "`reward_model_type` cannot be lora for Freeze/Full PPO training.")

        if self.wm_stage == "dpo" and self.wm_dpo_loss != "sigmoid" and self.wm_dpo_label_smoothing > 1e-6:
            raise ValueError(
                "`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.wm_use_llama_pro and self.wm_finetuning_type == "full":
            raise ValueError(
                "`use_llama_pro` is only valid for the Freeze or LoRA method.")

        if self.wm_use_galore and self.wm_finetuning_type == "lora":
            raise ValueError("Cannot use LoRA with GaLore together.")

    def save_to_json(self, json_path: str):
        r"""Saves the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        r"""Creates an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()

        return cls(**json.loads(text))


@dataclass
class PolicyFreezeArguments:
    r"""
    Arguments pertaining to the freeze (partial-parameter) training.
    """

    policy_name_module_trainable: str = field(
        default="all",
        metadata={
            "help": """Name of trainable modules for partial-parameter (freeze) fine-tuning. \
                    Use commas to separate multiple modules. \
                    Use "all" to specify all the available modules. \
                    LLaMA choices: ["mlp", "self_attn"], \
                    BLOOM & Falcon & ChatGLM choices: ["mlp", "self_attention"], \
                    Qwen choices: ["mlp", "attn"], \
                    InternLM2 choices: ["feed_forward", "attention"], \
                    Others choices: the same as LLaMA."""
        },
    )
    policy_num_layer_trainable: int = field(
        default=2,
        metadata={
            "help": "The number of trainable layers for partial-parameter (freeze) fine-tuning."},
    )


@dataclass
class PolicyLoraArguments:
    r"""
    Arguments pertaining to the LoRA training.
    """

    policy_additional_target: Optional[str] = field(
        default=None,
        metadata={
            "help": "Name(s) of modules apart from LoRA layers to be set as trainable and saved in the final checkpoint."
        },
    )
    policy_lora_alpha: Optional[int] = field(
        default=None,
        metadata={
            "help": "The scale factor for LoRA fine-tuning (default: lora_rank * 2)."},
    )
    policy_lora_dropout: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for the LoRA fine-tuning."},
    )
    policy_lora_rank: int = field(
        default=8,
        metadata={"help": "The intrinsic dimension for LoRA fine-tuning."},
    )
    policy_lora_target: str = field(
        default="all",
        metadata={
            "help": """Name(s) of target modules to apply LoRA. \
                    Use commas to separate multiple modules. \
                    Use "all" to specify all the linear modules. \
                    LLaMA choices: ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                    BLOOM & Falcon & ChatGLM choices: ["query_key_value", "dense", "dense_h_to_4h", "dense_4h_to_h"], \
                    Baichuan choices: ["W_pack", "o_proj", "gate_proj", "up_proj", "down_proj"], \
                    Qwen choices: ["c_attn", "attn.c_proj", "w1", "w2", "mlp.c_proj"], \
                    InternLM2 choices: ["wqkv", "wo", "w1", "w2", "w3"], \
                    Others choices: the same as LLaMA."""
        },
    )
    policy_loraplus_lr_ratio: Optional[float] = field(
        default=None,
        metadata={"help": "LoRA plus learning rate ratio (lr_B / lr_A)."},
    )
    policy_loraplus_lr_embedding: float = field(
        default=1e-6,
        metadata={"help": "LoRA plus learning rate for lora embedding layers."},
    )
    policy_use_rslora: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use the rank stabilization scaling factor for LoRA layer."},
    )
    policy_use_dora: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to use the weight-decomposed lora method (DoRA)."},
    )
    policy_create_new_adapter: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to create a new adapter with randomly initialized weight."},
    )


@dataclass
class PolicyRLHFArguments:
    r"""
    Arguments pertaining to the PPO and DPO training.
    """

    policy_dpo_beta: float = field(
        default=0.1,
        metadata={"help": "The beta parameter for the DPO loss."},
    )
    policy_dpo_loss: Literal["sigmoid", "hinge", "ipo", "kto_pair"] = field(
        default="sigmoid",
        metadata={"help": "The type of DPO loss to use."},
    )
    policy_dpo_label_smoothing: float = field(
        default=0.0,
        metadata={
            "help": "The robust DPO label smoothing parameter in cDPO that should be between 0 and 0.5."},
    )
    policy_dpo_ftx: float = field(
        default=0.0,
        metadata={
            "help": "The supervised fine-tuning loss coefficient in DPO training."},
    )
    policy_orpo_beta: float = field(
        default=0.1,
        metadata={
            "help": "The beta (lambda) parameter in ORPO loss representing the weight of the SFT loss."},
    )
    policy_ppo_buffer_size: int = field(
        default=1,
        metadata={
            "help": "The number of mini-batches to make experience buffer in a PPO optimization step."},
    )
    policy_ppo_epochs: int = field(
        default=4,
        metadata={
            "help": "The number of epochs to perform in a PPO optimization step."},
    )
    policy_ppo_score_norm: bool = field(
        default=False,
        metadata={"help": "Use score normalization in PPO training."},
    )
    policy_ppo_target: float = field(
        default=6.0,
        metadata={
            "help": "Target KL value for adaptive KL control in PPO training."},
    )
    policy_ppo_whiten_rewards: bool = field(
        default=False,
        metadata={
            "help": "Whiten the rewards before compute advantages in PPO training."},
    )
    policy_ref_model: Optional[str] = field(
        default=None,
        metadata={
            "help": "Path to the reference model used for the PPO or DPO training."},
    )
    policy_ref_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reference model."},
    )
    policy_ref_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reference model."},
    )
    policy_reward_model: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the reward model used for the PPO training."},
    )
    policy_reward_model_adapters: Optional[str] = field(
        default=None,
        metadata={"help": "Path to the adapters of the reward model."},
    )
    policy_reward_model_quantization_bit: Optional[int] = field(
        default=None,
        metadata={"help": "The number of bits to quantize the reward model."},
    )
    policy_reward_model_type: Literal["lora", "full", "api"] = field(
        default="lora",
        metadata={
            "help": "The type of the reward model in PPO training. Lora model only supports lora training."},
    )


@dataclass
class PolicyGaloreArguments:
    r"""
    Arguments pertaining to the GaLore algorithm.
    """

    policy_use_galore: bool = field(
        default=False,
        metadata={"help": "Whether or not to use gradient low-Rank projection."},
    )
    policy_galore_target: str = field(
        default="all",
        metadata={
            "help": """Name(s) of modules to apply GaLore. Use commas to separate multiple modules. \
                    Use "all" to specify all the linear modules."""
        },
    )
    policy_galore_rank: int = field(
        default=16,
        metadata={"help": "The rank of GaLore gradients."},
    )
    policy_galore_update_interval: int = field(
        default=200,
        metadata={"help": "Number of steps to update the GaLore projection."},
    )
    policy_galore_scale: float = field(
        default=0.25,
        metadata={"help": "GaLore scaling coefficient."},
    )
    policy_galore_proj_type: Literal["std", "reverse_std", "right", "left", "full"] = field(
        default="std",
        metadata={"help": "Type of GaLore projection."},
    )
    policy_galore_layerwise: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to enable layer-wise update to further save memory."},
    )


@dataclass
class PolicyFinetuningArguments(PolicyFreezeArguments, PolicyLoraArguments, PolicyRLHFArguments, PolicyGaloreArguments):
    r"""
    Arguments pertaining to which techniques we are going to fine-tuning with.
    """

    policy_pure_bf16: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to train model in purely bf16 precision (without AMP)."},
    )
    policy_stage: Literal["pt", "sft", "rm", "ppo", "dpo", "orpo", "oracle"] = field(
        default="sft",
        metadata={"help": "Which stage will be performed in training."},
    )
    policy_finetuning_type: Literal["lora", "freeze", "full"] = field(
        default="lora",
        metadata={"help": "Which fine-tuning method to use."},
    )
    policy_use_llama_pro: bool = field(
        default=False,
        metadata={
            "help": "Whether or not to make only the parameters in the expanded blocks trainable."},
    )
    policy_plot_loss: bool = field(
        default=False,
        metadata={"help": "Whether or not to save the training loss curves."},
    )

    def __post_init__(self):
        def split_arg(arg):
            if isinstance(arg, str):
                return [item.strip() for item in arg.split(",")]
            return arg

        self.policy_name_module_trainable = split_arg(
            self.policy_name_module_trainable)
        self.policy_lora_alpha = self.policy_lora_alpha or self.policy_lora_rank * 2
        self.policy_lora_target = split_arg(self.policy_lora_target)
        self.policy_additional_target = split_arg(
            self.policy_additional_target)
        self.policy_galore_target = split_arg(self.policy_galore_target)

        assert self.policy_finetuning_type in [
            "lora", "freeze", "full"], "Invalid fine-tuning method."
        assert self.policy_ref_model_quantization_bit in [
            None, 8, 4], "We only accept 4-bit or 8-bit quantization."
        assert self.policy_reward_model_quantization_bit in [
            None, 8, 4], "We only accept 4-bit or 8-bit quantization."

        if self.policy_stage == "dpo" and self.policy_dpo_loss != "sigmoid" and self.policy_dpo_label_smoothing > 1e-6:
            raise ValueError(
                "`dpo_label_smoothing` is only valid for sigmoid loss function.")

        if self.policy_use_llama_pro and self.policy_finetuning_type == "full":
            raise ValueError(
                "`use_llama_pro` is only valid for the Freeze or LoRA method.")

        if self.policy_use_galore and self.policy_finetuning_type == "lora":
            raise ValueError("Cannot use LoRA with GaLore together.")

    def save_to_json(self, json_path: str):
        r"""Saves the content of this instance in JSON format inside `json_path`."""
        json_string = json.dumps(asdict(self), indent=2, sort_keys=True) + "\n"
        with open(json_path, "w", encoding="utf-8") as f:
            f.write(json_string)

    @classmethod
    def load_from_json(cls, json_path: str):
        r"""Creates an instance from the content of `json_path`."""
        with open(json_path, "r", encoding="utf-8") as f:
            text = f.read()

        return cls(**json.loads(text))
