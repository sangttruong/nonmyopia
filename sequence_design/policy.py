import math
import re
import gc
import os
import sys
import copy
import time
import random
import requests
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import torch
from tqdm import tqdm
import editdistance

from transformers import GenerationConfig, Trainer, TrainerControl, TrainerState
from transformers.optimization import get_scheduler
from transformers.trainer_pt_utils import remove_dummy_checkpoint
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR
from transformers.utils import SAFE_WEIGHTS_NAME, WEIGHTS_NAME
from trl import PPOConfig, PPOTrainer
from trl.core import PPODecorators, logprobs_from_logits

from llmtuner.extras.callbacks import FixValueHeadModelCallback, LogCallback
from llmtuner.extras.logging import get_logger
from llmtuner.model import load_model, load_tokenizer
from llmtuner.extras.misc import AverageMeter, count_parameters, get_current_device, get_logits_processor
from llmtuner.train.utils import create_custom_optimzer, create_custom_scheduler
from llmtuner.train.ppo.utils import dump_layernorm, get_rewards_from_server, replace_model, restore_layernorm
from llmtuner.hparams import ModelArguments

from openai import OpenAI
from configs import HISTORY_FORMAT, POLICY_PROMPT
from utils import run_server, shutdown_server

if TYPE_CHECKING:
    from datasets import Dataset
    from transformers import DataCollatorWithPadding, PreTrainedTokenizer, Seq2SeqTrainingArguments, TrainerCallback
    from trl import AutoModelForCausalLMWithValueHead

    from src.llmtunner.hparams import FinetuningArguments, GeneratingArguments


logger = get_logger(__name__)

class Policy:
    def __init__(self, model_args, finetuning_args, data_args):
        self.tokenizer = None
        self.model = None
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.data_args = data_args

        self.__prompt__ = POLICY_PROMPT
        self.__history__ = HISTORY_FORMAT

    def __verify_output__(self, X: List[str], Y: List[str]):
        return [editdistance.eval(x, y) for x, y in zip(X, Y)]

    def load(self, iteration=0):
        self.tokenizer = load_tokenizer(self.model_args)
        model_args = copy.deepcopy(self.model_args)
        if iteration == 0:
            model_args.adapter_name_or_path = None
            
        self.model = load_model(self.tokenizer,
                                model_args,
                                self.finetuning_args,
                                is_trainable=True,
                                add_valuehead=True
                                )
    def unload(self):
        del self.tokenizer
        del self.model
        self.tokenizer = None
        self.model = None
        gc.collect()
        torch.cuda.empty_cache()

    def load_inference(self):
        api_port = 1337
        deploy_command = f"""API_PORT={api_port} python src/api_demo.py \
                            --model_name_or_path {self.model_args.model_name_or_path} \
                            --adapter_name_or_path {self.model_args.adapter_name_or_path[0]} \
                            --template {self.data_args.template} \
                            --infer_backend vllm \
                            --vllm_gpu_util {self.model_args.vllm_gpu_util} \
                            --temperature 0.6 \
                            --top_k 50 \
                            --top_p 0.9 \
                            --vllm_enforce_eager"""
        
        print("Deploying LLM...")
        server_process = run_server(deploy_command)
        time.sleep(5)

        url = f"http://localhost:{api_port}/v1/models"
        while True:
            try:
                print("Waiting for server...")
                response = requests.request("GET", url, headers={}, data={})
                if response.status_code == 200:
                    break
            except:
                time.sleep(2)

        return server_process

    def unload_inference(self, server_process):
        # del self.model
        # self.model = None
        # gc.collect()
        # torch.cuda.empty_cache()

        # Shutdown server
        shutdown_server(server_process)

    def format_prompt(self, X: List[str], prevX: List[List[str]], prevY: List[List[float]]):
        # prevX, prevY: n_steps x n_protein
        # histories = [[] for _ in range(len(X))]
        # for stepX, stepY in zip(prevX, prevY):
        #     for i, (pX, pY) in enumerate(zip(stepX, stepY)):
        #         histories[i].append(self.__history__.format(protein=pX, fluorescence=pY))

        # histories = ['\n'.join(h) for h in histories]
        # prompt = [self.__prompt__.format(history=h, protein=x) for h, x in zip(histories, X)]

        prompt = [self.__prompt__.format(protein=x) for x in X]
        return prompt

    def post_process(self, generation):
        return re.findall("[A-Z]{230,}", generation)
        
    def generate(self, X: List[str], prevX: List[List[str]], prevY: List[List[float]], generating_args={}, **kwargs):
        prompts = self.format_prompt(X, prevX, prevY)
        max_retry = generating_args.top_k
        api_port = 1337
        outputs = []

        client = OpenAI(base_url=f"http://localhost:{api_port}/v1", api_key="token-abc123")

        for pi, prompt in enumerate(tqdm(prompts)):
            trying_time = 0
            while trying_time < max_retry:
                completion = client.chat.completions.create(
                    model=self.model_args.model_name_or_path,
                    messages=[{"role": "user", "content": prompt}]
                )
                generations = completion.choices[0].message.content
                generations = self.post_process(generations)
                
                generations_ed = self.__verify_output__([X[pi]] * len(generations), generations)
                filtered_generations = [g for g, ed in zip(generations, generations_ed) if ed == 1]
                
                if len(filtered_generations) > 0:
                    outputs.append(random.choice(filtered_generations))
                    break
                else:
                    trying_time += 1

            if trying_time == max_retry:
                outputs.append(X[pi])

        return outputs

        
        
    # def generate(self, X: List[str], prevX: List[List[str]], prevY: List[List[float]], generating_args={}, **kwargs):
    #     n_sequences = len(X)
    #     prompt = self.format_prompt(X, prevX, prevY)
    #     prompt_tokenized = self.tokenizer(prompt, return_tensors=None, padding=True)
    #     generation_config = GenerationConfig(**(generating_args.to_dict()))
    #     invalid_idx = list(range(n_sequences))
    #     outputs = {}
        
    #     while len(invalid_idx) > 0:
    #         model_inputs = {
    #             k: torch.tensor([v[i] for i in range(len(v)) if i in invalid_idx], device=self.model.pretrained_model.device) for k,v in prompt_tokenized.items()
    #         }
    #         generations = self.model.pretrained_model.generate(
    #             model_inputs["input_ids"], generation_config=generation_config, **kwargs)
    #         breakpoint()
    #         generations_ed = self.__verify_output__(X, generations)
    #         valid_idx = []

    #         for idx, generation, ed in zip(invalid_idx, generations, generations_ed):
    #             if ed <= 1:
    #                 outputs[idx] = generation
    #                 valid_idx.append(idx)

    #         for idx in valid_idx:
    #             invalid_idx.remove(idx)

    #     outputs = [outputs[i] for i in range(n_sequences)]
    #     return outputs


class PolicyPPOTrainer(PPOTrainer, Trainer):
    r"""
    Inherits PPOTrainer.
    """

    def __init__(
        self,
        model_args: "ModelArguments",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
        generating_args: "GeneratingArguments",
        callbacks: List["TrainerCallback"],
        model: "AutoModelForCausalLMWithValueHead",
        ref_model: Optional["AutoModelForCausalLMWithValueHead"],
        tokenizer: "PreTrainedTokenizer",
        dataset: "Dataset",
        data_collator: "DataCollatorWithPadding",
    ):
        backward_batch_size = training_args.per_device_train_batch_size * \
            training_args.gradient_accumulation_steps
        ppo_config = PPOConfig(
            model_name=model_args.model_name_or_path,
            learning_rate=training_args.learning_rate,
            mini_batch_size=training_args.per_device_train_batch_size,
            batch_size=backward_batch_size * finetuning_args.ppo_buffer_size,
            gradient_accumulation_steps=training_args.gradient_accumulation_steps,
            ppo_epochs=finetuning_args.ppo_epochs,
            max_grad_norm=training_args.max_grad_norm,
            seed=training_args.seed,
            optimize_device_cache=True,
            target=finetuning_args.ppo_target,
            use_score_scaling=finetuning_args.ppo_score_norm,
            use_score_norm=finetuning_args.ppo_score_norm,
            whiten_rewards=finetuning_args.ppo_whiten_rewards,
            accelerator_kwargs={"step_scheduler_with_optimizer": False},
            log_with=training_args.report_to[0] if training_args.report_to else None,
            project_kwargs={"logging_dir": training_args.logging_dir},
            remove_unused_columns=training_args.remove_unused_columns
        )

        # Create optimizer and scheduler
        if training_args.max_steps > 0:
            num_training_steps = training_args.max_steps
        else:
            total_train_batch_size = backward_batch_size * \
                finetuning_args.ppo_buffer_size * training_args.world_size
            num_training_steps = training_args.num_train_epochs * \
                math.ceil(len(dataset) / total_train_batch_size)

        optimizer = self.create_optimizer(
            model, training_args, finetuning_args)
        scheduler = self.create_scheduler(
            training_args, num_training_steps, optimizer)
        
        PPOTrainer.__init__(
            self,
            config=ppo_config,
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            data_collator=data_collator,
            lr_scheduler=scheduler,
        )

        self.args = training_args
        self.model_args = model_args
        self.finetuning_args = finetuning_args
        self.current_device = get_current_device()  # patch for deepspeed training

        self.generation_config = GenerationConfig(
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=[self.tokenizer.eos_token_id] +
            self.tokenizer.additional_special_tokens_ids,
            **generating_args.to_dict(),
        )

        self.state = TrainerState()
        self.control = TrainerControl()
        self.log_callback, self.save_callback = callbacks[0], callbacks[1]
        assert isinstance(self.log_callback, LogCallback) and isinstance(
            self.save_callback, FixValueHeadModelCallback)

        if self.args.max_steps > 0:
            logger.info(
                "max_steps is given, it will override any value given in num_train_epochs")


    def ppo_train(self, resume_from_checkpoint: Optional[str] = None) -> None:
        r"""
        Implements training loop for the PPO stage, like _inner_training_loop() in Huggingface's Trainer.
        """
        if resume_from_checkpoint is not None:
            raise ValueError(
                "`resume_from_checkpoint` will be supported in the future version.")

        total_train_batch_size = (
            self.args.per_device_train_batch_size
            * self.args.gradient_accumulation_steps
            * self.finetuning_args.ppo_buffer_size
            * self.args.world_size
        )
        if self.args.max_steps > 0:
            num_examples = total_train_batch_size * self.args.max_steps
            num_train_epochs = sys.maxsize
            max_steps = self.args.max_steps
            steps_in_epoch = self.args.max_steps
        else:
            len_dataloader = len(self.dataloader)
            num_examples = len(self.dataset)
            num_train_epochs = self.args.num_train_epochs
            max_steps = math.ceil(num_train_epochs * len_dataloader)
            steps_in_epoch = len_dataloader

        self.state.max_steps = max_steps
        self.state.num_train_epochs = num_train_epochs
        self.state.is_local_process_zero = self.is_local_process_zero()
        self.state.is_world_process_zero = self.is_world_process_zero()

        if self.is_world_process_zero():
            logger.info("***** Running training *****")
            logger.info("  Num examples = {}".format(num_examples))
            logger.info("  Num Epochs = {}".format(num_train_epochs))
            logger.info("  Instantaneous batch size per device = {}".format(
                self.args.per_device_train_batch_size))
            logger.info(
                "  Total train batch size (w. parallel, buffer, distributed & accumulation) = {}".format(
                    total_train_batch_size
                )
            )
            logger.info("  Gradient Accumulation steps = {}".format(
                self.args.gradient_accumulation_steps))
            logger.info("  Num optimization epochs per batch = {}".format(
                self.finetuning_args.ppo_epochs))
            logger.info("  Total training steps = {}".format(max_steps))
            logger.info("  Number of trainable parameters = {}".format(
                count_parameters(self.model)[0]))

        unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(
            self.model)
        dataiter = iter(self.dataloader)
        loss_meter = AverageMeter()
        reward_meter = AverageMeter()
        self.log_callback.on_train_begin(self.args, self.state, self.control)

        for step in tqdm(range(max_steps), disable=not self.is_local_process_zero()):
            try:
                batch = next(dataiter)
            except StopIteration:
                dataiter = iter(self.dataloader)
                batch = next(dataiter)

            # Get inputs
            queries, responses, rewards = self.get_inputs(batch)

            # Cast to training mode
            unwrapped_model.gradient_checkpointing_enable()
            unwrapped_model.config.use_cache = False
            self.model.train()

            # Run PPO step
            stats = self.step(queries, responses, rewards)
            self.tokenizer.padding_side = "left"  # restore padding side
            loss_meter.update(float(stats["ppo/loss/total"]), n=len(rewards))
            reward_meter.update(torch.stack(
                rewards).mean().item(), n=len(rewards))

            if self.config.log_with is not None:
                try:
                    batch["query"] = self.tokenizer.batch_decode(
                        queries, skip_special_tokens=True)
                    batch["response"] = self.tokenizer.batch_decode(
                        responses, skip_special_tokens=True)
                    self.log_stats(stats, batch, rewards)
                except Exception:
                    logger.warning(
                        "Failed to save stats due to unknown errors.")

            self.state.global_step += 1
            self.log_callback.on_step_end(self.args, self.state, self.control)

            if self.is_local_process_zero() and (step + 1) % self.args.logging_steps == 0:
                logs = dict(
                    loss=round(loss_meter.avg, 4),
                    reward=round(reward_meter.avg, 4),
                    learning_rate=stats["ppo/learning_rate"],
                    epoch=round(step / steps_in_epoch, 2),
                )
                tqdm.write(str(logs))
                logs["step"] = step
                self.state.log_history.append(logs)
                self.log_callback.on_log(self.args, self.state, self.control)
                loss_meter.reset()
                reward_meter.reset()

            if (step + 1) % self.args.save_steps == 0:  # save checkpoint
                self.save_model(
                    os.path.join(
                        self.args.output_dir, "{}-{}".format(PREFIX_CHECKPOINT_DIR, self.state.global_step))
                )
                self.save_callback.on_save(
                    self.args, self.state, self.control, model=self.accelerator.unwrap_model(
                        self.model)
                )

            if self.control.should_epoch_stop or self.control.should_training_stop:
                break

        self.log_callback.on_train_end(self.args, self.state, self.control)
        self.save_callback.on_train_end(
            self.args, self.state, self.control, model=self.accelerator.unwrap_model(
                self.model)
        )

    def create_optimizer(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        training_args: "Seq2SeqTrainingArguments",
        finetuning_args: "FinetuningArguments",
    ) -> "torch.optim.Optimizer":
        optimizer = create_custom_optimzer(
            model, training_args, finetuning_args)
        if optimizer is None:
            decay_params, nodecay_params = [], []
            decay_param_names = self.get_decay_parameter_names(model)
            for name, param in model.named_parameters():
                if param.requires_grad:
                    if name in decay_param_names:
                        decay_params.append(param)
                    else:
                        nodecay_params.append(param)

            optim_class, optim_kwargs = Trainer.get_optimizer_cls_and_kwargs(
                training_args)
            param_groups = [
                dict(params=nodecay_params),
                dict(params=decay_params, weight_decay=training_args.weight_decay),
            ]
            optimizer = optim_class(param_groups, **optim_kwargs)

        return optimizer

    def create_scheduler(
        self, training_args: "Seq2SeqTrainingArguments", num_training_steps: int, optimizer: "torch.optim.Optimizer"
    ) -> "torch.optim.lr_scheduler.LRScheduler":
        create_custom_scheduler(training_args, num_training_steps, optimizer)
        lr_scheduler = get_scheduler(
            training_args.lr_scheduler_type,
            optimizer=optimizer,
            num_warmup_steps=training_args.get_warmup_steps(
                num_training_steps),
            num_training_steps=num_training_steps,
        )
        return lr_scheduler

    @torch.no_grad()
    def get_inputs(self, batch: Dict[str, torch.Tensor]) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        r"""
        Generates model's responses given queries.
        """
        queries = []
        responses = []
        rewards = []
        for i in range(len(batch)):
            queries.append(torch.tensor(batch[i][0]['input_ids']))
            responses.append(torch.tensor(batch[i][0]['output_ids']))
            rewards.append(torch.tensor(batch[i][0]['labels']))
    
        return queries, responses, rewards


    @PPODecorators.empty_device_cache()
    def batched_forward_pass(
        self,
        model: "AutoModelForCausalLMWithValueHead",
        queries: torch.Tensor,
        responses: torch.Tensor,
        model_inputs: dict,
        return_logits: bool = False,
        response_masks: Optional[torch.Tensor] = None,
    ):
        r"""
        Calculates model outputs in multiple batches.

        Subclass and override to inject custom behavior.
        """
        bs = len(queries)
        fbs = self.config.mini_batch_size
        all_logprobs = []
        all_logits = []
        all_masks = []
        all_values = []

        for i in range(math.ceil(bs / fbs)):
            input_kwargs = {
                key: value[i * fbs: (i + 1) * fbs] for key, value in model_inputs.items()}
            query_batch = queries[i * fbs: (i + 1) * fbs]
            response_batch = responses[i * fbs: (i + 1) * fbs]
            if response_masks is not None:
                response_masks_batch = response_masks[i * fbs: (i + 1) * fbs]
            input_ids = input_kwargs["input_ids"]
            attention_mask = input_kwargs["attention_mask"]

            # support bf16
            with torch.cuda.amp.autocast(dtype=self.model_args.compute_dtype):
                logits, _, values = model(**input_kwargs)

            unwrapped_model: "AutoModelForCausalLMWithValueHead" = self.accelerator.unwrap_model(
                self.model)
            if getattr(unwrapped_model.config, "model_type", None) == "chatglm":
                values = torch.transpose(values, 0, 1)

            logprobs = logprobs_from_logits(
                logits[:, :-1, :], input_ids[:, 1:])
            masks = torch.zeros_like(attention_mask)
            masks[:, :-1] = attention_mask[:, 1:]

            for j in range(len(query_batch)):
                start = len(query_batch[j]) - 1
                if attention_mask[j, 0] == 0:  # offset left padding
                    start += attention_mask[j, :].nonzero()[0].item()
                end = start + len(response_batch[j])

                if response_masks is not None:
                    response_masks_batch = torch.cat(
                        (torch.zeros_like(query_batch[j]), response_masks_batch[j]))[1:]

                masks[j, :start] = 0
                masks[j, end:] = 0
                if response_masks is not None:
                    masks[j, start:end] = masks[j, start:end] * \
                        response_masks_batch[j][start:end]

            if return_logits:
                all_logits.append(logits)
            else:
                del logits

            all_values.append(values)
            all_logprobs.append(logprobs)
            all_masks.append(masks)

        return (
            torch.cat(all_logprobs),
            torch.cat(all_logits)[:, :-1] if return_logits else None,
            torch.cat(all_values)[:, :-1],
            torch.cat(all_masks)[:, :-1],
        )

    def save_model(self, output_dir: Optional[str] = None) -> None:
        r"""
        Saves model checkpoint.

        Subclass and override to inject custom behavior.
        """
        if self.args.should_save:
            try:
                self._save(
                    output_dir, state_dict=self.accelerator.get_state_dict(self.model))
            except ValueError:
                logger.warning(
                    " stage3_gather_16bit_weights_on_model_save=false. Saving the full checkpoint instead,"
                    " use zero_to_fp32.py to recover weights"
                )
                self._save(output_dir, state_dict={})
                remove_dummy_checkpoint(
                    True, output_dir, [WEIGHTS_NAME, SAFE_WEIGHTS_NAME])
                self.model.save_checkpoint(output_dir)
