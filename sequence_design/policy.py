import os
import random
import re
from typing import List

import editdistance

from configs import HISTORY_FORMAT, POLICY_PROMPT
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams


class Policy:
    def __init__(self, config):
        self.config = config
        self.model = None
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.policy_model_name_or_path,
            use_fast=self.config.use_fast_tokenizer,
        )
        generation_config = GenerationConfig.from_pretrained(
            self.config.policy_model_name_or_path
        )
        self.sampling_params = SamplingParams(
            temperature=generation_config.temperature,
            top_p=generation_config.top_p,
            max_tokens=self.config.max_new_tokens,
        )
        self.__prompt__ = POLICY_PROMPT
        self.__history__ = HISTORY_FORMAT

    def __verify_output__(self, X: List[str], Y: List[str]):
        return [editdistance.eval(x, y) for x, y in zip(X, Y)]

    def load_inference(self, iteration):
        ckpt_dir = os.path.join(self.config.output_dir, f"{iteration}")
        self.model = LLM(
            ckpt_dir,
            gpu_memory_utilization=0.5,
            enforce_eager=True,
        )

    def unload_inference(self):
        del self.model
        self.model = None

    def format_prompt(self, prevX: List[List[str]], prevY: List[List[float]]):
        # prevX, prevY: n_steps x n_protein
        histories = [[] for _ in range(len(prevX[0]))]
        for stepX, stepY in zip(prevX, prevY):
            for i, (pX, pY) in enumerate(zip(stepX, stepY)):
                histories[i].append(
                    self.__history__.format(protein=pX, fluorescence=pY)
                )

        prompt = [
            self.__prompt__.format(history="\n".join(h), protein=p)
            for h, p in zip(histories, prevX[-1])
        ]
        return prompt

    def post_process(self, generations: List[str]):
        outputs = []
        for generation in generations:
            outputs.extend(re.findall("[A-Z]{230,}", generation))
        return outputs

    def generate(
        self,
        prevX: List[List[str]],
        prevY: List[List[float]],
        max_retry=8,
        **kwargs,
    ):
        X = prevX[-1]
        self.sampling_params.n = max_retry
        prompts = self.format_prompt(prevX, prevY)
        prompts = [
            self.tokenizer.apply_chat_template(
                [{"role": "user", "content": prompt}], tokenize=False
            )
            for prompt in prompts
        ]

        list_generations = self.model.generate(prompts, self.sampling_params)
        list_generations = [[go.text for go in g.outputs] for g in list_generations]
        list_generations = [self.post_process(g) for g in list_generations]

        outputs = []

        for pi, generations in enumerate(list_generations):
            generations_ed = self.__verify_output__(
                [X[pi]] * len(generations), generations
            )
            filtered_generations = [
                g for g, ed in zip(generations, generations_ed) if ed == 1
            ]

            if len(filtered_generations) > 0:
                outputs.append(random.choice(filtered_generations))
                break
            else:
                outputs.append(X[pi])

        return outputs
