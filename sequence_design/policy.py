import os
import random
import re
from typing import List

import editdistance

from transformers import AutoTokenizer, GenerationConfig
from utils import format_prompt
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

    def post_process(self, generations: List[str]):
        outputs = []
        for generation in generations:
            outputs.extend(re.findall("[A-Z]{220,}", generation))
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
        self.sampling_params.best_of = max_retry
        prompts = format_prompt(self.tokenizer, prevX, prevY)

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
