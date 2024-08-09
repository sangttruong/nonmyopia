import re
import os
import random
import editdistance
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer, GenerationConfig
from vllm import LLM, SamplingParams

from configs import HISTORY_FORMAT, POLICY_PROMPT


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

    def format_prompt(
        self, X: List[str], prevX: List[List[str]], prevY: List[List[float]]
    ):
        # prevX, prevY: n_steps x n_protein
        histories = [[] for _ in range(len(X))]
        for stepX, stepY in zip(prevX, prevY):
            for i, (pX, pY) in enumerate(zip(stepX, stepY)):
                histories[i].append(
                    self.__history__.format(protein=pX, fluorescence=pY)
                )

        histories = ["\n".join(h) for h in histories]
        prompt = [
            self.__prompt__.format(history=h, protein=x) for h, x in zip(histories, X)
        ]
        # prompt = [self.__prompt__.format(protein=x) for x in X]
        return prompt

    def post_process(self, generation):
        return re.findall("[A-Z]{230,}", generation)

    def generate(
        self,
        X: List[str],
        prevX: List[List[str]],
        prevY: List[List[float]],
        max_retry=4,
        **kwargs,
    ):
        prompts = self.format_prompt(X, prevX, prevY)
        outputs = []

        for pi, prompt in enumerate(tqdm(prompts)):
            trying_time = 0
            while trying_time < max_retry:
                prompt = self.tokenizer.apply_chat_template(
                    [{"role": "user", "content": prompt}], tokenize=False
                )

                generations = self.model.generate(prompt, self.sampling_params)
                generations = [g.outputs[0].text for g in generations][0]
                generations = self.post_process(generations)

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
                    trying_time += 1

            if trying_time == max_retry:
                outputs.append(X[pi])

        return outputs
