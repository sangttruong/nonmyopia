import re
import time
import random
import requests
import editdistance
from typing import List
from tqdm import tqdm
from transformers import AutoTokenizer

from openai import OpenAI
from configs import HISTORY_FORMAT, POLICY_PROMPT
from utils import run_server, shutdown_server


class Policy:
    def __init__(self, config):
        self.config = config
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.config.policy_model_name_or_path,
            use_fast=self.config.use_fast_tokenizer,
        )
        self.__prompt__ = POLICY_PROMPT
        self.__history__ = HISTORY_FORMAT

    def __verify_output__(self, X: List[str], Y: List[str]):
        return [editdistance.eval(x, y) for x, y in zip(X, Y)]

    def load_inference(self):
        api_port = 1337
        deploy_command = f"""API_PORT={api_port} python src/api_demo.py \
                            --model_name_or_path {self.config.policy_model_name_or_path} \
                            --infer_backend vllm \
                            --vllm_gpu_util {self.config.vllm_gpu_util} \
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
        # Shutdown server
        shutdown_server(server_process)

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
        api_port = 1337
        outputs = []

        client = OpenAI(
            base_url=f"http://localhost:{api_port}/v1", api_key="token-abc123"
        )

        for pi, prompt in enumerate(tqdm(prompts)):
            trying_time = 0
            while trying_time < max_retry:
                completion = client.chat.completions.create(
                    model=self.config.policy_model_name_or_path,
                    messages=[{"role": "user", "content": prompt}],
                )
                generations = completion.choices[0].message.content
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
