import pickle
import re

import joblib
import numpy as np
import torch
import yaml
from botorch.acquisition import (
    qExpectedImprovement,
    qKnowledgeGradient,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
)
from datasets import Dataset
from embed_text_package.embed_text import Embedder
from llmppo.reward_model import RewardModelTemplate
from torch.utils.data import DataLoader


class Acqf(RewardModelTemplate):
    def __init__(self, config):
        with open(config, "r", encoding="utf8") as stream:
            self.config = yaml.safe_load(stream)
        self.load()

    async def compute(self, messages):
        """
        This is implementation of your reward model.
        """
        messages = [self.post_process(x) for x in messages]
        ds = Dataset.from_dict({"text": messages})
        ds_emb = (
            self.embedder.get_embeddings(
                DataLoader(ds, batch_size=1),
                self.embedder.which_model,
                ["text"],
            )
            .data["text"]
            .to_pylist()
        )
        return self.bo_acqf(torch.Tensor(ds_emb).unsqueeze(-2))

    def load(self):
        """
        If you want to load something
        """
        self.embedder = Embedder()
        self.embedder.load("google/gemma-7b")

        self.model = joblib.load(f"{self.config['output_dir']}/reward_model.joblib")

    def unload(self):
        """
        If you want to unload something
        """
        self.embedder.unload()

    def post_process(self, generation):
        # Pick the lastest protein-like sequence
        return re.findall("[A-Z]{230,}", generation)[-1]


class qKG(Acqf):
    def __init__(self, config):
        super().__init__(config)
        self.bo_acqf = qKnowledgeGradient(
            model=self.model,
            num_fantasies=1,
        )


class qEI(Acqf):
    def __init__(self, config):
        super().__init__(config)
        buffer_file = (
            self.config["output_dir"][: self.config["output_dir"].rfind("/")]
            + "/buffer.pkl"
        )
        buffer = pickle.load(open(buffer_file, "rb"))
        best_f = np.array(buffer["y"])
        best_f = np.max(best_f, axis=0)
        self.bo_acqf = qExpectedImprovement(
            model=self.model, best_f=torch.tensor(best_f)
        )


class qPI(Acqf):
    def __init__(self, config):
        super().__init__(config)
        buffer_file = (
            self.config["output_dir"][: self.config["output_dir"].rfind("/")]
            + "/buffer.pkl"
        )
        buffer = pickle.load(open(buffer_file, "rb"))
        best_f = np.array(buffer["y"])
        best_f = np.max(best_f, axis=0)
        self.bo_acqf = qProbabilityOfImprovement(
            model=self.model, best_f=torch.tensor(best_f)
        )


class qSR(Acqf):
    def __init__(self, config):
        super().__init__(config)
        self.bo_acqf = qSimpleRegret(model=self.model)


class qUCB(Acqf):
    def __init__(self, config):
        super().__init__(config)
        self.bo_acqf = qUpperConfidenceBound(model=self.model, beta=0.1)


class qMultiStepHEntropySearch(Acqf):
    def __init__(self, config):
        super().__init__(config)

    def bo_acqf(self, X):
        y = (
            self.model.sample(
                torch.tensor(X),
                sample_size=64,
            )
            .mean(0)
            .squeeze(-1)
            .float()
            .detach()
            .cpu()
        )
        return y
