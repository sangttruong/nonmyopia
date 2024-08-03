import json
import torch
import joblib
from datasets import Dataset
from torch.utils.data import DataLoader
from botorch.acquisition import (
    qExpectedImprovement,
    qKnowledgeGradient,
    qMultiStepLookahead,
    qProbabilityOfImprovement,
    qSimpleRegret,
    qUpperConfidenceBound,
    qNegIntegratedPosteriorVariance,
)
from embed_text_package.embed_text import Embedder
from llmppo.reward_model import RewardModelTemplate


class Acqf(RewardModelTemplate):
    def __init__(self):
        self.load()

    async def compute(self, messages):
        """
        This is implementation of your reward model.
        """
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
        return self.bo_acqf(torch.Tensor(ds_emb))

    def load(self):
        """
        If you want to load something
        """
        self.embedder = Embedder()
        self.embedder.load("google/gemma-7b-it")

        self.model = joblib.load("ckpts/reward_model/model.joblib")

    def unload(self):
        """
        If you want to unload something
        """
        self.embedder.unload()


class EI(Acqf):
    def __init__(self):
        super().__init__()
        self.results = json.load("ckpts/results.json")
        self.bo_acqf = qExpectedImprovement(
            model=self.model, best_f=self.results["best_f"]
        )


class SR(Acqf):
    def __init__(self):
        super().__init__()
        self.bo_acqf = qSimpleRegret(model=self.model)
