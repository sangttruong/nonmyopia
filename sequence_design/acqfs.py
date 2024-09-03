import pickle
import re

import editdistance

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
from lampo.reward_model import RewardModelTemplate
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
        print(messages)
        messages = [self.post_process(x[-1]) for x in messages]
        sequences = [x[0] for x in messages]
        num_sequences = [x[1] for x in messages]

        ds = Dataset.from_dict({"text": sequences})
        ds_emb = (
            self.embedder.get_embeddings(
                DataLoader(ds, batch_size=1),
                self.embedder.which_model,
                ["text"],
            )
            .data["text"]
            .to_pylist()
        )
        output = self.bo_acqf(torch.Tensor(ds_emb).unsqueeze(-2))
        for i, ns in enumerate(num_sequences):
            if ns != 1:
                output[i] = -10
        return output

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
        pp_generation = re.findall("[A-Z]{230,}", generation)
        return pp_generation[-1], len(pp_generation)


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


def spotlight_cost_fn(msg) -> bool:
    """
    This function checks whether the generated sequences differ from latest sequence by 1 character or not.

    msg: List[str]: A converasation containing all rounds. E.g. [q1, a1, q2, a2]
    """
    # Take the latest two sequences
    latest_sequence = re.findall("[A-Z]{230,}", msg[-1])
    latest_sequence = latest_sequence[0] if len(latest_sequence) == 1 else ""
    if len(msg) > 3:
        msg_idx = -3
    else:
        msg_idx = 0
    semi_latest_sequence = re.findall("[A-Z]{230,}", msg[msg_idx])
    semi_latest_sequence = semi_latest_sequence[-1] if semi_latest_sequence else ""

    if latest_sequence == "" and semi_latest_sequence == "":
        return 0

    return editdistance.eval(latest_sequence, semi_latest_sequence) == 1
