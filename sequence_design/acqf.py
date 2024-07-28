import json
import joblib
from llmppo.reward_model import RewardModelTemplate

class EI(RewardModelTemplate):
    def __init__(self):
        self.load()
        
    async def compute(self, messages):
        """
        This is implementation of your reward model.
        """
        return self.bo_acqf(messages)

    def load(self):
        """
        If you want to load something
        """
        self.model = joblib.load("ckpts/reward_model/model.joblib")
        self.results = json.load("ckpts/results.json")
        self.bo_acqf = qExpectedImprovement(
            model=self.model,
            best_f=self.results["best_f"]
        )

    def unload(self):
        """
        If you want to unload something
        """
        pass