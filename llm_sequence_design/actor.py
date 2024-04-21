from acqfs import (
    acqf_hes,
)


class Actor:
    def __init__(self, model_args, finetuning_args):
        self.args = model_args
        self.acqf = acqf_hes

    def query(self):
        pass
