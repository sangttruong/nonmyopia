import dill as pickle
from botorch.test_functions.synthetic import Ackley


def make(env_name, x_dim, bounds):
    if env_name == "Ackley":
        f_ = Ackley(dim=x_dim, negate=False)
        f_.bounds[0, :].fill_(bounds[0])
        f_.bounds[1, :].fill_(bounds[1])
        return f_

    elif env_name == "chemical":
        with open("examples/semisynthetic.pt", "rb") as file_handle:
            return pickle.load(file_handle)
    else:
        raise NotImplementedError
