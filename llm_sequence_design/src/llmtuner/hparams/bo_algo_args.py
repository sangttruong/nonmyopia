from dataclasses import asdict, dataclass, field
from typing import Literal, Optional, List


@dataclass
class BOArguments:
    r"""
    Arguments pertaining to the BO hyperparameters.
    """

    algos: List[str] = field(
        default=["HES"],
        metadata={
            "help": "The list of algorithms to use for BO. \
                    Choices: ['HES', 'qKG', 'qEI', 'qPI', 'qSR', 'qUCB', 'qMSL', 'qNIPV']"
        },
    )

    algo_ts: bool = field(
        default=True,
        metadata={
            "help": "Whether to use Thompson sampling for BO."
        },
    )

    algo_n_iterations: int = field(
        default=None,
        metadata={
            "help": "The number of iterations for the BO algorithm."
        },
    )

    algo_lookahead_steps: int = field(
        default=None,
        metadata={
            "help": "The number of lookahead steps for the BO algorithm."
        },
    )

    cost_spotlight_k: int = field(
        default=100,
        metadata={
            "help": "The k factor to use for spotlighting the cost."
        },
    )

    cost_p_norm: float = field(
        default=2.0,
        metadata={
            "help": "The p-norm to use for the cost."
        },
    )

    cost_max_noise: float = field(
        default=1e-5,
        metadata={
            "help": "The maximum noise to use for the cost."
        },
    )

    cost_discount: float = field(
        default=0.0,
        metadata={
            "help": "The discount factor to use for the cost."
        },
    )

    cost_discount_threshold: float = field(
        default=-1,
        metadata={
            "help": "The discount threshold to use for the cost."
        },
    )
