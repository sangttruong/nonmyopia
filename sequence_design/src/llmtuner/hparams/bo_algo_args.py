from dataclasses import asdict, dataclass, field
from typing import Literal, Optional, List


@dataclass
class BOArguments:
    r"""
    Arguments pertaining to the BO hyperparameters.
    """

    algo: str = field(
        default="HES",
        metadata={
            "help": "The list of algorithms to use for BO. \
                    Choices: ['HES', 'qKG', 'qEI', 'qPI', 'qSR', 'qUCB', 'qMSL', 'qNIPV']"
        },
    )

    algo_ts: Optional[bool] = field(
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

    initinal_sequences: int = field(
        default=None,
        metadata={
            "help": "The number of initial sequences for training WM."
        },
    )

    n_sequences: int = field(
        default=None,
        metadata={
            "help": "The number of sequences to optimzie."
        },
    )

    n_restarts: int = field(
        default=None,
        metadata={
            "help": "The number of restarts."
        },
    )

    rollout_sequences: int = field(
        default=None,
        metadata={
            "help": "The number of rollout sequence used to train policy."
        },
    )

    cost_spotlight_k: Optional[int] = field(
        default=100,
        metadata={
            "help": "The k factor to use for spotlighting the cost."
        },
    )

    cost_p_norm: Optional[float] = field(
        default=2.0,
        metadata={
            "help": "The p-norm to use for the cost."
        },
    )

    cost_max_noise: Optional[float] = field(
        default=1e-5,
        metadata={
            "help": "The maximum noise to use for the cost."
        },
    )

    cost_discount: Optional[float] = field(
        default=0.0,
        metadata={
            "help": "The discount factor to use for the cost."
        },
    )

    cost_discount_threshold: Optional[float] = field(
        default=-1,
        metadata={
            "help": "The discount threshold to use for the cost."
        },
    )
