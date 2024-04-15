import torch
import numpy as np
import re
import numpy as np
from itertools import groupby
from abc import abstractmethod
from _8_tools import Absolut

nm_AAs = 20

COUNT_AA = 5  # maximum number of consecutive AAs
N_glycosylation_pattern = "N[^P][ST][^P]"

# Possible amino acids
aas = "ACDEFGHIKLMNPQRSTVWY"

# Mapping from amino acids to indices
aa_to_idx = {aa: idx for idx, aa in enumerate(aas)}

# Mapping from indices to amino acids
idx_to_aa = {i: aa for aa, i in aa_to_idx.items()}


class TestFunction:
    """
    The abstract class for all benchmark functions acting as objective functions for BO.
    Note that we assume all problems will be minimization problem, so convert maximisation problems as appropriate.
    """

    # this should be changed if we are tackling a mixed, or continuous problem, for e.g.
    problem_type = "categorical"

    def __init__(self, normalise=True, **kwargs):
        self.normalise = normalise
        self.n_vertices = None
        self.config = None
        self.dim = None
        self.continuous_dims = None
        self.categorical_dims = None
        self.int_constrained_dims = None

    def _check_int_constrained_dims(self):
        if self.int_constrained_dims is None:
            return
        assert self.continuous_dims is not None, (
            "int_constrained_dims must be a subset of the continuous_dims, "
            "but continuous_dims is not supplied!"
        )
        int_dims_np = np.asarray(self.int_constrained_dims)
        cont_dims_np = np.asarray(self.continuous_dims)
        assert np.all(np.in1d(int_dims_np, cont_dims_np)), (
            "all continuous dimensions with integer "
            "constraint must be themselves contained in the "
            "continuous_dimensions!"
        )

    @abstractmethod
    def compute(self, x, normalise=None):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)


class AntBO(TestFunction):
    """
    AntBO Class
    """

    # this should be changed if we are tackling a mixed, or continuous problem, for e.g.
    problem_type = "categorical"

    def __init__(self, n_categories, seq_len, bbox=None, normalise=True):
        super(AntBO, self).__init__(normalise)
        self.dtype = torch.float32
        self.device = "cpu"
        self.bbox = bbox
        self.n_vertices = n_categories
        self.config = self.n_vertices
        self.dim = seq_len
        self.categorical_dims = np.arange(self.dim)
        if self.bbox["tool"] == "Absolut":
            self.fbox = Absolut(self.bbox)
        else:
            assert 0, f"{self.config['tool']} Not Implemented"

    def __call__(self, x, negate=True):
        """
        x: categorical vector
        """
        energy, _ = self.fbox.Energy(x)
        energy = torch.tensor(energy, dtype=self.dtype).to(self.device)

        # Negate engergy, because our settings is maximization
        if negate:
            energy = -energy

        return energy

    def idx_to_seq(self, x):
        seqs = []
        for seq in x:
            seqs.append("".join(self.fbox.idx_to_AA[int(aa)] for aa in seq))
        return seqs

    def to(self, dtype, device):
        self.dtype = dtype
        self.device = device
        return self


def check_constraint_satisfaction(x):
    # Constraints on CDR3 sequence

    aa_seq = "".join(idx_to_aa[int(aa)] for aa in x)

    # Charge of AA
    # prot = ProteinAnalysis(aa_seq)
    # charge = prot.charge_at_pH(7.4)

    # check constraint 1 : charge between -2.0 and 2.0
    charge = 0
    for char in aa_seq:
        charge += (
            int(char == "R" or char == "K")
            + 0.1 * int(char == "H")
            - int(char == "D" or char == "E")
        )

    if charge > 2.0 or charge < -2.0:
        return False

    # check constraint 2 : does not contain N-X-S/T pattern. This looks for the single letter code N, followed by any
    # character that is not P, followed by either an S or a T, followed by any character that is not a P. Source
    # https://towardsdatascience.com/using-regular-expression-in-genetics-with-python-175e2b9395c2
    if re.search(N_glycosylation_pattern, aa_seq):
        return False

    # check constraint 3 : any amino acid should not repeat more than 5 times
    # Maximum number of the same subsequent AAs
    count = max([sum(1 for _ in group) for _, group in groupby(x)])
    if count > COUNT_AA:
        return False

    # # Check the instability index
    # prot = ProteinAnalysis(aa_seq)
    # instability = prot.instability_index()
    # if (instability > 40):
    #     return False

    return True


def check_constraint_satisfaction_batch(x):
    constraints_satisfied = list(map(lambda seq: check_constraint_satisfaction(seq), x))
    return np.array(constraints_satisfied)


def generate_random_X(n, seq_length):
    # Initialize random X
    X_next = np.random.randint(low=0, high=nm_AAs, size=(n, seq_length))
    # >> 11 x 20

    # Check for constraint violation
    constraints_violated = np.logical_not(check_constraint_satisfaction_batch(X_next))

    # Continue until all samples satisfy the constraints
    while np.sum(constraints_violated) != 0:
        # Generate new samples for the ones that violate the constraints
        X_next[constraints_violated] = np.random.randint(
            low=0, high=nm_AAs, size=(np.sum(constraints_violated), seq_length)
        )

        # Check for constraint violation
        constraints_violated = np.logical_not(
            check_constraint_satisfaction_batch(X_next)
        )

    return torch.tensor(X_next)
