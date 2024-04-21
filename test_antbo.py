import re
import torch
import numpy as np
from itertools import groupby
from _9_semifuncs import AntBO

n_suggestions = 1
seq_length = 11
nm_AAs = 20
bbox = {
    "tool": "Absolut",
    "antigen": "1ADQ_A",
    # Put path to Absolut (/ABS/PATH/TO/Absolut/)
    "path": "/Users/ducnguyen/Research/Absolut",
    "process": 4,  # Number of cores
    "startTask": 0,  # start core id
}
device = "cpu"
# antigens = ['1ADQ_A', '1FBI_X', '1HOD_C', '1NSN_S', '1OB1_C', '1WEJ_F', '2YPV_A', '3RAJ_A', '3VRL_C', '2DD8_S', '1S78_B', '2JEL_P']

# ==============================================================================
# Support functions and variables
# ==============================================================================

COUNT_AA = 5  # maximum number of consecutive AAs
N_glycosylation_pattern = "N[^P][ST][^P]"

# Possible amino acids
aas = "ACDEFGHIKLMNPQRSTVWY"

# Mapping from amino acids to indices
aa_to_idx = {aa: idx for idx, aa in enumerate(aas)}

# Mapping from indices to amino acids
idx_to_aa = {i: aa for aa, i in aa_to_idx.items()}


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
    constraints_satisfied = list(
        map(lambda seq: check_constraint_satisfaction(seq), x))
    return np.array(constraints_satisfied)


# ==============================================================================
# Main code
# ==============================================================================

# Initialize random X
X_next = np.random.randint(
    low=0, high=nm_AAs, size=(n_suggestions, seq_length))
# >> 11 x 1
# Vocab: 20

# Check for constraint violation
constraints_violated = np.logical_not(
    check_constraint_satisfaction_batch(X_next))

# Continue until all samples satisfy the constraints
while np.sum(constraints_violated) != 0:
    # Generate new samples for the ones that violate the constraints
    X_next[constraints_violated] = np.random.randint(
        low=0, high=nm_AAs, size=(np.sum(constraints_violated), seq_length)
    )

    # Check for constraint violation
    constraints_violated = np.logical_not(
        check_constraint_satisfaction_batch(X_next))


f_ = AntBO(
    device=device,
    n_categories=np.array([nm_AAs] * seq_length),
    seq_len=seq_length,
    bbox=bbox,
    normalise=False,
)

y = f_(torch.tensor(X_next, device=device))
print("y =", y.item())
