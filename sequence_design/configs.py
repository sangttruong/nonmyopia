HISTORY_FORMAT = """Protein: {protein} - Fluorescence: {fluorescence}"""

POLICY_PROMPT = """Edit 1 amino acid in the below protein sequence to create a new protein with higher fluorescence. The amino acid must be in set {{L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C, X, B, U, Z, O}}. Please return only the modified protein sequence.
Given the following protein sequences and their corresponding fluorescence value.
{history}
Protein sequence: {protein}
Modified protein sequence: """

ALLOWED_TOKENS = [
    "L",
    "A",
    "G",
    "V",
    "S",
    "E",
    "R",
    "T",
    "I",
    "D",
    "P",
    "K",
    "Q",
    "N",
    "F",
    "Y",
    "M",
    "H",
    "W",
    "C",
    "X",
    "B",
    "U",
    "Z",
    "O",
]
