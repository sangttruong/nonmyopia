HISTORY_FORMAT = """Protein: {protein} - Fluorescence: {fluorescence}"""

POLICY_PROMPT = """Edit 1 amino acid in the below protein sequence to create a new protein with higher fluorescence. The amino acid must be in set {{L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C, X, B, U, Z, O}}. Please return only the modified protein sequence.
Given the following protein sequences and their corresponding fluorescence value.
{history}
Protein sequence: {protein}
Modified protein sequence: """

LOOKAHEAD_PROMPT = "Based on above generated protein, please continue to edit 1 amino acid for higher fluorescence."
TEMPLATED_LOOKAHEAD_PROMPT = {
    "llama-3": f"<|start_header_id|>user<|end_header_id|>\n\n{LOOKAHEAD_PROMPT}<|eot_id|>",
    "qwen": f"<|im_start|>user\n{LOOKAHEAD_PROMPT}<|im_end|>\n",
}

SYSTEM_PROMPT = "You are a helpful assistant who works in a protein engineering lab. We are trying to edit a given protein by a sequence of 1-step protein editing, known as mutation. You need to use your knowledge to help me propose suitable protein editing. Going from an initial protein to an optimal one can take many steps."

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
