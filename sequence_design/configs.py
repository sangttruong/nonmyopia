SYSTEM_PROMPT = "You are a helpful assistant who works in a protein engineering lab. We are trying to edit a given protein by a sequence of 1-step protein editing, known as mutation. You need to use your knowledge to help me propose suitable protein editing. Going from an initial protein to an optimal one can take many steps."

TEMPLATED_RESPONSE = {
    "llama-3": """{protein}<|eot_id|>""",
    "qwen": """{protein}<|im_end|>\n""",
}

LOOKAHEAD_PROMPT = """Fluorescence level of the above protein: {reward}

Based on the above protein sequence and its fluorescence value, edit 1 amino acid to achieve higher fluorescence. You must only return the modified protein sequence and nothing else.
Modified protein sequence: """

TEMPLATED_LOOKAHEAD_PROMPT = {
    "llama-3": f"<|start_header_id|>user<|end_header_id|>\n\n{LOOKAHEAD_PROMPT}<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n",
    "qwen": f"<|im_start|>user\n{LOOKAHEAD_PROMPT}<|im_end|>\n<|im_start|>assistant\n",
}
