initinal_samples = 10
n_sequences = 100


POLICY_PROMPT = '''Edit 1 amino acid in below protein sequence to create a new protein with higher fluorescence. The amino acid mus be in set {{L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C, X, B, U, Z, O}}. Please return only the modified protein sequence.
Protein sequence: {protein}
Modified protein sequence: '''

ALLOWED_TOKENS = ['L', 'A', 'G', 'V', 'S', 'E', 'R', 'T', 'I', 'D', 'P', 'K', 'Q', 'N', 'F', 'Y', 'M', 'H', 'W', 'C', 'X', 'B', 'U', 'Z', 'O']

TRAINING_DATA_BUFFER = "data/training_data_buffer.pkl"
TESTING_DATA_BUFFER = "data/testing_data_buffer.pkl"
