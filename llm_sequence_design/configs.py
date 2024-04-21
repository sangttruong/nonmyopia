initinal_samples = 500
samples_per_iteration = 100
number_of_iterations = 20


POLICY_PROMPT = '''Edit 1 amino acid in below protein sequence to create a new protein with higher fluorescence. The amino acid mus be in set {L, A, G, V, S, E, R, T, I, D, P, K, Q, N, F, Y, M, H, W, C, X, B, U, Z, O}. Please return only the modified protein sequence.
Protein sequence: {protein}
Modified protein sequence: '''

TRAINING_DATA_BUFFER = "data/training_data_buffer.pkl"
TESTING_DATA_BUFFER = "data/testing_data_buffer.pkl"
