from itertools import product

import joblib
import networkx as nx
import torch
from datasets import Dataset
from Levenshtein import distance
from tqdm import tqdm
from utils import get_embedding_from_server

INITIAL_SEQ = "SKGEELFTGVVPILVELGGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRFPDHMKQHDFFKSAMPEGYVQERTIFSKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

ALLOWED_POS = [
    # 187,205,218,235
    212,
    215,
    209,
    196,
    154,
    132,
    141,
    131,
    171,
    172,
    189,
    116,  # 101, 179,
]

ALLOWED_AA = [
    # "A", "V", "I", "L", "M", "F", "Y", "W"
    "D",
    "E",
]


# Create all possible mutants
def generate_mutations(s, pos, chars):
    s_list = list(s)  # Convert string to list for mutability
    mutations = []

    # Generate all possible combinations of chars for the given positions
    for combo in tqdm(product(chars, repeat=len(pos))):
        # Create a copy of the string list
        s_copy = s_list[:]
        # Replace characters at each position with corresponding char from the combination
        for i, p in enumerate(pos):
            s_copy[p] = combo[i]
        # Append the mutated string to the results list
        mutations.append("".join(s_copy))

    return mutations


print("Creating sequences...")
mutants = generate_mutations(INITIAL_SEQ, ALLOWED_POS, ALLOWED_AA)


# Create a graph
G = nx.Graph()
for nid, mutant in enumerate(mutants):
    G.add_node(nid, text=mutant)

# Add edges if two nodes differ by one amino acid
print("Computing edit distance...")
for i, node1 in tqdm(G.nodes(data=True)):
    for j, node2 in G.nodes(data=True):
        if i == j:
            continue
        diff = 0
        if distance(node1["text"], node2["text"]) == 1:
            G.add_edge(i, j)

reward_model = joblib.load("ckpts/oracle/model.joblib")

# Get the reward for each node
data_dict = {
    "inputs_embeds": [],
    "text": [],
    "reward": [],
}

rewards = {}
list_nids = []
list_mtts = []
count = 0
print("Computing reward...")
for nid in tqdm(G.nodes):
    list_nids.append(nid)
    list_mtts.append(mutants[nid])

    if len(list_nids) == 4 or count == G.number_of_nodes():
        mutant_embs = get_embedding_from_server(
            server_url="http://skampere1:1337", list_sequences=list_mtts
        )
        mutant_ys = (
            reward_model.sample(
                torch.tensor(mutant_embs).unsqueeze(-2),
                sample_size=64,
            )
            .mean(0)
            .squeeze(-1)
            .float()
            .detach()
            .cpu()
        )
        for ni, mt, emb, y in zip(list_nids, list_mtts, mutant_embs, mutant_ys):
            rewards[ni] = y.item()
            data_dict["inputs_embeds"].append(emb)
            data_dict["text"].append(mt)
            data_dict["reward"].append(y.item())

        count += len(list_nids)
        list_nids = []
        list_mtts = []

ds = Dataset.from_dict(data_dict)
ds = ds.train_test_split(test_size=0.1)
ds.push_to_hub(f"stair-lab/semi_synthetic_protein_2p{len(ALLOWED_POS)}_gemma_7b")

# Set reward as node attribute
nx.set_node_attributes(G, rewards, "reward")

# Save the graph in gexf format
nx.write_gexf(G, f"mutants_2p{len(ALLOWED_POS)}.gexf")
