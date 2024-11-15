import argparse
from itertools import product

import joblib
import networkx as nx
import torch
from datasets import Dataset
from Levenshtein import distance
from tqdm import tqdm
from utils import get_embedding_from_server

parser = argparse.ArgumentParser()
parser.add_argument("--hf_org", type=str, required=True)
parser.add_argument("--mutant_ver", type=str, default="v1")
parser.add_argument("--server_url", type=str, default="http://localhost:1337")
parser.add_argument("--batch_size", type=int, default=4)
args = parser.parse_args()

if args.mutant_ver == "v1":
    ENV_SEQ = "SKGEELFTGVVPILVELGGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRFPDHMKQHDFFKSAMPEGYVQERTIFSKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

    ALLOWED_POS = [
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
        116,
    ]

    ALLOWED_AA = [
        "D",
        "E",
    ]

elif args.mutant_ver == "v2":
    ENV_SEQ = "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRYPDHMKQHDFFKSAMPEGYVQERTIFFKDDGNYKTRAEVKFEGDTLVNRIELKGIVFKEDGNTLGHKLEYNYNSHNVYIMADEQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

    ALLOWED_POS = [
        2,
        8,
        11,
        18,
        33,
        52,
        54,
        56,
        114,
        158,
        187,
        190,
    ]

    ALLOWED_AA = [
        "G",
        "P",
    ]

elif args.mutant_ver == "v3":
    ENV_SEQ = "SKGEELFTGVVPILVELDGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCSSRYPDHMKQHDFFKSAMPEGYVQERTLFFKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADKQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDNHYLSTQSALSKDPNEKRDHMVLLEFVTAAGITHGMDELYK"

    ALLOWED_POS = [0, 26, 28, 41, 60, 61, 63, 84, 116, 201, 203, 223]

    ALLOWED_AA = [
        "S",
        "T",
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
            if s_copy[p] not in ALLOWED_AA:
                raise ValueError(f"Invalid amino acid at position {p}")
            s_copy[p] = combo[i]
        # Append the mutated string to the results list
        mutations.append("".join(s_copy))

    return mutations


print("Creating sequences...")
mutants = generate_mutations(ENV_SEQ, ALLOWED_POS, ALLOWED_AA)


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

    if len(list_nids) == args.batch_size or count == G.number_of_nodes():
        mutant_embs = get_embedding_from_server(
            server_url=args.server_url, list_sequences=list_mtts
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
ds.push_to_hub(
    f"{args.hf_org}/semi_synthetic_protein_2p{len(ALLOWED_POS)}_{args.mutant_ver}_gemma_7b"
)

# Set reward as node attribute
nx.set_node_attributes(G, rewards, "reward")

# Save the graph in gexf format
nx.write_gexf(G, f"mutants_2p{len(ALLOWED_POS)}_{args.mutant_ver}.gexf")
