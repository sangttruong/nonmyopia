import collections
import pickle
import random

import joblib
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import torch
from datasets import Dataset, load_dataset
from tqdm import tqdm
from tueplots import bundles

plt.rcParams.update(bundles.neurips2024())

# # Example adjacency matrix
# oracle = joblib.load("ckpts/oracle/model.joblib")
# adj_matrix = pickle.load(
#     open("dist_mat.pkl", "rb")
# )
# ds = load_dataset("stair-lab/proteinea_fluorescence-gemma-7b-embedding")
# # node_values = np.array(ds["train"]["reward"])

# selected_idx = [i for i, s in enumerate(ds["train"]["text"]) if len(s) == 237]
# remove_idx = list(set(range(adj_matrix.shape[0])) - set(selected_idx))
# selected_ds = ds["train"].select(selected_idx)
# adj_matrix = np.delete(adj_matrix, remove_idx, axis=1)
# adj_matrix = np.delete(adj_matrix, remove_idx, axis=0)

# adj_matrix[adj_matrix > 1] = 0
# col_idx = np.argwhere(np.all(adj_matrix[..., :] == 0, axis=0))
# row_idx = np.argwhere(np.all(adj_matrix== 0, axis=1))
# selected_idx = list(set(range(adj_matrix.shape[0])) - set(col_idx.flatten().tolist()))

# adj_matrix = np.delete(adj_matrix, col_idx, axis=1)
# adj_matrix = np.delete(adj_matrix, row_idx, axis=0)

# selected_ds = selected_ds.select(selected_idx)
# node_values = oracle.predict(
#         torch.Tensor(selected_ds["inputs_embeds"])
# ).numpy()
# node_values = (1/(1+np.exp(-node_values))) * 6 - 3
# min_value = node_values.min()
# # adj_matrix = adj_matrix[:100, :100]
# # node_values = node_values[:100]

# # Convert adjacency matrix to a NetworkX graph
# G = nx.from_numpy_array(adj_matrix)
# selected_ds = (
#     selected_ds.remove_columns("inputs_embeds").remove_columns("reward")
#     .add_column("reward", node_values)
# )
# node_attr_dict = selected_ds.to_pandas().to_dict(orient='index')
# nx.set_node_attributes(G, node_attr_dict)

# nx.write_gexf(G, "dist_mat.gexf")
# print("DONE")


# # removed_node_idx = []
# # for component in list(nx.connected_components(G)):
# #     if len(component) < 10:
# #         for node in component:
# #             removed_node_idx.append(node)
# #             G.remove_node(node)


def find_node_with_text(graph, text_value):
    for node, data in graph.nodes(data=True):
        if data.get("text") == text_value:
            return node
    return None


# K_steps = 5
# init_seq = "SKGEELFTGVVPILVELGGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRFPDHMKQHDFFKSAMPEGYVQERTIFSKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEDGNILGHKLEYNYNSHNVYIMADDQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDEHYLSTQSALSKDDNEERDEMVLLEFVTAAGITHGMDELYK"

# K_steps = 8
# init_seq = "SKGEELFTGVVPILVELGGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRFPDHMKQHDFFKSAMPEGYVQERTIFSKDDGNYKTRAEVKFEGDTLVNRIELKGIDFKEEDNILGHKLEENYNSHNVYIMADDQKNGIKVNFKIRHNIEDGSVQLADHYQQNTPIGDGPVLLPDDHYLSTQSALSKDDNEDRDEMVLLEFVTAAGITHGMDELYK"

K_steps = 12
init_seq = "SKGEELFTGVVPILVELGGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRFPDHMKQHDFFKSAMPEGYVQERTIFSKDDGNYKTRAEVKFEGDELVNRIELKGIDFKEEENILGHKLEENYNSHNVYIMADDQKNGIKVNFKIRHNIEDDSVQLADHYQQNTPIGDEPVLLPDDHYLSTQSALSKDDNEDRDEMVLLEFVTAAGITHGMDELYK"

# K_steps = 14
# init_seq = "SKGEELFTGVVPILVELGGDVNGHKFSVSGEGEGDATYGKLTLKFICTTGKLPVPWPTLVTTLSYGVQCFSRFPDHMKQHDFFKSAMPEGYVQERTIFSKDEGNYKTRAEVKFEGDELVNRIELKGIDFKEDENILGHKLEDNYNSHNVYIMADDQKNGIKVNFKIRHNIEDDSVQLADDYQQNTPIGDDPVLLPDEHYLSTQSALSKDDNEDRDEMVLLEFVTAAGITHGMDELYK"

G = nx.read_gexf(f"mutants/mutants_2p{K_steps}.gexf")

max_value = -100
min_value = 100
min_node = None
max_node = None
node_values = []
for node in G.nodes:
    node_values.append(G.nodes[node]["reward"])
    if G.nodes[node]["reward"] < min_value:
        min_value = G.nodes[node]["reward"]
        min_node = node

    if G.nodes[node]["reward"] > max_value:
        max_value = G.nodes[node]["reward"]
        max_node = node

print("Max ID:", max_node)
print("Max seq:", G.nodes[max_node]["text"])
print("Max reward:", G.nodes[max_node]["reward"])

start_node = find_node_with_text(G, init_seq)
print("Init ID:", start_node)
print("Init seq:", G.nodes[start_node]["text"])
print("Init reward:", G.nodes[start_node]["reward"])


# Compute mean and std of rewards of all nodes that are K steps away from the start node
list_rws = []
hop_nodes = {}
visited = []
truncating = False
f = lambda x: 0.0025 * (x - 0.25) * (x - 2.25) * (x - 7) * (x - 11.25) * (x - 10)
# f = lambda x: 0.02 * (x - 1) * (x - 7) * (x - 10)

reward_by_node = {}
for hop in tqdm(range(0, K_steps + 1)):
    if hop == 0:
        reward_by_node[start_node] = G.nodes[start_node]["reward"] * 1 / 8 + f(hop)
        list_rws.append([reward_by_node[start_node]])
        hop_nodes[start_node] = 1
        print("New initial rw:", reward_by_node[start_node])
    else:
        # Find all nodes that are K steps away from the start node by using the BFS tree
        next_nodes = {}
        rewards = []
        if truncating:
            div_factor = max(hop_nodes.values()) / 256
        else:
            div_factor = 1
        for node, freq in hop_nodes.items():
            freq = int(freq // div_factor)
            if freq < 1:
                freq = 1

            # next_nodes[node] = freq
            # rewards.extend([reward_by_node[node]] * freq) # Stop at current

            for neighbor in G.neighbors(node):
                if neighbor not in hop_nodes and neighbor not in visited:
                    if neighbor not in next_nodes:
                        next_nodes[neighbor] = freq
                    else:
                        next_nodes[neighbor] += freq
                    if neighbor not in reward_by_node:
                        reward_by_node[neighbor] = G.nodes[neighbor][
                            "reward"
                        ] * 1 / 5 + f(hop)
                    rewards.extend([reward_by_node[neighbor]] * freq)

        visited = list(set(visited + list(hop_nodes.keys())))
        hop_nodes = next_nodes
        print("Hop:", sum(hop_nodes.values()))
        if sum(hop_nodes.values()) > 32768:
            truncating = True
        # Compute mean and std of rewards of all nodes that are K steps away from the start node
        list_rws.append(rewards)

import pickle

pickle.dump(reward_by_node, open("updated_rewards.pkl", "wb"))
# Plot the mean and std of rewards of all nodes that are K steps away from the start node
plt.figure(figsize=(10, 5))
plt.violinplot(list_rws, showmeans=False, showmedians=False, showextrema=True)
plt.xlabel("Edit distance", fontsize=18)
plt.ylabel("Flourecescence Level", fontsize=18)
plt.tick_params(labelsize=18)
plt.legend()
plt.savefig(f"plots/mutants_2p{K_steps}_hop.pdf")
plt.close()


# Update node rw
node_values = []
min_node = None
min_value = 100
for node in G.nodes:
    G.nodes[node]["reward"] = reward_by_node[node]
    node_values.append(reward_by_node[node])

    if G.nodes[node]["reward"] < min_value:
        min_value = G.nodes[node]["reward"]
        min_node = node

print(G.nodes[start_node]["reward"])
print(G.nodes[max_node]["reward"])
# Embed the graph in 2D space
pos = nx.bfs_layout(G, start=start_node)

# Normalize node values for color mapping
cmap = plt.get_cmap("coolwarm")

# Plot the graph
print("Drawing ...")
plt.figure(figsize=(50, 25))
nodes = nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_values, cmap=cmap)
edges = nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=2e-2)
cbar = plt.colorbar(nodes, label="Reward")
cbar.ax.tick_params(labelsize=100)
cbar.mappable.set_clim(min_value, 3)
plt.savefig(f"plots/mutants_2p{K_steps}.pdf")


# paths = nx.all_shortest_paths(G, start_node, max_node)
# paths = [x for x in paths]
# num_paths = len(paths)

# print("Drawing...")
# fig = plt.figure(figsize=(10, 5))
# for path in tqdm(paths):
#     idxs = list(range(len(path)))
#     if len(path) < K_steps + 1:
#         idxs[-1] = K_steps
#     plt.plot(idxs, [G.nodes[x]["reward"] for x in path], alpha=1e-4, color="grey")

#     mean = np.array(
# # plt.title(f"Start Node: {start_node}")
# plt.tick_params(axis="both", labelsize=18)
# plt.ylim(-3, 3)
# plt.savefig(f"plots/protein_candidates/{start_node}_np{len(paths)}.png", dpi=300)
# plt.close()


# list_start_nodes = []
# list_paths = {}
# count = 0
# while count < 10:
#     # start_node = str(start_node)
#     start_node = str(random.randint(0, 4095))
#     # start_node = find_node_with_text(G, init_seq)
#     paths = nx.all_shortest_paths(G, start_node, max_node)
#     paths = [x for x in paths]

#     is_4hop = True
#     for path in paths:
#         if len(path) < 11:
#             is_4hop = False
#             break

#     if not is_4hop:
#         continue

#     convex_arr = []
#     list_rws = []
#     for path in paths:
#         is_concave = 1
#         rws = [G.nodes[start_node]["reward"]]
#         for nid1, nid2 in zip(path[:-1], path[1:]):
#             rws.append(G.nodes[nid2]["reward"])
#             if G.nodes[nid1]["reward"] > G.nodes[nid2]["reward"]:
#                 is_concave = 0
#             # if not is_concave:
#             #     break
#         convex_arr.append(is_concave)
#         list_rws.append(rws)
#     convex_arr = np.array(convex_arr)
#     # list_rws = np.array(list_rws)

#     having_valley = np.all(np.array([x[1:10] for x in list_rws]) < 2., axis=1).sum()

#     if np.mean(convex_arr) < 0.0625 and having_valley:
#         num_paths = len(paths)
#         print("Concave Percentage:", np.mean(convex_arr))
#         print("=====================")
#         print(G.nodes[max_node]["text"])
#         print(G.nodes[start_node]["text"])
#         print("Num paths:", num_paths)
#         print("=====================")
#         list_start_nodes.append(start_node)
#         list_paths[start_node] = paths

#         print("Drawing...")
#         fig = plt.figure(figsize=(10, 5))
#         for path in tqdm(paths[:int(num_paths * 0.1)]):
#             idxs = list(range(len(path)))
#             if len(path) < K_steps + 1:
#                 idxs[-1] = K_steps
#             plt.plot(idxs, [G.nodes[x]["reward"] for x in path])
#         # plt.title(f"Start Node: {start_node}")
#         plt.tick_params(axis="both", labelsize=18)
#         plt.ylim(-3, 3)
#         plt.savefig(f"plots/protein_candidates/{start_node}_np{len(paths)}.png", dpi=300)
#         plt.close()

#         count += 1


# Filter 10 node in `list_start_nodes` that have the lowest reward
# list_start_nodes = sorted(
#     list_start_nodes, key=lambda x: G.nodes[x]["reward"])[:10]
# list_paths = {k: v for k, v in list_paths.items() if k in list_start_nodes}
