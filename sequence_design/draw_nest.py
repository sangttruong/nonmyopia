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
from configs import F
from datasets import Dataset, load_dataset
from Levenshtein import distance
from tqdm import tqdm
from tueplots import bundles, figsizes

plt.rcParams.update(bundles.iclr2024())

K_steps = 12


def find_node_with_text(graph, text_value):
    for node, data in graph.nodes(data=True):
        if data.get("text") == text_value:
            return node
    return None


def find_node_with_distance(graph, target_seq):
    list_nodes = []
    for node, data in graph.nodes(data=True):
        if distance(data.get("text"), target_seq) == K_steps:
            list_nodes.append(node)
    return list_nodes


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

start_node = find_node_with_distance(G, G.nodes[max_node]["text"])[0]
init_seq = G.nodes[start_node]["text"]
print("Init ID:", start_node)
print("Init seq:", G.nodes[start_node]["text"])
print("Init reward:", G.nodes[start_node]["reward"])


# Compute mean and std of rewards of all nodes that are K steps away from the start node
list_rws = []
hop_nodes = {}
visited = []
truncating = False

reward_by_node = {}
for hop in tqdm(range(0, K_steps + 1)):
    if hop == 0:
        reward_by_node[start_node] = G.nodes[start_node]["reward"] / 5 + F(hop)
        list_rws.append([reward_by_node[start_node]])
        hop_nodes[start_node] = 1
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
                        ] / 5 + F(hop)
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
overall_size = figsizes.iclr2024(nrows=1, ncols=2)["figure.figsize"]
plt.figure(figsize=[overall_size[0] / 2, overall_size[1]])
plt.violinplot(list_rws, showmeans=False, showmedians=False, showextrema=True)
list_means = []
for nodehop in list_rws:
    mean = np.mean(nodehop)
    list_means.append(mean)
plt.plot(list(range(1, len(list_means)+1)), list_means)
plt.xlabel("Edit distance")
plt.ylabel("Flourecescence Level")
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

print("New initial rw:", G.nodes[start_node]["reward"])
print("New maximal rw:", G.nodes[max_node]["reward"])

# Embed the graph in 2D space
pos = nx.bfs_layout(G, start=start_node)

# Normalize node values for color mapping
cmap = plt.get_cmap("coolwarm")

# Plot the graph
print("Drawing ...")
plt.figure(figsize=[10 * overall_size[0] / 2, 10 * overall_size[1]])
nodes = nx.draw_networkx_nodes(G, pos, node_size=500, node_color=node_values, cmap=cmap)
edges = nx.draw_networkx_edges(G, pos, edge_color="gray", alpha=2e-2)
cbar = plt.colorbar(nodes)
cbar.ax.tick_params(labelsize=50)
cbar.mappable.set_clim(min_value, 3)
plt.box(False)
plt.savefig(f"plots/mutants_2p{K_steps}.pdf")
