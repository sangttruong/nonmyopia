import numpy as np
from pathlib import Path
import pickle
from sklearn.manifold import MDS, LocallyLinearEmbedding
import sklearn.gaussian_process as gp
import matplotlib.pyplot as plt
from argparse import ArgumentParser
from src.mols.molecule import Reaction
from src.datasets.loaders import MolSampler
from src.forward_synth import RexgenForwardSynthesizer
from src.mols.mol_functions import get_objective_by_name
np.set_printoptions(suppress=True)


parser = ArgumentParser()
parser.add_argument('--dataset', default='chembl', type=str,
                    help='dataset: chembl or zinc250')
parser.add_argument('--seed', default=42, type=int,
                    help='sampling seed for the dataset')
parser.add_argument('--n_edge', default=100, type=int,
                    help='number of reactions')
parser.add_argument('--init_pool_size', default=10, type=int,
                    help='size of initial pool')
parser.add_argument('--objective', default='qed', type=str,
                    help='which objective function to use: qed or logp')

args = parser.parse_args()
np.random.seed(args.seed)


def get_distance_matrix(n_dim=10):
    """Return distance matrix."""
    default_dist = 10
    close_dist = 0.5
    n_links = 4
    dist_mat = default_dist * np.ones((n_dim, n_dim))
    link_tups = np.random.randint(n_dim, size=(n_links, 2)).tolist()
    print(f"link_tups = {link_tups}")
    for tup in link_tups:
        dist_mat[tup[0], tup[1]] = dist_mat[tup[1], tup[0]] = close_dist
    np.fill_diagonal(dist_mat, 0.0)
    return dist_mat


def get_adjacency_matrix(link_tups, edge_weight=1):
    """Return adjacency matrix."""
    # breakpoint()
    n_node = np.max(link_tups) + 1
    adj_mat = np.zeros((n_node, n_node))
    print(f"link_tups = {link_tups}")
    for tup in link_tups:
        adj_mat[tup[0], tup[1]] = edge_weight
    return adj_mat


obj_func = get_objective_by_name(args.objective)
synth = RexgenForwardSynthesizer()
sampler = MolSampler(args.dataset, sampling_seed=args.seed)
pool = sampler(args.init_pool_size)
link_tups = []
label = [obj_func(i) for i in pool]

for _ in range(args.n_edge):
    outcomes = []
    while not outcomes:
        # choose molecules to cross-over
        mols = np.random.choice(pool, size=2)
        # evolve
        reaction = Reaction(mols)
        try:
            outcomes = synth.predict_outcome(reaction, k=1)
        except RuntimeError as e:
            print('Synthesizer failed, restarting with another subset.')
            outcomes = []
        else:
            if not outcomes:
                print(
                    'Synthesizer returned an empty set of results, restarting with another subset.')

    top_pt = outcomes[0]
    pool.append(top_pt)
    label.append(obj_func(top_pt))

    index_reactant_1 = pool.index(mols[0])
    index_reactant_2 = pool.index(mols[1])
    index_product = pool.index(top_pt)
    link_tups.append([index_reactant_1, index_product])
    link_tups.append([index_reactant_2, index_product])

# Build distance matrix and embed it
X = get_adjacency_matrix(link_tups)
X_embed = LocallyLinearEmbedding(n_components=2).fit_transform(X)

# Print embedding
print(f"\nX =\n{X}")
print(f"\nX_embed=\n{X_embed}")

# Plotting
plt.plot(X_embed[:, 0], X_embed[:, 1], 'o')
for idx, x in enumerate(X_embed):
    plt.text(x[0], x[1], str(idx))

constant_kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3))
rbf_kernel = gp.kernels.RBF(10.0, (1e-3, 1e3))
kernel = constant_kernel * rbf_kernel
model = gp.GaussianProcessRegressor(
    kernel=kernel,
    n_restarts_optimizer=10,
    alpha=0.1,
    normalize_y=True
)
model.fit(X_embed, label)
with open("../semisynthetic.pt", "wb") as file_handle:
    pickle.dump(model, file_handle)

y_pred, std = model.predict(X_embed, return_std=True)
