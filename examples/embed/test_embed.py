import numpy as np
from sklearn.manifold import MDS
import matplotlib.pyplot as plt


np.set_printoptions(suppress=True)


seed = 11
np.random.seed(seed)


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
    """Return distance matrix."""
    n_node = np.max(link_tups)
    adj_mat = np.zeros((n_node, n_node))
    print(f"link_tups = {link_tups}")
    for tup in link_tups:
        adj_mat[tup[0], tup[1]] = edge_weight
    return adj_mat

# Build distance matrix and embed it
X = get_distance_matrix(n_dim=10)
X_embed = MDS(n_components=2, dissimilarity="precomputed").fit_transform(X)


# Print embedding
print(f"\nX =\n{X}")
print(f"\nX_embed=\n{X_embed}")


# Plotting
plt.plot(X_embed[:, 0], X_embed[:, 1], 'o')
for idx, x in enumerate(X_embed):
    plt.text(x[0], x[1], str(idx))
plt.show()
