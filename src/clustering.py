import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

def kmeans_cluster(X, n_clusters=5, random_state=42):
    # n_init=10 is compatible across scikit-learn versions
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    return km.fit_predict(X)

def save_tsne_plot(X, labels, out_path):
    """
    Saves t-SNE plot into results/latent_visualization/.
    Adjust perplexity safely if dataset is small.
    """
    n = X.shape[0]
    if n < 5:
        raise ValueError("Too few samples to run t-SNE. Need at least ~5 samples.")

    # perplexity must be < n_samples
    perplexity = min(30, max(2, (n - 1) // 3))

    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    X2 = tsne.fit_transform(X)

    plt.figure()
    plt.scatter(X2[:, 0], X2[:, 1], c=labels, s=10)
    plt.title("t-SNE visualization of clusters (VAE latent)")
    plt.tight_layout()

    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    plt.savefig(out_path, dpi=200)
    plt.close()
