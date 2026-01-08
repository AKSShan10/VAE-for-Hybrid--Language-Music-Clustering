# import os
# from dataclasses import dataclass
# from typing import Tuple

# import numpy as np
# import matplotlib.pyplot as plt
# import pandas as pd

# from sklearn.cluster import KMeans
# from sklearn.decomposition import PCA
# from sklearn.manifold import TSNE
# from sklearn.metrics import silhouette_score
# from src.dataset import load_spectrogram_tensor, load_lyrics_map, align_lyrics_to_audio_ids
# from src.vae_conv import build_conv_vae
# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import TfidfVectorizer


# from sklearn.preprocessing import StandardScaler
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.cluster import AgglomerativeClustering, DBSCAN
# from src.evaluation import clustering_metrics_medium
# from src.dataset import load_spectrogram_tensor, load_lyrics_map, align_lyrics_to_audio_ids
# from src.vae_conv import build_conv_vae


# try:
#     import umap
#     HAS_UMAP = True
# except Exception:
#     HAS_UMAP = False

# import tensorflow as tf

# from src.vae import build_vae, VAE
# from src.evaluation import clustering_metrics


# @dataclass
# class TrainConfig:
#     latent_dim: int = 32
#     hidden_dim: int = 256
#     lr: float = 1e-3
#     batch_size: int = 128
#     epochs: int = 150
#     seed: int = 42

#     beta: float = 0.5
#     kl_warmup_steps: int = 600  # tuned for ~999 samples and batch_size=128


# def set_seed(seed: int):
#     np.random.seed(seed)
#     tf.random.set_seed(seed)


# def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
#     mean = X.mean(axis=0, keepdims=True)
#     std = X.std(axis=0, keepdims=True) + 1e-8
#     return (X - mean) / std, mean, std


# def embed_2d(A: np.ndarray, method: str = "tsne", seed: int = 42) -> np.ndarray:
#     method = method.lower()
#     if method == "umap":
#         if not HAS_UMAP:
#             raise RuntimeError("umap-learn not installed. Install it or use t-SNE.")
#         reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed)
#         return reducer.fit_transform(A)

#     perplexity = min(30, max(5, A.shape[0] // 20))
#     tsne = TSNE(n_components=2, random_state=seed, init="pca", perplexity=perplexity)
#     return tsne.fit_transform(A)


# # def save_scatter(points_2d: np.ndarray, labels: np.ndarray, out_path: str, title: str):
# #     os.makedirs(os.path.dirname(out_path), exist_ok=True)
# #     plt.figure()
# #     plt.scatter(points_2d[:, 0], points_2d[:, 1], c=labels, s=10)
# #     plt.title(title)
# #     plt.tight_layout()
# #     plt.savefig(out_path, dpi=200)
# #     plt.close()

# # def save_scatter(
# #     points_2d: np.ndarray,
# #     labels: np.ndarray,
# #     out_path: str,
# #     title: str,
# #     show_centroids: bool = True,
# #     alpha: float = 0.7,
# # ):
# #     os.makedirs(os.path.dirname(out_path), exist_ok=True)

# #     plt.figure(figsize=(7, 6))

# #     scatter = plt.scatter(
# #         points_2d[:, 0],
# #         points_2d[:, 1],
# #         c=labels,
# #         cmap="tab10",
# #         s=14,
# #         alpha=alpha,
# #         edgecolors="none",
# #     )

# #     if show_centroids:
# #         unique_labels = np.unique(labels)
# #         centroids = np.array([
# #             points_2d[labels == k].mean(axis=0)
# #             for k in unique_labels if k != -1
# #         ])
# #         plt.scatter(
# #             centroids[:, 0],
# #             centroids[:, 1],
# #             c="black",
# #             s=120,
# #             marker="x",
# #             linewidths=2,
# #             label="Centroid",
# #         )

# #     plt.title(title, fontsize=12)
# #     plt.xlabel("Component 1")
# #     plt.ylabel("Component 2")
# #     plt.colorbar(scatter, label="Cluster label")

# #     plt.tight_layout()
# #     plt.savefig(out_path, dpi=300)
# #     plt.close()

# #     print("Saved:", out_path)

# def pick_best_k_by_silhouette(Z: np.ndarray, k_min: int = 6, k_max: int = 14, seed: int = 42) -> int:
#     best_k = k_min
#     best_s = -1.0
#     for k in range(k_min, k_max + 1):
#         labels = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(Z)
#         s = silhouette_score(Z, labels)
#         if s > best_s:
#             best_s = s
#             best_k = k
#     print(f"Best k by silhouette in [{k_min},{k_max}]: k={best_k}, silhouette={best_s:.4f}")
#     return best_k



# def save_scatter(
#     points_2d: np.ndarray,
#     labels: np.ndarray,
#     out_path: str,
#     title: str,
#     alpha: float = 0.85,
#     point_size: int = 18,
#     show_colorbar: bool = False,   # set True if you want
# ):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)

#     plt.figure(figsize=(8, 6))
#     sc = plt.scatter(
#         points_2d[:, 0],
#         points_2d[:, 1],
#         c=labels,
#         s=point_size,
#         alpha=alpha,
#         edgecolors="none",
#     )

#     plt.title(title, fontsize=12)
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")

#     # Optional: no colorbar like your example
#     if show_colorbar:
#         plt.colorbar(sc, label="Cluster label")

#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300)
#     plt.close()

#     print("Saved:", out_path)


# # def save_side_by_side(
# #     A_2d: np.ndarray,
# #     A_labels: np.ndarray,
# #     B_2d: np.ndarray,
# #     B_labels: np.ndarray,
# #     titles: tuple[str, str],
# #     out_path: str,
# # ):
# #     os.makedirs(os.path.dirname(out_path), exist_ok=True)

# #     fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

# #     for ax, pts, labels, title in zip(
# #         axes,
# #         [A_2d, B_2d],
# #         [A_labels, B_labels],
# #         titles
# #     ):
# #         sc = ax.scatter(
# #             pts[:, 0],
# #             pts[:, 1],
# #             c=labels,
# #             cmap="tab10",
# #             s=14,
# #             alpha=0.7,
# #         )

# #         # centroids
# #         for k in np.unique(labels):
# #             if k == -1:
# #                 continue
# #             c = pts[labels == k].mean(axis=0)
# #             ax.scatter(c[0], c[1], c="black", s=120, marker="x")

# #         ax.set_title(title)
# #         ax.set_xlabel("Component 1")
# #         ax.set_ylabel("Component 2")

# #     fig.colorbar(sc, ax=axes, label="Cluster label")
# #     plt.tight_layout()
# #     plt.savefig(out_path, dpi=300)
# #     plt.close()

# #     print("Saved:", out_path)

# def save_side_by_side(
#     A_2d: np.ndarray,
#     A_labels: np.ndarray,
#     B_2d: np.ndarray,
#     B_labels: np.ndarray,
#     titles: tuple[str, str],
#     out_path: str,
#     alpha: float = 0.85,
#     point_size: int = 18,
# ):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)

#     fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)

#     for ax, pts, labels, title in zip(
#         axes,
#         [A_2d, B_2d],
#         [A_labels, B_labels],
#         titles
#     ):
#         ax.scatter(
#             pts[:, 0],
#             pts[:, 1],
#             c=labels,
#             s=point_size,
#             alpha=alpha,
#             edgecolors="none",
#         )
#         ax.set_title(title, fontsize=12)
#         ax.set_xlabel("Component 1")
#         ax.set_ylabel("Component 2")

#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300)
#     plt.close()

#     print("Saved:", out_path)




# def pick_best_k_by_silhouette(Z: np.ndarray, k_min: int = 6, k_max: int = 14, seed: int = 42) -> int:
#     best_k = k_min
#     best_s = -1.0
#     for k in range(k_min, k_max + 1):
#         labels = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(Z)
#         s = silhouette_score(Z, labels)
#         if s > best_s:
#             best_s = s
#             best_k = k
#     print(f"Best k by silhouette in [{k_min},{k_max}]: k={best_k}, silhouette={best_s:.4f}")
#     return best_k


# # def run_easy_task(
# #     X_raw: np.ndarray,
# #     n_clusters: int | None = None,   # if None, we choose by silhouette
# #     viz_method: str = "tsne",
# #     results_dir: str = "results",
# # ) -> pd.DataFrame:
# #     cfg = TrainConfig()
# #     set_seed(cfg.seed)

# #     # Standardize
# #     X, _, _ = standardize(X_raw)

# #     # Build + train VAE
# #     encoder, decoder = build_vae(
# #         input_dim=X.shape[1],
# #         latent_dim=cfg.latent_dim,
# #         hidden_dim=cfg.hidden_dim,
# #     )

# #     vae = VAE(encoder, decoder, beta=cfg.beta, kl_warmup_steps=cfg.kl_warmup_steps)
# #     vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr))
# #     vae.fit(X, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=1)

# #     # Latent mean (mu)
# #     mu, logvar, z = encoder.predict(X, batch_size=cfg.batch_size, verbose=0)

# #     # Choose k if not provided
# #     if n_clusters is None:
# #         n_clusters = pick_best_k_by_silhouette(mu, k_min=6, k_max=14, seed=cfg.seed)

# #     # VAE latent + KMeans
# #     km_latent = KMeans(n_clusters=n_clusters, random_state=cfg.seed, n_init=20)
# #     labels_latent = km_latent.fit_predict(mu)
# #     m_latent = clustering_metrics(mu, labels_latent)

# #     # PCA + KMeans baseline (use up to 32 components for fairness)
# #     pca = PCA(n_components=min(32, X.shape[1]), random_state=cfg.seed)
# #     X_pca = pca.fit_transform(X)

# #     if n_clusters is None:
# #         # not reached, but kept for logical completeness
# #         n_clusters = pick_best_k_by_silhouette(X_pca, k_min=6, k_max=14, seed=cfg.seed)

# #     km_pca = KMeans(n_clusters=n_clusters, random_state=cfg.seed, n_init=20)
# #     labels_pca = km_pca.fit_predict(X_pca)
# #     m_pca = clustering_metrics(X_pca, labels_pca)

# #     # Visualizations
# #     # mu_2d = embed_2d(mu, method=viz_method, seed=cfg.seed)
# #     # out1 = os.path.join(results_dir, "latent_visualization", f"vae_latent_{viz_method}.png")
# #     # save_scatter(mu_2d, labels_latent, out1, f"VAE Latent(mu) + KMeans, k={n_clusters} ({viz_method.upper()})")

# #     # Xpca_2d = embed_2d(X_pca, method=viz_method, seed=cfg.seed)
# #     # out2 = os.path.join(results_dir, "latent_visualization", f"pca_{viz_method}.png")
# #     # save_scatter(Xpca_2d, labels_pca, out2, f"PCA + KMeans, k={n_clusters} ({viz_method.upper()})")
    
# #     mu_2d = embed_2d(mu, method=viz_method, seed=cfg.seed)
# #     Xpca_2d = embed_2d(X_pca, method=viz_method, seed=cfg.seed)
    
# #     lv_dir = os.path.join(results_dir, "latent_visualization")
    
# #     # Individual plots
# #     out_vae = os.path.join(lv_dir, f"vae_latent_{viz_method}.png")
# #     out_pca = os.path.join(lv_dir, f"pca_{viz_method}.png")
    
# #     save_scatter(
# #         mu_2d,
# #         labels_latent,
# #         out_vae,
# #         f"VAE Latent Space (μ) + KMeans (k={n_clusters}) [{viz_method.upper()}]"
# #     )
    
# #     save_scatter(
# #         Xpca_2d,
# #         labels_pca,
# #         out_pca,
# #         f"PCA + KMeans (k={n_clusters}) [{viz_method.upper()}]"
# #     )
    
# #     # Side-by-side comparison (this is the key improvement)
# #     out_compare = os.path.join(lv_dir, f"compare_vae_vs_pca_{viz_method}.png")
    
# #     save_side_by_side(
# #         mu_2d,
# #         labels_latent,
# #         Xpca_2d,
# #         labels_pca,
# #         titles=(
# #             "VAE Latent Space (μ)",
# #             "PCA Projection"
# #         ),
# #         out_path=out_compare,
# #     )


# #     df = pd.DataFrame([
# #         {"method": f"VAE(mu)+KMeans(k={n_clusters})", **m_latent},
# #         {"method": f"PCA+KMeans(k={n_clusters})", **m_pca},
# #     ])

# #     os.makedirs(results_dir, exist_ok=True)
# #     out_csv = os.path.join(results_dir, "clustering_metrics.csv")
# #     df.to_csv(out_csv, index=False)

# #     # print("Saved:", out_csv)
# #     # print("Saved:", out1)
# #     # print("Saved:", out2)
# #     # return df
# #     df.to_csv(out_csv, index=False)

# #     print("Saved:", out_csv)
# #     print("Saved:", out_vae)
# #     print("Saved:", out_pca)
# #     print("Saved:", out_compare)
    
# #     return df




# def run_easy_task(
#     X_raw: np.ndarray,
#     n_clusters: int | None = None,   # if None, we choose by silhouette
#     viz_method: str = "tsne",         # viz_method set to "tsne" by default (change to "umap" if needed)
#     results_dir: str = "results",
# ) -> pd.DataFrame:
#     cfg = TrainConfig()
#     set_seed(cfg.seed)

#     # Standardize
#     X, _, _ = standardize(X_raw)

#     # Build + train VAE
#     encoder, decoder = build_vae(
#         input_dim=X.shape[1],
#         latent_dim=cfg.latent_dim,
#         hidden_dim=cfg.hidden_dim,
#     )

#     vae = VAE(encoder, decoder, beta=cfg.beta, kl_warmup_steps=cfg.kl_warmup_steps)
#     vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr))
#     vae.fit(X, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=1)

#     # Latent mean (mu)
#     mu, logvar, z = encoder.predict(X, batch_size=cfg.batch_size, verbose=0)

#     # Choose k if not provided
#     if n_clusters is None:
#         n_clusters = pick_best_k_by_silhouette(mu, k_min=6, k_max=14, seed=cfg.seed)

#     # VAE latent + KMeans
#     km_latent = KMeans(n_clusters=n_clusters, random_state=cfg.seed, n_init=20)
#     labels_latent = km_latent.fit_predict(mu)
#     m_latent = clustering_metrics(mu, labels_latent)

#     # PCA + KMeans baseline (use up to 32 components for fairness)
#     pca = PCA(n_components=min(32, X.shape[1]), random_state=cfg.seed)
#     X_pca = pca.fit_transform(X)

#     if n_clusters is None:
#         # not reached, but kept for logical completeness
#         n_clusters = pick_best_k_by_silhouette(X_pca, k_min=6, k_max=14, seed=cfg.seed)

#     km_pca = KMeans(n_clusters=n_clusters, random_state=cfg.seed, n_init=20)
#     labels_pca = km_pca.fit_predict(X_pca)
#     m_pca = clustering_metrics(X_pca, labels_pca)

#     # Visualizations
#     # mu_2d = embed_2d(mu, method=viz_method, seed=cfg.seed)
#     # out1 = os.path.join(results_dir, "latent_visualization", f"vae_latent_{viz_method}.png")
#     # save_scatter(mu_2d, labels_latent, out1, f"VAE Latent(mu) + KMeans, k={n_clusters} ({viz_method.upper()})")

#     # Xpca_2d = embed_2d(X_pca, method=viz_method, seed=cfg.seed)
#     # out2 = os.path.join(results_dir, "latent_visualization", f"pca_{viz_method}.png")
#     # save_scatter(Xpca_2d, labels_pca, out2, f"PCA + KMeans, k={n_clusters} ({viz_method.upper()})")
    
#     mu_2d = embed_2d(mu, method=viz_method, seed=cfg.seed)
#     Xpca_2d = embed_2d(X_pca, method=viz_method, seed=cfg.seed)
    
#     lv_dir = os.path.join(results_dir, "latent_visualization")
    
#     # Individual plots for VAE + KMeans and PCA + KMeans
#     out_vae = os.path.join(lv_dir, f"vae_latent_{viz_method}.png")
#     out_pca = os.path.join(lv_dir, f"pca_{viz_method}.png")
    
#     save_scatter(
#         mu_2d,
#         labels_latent,
#         out_vae,
#         f"VAE Latent Space (μ) + KMeans (k={n_clusters}) [{viz_method.upper()}]"
#     )
    
#     save_scatter(
#         Xpca_2d,
#         labels_pca,
#         out_pca,
#         f"PCA + KMeans (k={n_clusters}) [{viz_method.upper()}]"
#     )
    
#     # t-SNE visualization
#     mu_2d_tsne = embed_2d(mu, method="tsne", seed=cfg.seed)  # Apply t-SNE on VAE latent space
#     Xpca_2d_tsne = embed_2d(X_pca, method="tsne", seed=cfg.seed)  # Apply t-SNE on PCA projection
    
#     out_vae_tsne = os.path.join(lv_dir, "vae_latent_tsne.png")
#     out_pca_tsne = os.path.join(lv_dir, "pca_tsne.png")
    
#     # Save t-SNE plots
#     save_scatter(
#         mu_2d_tsne,
#         labels_latent,
#         out_vae_tsne,
#         f"VAE Latent Space (μ) + KMeans (k={n_clusters}) [t-SNE]"
#     )
    
#     save_scatter(
#         Xpca_2d_tsne,
#         labels_pca,
#         out_pca_tsne,
#         f"PCA + KMeans (k={n_clusters}) [t-SNE]"
#     )
    
#     # Side-by-side comparison (this is the key improvement)
#     out_compare = os.path.join(lv_dir, f"compare_vae_vs_pca_{viz_method}.png")
    
#     save_side_by_side(
#         mu_2d,
#         labels_latent,
#         Xpca_2d,
#         labels_pca,
#         titles=(
#             f"VAE Latent Space (μ) + KMeans (k={n_clusters}) [{viz_method.upper()}]",
#             f"PCA + KMeans (k={n_clusters}) [{viz_method.upper()}]",
#         ),
#         out_path=out_compare,
#     )

#     # Side-by-side comparison for t-SNE
#     out_compare_tsne = os.path.join(lv_dir, f"compare_vae_vs_pca_tsne.png")
    
#     save_side_by_side(
#         mu_2d_tsne,
#         labels_latent,
#         Xpca_2d_tsne,
#         labels_pca,
#         titles=(
#             "VAE Latent Space (μ) [t-SNE]",
#             "PCA Projection [t-SNE]"
#         ),
#         out_path=out_compare_tsne,
#     )

#     # Final metrics DataFrame (with Silhouette, DB, etc.)
#     df = pd.DataFrame([
#         {"method": f"VAE(mu)+KMeans(k={n_clusters})", **m_latent},
#         {"method": f"PCA+KMeans(k={n_clusters})", **m_pca},
#     ])

#     os.makedirs(results_dir, exist_ok=True)
#     out_csv = os.path.join(results_dir, "clustering_metrics.csv")
#     df.to_csv(out_csv, index=False)

#     print("Saved:", out_csv)
#     print("Saved:", out_vae)
#     print("Saved:", out_pca)
#     print("Saved:", out_vae_tsne)
#     print("Saved:", out_pca_tsne)
#     print("Saved:", out_compare)
#     print("Saved:", out_compare_tsne)
    
#     return df




# #Medium
# def _standardize_2d(X: np.ndarray) -> np.ndarray:
#     return StandardScaler().fit_transform(X)


# # def _lyrics_tfidf_embeddings(texts, max_features=5000) -> np.ndarray:
# #     vec = TfidfVectorizer(
# #         max_features=max_features,
# #         stop_words="english",
# #         ngram_range=(1, 2),
# #         min_df=2
# #     )
# #     X = vec.fit_transform(texts)
# #     return X.toarray().astype(np.float32)

# def _lyrics_tfidf_embeddings(texts, max_features=5000) -> np.ndarray:
#     cleaned = [t for t in texts if isinstance(t, str) and t.strip()]

#     if len(set(cleaned)) <= 1:
#         # All lyrics identical or empty → return a constant feature
#         return np.zeros((len(texts), 1), dtype=np.float32)

#     vec = TfidfVectorizer(
#         max_features=max_features,
#         stop_words="english",
#         ngram_range=(1, 2),
#         min_df=1
#     )

#     X = vec.fit_transform(cleaned).toarray().astype(np.float32)

#     out = np.zeros((len(texts), X.shape[1]), dtype=np.float32)
#     j = 0
#     for i, t in enumerate(texts):
#         if isinstance(t, str) and t.strip():
#             out[i] = X[j]
#             j += 1
#     return out


# def _cluster(
#     Z: np.ndarray,
#     method: str,
#     n_clusters: int,
#     dbscan_eps: float,
#     dbscan_min_samples: int
# ) -> np.ndarray:
#     method = method.lower()
#     if method == "kmeans":
#         return KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(Z)
#     if method == "agglomerative":
#         return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(Z)
#     if method == "dbscan":
#         return DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(Z)
#     raise ValueError(f"Unknown clustering method: {method}")



# # def run_medium_task(
# #     audio_dir: str,
# #     lyrics_dir: str | None = None,
# #     max_files: int = 2000,
# #     sr: int = 22050,
# #     duration: float = 30.0,
# #     n_mels: int = 64,
# #     max_frames: int = 256,
# #     latent_dim: int = 32,
# #     beta: float = 0.5,
# #     kl_warmup_steps: int = 800,
# #     epochs: int = 60,
# #     batch_size: int = 32,
# #     n_clusters: int = 10,
# #     clustering_methods=("kmeans", "agglomerative", "dbscan"),
# #     dbscan_eps: float = 0.9,
# #     dbscan_min_samples: int = 10,
# #     true_labels: np.ndarray | None = None,
# # ):
# #     """
# #     Medium Task:
# #       - Conv-VAE on log-mel spectrograms => mu_audio
# #       - Lyrics TF-IDF embeddings (optional) => z_lyrics
# #       - Hybrid fusion: concat([mu_audio, z_lyrics]) then standardize
# #       - Cluster: KMeans, Agglomerative, DBSCAN
# #       - Metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI(if true labels)
# #     Returns:
# #       ids, Z_hybrid, labels_by_method, metrics_by_method, history
# #     """
# #     # 1) Load spectrogram tensor
# #     X_spec, ids = load_spectrogram_tensor(
# #         audio_dir=audio_dir,
# #         max_files=max_files,
# #         sr=sr,
# #         duration=duration,
# #         n_mels=n_mels,
# #         max_frames=max_frames,
# #     )

# #     # 2) Train Conv-VAE
# #     encoder, decoder, cvae = build_conv_vae(
# #         input_shape=(n_mels, max_frames, 1),
# #         latent_dim=latent_dim,
# #         beta=beta,
# #         kl_warmup_steps=kl_warmup_steps,
# #     )
# #     # cvae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
# #     cvae.compile(optimizer=tf.keras.optimizers.Adam(3e-4))

# #     history = cvae.fit(X_spec, epochs=epochs, batch_size=batch_size, verbose=1)

# #     mu_audio, logvar, z = encoder.predict(X_spec, batch_size=batch_size, verbose=0)
# #     mu_audio = _standardize_2d(mu_audio)

# #     # 3) Lyrics embeddings (optional)
# #     if lyrics_dir:
# #         lyr_map = load_lyrics_map(lyrics_dir)
# #         texts = align_lyrics_to_audio_ids(ids, lyr_map)
        
# #         z_lyrics = _lyrics_tfidf_embeddings(texts, max_features=5000)
# #         z_lyrics = _standardize_2d(z_lyrics)
# #         Z_hybrid = np.concatenate([mu_audio, z_lyrics], axis=1)
# #     else:
# #         Z_hybrid = mu_audio

# #     Z_hybrid = _standardize_2d(Z_hybrid)

# #     # 4) Clustering + metrics
# #     labels_by_method = {}
# #     metrics_by_method = {}

# #     for m in clustering_methods:
# #         labels = _cluster(
# #             Z_hybrid,
# #             method=m,
# #             n_clusters=n_clusters,
# #             dbscan_eps=dbscan_eps,
# #             dbscan_min_samples=dbscan_min_samples,
# #         )
# #         labels_by_method[m] = labels
# #         metrics_by_method[m] = clustering_metrics_medium(Z_hybrid, labels, true_labels=true_labels)


# #     return ids, Z_hybrid, labels_by_method, metrics_by_method, history



# def run_medium_task(
#     audio_dir: str,
#     lyrics_dir: str | None = None,
#     max_files: int = 2000,
#     sr: int = 22050,
#     duration: float = 30.0,
#     n_mels: int = 64,
#     max_frames: int = 256,
#     latent_dim: int = 32,
#     beta: float = 0.1,
#     kl_warmup_steps: int = 4000,
#     epochs: int = 60,
#     batch_size: int = 32,
#     n_clusters: int = 10,
#     clustering_methods=("kmeans", "agglomerative", "dbscan"),
#     dbscan_eps: float = 0.35,  # Tuning eps for DBSCAN
#     dbscan_min_samples: int = 10,
#     true_labels: np.ndarray | None = None,
# ):
#     """
#     Medium Task:
#       - Conv-VAE on log-mel spectrograms => mu_audio
#       - Lyrics TF-IDF embeddings (optional) => z_lyrics
#       - Hybrid fusion: concat([mu_audio, z_lyrics]) then standardize
#       - Cluster: KMeans, Agglomerative, DBSCAN
#       - Metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI(if true labels)
#     Returns:
#       ids, Z_hybrid, labels_by_method, metrics_by_method, history
#     """
#     # 1) Load spectrogram tensor
#     X_spec, ids = load_spectrogram_tensor(
#         audio_dir=audio_dir,
#         max_files=max_files,
#         sr=sr,
#         duration=duration,
#         n_mels=n_mels,
#         max_frames=max_frames,
#     )

#     # 2) Train Conv-VAE
#     encoder, decoder, cvae = build_conv_vae(
#         input_shape=(n_mels, max_frames, 1),
#         latent_dim=latent_dim,
#         beta=beta,
#         kl_warmup_steps=kl_warmup_steps,
#     )
#     cvae.compile(optimizer=tf.keras.optimizers.Adam(3e-4))

#     history = cvae.fit(X_spec, epochs=epochs, batch_size=batch_size, verbose=1)

#     mu_audio, logvar, z = encoder.predict(X_spec, batch_size=batch_size, verbose=0)
#     mu_audio = _standardize_2d(mu_audio)

#     # 3) Lyrics embeddings (optional)
#     if lyrics_dir:
#         lyr_map = load_lyrics_map(lyrics_dir)
#         texts = align_lyrics_to_audio_ids(ids, lyr_map)
        
#         z_lyrics = _lyrics_tfidf_embeddings(texts, max_features=5000)
#         z_lyrics = _standardize_2d(z_lyrics)
#         Z_hybrid = np.concatenate([mu_audio, z_lyrics], axis=1)
#     else:
#         Z_hybrid = mu_audio

#     Z_hybrid = _standardize_2d(Z_hybrid)

#     # 4) Clustering + metrics
#     labels_by_method = {}
#     metrics_by_method = {}

#     for m in clustering_methods:
#         labels = _cluster(
#             Z_hybrid,
#             method=m,
#             n_clusters=n_clusters,
#             dbscan_eps=dbscan_eps,
#             dbscan_min_samples=dbscan_min_samples,
#         )

#         # Debugging prints to verify clustering results
#         print(f"[{m}] - unique labels: {np.unique(labels)}")
#         print(f"[{m}] - cluster sizes: {dict(zip(*np.unique(labels, return_counts=True)))}")
#         if m == "dbscan":
#             print(f"[{m}] - noise points: {np.sum(labels == -1)}")  # DBSCAN noise points

#         labels_by_method[m] = labels
#         metrics_by_method[m] = clustering_metrics_medium(Z_hybrid, labels, true_labels=true_labels)

#     return ids, Z_hybrid, labels_by_method, metrics_by_method, history

 
# def pick_best_k_by_silhouette(Z: np.ndarray, k_min: int = 6, k_max: int = 14, seed: int = 42) -> int:
#     best_k = k_min
#     best_s = -1.0
#     for k in range(k_min, k_max + 1):
#         labels = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(Z)
#         s = silhouette_score(Z, labels)
#         if s > best_s:
#             best_s = s
#             best_k = k
#     print(f"Best k by silhouette in [{k_min},{k_max}]: k={best_k}, silhouette={best_s:.4f}")
#     return best_k





import os
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score
from src.dataset import load_spectrogram_tensor, load_lyrics_map, align_lyrics_to_audio_ids
from src.vae_conv import build_conv_vae
from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer


from sklearn.preprocessing import StandardScaler
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import AgglomerativeClustering, DBSCAN
from src.evaluation import clustering_metrics_medium
from src.dataset import load_spectrogram_tensor, load_lyrics_map, align_lyrics_to_audio_ids
from src.vae_conv import build_conv_vae


try:
    import umap
    HAS_UMAP = True
except Exception:
    HAS_UMAP = False

import tensorflow as tf

from src.vae import build_vae, VAE
from src.evaluation import clustering_metrics


@dataclass
class TrainConfig:
    latent_dim: int = 32
    hidden_dim: int = 256
    lr: float = 1e-3
    batch_size: int = 128
    epochs: int = 150
    seed: int = 42

    beta: float = 0.5
    kl_warmup_steps: int = 600  # tuned for ~999 samples and batch_size=128


def set_seed(seed: int):
    np.random.seed(seed)
    tf.random.set_seed(seed)


def standardize(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = X.mean(axis=0, keepdims=True)
    std = X.std(axis=0, keepdims=True) + 1e-8
    return (X - mean) / std, mean, std


def embed_2d(A: np.ndarray, method: str = "tsne", seed: int = 42) -> np.ndarray:
    method = method.lower()
    if method == "umap":
        if not HAS_UMAP:
            raise RuntimeError("umap-learn not installed. Install it or use t-SNE.")
        reducer = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=seed)
        return reducer.fit_transform(A)

    perplexity = min(30, max(5, A.shape[0] // 20))
    tsne = TSNE(n_components=2, random_state=seed, init="pca", perplexity=perplexity)
    return tsne.fit_transform(A)


# def save_scatter(points_2d: np.ndarray, labels: np.ndarray, out_path: str, title: str):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)
#     plt.figure()
#     plt.scatter(points_2d[:, 0], points_2d[:, 1], c=labels, s=10)
#     plt.title(title)
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=200)
#     plt.close()

# def save_scatter(
#     points_2d: np.ndarray,
#     labels: np.ndarray,
#     out_path: str,
#     title: str,
#     show_centroids: bool = True,
#     alpha: float = 0.7,
# ):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)

#     plt.figure(figsize=(7, 6))

#     scatter = plt.scatter(
#         points_2d[:, 0],
#         points_2d[:, 1],
#         c=labels,
#         cmap="tab10",
#         s=14,
#         alpha=alpha,
#         edgecolors="none",
#     )

#     if show_centroids:
#         unique_labels = np.unique(labels)
#         centroids = np.array([
#             points_2d[labels == k].mean(axis=0)
#             for k in unique_labels if k != -1
#         ])
#         plt.scatter(
#             centroids[:, 0],
#             centroids[:, 1],
#             c="black",
#             s=120,
#             marker="x",
#             linewidths=2,
#             label="Centroid",
#         )

#     plt.title(title, fontsize=12)
#     plt.xlabel("Component 1")
#     plt.ylabel("Component 2")
#     plt.colorbar(scatter, label="Cluster label")

#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300)
#     plt.close()

#     print("Saved:", out_path)


def save_scatter(
    points_2d: np.ndarray,
    labels: np.ndarray,
    out_path: str,
    title: str,
    alpha: float = 0.85,
    point_size: int = 18,
    show_colorbar: bool = False,   # set True if you want
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    plt.figure(figsize=(8, 6))
    sc = plt.scatter(
        points_2d[:, 0],
        points_2d[:, 1],
        c=labels,
        s=point_size,
        alpha=alpha,
        edgecolors="none",
    )

    plt.title(title, fontsize=12)
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")

    # Optional: no colorbar like your example
    if show_colorbar:
        plt.colorbar(sc, label="Cluster label")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)


# def save_side_by_side(
#     A_2d: np.ndarray,
#     A_labels: np.ndarray,
#     B_2d: np.ndarray,
#     B_labels: np.ndarray,
#     titles: tuple[str, str],
#     out_path: str,
# ):
#     os.makedirs(os.path.dirname(out_path), exist_ok=True)

#     fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

#     for ax, pts, labels, title in zip(
#         axes,
#         [A_2d, B_2d],
#         [A_labels, B_labels],
#         titles
#     ):
#         sc = ax.scatter(
#             pts[:, 0],
#             pts[:, 1],
#             c=labels,
#             cmap="tab10",
#             s=14,
#             alpha=0.7,
#         )

#         # centroids
#         for k in np.unique(labels):
#             if k == -1:
#                 continue
#             c = pts[labels == k].mean(axis=0)
#             ax.scatter(c[0], c[1], c="black", s=120, marker="x")

#         ax.set_title(title)
#         ax.set_xlabel("Component 1")
#         ax.set_ylabel("Component 2")

#     fig.colorbar(sc, ax=axes, label="Cluster label")
#     plt.tight_layout()
#     plt.savefig(out_path, dpi=300)
#     plt.close()

#     print("Saved:", out_path)

def save_side_by_side(
    A_2d: np.ndarray,
    A_labels: np.ndarray,
    B_2d: np.ndarray,
    B_labels: np.ndarray,
    titles: tuple[str, str],
    out_path: str,
    alpha: float = 0.85,
    point_size: int = 18,
):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=False, sharey=False)

    for ax, pts, labels, title in zip(
        axes,
        [A_2d, B_2d],
        [A_labels, B_labels],
        titles
    ):
        ax.scatter(
            pts[:, 0],
            pts[:, 1],
            c=labels,
            s=point_size,
            alpha=alpha,
            edgecolors="none",
        )
        ax.set_title(title, fontsize=12)
        ax.set_xlabel("Component 1")
        ax.set_ylabel("Component 2")

    plt.tight_layout()
    plt.savefig(out_path, dpi=300)
    plt.close()

    print("Saved:", out_path)




def pick_best_k_by_silhouette(Z: np.ndarray, k_min: int = 6, k_max: int = 14, seed: int = 42) -> int:
    best_k = k_min
    best_s = -1.0
    for k in range(k_min, k_max + 1):
        labels = KMeans(n_clusters=k, random_state=seed, n_init=20).fit_predict(Z)
        s = silhouette_score(Z, labels)
        if s > best_s:
            best_s = s
            best_k = k
    print(f"Best k by silhouette in [{k_min},{k_max}]: k={best_k}, silhouette={best_s:.4f}")
    return best_k


def run_easy_task(
    X_raw: np.ndarray,
    n_clusters: int | None = None,   # if None, we choose by silhouette
    viz_method: str = "tsne",
    results_dir: str = "results",
) -> pd.DataFrame:
    cfg = TrainConfig()
    set_seed(cfg.seed)

    # Standardize
    X, _, _ = standardize(X_raw)

    # Build + train VAE
    encoder, decoder = build_vae(
        input_dim=X.shape[1],
        latent_dim=cfg.latent_dim,
        hidden_dim=cfg.hidden_dim,
    )

    vae = VAE(encoder, decoder, beta=cfg.beta, kl_warmup_steps=cfg.kl_warmup_steps)
    vae.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=cfg.lr))
    vae.fit(X, epochs=cfg.epochs, batch_size=cfg.batch_size, verbose=1)

    # Latent mean (mu)
    mu, logvar, z = encoder.predict(X, batch_size=cfg.batch_size, verbose=0)

    # Choose k if not provided
    if n_clusters is None:
        n_clusters = pick_best_k_by_silhouette(mu, k_min=6, k_max=14, seed=cfg.seed)

    # VAE latent + KMeans
    km_latent = KMeans(n_clusters=n_clusters, random_state=cfg.seed, n_init=20)
    labels_latent = km_latent.fit_predict(mu)
    m_latent = clustering_metrics(mu, labels_latent)

    # PCA + KMeans baseline (use up to 32 components for fairness)
    pca = PCA(n_components=min(32, X.shape[1]), random_state=cfg.seed)
    X_pca = pca.fit_transform(X)

    if n_clusters is None:
        # not reached, but kept for logical completeness
        n_clusters = pick_best_k_by_silhouette(X_pca, k_min=6, k_max=14, seed=cfg.seed)

    km_pca = KMeans(n_clusters=n_clusters, random_state=cfg.seed, n_init=20)
    labels_pca = km_pca.fit_predict(X_pca)
    m_pca = clustering_metrics(X_pca, labels_pca)

    # Visualizations
    # mu_2d = embed_2d(mu, method=viz_method, seed=cfg.seed)
    # out1 = os.path.join(results_dir, "latent_visualization", f"vae_latent_{viz_method}.png")
    # save_scatter(mu_2d, labels_latent, out1, f"VAE Latent(mu) + KMeans, k={n_clusters} ({viz_method.upper()})")

    # Xpca_2d = embed_2d(X_pca, method=viz_method, seed=cfg.seed)
    # out2 = os.path.join(results_dir, "latent_visualization", f"pca_{viz_method}.png")
    # save_scatter(Xpca_2d, labels_pca, out2, f"PCA + KMeans, k={n_clusters} ({viz_method.upper()})")
    
    mu_2d = embed_2d(mu, method=viz_method, seed=cfg.seed)
    Xpca_2d = embed_2d(X_pca, method=viz_method, seed=cfg.seed)
    
    lv_dir = os.path.join(results_dir, "latent_visualization")
    
    # Individual plots
    out_vae = os.path.join(lv_dir, f"vae_latent_{viz_method}.png")
    out_pca = os.path.join(lv_dir, f"pca_{viz_method}.png")
    
    save_scatter(
        mu_2d,
        labels_latent,
        out_vae,
        f"VAE Latent Space (μ) + KMeans (k={n_clusters}) [{viz_method.upper()}]"
    )
    
    save_scatter(
        Xpca_2d,
        labels_pca,
        out_pca,
        f"PCA + KMeans (k={n_clusters}) [{viz_method.upper()}]"
    )
    
    # Side-by-side comparison (this is the key improvement)
    out_compare = os.path.join(lv_dir, f"compare_vae_vs_pca_{viz_method}.png")
    
    save_side_by_side(
        mu_2d,
        labels_latent,
        Xpca_2d,
        labels_pca,
        titles=(
            "VAE Latent Space (μ)",
            "PCA Projection"
        ),
        out_path=out_compare,
    )


    df = pd.DataFrame([
        {"method": f"VAE(mu)+KMeans(k={n_clusters})", **m_latent},
        {"method": f"PCA+KMeans(k={n_clusters})", **m_pca},
    ])

    os.makedirs(results_dir, exist_ok=True)
    out_csv = os.path.join(results_dir, "clustering_metrics.csv")
    df.to_csv(out_csv, index=False)

    # print("Saved:", out_csv)
    # print("Saved:", out1)
    # print("Saved:", out2)
    # return df
    df.to_csv(out_csv, index=False)

    print("Saved:", out_csv)
    print("Saved:", out_vae)
    print("Saved:", out_pca)
    print("Saved:", out_compare)
    
    return df



#Medium
def _standardize_2d(X: np.ndarray) -> np.ndarray:
    return StandardScaler().fit_transform(X)


# def _lyrics_tfidf_embeddings(texts, max_features=5000) -> np.ndarray:
#     vec = TfidfVectorizer(
#         max_features=max_features,
#         stop_words="english",
#         ngram_range=(1, 2),
#         min_df=2
#     )
#     X = vec.fit_transform(texts)
#     return X.toarray().astype(np.float32)

def _lyrics_tfidf_embeddings(texts, max_features=5000) -> np.ndarray:
    cleaned = [t for t in texts if isinstance(t, str) and t.strip()]

    if len(set(cleaned)) <= 1:
        # All lyrics identical or empty → return a constant feature
        return np.zeros((len(texts), 1), dtype=np.float32)

    vec = TfidfVectorizer(
        max_features=max_features,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=1
    )

    X = vec.fit_transform(cleaned).toarray().astype(np.float32)

    out = np.zeros((len(texts), X.shape[1]), dtype=np.float32)
    j = 0
    for i, t in enumerate(texts):
        if isinstance(t, str) and t.strip():
            out[i] = X[j]
            j += 1
    return out

# This one used for the medium task. Will be updated for the hard task.
# def _cluster(
#     Z: np.ndarray,
#     method: str,
#     n_clusters: int,
#     dbscan_eps: float,
#     dbscan_min_samples: int
# ) -> np.ndarray:
#     method = method.lower()
#     if method == "kmeans":
#         return KMeans(n_clusters=n_clusters, random_state=42, n_init=20).fit_predict(Z)
#     if method == "agglomerative":
#         return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(Z)
#     if method == "dbscan":
#         return DBSCAN(eps=dbscan_eps, min_samples=dbscan_min_samples).fit_predict(Z)
#     raise ValueError(f"Unknown clustering method: {method}")



def run_medium_task(
    audio_dir: str,
    lyrics_dir: str | None = None,
    max_files: int = 2000,
    sr: int = 22050,
    duration: float = 30.0,
    n_mels: int = 64,
    max_frames: int = 256,
    latent_dim: int = 32,
    beta: float = 0.5,
    kl_warmup_steps: int = 800,
    epochs: int = 60,
    batch_size: int = 32,
    n_clusters: int = 10,
    clustering_methods=("kmeans", "agglomerative", "dbscan"),
    dbscan_eps: float = 0.9,
    dbscan_min_samples: int = 10,
    true_labels: np.ndarray | None = None,
):
    """
    Medium Task:
      - Conv-VAE on log-mel spectrograms => mu_audio
      - Lyrics TF-IDF embeddings (optional) => z_lyrics
      - Hybrid fusion: concat([mu_audio, z_lyrics]) then standardize
      - Cluster: KMeans, Agglomerative, DBSCAN
      - Metrics: Silhouette, Calinski-Harabasz, Davies-Bouldin, ARI(if true labels)
    Returns:
      ids, Z_hybrid, labels_by_method, metrics_by_method, history
    """
    # 1) Load spectrogram tensor
    X_spec, ids = load_spectrogram_tensor(
        audio_dir=audio_dir,
        max_files=max_files,
        sr=sr,
        duration=duration,
        n_mels=n_mels,
        max_frames=max_frames,
    )

    # 2) Train Conv-VAE
    encoder, decoder, cvae = build_conv_vae(
        input_shape=(n_mels, max_frames, 1),
        latent_dim=latent_dim,
        beta=beta,
        kl_warmup_steps=kl_warmup_steps,
    )
    # cvae.compile(optimizer=tf.keras.optimizers.Adam(1e-3))
    cvae.compile(optimizer=tf.keras.optimizers.Adam(3e-4))

    history = cvae.fit(X_spec, epochs=epochs, batch_size=batch_size, verbose=1)

    mu_audio, logvar, z = encoder.predict(X_spec, batch_size=batch_size, verbose=0)
    mu_audio = _standardize_2d(mu_audio)

    # 3) Lyrics embeddings (optional)
    if lyrics_dir:
        lyr_map = load_lyrics_map(lyrics_dir)
        texts = align_lyrics_to_audio_ids(ids, lyr_map)
        
        z_lyrics = _lyrics_tfidf_embeddings(texts, max_features=5000)
        z_lyrics = _standardize_2d(z_lyrics)
        Z_hybrid = np.concatenate([mu_audio, z_lyrics], axis=1)
    else:
        Z_hybrid = mu_audio

    Z_hybrid = _standardize_2d(Z_hybrid)

    # 4) Clustering + metrics
    labels_by_method = {}
    metrics_by_method = {}

    for m in clustering_methods:
        labels = _cluster(
            Z_hybrid,
            method=m,
            n_clusters=n_clusters,
            dbscan_eps=dbscan_eps,
            dbscan_min_samples=dbscan_min_samples,
        )
        labels_by_method[m] = labels
        metrics_by_method[m] = clustering_metrics_medium(Z_hybrid, labels, true_labels=true_labels)


    return ids, Z_hybrid, labels_by_method, metrics_by_method, history



# ##################################################################################

# Hard Task

# ###################################################################################
import librosa
import numpy as np

def extract_audio_features(audio_dir, max_files=2000, sr=22050, duration=30.0, n_mels=64, max_frames=256):
    """
    Extracts audio features (log-mel spectrograms or MFCC) from the audio files.
    """
    # Initialize lists to store audio features and ids
    audio_features = []
    audio_ids = []

    # Loop over the audio files and extract features
    for i, audio_file in enumerate(os.listdir(audio_dir)):
        if i >= max_files:
            break
        # Load the audio file
        y, sr = librosa.load(os.path.join(audio_dir, audio_file), sr=sr, duration=duration)

        # Extract log-mel spectrogram
        mel_spectrogram = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=n_mels, fmax=8000)
        mel_spectrogram = librosa.power_to_db(mel_spectrogram, ref=np.max)

        # Ensure the spectrogram has a fixed length by truncating or padding
        mel_spectrogram = mel_spectrogram[:, :max_frames] if mel_spectrogram.shape[1] >= max_frames else \
                          np.pad(mel_spectrogram, ((0, 0), (0, max_frames - mel_spectrogram.shape[1])), mode='constant')

        audio_features.append(mel_spectrogram.flatten())
        audio_ids.append(audio_file)
    
    return np.array(audio_features), audio_ids



# from sklearn.feature_extraction.text import TfidfVectorizer
# import os

# def extract_lyrics_features(lyrics_dir, max_features=5000):
#     """
#     Extracts TF-IDF features from the lyrics.
#     """
#     lyrics = []
#     for lyric_file in os.listdir(lyrics_dir):
#         with open(os.path.join(lyrics_dir, lyric_file), 'r', encoding='utf-8') as file:
#             lyrics.append(file.read())

#     # Use TF-IDF to convert text data into numerical features
#     tfidf = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2), min_df=1)
#     lyrics_features = tfidf.fit_transform(lyrics).toarray()

#     return lyrics_features

import chardet
from sklearn.feature_extraction.text import TfidfVectorizer

def extract_lyrics_features(lyrics_dir, max_features=5000):
    lyrics = []
    for lyric_file in os.listdir(lyrics_dir):
        try:
            # Detect file encoding
            with open(os.path.join(lyrics_dir, lyric_file), 'rb') as file:
                raw_data = file.read()
                result = chardet.detect(raw_data)
                encoding = result['encoding']

            # Open file with the detected encoding
            with open(os.path.join(lyrics_dir, lyric_file), 'r', encoding=encoding) as file:
                lyrics.append(file.read())
        
        except Exception as e:
            print(f"Error reading {lyric_file}: {e}")
            continue
    
    tfidf = TfidfVectorizer(max_features=max_features, stop_words='english', ngram_range=(1, 2), min_df=1)
    lyrics_features = tfidf.fit_transform(lyrics).toarray()

    return lyrics_features



from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.preprocessing import StandardScaler
import numpy as np
from src.evaluation import clustering_metrics_medium

# def multi_modal_clustering(audio_features, lyrics_features, n_clusters=10, clustering_methods=("kmeans", "agglomerative", "dbscan")):
#     """
#     Performs multi-modal clustering combining audio and lyrics features.
#     """
#     # Combine audio features and lyrics features into a single hybrid feature vector
#     hybrid_features = np.concatenate([audio_features, lyrics_features], axis=1)

#     # Standardize the combined features
#     hybrid_features = StandardScaler().fit_transform(hybrid_features)

#     # Perform clustering using various methods
#     labels_by_method = {}
#     metrics_by_method = {}

#     for method in clustering_methods:
#         labels = _cluster(hybrid_features, method, n_clusters)
#         labels_by_method[method] = labels
#         metrics_by_method[method] = clustering_metrics_medium(hybrid_features, labels)

#     return labels_by_method, metrics_by_method

# def _cluster(Z: np.ndarray, method: str, n_clusters: int) -> np.ndarray:
#     """
#     Cluster data using different clustering methods (KMeans, Agglomerative, DBSCAN).
#     """
#     method = method.lower()
#     if method == "kmeans":
#         return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(Z)
#     elif method == "agglomerative":
#         return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(Z)
#     elif method == "dbscan":
#         return DBSCAN(eps=0.35, min_samples=10).fit_predict(Z)
#     else:
#         raise ValueError(f"Unknown clustering method: {method}")


from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering, DBSCAN
from sklearn.metrics import silhouette_score
import numpy as np
from src.evaluation import clustering_metrics_medium

def multi_modal_clustering(audio_features, genre_features, n_clusters=10, clustering_methods=("kmeans", "agglomerative", "dbscan")):
    """
    Performs multi-modal clustering combining audio and genre features.
    """
    # Combine audio features and genre features into a single hybrid feature vector
    hybrid_features = np.concatenate([audio_features, genre_features], axis=1)

    # Standardize the combined features
    hybrid_features = StandardScaler().fit_transform(hybrid_features)

    # Perform clustering using various methods
    labels_by_method = {}
    metrics_by_method = {}

    for method in clustering_methods:
        labels = _cluster(hybrid_features, method, n_clusters)
        labels_by_method[method] = labels
        metrics_by_method[method] = clustering_metrics_medium(hybrid_features, labels)

    return labels_by_method, metrics_by_method

def _cluster(Z: np.ndarray, method: str, n_clusters: int) -> np.ndarray:
    """
    Cluster data using different clustering methods (KMeans, Agglomerative, DBSCAN).
    """
    method = method.lower()
    if method == "kmeans":
        return KMeans(n_clusters=n_clusters, random_state=42).fit_predict(Z)
    elif method == "agglomerative":
        return AgglomerativeClustering(n_clusters=n_clusters).fit_predict(Z)
    elif method == "dbscan":
        return DBSCAN(eps=0.35, min_samples=10).fit_predict(Z)
    else:
        raise ValueError(f"Unknown clustering method: {method}")





# //////////////////////////////////////////////////////////////////////////////////////////
"""
Multi-modal clustering module for audio and lyrics features.
"""

import numpy as np
from sklearn.cluster import KMeans, SpectralClustering
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras import layers, Model


def extract_vae_latent_features(encoder, audio_features, lyrics_features):
    """
    Extract latent representations from the trained VAE encoder.
    
    Args:
        encoder: Trained VAE encoder model
        audio_features: Audio feature array (N, audio_dim)
        lyrics_features: Lyrics feature array (N, lyrics_dim)
    
    Returns:
        latent_features: Latent representations (N, latent_dim)
    """
    mu, logvar, z = encoder.predict([audio_features, lyrics_features])
    # Use the mean (mu) as the latent representation for clustering
    return mu


def multi_modal_clustering(latent_features, n_clusters=10, method='kmeans', random_state=42):
    """
    Perform clustering on latent features.
    
    Args:
        latent_features: Latent representations (N, latent_dim)
        n_clusters: Number of clusters
        method: Clustering method ('kmeans' or 'spectral')
        random_state: Random seed
    
    Returns:
        cluster_labels: Cluster assignments (N,)
        clusterer: Fitted clustering model
    """
    if method == 'kmeans':
        clusterer = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    elif method == 'spectral':
        clusterer = SpectralClustering(n_clusters=n_clusters, random_state=random_state, 
                                       affinity='nearest_neighbors')
    else:
        raise ValueError(f"Unknown clustering method: {method}")
    
    cluster_labels = clusterer.fit_predict(latent_features)
    return cluster_labels, clusterer


def pca_kmeans_clustering(audio_features, lyrics_features, n_clusters=10, n_components=32, random_state=42):
    """
    Baseline: PCA + K-Means clustering.
    
    Args:
        audio_features: Audio feature array (N, audio_dim)
        lyrics_features: Lyrics feature array (N, lyrics_dim)
        n_clusters: Number of clusters
        n_components: Number of PCA components
        random_state: Random seed
    
    Returns:
        cluster_labels: Cluster assignments
        pca_features: PCA-transformed features
        pca_model: Fitted PCA model
    """
    # Concatenate features
    combined_features = np.concatenate([audio_features, lyrics_features], axis=1)
    
    # Standardize
    scaler = StandardScaler()
    combined_features_scaled = scaler.fit_transform(combined_features)
    
    # Apply PCA
    pca = PCA(n_components=n_components, random_state=random_state)
    pca_features = pca.fit_transform(combined_features_scaled)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(pca_features)
    
    print(f"PCA explained variance ratio: {pca.explained_variance_ratio_.sum():.4f}")
    
    return cluster_labels, pca_features, pca


def build_autoencoder(input_dim, latent_dim=32):
    """
    Build a simple autoencoder for baseline comparison.
    
    Args:
        input_dim: Input feature dimension
        latent_dim: Latent space dimension
    
    Returns:
        encoder: Encoder model
        autoencoder: Full autoencoder model
    """
    # Encoder
    input_layer = layers.Input(shape=(input_dim,))
    x = layers.Dense(512, activation='relu')(input_layer)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    latent = layers.Dense(latent_dim, activation='relu', name='latent')(x)
    
    encoder = Model(input_layer, latent, name='encoder')
    
    # Decoder
    x = layers.Dense(256, activation='relu')(latent)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(512, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    output_layer = layers.Dense(input_dim, activation='sigmoid')(x)
    
    autoencoder = Model(input_layer, output_layer, name='autoencoder')
    
    return encoder, autoencoder


def autoencoder_kmeans_clustering(audio_features, lyrics_features, n_clusters=10, 
                                   latent_dim=32, epochs=50, batch_size=32, random_state=42):
    """
    Baseline: Autoencoder + K-Means clustering.
    
    Args:
        audio_features: Audio feature array (N, audio_dim)
        lyrics_features: Lyrics feature array (N, lyrics_dim)
        n_clusters: Number of clusters
        latent_dim: Latent space dimension
        epochs: Training epochs
        batch_size: Batch size
        random_state: Random seed
    
    Returns:
        cluster_labels: Cluster assignments
        latent_features: Latent representations
        encoder: Trained encoder
    """
    # Concatenate and normalize features
    combined_features = np.concatenate([audio_features, lyrics_features], axis=1)
    scaler = StandardScaler()
    combined_features_scaled = scaler.fit_transform(combined_features)
    
    # Build and train autoencoder
    input_dim = combined_features_scaled.shape[1]
    encoder, autoencoder = build_autoencoder(input_dim, latent_dim)
    
    autoencoder.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
                       loss='mse')
    
    print("Training Autoencoder...")
    autoencoder.fit(combined_features_scaled, combined_features_scaled,
                   epochs=epochs, batch_size=batch_size, verbose=0)
    
    # Extract latent features
    latent_features = encoder.predict(combined_features_scaled, verbose=0)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(latent_features)
    
    return cluster_labels, latent_features, encoder


def spectral_feature_clustering(audio_features, n_clusters=10, random_state=42):
    """
    Baseline: Direct clustering on audio spectral features.
    
    Args:
        audio_features: Audio feature array (N, audio_dim)
        n_clusters: Number of clusters
        random_state: Random seed
    
    Returns:
        cluster_labels: Cluster assignments
    """
    # Standardize features
    scaler = StandardScaler()
    audio_features_scaled = scaler.fit_transform(audio_features)
    
    # K-Means clustering
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    cluster_labels = kmeans.fit_predict(audio_features_scaled)
    
    return cluster_labels


def determine_optimal_clusters(latent_features, max_clusters=20, method='elbow'):
    """
    Determine optimal number of clusters using elbow method or silhouette analysis.
    
    Args:
        latent_features: Feature array (N, feature_dim)
        max_clusters: Maximum number of clusters to try
        method: 'elbow' or 'silhouette'
    
    Returns:
        inertias or silhouette_scores: Scores for each k
    """
    from sklearn.metrics import silhouette_score
    
    scores = []
    k_range = range(2, max_clusters + 1)
    
    for k in k_range:
        kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = kmeans.fit_predict(latent_features)
        
        if method == 'elbow':
            scores.append(kmeans.inertia_)
        elif method == 'silhouette':
            score = silhouette_score(latent_features, labels)
            scores.append(score)
    
    return list(k_range), scores

