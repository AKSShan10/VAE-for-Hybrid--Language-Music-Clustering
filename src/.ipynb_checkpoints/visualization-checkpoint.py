"""
Visualization functions for clustering analysis.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
import pandas as pd


def plot_latent_space_2d(latent_features, cluster_labels, method='tsne', 
                          title='Latent Space Visualization', 
                          save_path=None, figsize=(12, 8)):
    """
    Visualize latent space in 2D using t-SNE or PCA.
    
    Args:
        latent_features: Latent representations (N, latent_dim)
        cluster_labels: Cluster assignments (N,)
        method: Dimensionality reduction method ('tsne' or 'pca')
        title: Plot title
        save_path: Path to save figure
        figsize: Figure size
    """
    # Reduce to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = reducer.fit_transform(latent_features)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(latent_features)
    else:
        raise ValueError(f"Unknown method: {method}")
    
    # Create plot
    plt.figure(figsize=figsize)
    scatter = plt.scatter(features_2d[:, 0], features_2d[:, 1], 
                         c=cluster_labels, cmap='tab20', 
                         alpha=0.6, s=50, edgecolors='k', linewidth=0.5)
    
    plt.colorbar(scatter, label='Cluster ID')
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    return features_2d


def plot_latent_space_with_metadata(latent_features, metadata_labels, 
                                     metadata_name='Genre',
                                     method='tsne', save_path=None, figsize=(12, 8)):
    """
    Visualize latent space colored by metadata (e.g., genre, language).
    
    Args:
        latent_features: Latent representations (N, latent_dim)
        metadata_labels: Metadata category labels (N,)
        metadata_name: Name of the metadata
        method: Dimensionality reduction method
        save_path: Path to save figure
        figsize: Figure size
    """
    # Reduce to 2D
    if method == 'tsne':
        reducer = TSNE(n_components=2, random_state=42, perplexity=30)
        features_2d = reducer.fit_transform(latent_features)
    elif method == 'pca':
        reducer = PCA(n_components=2, random_state=42)
        features_2d = reducer.fit_transform(latent_features)
    
    # Create plot
    plt.figure(figsize=figsize)
    
    unique_labels = np.unique(metadata_labels)
    colors = plt.cm.tab20(np.linspace(0, 1, len(unique_labels)))
    
    for i, label in enumerate(unique_labels):
        mask = metadata_labels == label
        plt.scatter(features_2d[mask, 0], features_2d[mask, 1],
                   c=[colors[i]], label=label, alpha=0.6, s=50,
                   edgecolors='k', linewidth=0.5)
    
    plt.xlabel(f'{method.upper()} Component 1', fontsize=12)
    plt.ylabel(f'{method.upper()} Component 2', fontsize=12)
    plt.title(f'Latent Space by {metadata_name}', fontsize=14, fontweight='bold')
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_cluster_distribution(cluster_labels, metadata_labels, 
                               metadata_name='Genre', save_path=None, figsize=(14, 8)):
    """
    Plot cluster distribution across metadata categories.
    
    Args:
        cluster_labels: Cluster assignments (N,)
        metadata_labels: Metadata category labels (N,)
        metadata_name: Name of the metadata
        save_path: Path to save figure
        figsize: Figure size
    """
    # Create DataFrame
    df = pd.DataFrame({
        'Cluster': cluster_labels,
        metadata_name: metadata_labels
    })
    
    # Create crosstab
    crosstab = pd.crosstab(df['Cluster'], df[metadata_name])
    
    # Plot stacked bar chart
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    
    # Absolute counts
    crosstab.plot(kind='bar', stacked=True, ax=ax1, colormap='tab20')
    ax1.set_xlabel('Cluster ID', fontsize=12)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.set_title(f'Cluster Distribution by {metadata_name} (Absolute)', 
                  fontsize=13, fontweight='bold')
    ax1.legend(title=metadata_name, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Normalized (percentage)
    crosstab_norm = crosstab.div(crosstab.sum(axis=1), axis=0) * 100
    crosstab_norm.plot(kind='bar', stacked=True, ax=ax2, colormap='tab20')
    ax2.set_xlabel('Cluster ID', fontsize=12)
    ax2.set_ylabel('Percentage (%)', fontsize=12)
    ax2.set_title(f'Cluster Distribution by {metadata_name} (Normalized)', 
                  fontsize=13, fontweight='bold')
    ax2.legend(title=metadata_name, bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=8)
    ax2.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_vae_reconstructions(decoder, latent_features, lyrics_features, 
                              original_audio, n_samples=5, 
                              save_path=None, figsize=(15, 10)):
    """
    Visualize VAE reconstructions vs original audio features.
    
    Args:
        decoder: Trained VAE decoder
        latent_features: Latent representations (N, latent_dim)
        lyrics_features: Lyrics features (N, lyrics_dim)
        original_audio: Original audio features (N, audio_dim)
        n_samples: Number of samples to visualize
        save_path: Path to save figure
        figsize: Figure size
    """
    # Randomly select samples
    indices = np.random.choice(len(latent_features), n_samples, replace=False)
    
    # Generate reconstructions
    reconstructed = decoder.predict([latent_features[indices], lyrics_features[indices]], verbose=0)
    
    # Create subplots
    fig, axes = plt.subplots(n_samples, 2, figsize=figsize)
    
    if n_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i, idx in enumerate(indices):
        # Original
        axes[i, 0].plot(original_audio[idx], alpha=0.7)
        axes[i, 0].set_title(f'Original Sample {idx}', fontsize=10)
        axes[i, 0].set_ylabel('Amplitude', fontsize=9)
        axes[i, 0].grid(True, alpha=0.3)
        
        # Reconstructed
        axes[i, 1].plot(reconstructed[i], alpha=0.7, color='orange')
        axes[i, 1].set_title(f'Reconstructed Sample {idx}', fontsize=10)
        axes[i, 1].set_ylabel('Amplitude', fontsize=9)
        axes[i, 1].grid(True, alpha=0.3)
        
        if i == n_samples - 1:
            axes[i, 0].set_xlabel('Feature Index', fontsize=9)
            axes[i, 1].set_xlabel('Feature Index', fontsize=9)
    
    plt.suptitle('VAE Reconstruction Examples', fontsize=14, fontweight='bold', y=1.0)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_clustering_comparison(comparison_df, save_path=None, figsize=(12, 6)):
    """
    Compare different clustering methods using bar charts.
    
    Args:
        comparison_df: DataFrame from compare_clustering_methods
        save_path: Path to save figure
        figsize: Figure size
    """
    metrics = ['Silhouette', 'NMI', 'ARI', 'Purity']
    available_metrics = [m for m in metrics if m in comparison_df.columns]
    
    fig, axes = plt.subplots(1, len(available_metrics), figsize=figsize)
    
    if len(available_metrics) == 1:
        axes = [axes]
    
    for i, metric in enumerate(available_metrics):
        ax = axes[i]
        data = comparison_df[['Method', metric]].dropna()
        
        bars = ax.bar(data['Method'], data[metric], color=plt.cm.viridis(np.linspace(0.3, 0.9, len(data))))
        ax.set_ylabel(metric, fontsize=11)
        ax.set_title(f'{metric} Score', fontsize=12, fontweight='bold')
        ax.set_ylim(0, 1)
        ax.grid(True, alpha=0.3, axis='y')
        ax.set_xticklabels(data['Method'], rotation=45, ha='right', fontsize=9)
        
        # Add value labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.suptitle('Clustering Methods Comparison', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_elbow_curve(k_range, inertias, save_path=None, figsize=(10, 6)):
    """
    Plot elbow curve for determining optimal number of clusters.
    
    Args:
        k_range: Range of k values
        inertias: Inertia values for each k
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(k_range, inertias, 'bo-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Inertia', fontsize=12)
    plt.title('Elbow Method for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_silhouette_analysis(k_range, silhouette_scores, save_path=None, figsize=(10, 6)):
    """
    Plot silhouette scores for different k values.
    
    Args:
        k_range: Range of k values
        silhouette_scores: Silhouette scores for each k
        save_path: Path to save figure
        figsize: Figure size
    """
    plt.figure(figsize=figsize)
    plt.plot(k_range, silhouette_scores, 'go-', linewidth=2, markersize=8)
    plt.xlabel('Number of Clusters (k)', fontsize=12)
    plt.ylabel('Silhouette Score', fontsize=12)
    plt.title('Silhouette Analysis for Optimal k', fontsize=14, fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Mark the maximum
    max_idx = np.argmax(silhouette_scores)
    plt.axvline(x=k_range[max_idx], color='r', linestyle='--', alpha=0.7, 
                label=f'Optimal k={k_range[max_idx]}')
    plt.legend(fontsize=10)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_confusion_matrix(true_labels, cluster_labels, 
                          save_path=None, figsize=(12, 10)):
    """
    Plot confusion matrix between true labels and cluster assignments.
    
    Args:
        true_labels: Ground truth labels (N,)
        cluster_labels: Cluster assignments (N,)
        save_path: Path to save figure
        figsize: Figure size
    """
    from sklearn.metrics import confusion_matrix
    
    # Create confusion matrix
    cm = confusion_matrix(true_labels, cluster_labels)
    
    # Normalize by row (true label)
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=figsize)
    sns.heatmap(cm_normalized, annot=True, fmt='.2f', cmap='YlOrRd', 
                cbar_kws={'label': 'Proportion'})
    plt.xlabel('Cluster ID', fontsize=12)
    plt.ylabel('True Label', fontsize=12)
    plt.title('Confusion Matrix (Normalized by True Label)', 
              fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()