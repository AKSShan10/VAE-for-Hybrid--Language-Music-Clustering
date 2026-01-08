from typing import Dict, Optional
import numpy as np

from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    adjusted_rand_score,
)


def clustering_metrics(X: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    """
    Easy Task metrics: Silhouette + Calinski-Harabasz.
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    unique = np.unique(labels)
    if unique.size < 2:
        return {"silhouette": float("nan"), "calinski_harabasz": float("nan")}

    return {
        "silhouette": float(silhouette_score(X, labels)),
        "calinski_harabasz": float(calinski_harabasz_score(X, labels)),
    }


def clustering_metrics_medium(
    X: np.ndarray,
    labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Medium Task metrics:
      - Silhouette (higher better) if >=2 clusters
      - Davies-Bouldin (lower better) if >=2 clusters
      - ARI (higher better) if true_labels provided
    Also includes:
      - cluster_count (excluding DBSCAN noise label -1)
      - noise_ratio (fraction of points labeled -1)
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    unique = np.unique(labels)
    n_clusters = int((unique[unique != -1]).size)  # exclude noise label -1
    noise_ratio = float(np.mean(labels == -1)) if labels.size > 0 else 0.0

    out: Dict[str, float] = {
        "cluster_count": float(n_clusters),
        "noise_ratio": float(noise_ratio),
        "ari": float("nan"),
    }

    # Undefined if <2 clusters (excluding noise), or if everything is noise
    if n_clusters < 2:
        out["silhouette"] = float("nan")
        out["davies_bouldin"] = float("nan")
    else:
        out["silhouette"] = float(silhouette_score(X, labels))
        out["davies_bouldin"] = float(davies_bouldin_score(X, labels))

    if true_labels is not None:
        true_labels = np.asarray(true_labels)
        out["ari"] = float(adjusted_rand_score(true_labels, labels))

    return out




# from sklearn.metrics import silhouette_score, davies_bouldin_score, adjusted_rand_score, normalized_mutual_info_score
# from sklearn.metrics.cluster import contingency_matrix
# from typing import Optional
# import numpy as np

# def clustering_metrics_hard(
#     X: np.ndarray,
#     labels: np.ndarray,
#     true_labels: Optional[np.ndarray] = None
# ) -> dict:
#     """
#     Hard Task metrics:
#       - Silhouette (higher better) [only if >=2 clusters]
#       - Davies-Bouldin (lower better) [only if >=2 clusters]
#       - ARI (higher better) if true_labels provided
#       - NMI (higher better) if true_labels provided
#       - Cluster Purity
#     """
#     X = np.asarray(X)
#     labels = np.asarray(labels)
    
#     unique = np.unique(labels)
#     n_clusters = int((unique[unique != -1]).size)  # Excluding DBSCAN noise label (-1)
#     noise_ratio = float(np.mean(labels == -1)) if labels.size > 0 else 0.0
    
#     out = {
#         "cluster_count": float(n_clusters),
#         "noise_ratio": float(noise_ratio),
#     }

#     # Silhouette and Davies-Bouldin (if >= 2 clusters)
#     if n_clusters < 2:
#         out.update({
#             "silhouette": float("nan"),
#             "davies_bouldin": float("nan"),
#         })
#     else:
#         out["silhouette"] = float(silhouette_score(X, labels))
#         out["davies_bouldin"] = float(davies_bouldin_score(X, labels))

#     if true_labels is not None:
#         true_labels = np.asarray(true_labels)
#         out["ari"] = adjusted_rand_score(true_labels, labels)
#         out["nmi"] = normalized_mutual_info_score(true_labels, labels)
        
#         # Cluster Purity
#         out["cluster_purity"] = cluster_purity(true_labels, labels)

#     return out

# def cluster_purity(true_labels, predicted_labels):
#     """
#     Compute the Cluster Purity score.
#     """
#     contingency = contingency_matrix(true_labels, predicted_labels)
#     purity = np.sum(np.amax(contingency, axis=0)) / np.sum(contingency)
#     return purity

from typing import Dict, Optional
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, normalized_mutual_info_score, adjusted_rand_score

def clustering_metrics_hard(
    X: np.ndarray,
    labels: np.ndarray,
    true_labels: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """
    Hard Task metrics:
      - Silhouette (higher better) [only if >=2 clusters]
      - Davies-Bouldin (lower better) [only if >=2 clusters]
      - ARI (higher better) if true_labels provided
      - Cluster Purity (higher better)
    """
    X = np.asarray(X)
    labels = np.asarray(labels)

    unique = np.unique(labels)
    n_clusters = int((unique[unique != -1]).size)  # excluding DBSCAN noise label -1
    noise_ratio = float(np.mean(labels == -1)) if labels.size > 0 else 0.0

    out: Dict[str, float] = {
        "cluster_count": float(n_clusters),
        "noise_ratio": float(noise_ratio),
    }

    if n_clusters < 2:
        out.update({
            "silhouette": float("nan"),
            "davies_bouldin": float("nan"),
            "ari": float("nan"),
        })
    else:
        out["silhouette"] = float(silhouette_score(X, labels))
        out["davies_bouldin"] = float(normalized_mutual_info_score(X, labels))

    if true_labels is not None:
        true_labels = np.asarray(true_labels)
        out["ari"] = float(adjusted_rand_score(true_labels, labels))
    else:
        out["ari"] = float("nan")

    return out








# ///////////////////////////////////////////////////////////////
"""
Evaluation metrics for clustering quality assessment.
"""

import numpy as np
from sklearn.metrics import (
    silhouette_score, 
    normalized_mutual_info_score,
    adjusted_rand_score,
    confusion_matrix
)
from collections import Counter


def calculate_silhouette_score(features, cluster_labels):
    """
    Calculate Silhouette Score.
    Measures how similar objects are to their own cluster compared to other clusters.
    Range: [-1, 1], higher is better.
    
    Args:
        features: Feature array (N, feature_dim)
        cluster_labels: Cluster assignments (N,)
    
    Returns:
        silhouette: Silhouette score
    """
    if len(np.unique(cluster_labels)) < 2:
        return 0.0
    
    silhouette = silhouette_score(features, cluster_labels)
    return silhouette


def calculate_nmi(true_labels, cluster_labels):
    """
    Calculate Normalized Mutual Information (NMI).
    Measures mutual information between true labels and cluster assignments.
    Range: [0, 1], higher is better.
    
    Args:
        true_labels: Ground truth labels (N,)
        cluster_labels: Cluster assignments (N,)
    
    Returns:
        nmi: NMI score
    """
    nmi = normalized_mutual_info_score(true_labels, cluster_labels, average_method='arithmetic')
    return nmi


def calculate_ari(true_labels, cluster_labels):
    """
    Calculate Adjusted Rand Index (ARI).
    Measures similarity between two clusterings adjusted for chance.
    Range: [-1, 1], higher is better (1 = perfect match).
    
    Args:
        true_labels: Ground truth labels (N,)
        cluster_labels: Cluster assignments (N,)
    
    Returns:
        ari: ARI score
    """
    ari = adjusted_rand_score(true_labels, cluster_labels)
    return ari


def calculate_cluster_purity(true_labels, cluster_labels):
    """
    Calculate Cluster Purity.
    Measures the extent to which clusters contain a single class.
    Range: [0, 1], higher is better.
    
    Args:
        true_labels: Ground truth labels (N,)
        cluster_labels: Cluster assignments (N,)
    
    Returns:
        purity: Purity score
    """
    # Create confusion matrix
    contingency_matrix = confusion_matrix(true_labels, cluster_labels)
    
    # Sum the maximum value in each column (cluster)
    purity = np.sum(np.amax(contingency_matrix, axis=0)) / np.sum(contingency_matrix)
    
    return purity


def evaluate_clustering(features, cluster_labels, true_labels=None):
    """
    Comprehensive clustering evaluation.
    
    Args:
        features: Feature array (N, feature_dim)
        cluster_labels: Cluster assignments (N,)
        true_labels: Ground truth labels (optional, N,)
    
    Returns:
        metrics: Dictionary of evaluation metrics
    """
    metrics = {}
    
    # Silhouette Score (unsupervised)
    metrics['silhouette_score'] = calculate_silhouette_score(features, cluster_labels)
    
    # If ground truth labels are available
    if true_labels is not None:
        metrics['nmi'] = calculate_nmi(true_labels, cluster_labels)
        metrics['ari'] = calculate_ari(true_labels, cluster_labels)
        metrics['purity'] = calculate_cluster_purity(true_labels, cluster_labels)
    
    # Cluster statistics
    unique_clusters, cluster_counts = np.unique(cluster_labels, return_counts=True)
    metrics['n_clusters'] = len(unique_clusters)
    metrics['min_cluster_size'] = cluster_counts.min()
    metrics['max_cluster_size'] = cluster_counts.max()
    metrics['mean_cluster_size'] = cluster_counts.mean()
    metrics['std_cluster_size'] = cluster_counts.std()
    
    return metrics


def print_evaluation_metrics(metrics, method_name="Clustering"):
    """
    Pretty print evaluation metrics.
    
    Args:
        metrics: Dictionary of metrics
        method_name: Name of the clustering method
    """
    print(f"\n{'='*60}")
    print(f"{method_name} Evaluation Metrics")
    print(f"{'='*60}")
    
    # Unsupervised metrics
    print(f"\nUnsupervised Metrics:")
    print(f"  Silhouette Score:        {metrics.get('silhouette_score', 'N/A'):.4f}")
    
    # Supervised metrics (if available)
    if 'nmi' in metrics:
        print(f"\nSupervised Metrics (with ground truth):")
        print(f"  Normalized Mutual Info:  {metrics['nmi']:.4f}")
        print(f"  Adjusted Rand Index:     {metrics['ari']:.4f}")
        print(f"  Cluster Purity:          {metrics['purity']:.4f}")
    
    # Cluster statistics
    print(f"\nCluster Statistics:")
    print(f"  Number of Clusters:      {metrics['n_clusters']}")
    print(f"  Min Cluster Size:        {metrics['min_cluster_size']}")
    print(f"  Max Cluster Size:        {metrics['max_cluster_size']}")
    print(f"  Mean Cluster Size:       {metrics['mean_cluster_size']:.2f}")
    print(f"  Std Cluster Size:        {metrics['std_cluster_size']:.2f}")
    print(f"{'='*60}\n")


def compare_clustering_methods(results_dict):
    """
    Compare multiple clustering methods side by side.
    
    Args:
        results_dict: Dictionary mapping method names to their metrics
    
    Returns:
        comparison_df: Pandas DataFrame with comparison
    """
    import pandas as pd
    
    comparison_data = []
    
    for method_name, metrics in results_dict.items():
        row = {
            'Method': method_name,
            'Silhouette': metrics.get('silhouette_score', np.nan),
            'NMI': metrics.get('nmi', np.nan),
            'ARI': metrics.get('ari', np.nan),
            'Purity': metrics.get('purity', np.nan),
            'N_Clusters': metrics.get('n_clusters', np.nan),
        }
        comparison_data.append(row)
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by Silhouette Score (or another metric)
    comparison_df = comparison_df.sort_values('Silhouette', ascending=False)
    
    return comparison_df


def cluster_distribution_analysis(cluster_labels, metadata_labels, metadata_name="Genre"):
    """
    Analyze how clusters are distributed across metadata categories (e.g., genres, languages).
    
    Args:
        cluster_labels: Cluster assignments (N,)
        metadata_labels: Metadata category labels (N,)
        metadata_name: Name of the metadata (e.g., "Genre", "Language")
    
    Returns:
        distribution: Dictionary mapping cluster_id -> {metadata: count}
    """
    unique_clusters = np.unique(cluster_labels)
    distribution = {}
    
    for cluster_id in unique_clusters:
        cluster_mask = cluster_labels == cluster_id
        metadata_in_cluster = metadata_labels[cluster_mask]
        metadata_counts = Counter(metadata_in_cluster)
        distribution[cluster_id] = dict(metadata_counts)
    
    return distribution


def print_cluster_distribution(distribution, metadata_name="Category", top_k=3):
    """
    Print cluster distribution analysis.
    
    Args:
        distribution: Dictionary from cluster_distribution_analysis
        metadata_name: Name of the metadata
        top_k: Number of top categories to show per cluster
    """
    print(f"\n{'='*60}")
    print(f"Cluster Distribution by {metadata_name}")
    print(f"{'='*60}\n")
    
    for cluster_id, metadata_counts in sorted(distribution.items()):
        total_count = sum(metadata_counts.values())
        print(f"Cluster {cluster_id} (Size: {total_count}):")
        
        # Sort by count and get top_k
        sorted_metadata = sorted(metadata_counts.items(), key=lambda x: x[1], reverse=True)[:top_k]
        
        for metadata, count in sorted_metadata:
            percentage = (count / total_count) * 100
            print(f"  {metadata:20s}: {count:4d} ({percentage:5.1f}%)")
        
        print()
