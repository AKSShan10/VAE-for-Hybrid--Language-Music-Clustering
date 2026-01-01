import os
import csv
from sklearn.metrics import silhouette_score, calinski_harabasz_score

def easy_metrics(X, labels):
    sil = silhouette_score(X, labels)
    ch = calinski_harabasz_score(X, labels)
    return sil, ch

def write_easy_metrics_csv(rows, out_csv="results/clustering_metrics.csv"):
    """
    rows: list of dicts with keys:
      Method, Silhouette Score, Calinski-Harabasz Index
    """
    os.makedirs(os.path.dirname(out_csv), exist_ok=True)
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["Method", "Silhouette Score", "Calinski-Harabasz Index"],
        )
        writer.writeheader()
        writer.writerows(rows)
