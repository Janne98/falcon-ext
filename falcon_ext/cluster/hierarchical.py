import numpy as np

from typing import Dict, Tuple
from collections import Counter

from sklearn.cluster import AgglomerativeClustering
from scipy.cluster.hierarchy import dendrogram
import matplotlib
import matplotlib.pyplot as plt

import config
from config import *


matplotlib.use('TkAgg')

def generate_clusters(dist_matrix: np.ndarray, 
                      linkage:str, 
                      distance_threshold: float,
                      min_cluster_size: int) -> AgglomerativeClustering:
    """
    Generate clusters using agglomerative (hierarchical) clustering.

    Parameters
    ----------
    dist_matrix : np.ndarray
        distance matrix used for clustering.
    linkage: str
        linkage method (see sklear documentation).
    distance_threshold: float
        distance above which clusters will not be merged (see sklearn docs).

    Returns
    -------
    AgglomerativeClustering
        clustering of spectra.
    """
    # print(f"{linkage}-linkage hierarchical clustering...")
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed", 
        linkage=linkage,
        distance_threshold=distance_threshold,
        compute_distances=True).fit(dist_matrix)
    
    new_labels = _post_process_clusters(clustering.labels_, min_cluster_size)
    clustering.labels_ = new_labels
    clustering.n_clusters_ = _count_clusters(new_labels)

    if config.plot_dendrogram:
        plot_dendrogram(clustering=clustering, labels=clustering.labels_)

    return clustering


def _post_process_clusters(labels: np.ndarray, min_cluster_size: int) -> np.ndarray:
    """
    Label clusters of size 1 as noise (-1).

    Parameters
    ----------
    labels : np.ndarray
        array of predicted labels.

    Returns
    -------
    np.ndarray
        new labels where samples in singleton clusters are replaced by -1 (noise).
    """
    # count occurences of labels
    c = Counter(labels)
    singleton_labels = [k for k, v in c.items() if v < min_cluster_size]
    # if label appears once, replace with -1 (noise sample)
    new_labels = [l if l not in singleton_labels else -1 for l in labels]
    return np.array(new_labels)


def _count_clusters(labels: np.ndarray) -> np.ndarray:
    return len(np.delete(np.unique(labels), -1))


# code from: 
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(clustering, **kwargs):
    """
    Plot a dendrogram of the clustering result.

    Parameters
    ----------
    clustering : AgglomerativeClustering
        clustering result.
    """
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(clustering.children_.shape[0])
    n_samples = len(clustering.labels_)
    for i, merge in enumerate(clustering.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [clustering.children_, clustering.distances_, counts]
    ).astype(float)

    # plot the corresponding dendrogram
    fig = plt.figure("Clustering dendrogram")
    dendrogram(linkage_matrix, **kwargs)
    fig.show()