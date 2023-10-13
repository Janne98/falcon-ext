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
                      linkage:str = "complete", 
                      distance_threshold: float = 1) -> AgglomerativeClustering:
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
    clustering = AgglomerativeClustering(
        n_clusters=None,
        metric="precomputed", 
        linkage=linkage,
        distance_threshold=distance_threshold,
        compute_distances=True).fit(dist_matrix)
    
    new_labels = singletons_to_noise(clustering.labels_)
    clustering.labels_ = new_labels

    if config.plot_dendrogram:
        plot_dendrogram(clustering=clustering, labels=clustering.labels_)
        
    return clustering


def get_medoids(dist_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
    """
    Get indices of medoids for each cluster.

    Parameters
    ----------
    dist_matrix : np.ndarray
        distance matrix used for clustering.
    labels : np.ndarray
        array of predicted labels.

    Returns
    -------
    np.ndarray
        indices of medoid spectra.
    """
    medoids = []

    cluster_dict = {}
    for cluster_label in labels:
        if cluster_label == -1:
            continue
        cluster_dict[cluster_label] = [idx for idx, label in enumerate(labels) \
                                       if label == cluster_label]

    for _, spectra in cluster_dict.items():
        dist_sums = []
        for spec in spectra:
            other_specs = [spec_idx for spec_idx in spectra if spec_idx != spec]
            dist_sums.append(sum(dist_matrix[spec][other_specs]))
        medoids.append(spectra[dist_sums.index(min(dist_sums))])

    return np.array(medoids)


def singletons_to_noise(labels: np.ndarray) -> np.ndarray:
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
    singleton_labels = [k for k, v in c.items() if v == 1]
    # if label appears once, replace with -1 (noise sample)
    new_labels = [l if l not in singleton_labels else -1 for l in labels]
    return np.array(new_labels)


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