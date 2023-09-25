import numpy as np
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple

from scipy.cluster.hierarchy import dendrogram
from sklearn.cluster import AgglomerativeClustering


def generate_clusters(
        dist_matrix: np.ndarray, 
        linkage:str = "complete", 
        distance_threshold: float = 0.6):
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
    return clustering


def get_medoids( 
        dist_matrix: np.ndarray, 
        clustering: AgglomerativeClustering) -> List[Tuple[int, int]]:
    """
    Get the index of the representative spectrum (medoid) for each cluster.

    Parameters
    ----------
    dist_matrix : np.ndarray
        distance matrix.
    clustering : AgglomerativeClustering
        result of clustering

    Returns
    -------
    List[Tuple[int, int]]
        list of (cluster label, spectrum index)-pairs, 
        indicating the representative spectrum for each cluster.
    """
    labels = clustering.labels_
    # create dict of {cluster_label : [spectrum_idx]}-format
    cluster_dict = {}
    for cluster_label in range(max(labels) + 1):
        cluster_dict[cluster_label] = [idx for idx, label in enumerate(labels) \
                                       if label == cluster_label]
    # print(cluster_dict)
    # create dict of {cluster_label : medoid_spectrum_idx}-format
    medoids = []
    for cluster, specs in cluster_dict.items():
        dist_sums = []
        if len(specs) < 2:
            medoids.append((cluster, specs[0]))
            continue
        for spec in specs:
            other_specs = [spec_idx for spec_idx in specs if spec_idx != spec]
            dist_sums.append(sum(dist_matrix[spec][other_specs]))
        medoids.append((cluster, specs[dist_sums.index(min(dist_sums))]))
    print(medoids)

    return medoids

# code from: 
# https://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_dendrogram.html
def plot_dendrogram(model, **kwargs):
    """
    Plot a dendrogram of the clustering result.

    Parameters
    ----------
    model : AgglomerativeClustering
        clustering result.
    """
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # np.clip(linkage_matrix,0,1,linkage_matrix)

    # plot the corresponding dendrogram
    #plt.ion()
    fig = plt.figure(1)
    dendrogram(linkage_matrix, **kwargs)
    fig.show()