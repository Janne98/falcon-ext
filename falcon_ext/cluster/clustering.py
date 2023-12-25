import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple

from sklearn.cluster import AgglomerativeClustering, DBSCAN

import collections

from . import dbscan
from . import hierarchical


ClusterResult = collections.namedtuple(
    'Clustering', ['labels', 'n_clusters', 'cluster_samples', 'noise_samples']
)


def generate_clusters(dist_matrix: np.ndarray, cluster_method: str, linkage: str, 
                      eps: float, min_cluster_size: int) -> ClusterResult:
    """
    Generate clusters using agglomerative (hierarchical) clustering or DBSCAN.

    Parameters
    ----------
    dist_matrix : np.ndarray
        distance matrix used for clustering.

    Returns
    -------
    Clustering
        clustering of spectra.
    """
    if cluster_method == 'hierarchical':
        result = hierarchical.generate_clusters(dist_matrix, 
                                                linkage, 
                                                eps,
                                                min_cluster_size)
        return ClusterResult(result.labels_, result.n_clusters_, 
                             _get_medoids(dist_matrix, result.labels_),
                             _get_noise_samples(result.labels_))
    
    elif cluster_method == 'DBSCAN':
        result = dbscan.generate_clusters(dist_matrix, eps, min_cluster_size)
        return ClusterResult(result.labels_, max(result.labels_),
                             _get_medoids(dist_matrix, result.labels_),
                             _get_noise_samples(result.labels_))
    
    else:
        raise ValueError(f'Unknown clustering method "{cluster_method}"')
    

def _get_medoids(dist_matrix: np.ndarray, labels: np.ndarray) -> np.ndarray:
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
    for cluster_label in np.unique(labels):
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

    return np.asarray(medoids, dtype=int)
    

def _get_noise_samples(labels: np.ndarray) -> np.ndarray:
    """
    Get indices of noise samples.

    Parameters
    ----------
    labels : np.ndarray
        array of predicted labels.

    Returns
    -------
    np.ndarray
        array of indices of noise samples (label = -1).
    """
    noise_idx = [idx for idx in range(len(labels)) if labels[idx] == -1]
    return np.asarray(noise_idx, dtype=int)


def clusters_to_csv(clustering: ClusterResult, idx_to_scan_map: List[int]) -> None:
    """
    Write cluster assignments to csv file.

    Parameters
    ----------
    clustering : ClusterResult
        clustering result.
    spec_map: List[int]
        list of scan ids mapping index in clustering to scan.
    """
    labels = clustering.labels
    cluster_assignments = pd.DataFrame({'scan_id': idx_to_scan_map, 
                                        'cluster_labels': labels})
    cluster_assignments.to_csv('cluster_assignments.csv', index=False)