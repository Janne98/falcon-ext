import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple

from sklearn.cluster import AgglomerativeClustering, DBSCAN

import collections

import config
from config import *
from . import dbscan
from . import hierarchical


ClusterResult = collections.namedtuple(
    'Clustering', ['labels', 'n_clusters', 'core_samples']
)


def generate_clusters(dist_matrix: np.ndarray) -> ClusterResult:
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
    if config.cluster_method == 'hierarchical':
        result = hierarchical.generate_clusters(dist_matrix, 
                                                config.linkage, 
                                                config.max_cluster_dist)
        return ClusterResult(result.labels_, result.n_clusters, 
                          hierarchical.get_medoids(dist_matrix, result.labels_))
    
    elif config.cluster_method == 'DBSCAN':
        result = dbscan.generate_clusters(dist_matrix, config.eps)
        return ClusterResult(result.labels_, max(result.labels_), result.core_sample_indices_)
    
    else:
        raise ValueError(f'Unknown clustering method "{config.cluster_method}"')


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