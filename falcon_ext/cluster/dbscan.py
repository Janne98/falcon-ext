import numpy as np

from sklearn.cluster import DBSCAN


def generate_clusters(dist_matrix: np.ndarray, eps: float, min_cluster_size: int) -> DBSCAN:
    """
    Generate clusters using DBSCAN.

    Parameters
    ----------
    dist_matrix : np.ndarray
        distance matrix used for clustering.
    eps : float
        The maximum distance between two samples for one to be considered as 
        in the neighborhood of the other. 

    Returns
    -------
    DBSCAN
        clustering of spectra.
    """
    print('DBSCAN clustering...')
    clustering = DBSCAN(eps=eps, 
                        min_samples=min_cluster_size, 
                        metric='precomputed', 
                        n_jobs=-1).fit(dist_matrix)
    return clustering