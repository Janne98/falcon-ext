import numpy as np

from sklearn.cluster import DBSCAN


def generate_clusters(dist_matrix: np.ndarray, eps: float) -> DBSCAN:
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
    clustering = DBSCAN(eps=eps, 
                        min_samples=2, 
                        metric='precomputed', 
                        n_jobs=-1).fit(dist_matrix)
    return clustering