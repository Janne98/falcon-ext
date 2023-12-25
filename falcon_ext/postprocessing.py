import numpy as np

from scipy.cluster.hierarchy import fcluster
from spectrum_utils.spectrum import MsmsSpectrum

from typing import List, Iterator, Tuple


def postprocess_clusters(spectra: List[MsmsSpectrum], cluster_labels: np.ndarray,
                         precursor_tol: float, tol_mode: str, min_samples: int) -> np.ndarray:
    """Postprocess the clusters obtained from the clustering algorithm.

    Parameters
    ----------
    spectra : list of :class:`~spectrum_utils.spectrum.MsmsSpectrum`
        List of spectra.
    cluster_labels : :class:`numpy.ndarray`
        Cluster labels. 
    precursor_tol : float
        The precursor m/z tolerance in Da.
    min_samples : int
        The minimum number of samples in a cluster.

    Returns
    -------
    :class:`numpy.ndarray`
        Postprocessed cluster labels.
    """
    order = np.argsort(cluster_labels)
    reverse_order = np.argsort(order)
    cluster_labels[:] = cluster_labels[order]

    if cluster_labels[-1] == -1:
        cluster_labels.fill(-1)
        noise_mask = np.ones_like(cluster_labels)
        n_clusters, n_noise = 0, len(noise_mask)
    else:
        n_clusters = []
        group_idx = _get_cluster_group_idx(cluster_labels)
        for start_i, stop_i in group_idx:
            n_c = _postprocess_cluster(spectra[start_i:stop_i], cluster_labels[start_i:stop_i], 
                                precursor_tol, tol_mode, min_samples)
            n_clusters.append(n_c)
        _assign_unique_cluster_labels(cluster_labels, group_idx, n_clusters, min_samples)
        cluster_labels[:] = cluster_labels[reverse_order]
        noise_mask = cluster_labels == -1
        n_clusters, n_noise = np.amax(cluster_labels) + 1, noise_mask.sum()
    cluster_labels[noise_mask] = np.arange(n_clusters, n_clusters + n_noise)

    return np.asarray(cluster_labels)


def _get_cluster_group_idx(cluster_labels: np.ndarray) -> Iterator[Tuple[int, int]]:
    """
    Get start and stop indexes for unique cluster labels.
    Parameters
    ----------
    cluster_labels : np.ndarray
        The ordered cluster labels (noise points are -1).
    Returns
    -------
    Iterator[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the unique cluster labels.
    """
    start_i = 0
    while cluster_labels[start_i] == -1 and start_i < cluster_labels.shape[0]:
        start_i += 1
    stop_i = start_i
    while stop_i < cluster_labels.shape[0]:
        start_i, label = stop_i, cluster_labels[stop_i]
        while stop_i < cluster_labels.shape[0] and cluster_labels[stop_i] == label:
            stop_i += 1
        yield start_i, stop_i


def _postprocess_cluster(spectra: List[MsmsSpectrum], cluster_labels: np.ndarray, 
                        precursor_tol: float, tol_mode: str, min_samples: int) -> int:
    """
    Agglomerative clustering of the precursor m/z's within each initial
    cluster to avoid that spectra within a cluster have an excessive precursor
    m/z difference.

    Parameters
    ----------
    spectra : list of :class:`~spectrum_utils.spectrum.MsmsSpectrum`
        List of spectra.
    cluster_labels : :class:`numpy.ndarray`
        Cluster labels. precursor : float
        The precursor m/z tolerance in Da.
    precursor_tol_mode : str
        The precursor m/z tolerance mode, either 'Da' or 'ppm'.
    min_samples : int
        The minimum number of samples in a cluster.

    Returns
    -------
    int
        The number of clusters after splitting on precursor m/z.
    """
    precursor_mzs = np.array([s.precursor_mz for s in spectra])
    cluster_labels.fill(-1)
    # No splitting needed if there are too few items in cluster.
    # This seems to happen sometimes despite that DBSCAN requires a higher
    # `min_samples`.
    if cluster_labels.shape[0] < min_samples:
        n_clusters = 0
    else:
        # Group items within the cluster based on their precursor m/z.
        # Precursor m/z's within a single group can't exceed the specified
        # precursor m/z tolerance (`distance_threshold`).
        # Subtract 1 because fcluster starts with cluster label 1 instead of 0
        # (like Scikit-Learn does).
        cluster_assignments = fcluster(
            _linkage(precursor_mzs, tol_mode), precursor_tol, 'distance') - 1

        n_clusters = cluster_assignments.max() + 1
        # Update cluster assignments.
        if n_clusters == 1:
            # Single homogeneous cluster.
            cluster_labels.fill(0)
        elif n_clusters == precursor_mzs.shape[0]:
            # Only singletons.
            n_clusters = 0
        else:
            unique, inverse, counts = np.unique(
                cluster_assignments, return_inverse=True, return_counts=True)
            non_noise_clusters = np.where(counts >= min_samples)[0]
            labels = -np.ones_like(unique)
            labels[non_noise_clusters] = np.unique(unique[non_noise_clusters],
                                                   return_inverse=True)[1]
            cluster_labels[:] = labels[inverse]
            n_clusters = len(non_noise_clusters)
    return n_clusters


def _linkage(values: np.ndarray, tol_mode: str = None) -> np.ndarray:
    """
    Perform hierarchical clustering of a one-dimensional m/z or RT array.

    Because the data is one-dimensional, no pairwise distance matrix needs to
    be computed, but rather sorting can be used.
    
    Parameters
    ----------
    values : np.ndarray
        The precursor m/z's for which pairwise distances are computed.
    tol_mode : str
        The unit of the tolerance ('Da' or 'ppm' for precursor m/z).

    Returns
    -------
    np.ndarray
        The hierarchical clustering encoded as a linkage matrix.
    """
    linkage = np.zeros((values.shape[0] - 1, 4), np.double)
    # min, max, cluster index, number of cluster elements
    # noinspection PyUnresolvedReferences
    clusters = [(values[i], values[i], i, 1) for i in np.argsort(values)]
    for it in range(values.shape[0] - 1):
        min_dist, min_i = np.inf, -1
        for i in range(len(clusters) - 1):
            dist = clusters[i + 1][1] - clusters[i][0]  # Always positive.
            if tol_mode == 'ppm':
                dist = dist / clusters[i][0] * 10 ** 6
            if dist < min_dist:
                min_dist, min_i = dist, i
        n_points = clusters[min_i][3] + clusters[min_i + 1][3]
        linkage[it, :] = [clusters[min_i][2], clusters[min_i + 1][2],
                          min_dist, n_points]
        clusters[min_i] = (clusters[min_i][0], clusters[min_i + 1][1],
                           values.shape[0] + it, n_points)
        del clusters[min_i + 1]

    return linkage


def _assign_unique_cluster_labels(cluster_labels: np.ndarray,
                                  group_idx: List[Tuple[int, int]],
                                  n_clusters: List[int],
                                  min_samples: int) -> None:
    """
    Make sure all cluster labels are unique after potential splitting of
    clusters to avoid excessive precursor m/z differences.

    Parameters
    ----------
    cluster_labels : np.ndarray
        Cluster labels per cluster grouping.
    group_idx : nb.typed.List[Tuple[int, int]]
        Tuples with the start index (inclusive) and end index (exclusive) of
        the cluster groupings.
    n_clusters: nb.typed.List[int]
        The number of clusters per cluster grouping.
    min_samples : int
        The minimum number of samples in a cluster.
    """
    current_label = 0
    for (start_i, stop_i), n_cluster in zip(group_idx, n_clusters):
        if n_cluster > 0 and stop_i - start_i >= min_samples:
            current_labels = cluster_labels[start_i:stop_i]
            current_labels[current_labels != -1] += current_label
            current_label += n_cluster
        else:
            cluster_labels[start_i:stop_i].fill(-1)