import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict

from spectrum_utils.spectrum import MsmsSpectrum

def network_from_distance_matrix(
        spectra: List[MsmsSpectrum], 
        dist_matrix: np.ndarray) -> None:
    """
    Plot the molecular network starting from the distance matrix.

    Parameters
    ----------
    spectra: List[MsmsSpectrum]
        List of MS/MS spectra.
    dist_matrix: np.ndarray
        pairwise distance matrix of the spectra.
    """
    graph = nx.Graph()
    graph.add_nodes_from(spectra)

    label_dict = {}
    for spec in spectra:
        label_dict[spec] = spec.precursor_mz

    graph = _add_edges(graph, spectra, dist_matrix)

    fig = plt.figure("Molecular network before clustering")
    nx.draw(graph, labels=label_dict, with_labels=True)
    fig.show()


def network_from_clusters(
        spectra: List[MsmsSpectrum],
        medoids: Dict[int, Tuple[int, int]],
        dist_matrix: np.ndarray) -> None:
    """
    Plot the molecular network from the distance matrix and cluster medoids.

    Parameters
    ----------
    spectra: List[MsmsSpectrum]
        List of MS/MS spectra.
    medoids: Dict[int, Tuple[int, int]]
        dictionary of {clusted_idx: (cluster size, medoid_spectrum_idx)}-format,
        contains the cluster size and spectrum index of the medoid for each cluster.
    dist_matrix: np.ndarray
        pairwise distance matrix of the spectra.
    """
    medoids_idx = [idx for _, (_, idx) in medoids.items()]
    spec_slice = [spectra[idx] for idx in medoids_idx]
    dist_slice = np.array([[dist_matrix[idx][idy] for idy in medoids_idx] \
                           for idx in medoids_idx])

    graph = nx.Graph()
    graph.add_nodes_from(spec_slice)

    graph = _add_edges(graph, spec_slice, dist_slice)

    label_dict = {}
    for spec in spec_slice:
        label_dict[spec] = spec.precursor_mz

    node_size = [cluster_size * 200 for _, (cluster_size, _) in medoids.items()] # default 300

    fig = plt.figure("Molecular network after clustering")
    nx.draw(graph, labels=label_dict, with_labels=True, node_size=node_size)
    fig.show()

    
def _add_edges(
    graph: nx.Graph, 
    spectra: List[MsmsSpectrum], 
    dist_matrix: np.ndarray) -> nx.Graph:
    """
    Add edges between noded in the network.

    Parameters
    ----------
    graph: nx.Graph
        graph to add edges to.
    spectra: List[MsmsSpectrum]
        List of MS/MS spectra.
    dist_matrix: np.ndarray
        pairwise distance matrix of the spectra.

    Returns
    -------
    nx.Graph
        graph with edges (if added any)
    """
    # only upper triangle of dist matrix, diagonal not included
    for i in range(dist_matrix.shape[0]):
        for j in range(i+1, dist_matrix.shape[1]):
            if dist_matrix[i][j] < 0.35:
                graph.add_edge(spectra[i], spectra[j])
    
    return graph