import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict

from spectrum_utils.spectrum import MsmsSpectrum


def network_from_distance_matrix(
        spectra: List[MsmsSpectrum], 
        dist_matrix: np.ndarray,
        max_edges: int,
        max_edge_dist: float) -> None:
    """
    Plot the molecular network starting from the distance matrix.

    Parameters
    ----------
    spectra: List[MsmsSpectrum]
        List of MS/MS spectra.
    dist_matrix: np.ndarray
        pairwise distance matrix of the spectra.
    max_edges:
        amount of edges to add for each node. If none, add all edges.
    max_edge_dist:
        maximum pairwise distance above which no edge will be added to the network.
    """
    graph = nx.Graph()
    graph.add_nodes_from(spectra)

    label_dict = {}
    for spec in spectra:
        label_dict[spec] = spec.precursor_mz

    graph = _add_edges(graph=graph, spectra=spectra, dist_matrix=dist_matrix, 
                       max_edges=max_edges, max_edge_dist=max_edge_dist)

    fig = plt.figure("Molecular network before clustering")
    nx.draw(graph, labels=label_dict, with_labels=True)
    fig.show()


def network_from_clusters(
        spectra: List[MsmsSpectrum],
        core_samples: np.ndarray,
        noise_samples: np.ndarray,
        dist_matrix: np.ndarray,
        max_edges: int,
        max_edge_dist: float) -> None:
    """
    Plot the molecular network from the distance matrix and cluster medoids.

    Parameters
    ----------
    spectra: List[MsmsSpectrum]
        List of MS/MS spectra.
    core_samples: np.ndarray
        list of indices of core samples.
    dist_matrix: np.ndarray
        pairwise distance matrix of the spectra.
    max_edges:
        amount of edges to add for each node. If none, add all edges.
    max_edge_dist:
        maximum pairwise distance above which no edge will be added to the network.
    """
    slice_samples = np.concatenate((core_samples, noise_samples))
    slice_samples.sort()
    spec_slice = [(idx, spectra[idx]) for idx in slice_samples]
    dist_slice = np.array([[dist_matrix[idx][idy] for idy in slice_samples] \
                           for idx in slice_samples])

    graph = nx.Graph()
    graph.add_nodes_from([x[1] for x in spec_slice])

    graph = _add_edges(graph=graph, spectra=[x[1] for x in spec_slice], 
                       dist_matrix=dist_slice, max_edges=max_edges, 
                       max_edge_dist=max_edge_dist)

    label_dict = {}
    node_size = []
    node_color = []
    for idx, spec in spec_slice:
        label_dict[spec] = spec.precursor_mz
        node_size.append(300.0 if idx in core_samples else 100.0)
        node_color.append('#1f78b4' if idx in core_samples else '#91aaa9')

    fig = plt.figure("Molecular network after clustering")
    nx.draw(graph, labels=label_dict, with_labels=True, 
            node_size=node_size, node_color=node_color) 
    fig.show()

    
def _add_edges(
    graph: nx.Graph, 
    spectra: List[MsmsSpectrum], 
    dist_matrix: np.ndarray, 
    max_edges: int,
    max_edge_dist: float) -> nx.Graph:
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
    max_edges:
        amount of edges to add for each node. If none, add all edges.
    max_edge_dist:
        maximum pairwise distance above which no edge will be added to the network.

    Returns
    -------
    nx.Graph
        graph with edges (if added any)
    """
    # only upper triangle of dist matrix, diagonal not included -> add once
    for i in range(dist_matrix.shape[0]):
        if max_edges and i+1 < dist_matrix.shape[0] and max_edges < len(dist_matrix[i][i+1:]):
            nearest_nodes = np.argpartition(dist_matrix[i][i+1:], max_edges)[:max_edges]
            for j in nearest_nodes:
                if dist_matrix[i][j+i+1] < max_edge_dist:
                    if max_edges:
                        graph.add_edge(spectra[i], spectra[j+i+1])
        else: 
            nearest_nodes = range(i+1, dist_matrix.shape[1])
            for j in nearest_nodes: 
                if dist_matrix[i][j] < max_edge_dist:
                    graph.add_edge(spectra[i], spectra[j])
    
    return graph