import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from typing import List, Tuple, Dict

from spectrum_utils.spectrum import MsmsSpectrum

import json


def network_from_distance_matrix(
        spectra: List[MsmsSpectrum], 
        dist_matrix: np.ndarray,
        match_matrix: np.ndarray,
        max_edges: int,
        max_edge_dist: float,
        min_matched_peaks: int) -> None:
    """
    Plot the molecular network starting from the distance matrix.

    Parameters
    ----------
    spectra: List[MsmsSpectrum]
        List of MS/MS spectra.
    dist_matrix: np.ndarray
        pairwise distance matrix of the spectra.
    match_matrix: np.ndarray
        number of matched peaks between spectra.
    max_edges:
        amount of edges to add for each node. If none, add all edges.
    max_edge_dist:
        maximum pairwise distance above which no edge will be added to the network.
    min_matched_peaks:
        minimum number of matched peaks to add an edge to the network.
    """
    graph = nx.Graph()
    graph.add_nodes_from(spectra)

    graph = _add_edges(graph=graph, spectra=spectra, dist_matrix=dist_matrix, 
                       match_matrix=match_matrix, max_edges=max_edges, 
                       max_edge_dist=max_edge_dist, min_matched_peaks=min_matched_peaks)

    label_dict = {}
    for idx, spec in enumerate(spectra):
        label_dict[spec] = spec.precursor_mz

    fig = plt.figure("Molecular network before clustering")
    nx.draw(graph, pos=nx.spring_layout(graph, k=0.7), labels=label_dict, 
            with_labels=True)
    fig.show()

    print('Number of nodes before clustering:', _count_nodes(graph))
    degree_dict = _network_degree_distribution(graph)
    _degree_distribution_plot(degree_dict, before_clustering=True)
    print('Average degree before clustering:', str(_average_degree(graph)))

    nx.write_gml(graph, 'network_before_clustering.gml')


def network_from_clusters(
        spectra: List[MsmsSpectrum],
        core_samples: np.ndarray,
        noise_samples: np.ndarray,
        dist_matrix: np.ndarray,
        match_matrix: np.ndarray,
        max_edges: int,
        max_edge_dist: float,
        min_matched_peaks: int) -> None:
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
    match_matrix: np.ndarray
        number of matched peaks between spectra.
    max_edges:
        amount of edges to add for each node. If none, add all edges.
    max_edge_dist:
        maximum pairwise distance above which no edge will be added to the network.
    min_matched_peaks:
        minimum number of matched peaks to add an edge to the network.
    """
    slice_samples = np.concatenate((core_samples, noise_samples))
    slice_samples.sort()
    spec_slice = [(idx, spectra[idx]) for idx in slice_samples]
    dist_slice = np.array([[dist_matrix[idx][idy] for idy in slice_samples] \
                           for idx in slice_samples])
    match_slice = np.array([[match_matrix[idx][idy] for idy in slice_samples] \
                           for idx in slice_samples])

    graph = nx.Graph()
    graph.add_nodes_from([x[1] for x in spec_slice])

    graph = _add_edges(graph=graph, spectra=[x[1] for x in spec_slice], 
                       dist_matrix=dist_slice, match_matrix=match_slice, 
                       max_edges=max_edges, max_edge_dist=max_edge_dist, 
                       min_matched_peaks=min_matched_peaks)

    label_dict = {}
    node_size = []
    node_color = []
    for idx, spec in spec_slice:
        label_dict[spec] = spec.precursor_mz
        node_size.append(300.0 if idx in core_samples else 150.0)
        node_color.append('#1f78b4' if idx in core_samples else '#91aaa9')

    fig = plt.figure("Molecular network after clustering")
    nx.draw(graph, pos=nx.spring_layout(graph, k=0.7), labels=label_dict,
            with_labels=True, node_size=node_size, node_color=node_color) #fruchterman_reingold_layout
    fig.show()

    print('Number of nodes after clustering:', _count_nodes(graph))
    degree_dict = _network_degree_distribution(graph)
    _degree_distribution_plot(degree_dict, before_clustering=False)
    print('Average degree after clustering:', str(_average_degree(graph)))

    nx.write_gml(graph, 'network_after_clustering.gml')

    
def _add_edges(
    graph: nx.Graph, 
    spectra: List[MsmsSpectrum], 
    dist_matrix: np.ndarray, 
    match_matrix: np.ndarray,
    max_edges: int,
    max_edge_dist: float,
    min_matched_peaks: int) -> nx.Graph:
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
    match_matrix: np.ndarray
        number of matched peaks between spectra.
    max_edges:
        amount of edges to add for each node. If none, add all edges.
    max_edge_dist:
        maximum pairwise distance above which no edge will be added to the network.
    min_matched_peaks:
        minimum number of matched peaks to add an edge to the network.

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
                if dist_matrix[i][j+i+1] < max_edge_dist and match_matrix[i][j+i+1] >= min_matched_peaks:
                    if max_edges:
                        graph.add_edge(spectra[i], spectra[j+i+1])
        else: 
            nearest_nodes = range(i+1, dist_matrix.shape[1])
            for j in nearest_nodes: 
                if dist_matrix[i][j] < max_edge_dist and match_matrix[i][j] >= min_matched_peaks:
                    graph.add_edge(spectra[i], spectra[j])
    
    return graph


def _count_nodes(graph: nx.Graph) -> int:
    """
    Count nodes in the graph.

    Parameters
    ----------
    graph: nx.Graph
        graph to count nodes in.

    Returns
    -------
    int
        number of nodes in the graph.
    """
    return len(graph.nodes)


def _network_degree_distribution(graph: nx.Graph) -> Dict[int, int]:
    """
    Calculate the degree distribution of the network.

    Parameters
    ----------
    graph: nx.Graph
        graph to calculate degree distribution for.

    Returns
    -------
    Dict[int, int]
        dictionary with degree distribution.
    """
    degree_dict = {}
    for node in graph.nodes:
        degree = graph.degree(node)
        if degree in degree_dict.keys():
            degree_dict[degree] += 1
        else:
            degree_dict[degree] = 1
    return degree_dict


def _degree_distribution_plot(degree_dict: Dict[int, int], before_clustering: bool) -> None:
    """
    Plot the degree distribution of the network.

    Parameters
    ----------
    degree_dict: Dict[int, int]
        dictionary with degree distribution.
    """
    if before_clustering:
        fig = plt.figure("Degree distribution before clustering")
    else:
        fig = plt.figure("Degree distribution after clustering")
    plt.bar(degree_dict.keys(), degree_dict.values())
    plt.xlabel('Degree')
    plt.ylabel('Number of nodes')
    fig.show()


def _average_degree(graph: nx.Graph) -> float:
    """
    Calculate the average degree of the network.

    Parameters
    ----------
    graph: nx.Graph
        graph to calculate average degree for.

    Returns
    -------
    float
        average degree of the network.
    """
    return sum([graph.degree(node) for node in graph.nodes]) / len(graph.nodes)


def _degree_standard_deviation(graph: nx.Graph) -> float:
    """
    Calculate the standard deviation of the degree of the network.

    Parameters
    ----------
    graph: nx.Graph
        graph to calculate standard deviation of degree for.

    Returns
    -------
    float
        standard deviation of the degree of the network.
    """
    return np.std([graph.degree(node) for node in graph.nodes]
)