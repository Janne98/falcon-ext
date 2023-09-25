import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

from typing import List

from spectrum_utils.spectrum import MsmsSpectrum

def network_from_distance_matrix(
        spectra: List[MsmsSpectrum], 
        dist_matrix: np.ndarray) -> None:

    graph = nx.Graph()
    graph.add_nodes_from(spectra)

    label_dict = {}
    for spec in spectra:
        label_dict[spec] = spec.precursor_mz

    graph = _add_edges(graph, spectra, dist_matrix)

    fig = plt.figure(2)
    nx.draw(graph, labels=label_dict, with_labels=True)
    fig.show()

    
def _add_edges(
    graph: nx.Graph, 
    spectra: List[MsmsSpectrum], 
    dist_matrix: np.ndarray) -> nx.Graph:

    # only upper triangle of dist matrix, diagonal not included
    for i in range(dist_matrix.shape[0]):
        for j in range(i+1, dist_matrix.shape[1]):
            if dist_matrix[i][j] < 0.35:
                graph.add_edge(spectra[i], spectra[j])
    
    return graph



