import os 
import sys
import logging

import itertools as it
import numpy as np
import matplotlib.pyplot as plt

from typing import List, Tuple, Union

from ms_io import mgf_io, mzml_io
from cluster import similarity, masking, clustering
from plot import network
from eval import eval
import config
from config import *
from preprocessing import preprocessing

from skopt import gp_minimize
import functools

logger = logging.getLogger('falcon_ext')

def main(args: Union[str, List[str]] = None) -> int:

    config.parse(args)

    # check filenames
    spec_filename = config.input_filenames
    anno_filename = config.annotations_file

    if not os.path.isfile(spec_filename):
        raise ValueError(f'Non-existing peak file (spec_filename)')
    if not os.path.isfile(anno_filename):
        raise ValueError(f'Non-existing annotations file (anno_filename)')
    if config.dist_matrix_file is not None:
        if not os.path.isfile(config.dist_matrix_file):
            raise ValueError(f'Non-existing distance matrix file (dist_matrix_file)')

    # read file and process spectra
    print('Reading MGF file ...')
    raw_spectra = mgf_io.get_spectra(source=spec_filename)
    spectra = list(preprocessing.process_all_spectra(raw_spectra, 
                                                    config.min_peaks, config.min_mz_range,
                                                    config.min_mz, config.max_mz,
                                                    config.remove_precursor_tol,
                                                    config.min_intensity, 
                                                    config.max_peaks_used, config.scaling))
    spectra = [spectrum for spectrum in spectra if spectrum is not None]
    spectra.sort(key=lambda x: x.precursor_mz)
    # spectra = spectra[30:36]
    # spectra = spectra[:200]

    scan_idx_list = [int(spec.identifier) for spec in spectra]

    if config.dist_matrix_file is not None and config.matches_matrix_file is not None:
        # read distance matrix file and create similarity matrix
        distance_matrix = similarity.load_matrix(config.dist_matrix_file)
        matches_matrix = similarity.load_matrix(config.matches_matrix_file)
        similarity_matrix = similarity.similarity_to_distance(distance_matrix)
    else:
        # calculate pairwise mod cos similarity
        print('Calculating modified cosine similarity ...')
        similarity_matrix, matches_matrix = similarity.create_mod_cos_similarity_matrix(spectra, 
                                                                        config.fragment_tol)
        distance_matrix = similarity.similarity_to_distance(similarity_matrix)
        if config.export_dist_matrix:
            similarity.save_matrix(distance_matrix, 'distance_matrix.npz')
            similarity.save_matrix(matches_matrix, 'matches_matrix.npz')

    # create masked distance matrix for clustering based on precursor mass
    print('Generating mask ...')
    mask = masking.generate_mask(spectra, config.precursor_tol[0], config.precursor_tol[1])
    masked_distance_matrix = similarity.similarity_to_distance(np.multiply(similarity_matrix, mask))
    # deal with floating point inaccuracy 
    # np.clip results in "ValueError: Linkage 'Z' uses the same cluster more than once." when plotting dendrogram
    masked_distance_matrix = np.where(masked_distance_matrix>0, masked_distance_matrix, 0)

    # cluster spectra (and plot dendrogram)
    print('Clustering...')
    cluster = clustering.generate_clusters(masked_distance_matrix, config.cluster_method, 
                                           config.linkage, config.eps, config.min_cluster_size)

    # plot molecular network before and after clustering
    print('---NETWORK EVALUATION---')
    network.network_from_distance_matrix(spectra, distance_matrix, matches_matrix, config.max_edges, 
                                         config.max_edge_dist, config.min_matched_peaks)
    network.network_from_clusters(spectra, cluster.cluster_samples, cluster.noise_samples, 
                                  distance_matrix, matches_matrix, config.max_edges, 
                                  config.max_edge_dist, config.min_matched_peaks)

    # evaluate clustering
    print('---CLUSTER EVALUATION---')
    eval.evaluate_clustering(anno_filename, cluster, scan_idx_list)

    # IO
    clustering.clusters_to_csv(cluster, scan_idx_list)

    plt.show() # keep figures alive

    return 0


def run_experiment(eps: float, min_cluster_size: int, masked_dist_matrix: np.ndarray, 
                   annotations_file: str, idx_scan_map: List[int], cluster_method: str, 
                   linkage: str) -> Tuple[int, float, float, float]:
    
    cluster = clustering.generate_clusters(masked_dist_matrix, cluster_method, 
                                           linkage, eps, min_cluster_size)
    eval_result = eval.evaluate_clustering(annotations_file, cluster, idx_scan_map)

    return eval_result


def get_pareto_frontier(arr: np.ndarray) -> np.ndarray:
    # Sort by the first column.
    arr_sorted = arr[arr[:, 0].argsort()]
    # Iteratively add points to the Pareto frontier.
    pareto_idx = [0]
    for i in range(1, arr_sorted.shape[0]):
        if (arr_sorted[i, 0] > arr_sorted[pareto_idx[-1], 0] and
                arr_sorted[i, 1] > arr_sorted[pareto_idx[-1], 1]):
            pareto_idx.append(i)
    return arr_sorted[pareto_idx]


if __name__ == '__main__':
	sys.exit(main())