import os 
import sys
import logging

import itertools as it
import numpy as np
import matplotlib.pyplot as plt


from typing import Dict, Iterator, List, Tuple, Union

import multiprocessing

from ms_io import mgf_io
from cluster import similarity, masking, clustering
from plot import network
from eval import eval

logger = logging.getLogger('falcon_ext')

def main(args: Union[str, List[str]] = None) -> int:

    # read file
    spec_filename = sys.argv[-2]
    anno_filename = sys.argv[-1]

    if not os.path.isfile(spec_filename):
        raise ValueError(f'Non-existing peak file (spec_filename)')
    if not os.path.isfile(anno_filename):
        raise ValueError(f'Non-existing annotations file (anno_filename)')

    print('Reading MGF file ...')

    spectra = list(mgf_io.get_spectra(spec_filename))
    spectra.sort(key=lambda x: x.precursor_mz)
    # spectra = spectra[33:40]
    spectra = spectra[:200]
    n_spectra = len(spectra)

    scan_idx_list = [int(spec.identifier) for spec in spectra]

    # calculate pairwise mod cos similarity
    print('Calculating modified cosine similarity ...')
    similarity_matrix = similarity.create_mod_cos_similarity_matrix(spectra)
    distance_matrix = similarity.similarity_to_distance(similarity_matrix)

    # create masked distance matrix for clustering based on precursor mass
    print('Generating mask ...')
    mask = masking.generate_mask(spectra, 0.05)
    masked_distance_matrix = similarity.similarity_to_distance(\
        np.multiply(similarity_matrix, mask))
    # deal with floating point inaccuracy 
    # np.clip results in "ValueError: Linkage 'Z' uses the same cluster more than once." when plotting dendrogram
    masked_distance_matrix = np.where(masked_distance_matrix>0, masked_distance_matrix, 0)

    # cluster spectra and plot dendrogram
    print('Clustering...')
    cluster = clustering.generate_clusters(masked_distance_matrix)
    clustering.plot_dendrogram(cluster, labels=cluster.labels_)

    # plot molecular network before and after clustering
    network.network_from_distance_matrix(spectra, distance_matrix, 0.2)
    # get cluster medoids
    medoids = clustering.get_medoids(masked_distance_matrix, cluster)
    network.network_from_clusters(spectra, medoids, distance_matrix, 0.2)

    # evaluate clustering
    print('Cluster evaluation...')
    eval.evaluate_clustering(anno_filename, cluster, scan_idx_list)

    clustering.clusters_to_csv(cluster, scan_idx_list)

    plt.show() # keep figures alive

    return 0


if __name__ == '__main__':
    sys.exit(main())