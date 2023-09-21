import os 
import sys
import logging

import itertools as it
import numpy as np

from typing import Dict, Iterator, List, Tuple, Union

import multiprocessing

from ms_io import mgf_io
from cluster import similarity, masking, clustering
from plot import network

logger = logging.getLogger('falcon_ext')

def main(args: Union[str, List[str]] = None) -> int:

    print(multiprocessing.cpu_count())

    # read file
    filename = sys.argv[-1]
    if not os.path.isfile(filename):
        raise ValueError(f'Non-existing peak file (filename)')

    print('Reading MGF file ...')

    spectra = list(mgf_io.get_spectra(filename))
    spectra.sort(key=lambda x: x.precursor_mz)
    spectra = spectra[33:40]
    n_spectra = len(spectra)

    # calculate pairwise mod cos similarity
    print('Calculating modified cosine similarity ...')

    similarity_matrix = similarity.create_mod_cos_dist_matrix(spectra)

    print('Generating mask ...')

    mask = masking.generate_mask(spectra, 0.05)
    print(mask)

    print([spec.precursor_mz for spec in spectra])

    distance_matrix = similarity.similarity_to_distance(
        np.multiply(similarity_matrix, mask))
    print(distance_matrix)

    cluster = clustering.generate_clusters(distance_matrix)
    #clustering.plot_dendrogram(cluster)

    #network.network_from_distance_matrix(spectra, distance_matrix)

    clustering.get_medoids(distance_matrix, cluster)
    return 0


if __name__ == '__main__':
    sys.exit(main())