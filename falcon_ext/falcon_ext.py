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
    spectra = spectra[:500]
    n_spectra = len(spectra)

    # calculate pairwise mod cos similarity
    print('Calculating modified cosine similarity ...')

    similarity_matrix = similarity.create_mod_cos_dist_matrix(spectra)

    print('Generating mask ...')

    mask = masking.generate_mask(spectra, 0.05)
    #print(mask)

    # print([spec.precursor_mz for spec in spectra])

    distance_matrix = similarity.similarity_to_distance(
        np.multiply(similarity_matrix, mask))
    # deal with floating point inaccuracy 
    # np.clip results in "ValueError: Linkage 'Z' uses the same cluster more than once." when plotting dendrogram
    distance_matrix = np.where(distance_matrix>0, distance_matrix, 0)

    cluster = clustering.generate_clusters(distance_matrix)
    clustering.plot_dendrogram(cluster, labels=cluster.labels_)

    network.network_from_distance_matrix(spectra, distance_matrix)

    medoids = clustering.get_medoids(distance_matrix, cluster)

    network.network_from_clusters(spectra, medoids, distance_matrix)

    eval.evaluate_clustering(anno_filename, cluster)

    plt.show() # keep figures alive

    return 0


if __name__ == '__main__':
    sys.exit(main())


"""
Traceback (most recent call last):
  File "/home/janne/falcon-ext/falcon_ext/falcon_ext.py", line 62, in <module>
    sys.exit(main())
             ^^^^^^
  File "/home/janne/falcon-ext/falcon_ext/falcon_ext.py", line 52, in main
    clustering.plot_dendrogram(cluster)
  File "/home/janne/falcon-ext/falcon_ext/cluster/clustering.py", line 115, in plot_dendrogram
    dendrogram(linkage_matrix, **kwargs)
  File "/home/janne/anaconda3/envs/falcon-env/lib/python3.11/site-packages/scipy/cluster/hierarchy.py", line 3307, in dendrogram
    is_valid_linkage(Z, throw=True, name='Z')
  File "/home/janne/anaconda3/envs/falcon-env/lib/python3.11/site-packages/scipy/cluster/hierarchy.py", line 2293, in is_valid_linkage
    raise ValueError('Linkage %suses the same cluster more than once.'
ValueError: Linkage 'Z' uses the same cluster more than once.
"""