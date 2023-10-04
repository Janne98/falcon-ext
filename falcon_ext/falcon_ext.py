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
import config
from config import *
from preprocessing import preprocessing

logger = logging.getLogger('falcon_ext')

def main(args: Union[str, List[str]] = None) -> int:

	config.parse(args)
	# read file
	spec_filename = config.input_filenames
	anno_filename = config.annotations_file

	if not os.path.isfile(spec_filename):
		raise ValueError(f'Non-existing peak file (spec_filename)')
	if not os.path.isfile(anno_filename):
		raise ValueError(f'Non-existing annotations file (anno_filename)')

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
	# spectra = spectra[33:40]
	spectra = spectra[:200]

	scan_idx_list = [int(spec.identifier) for spec in spectra]

	# calculate pairwise mod cos similarity
	print('Calculating modified cosine similarity ...')
	similarity_matrix = similarity.create_mod_cos_similarity_matrix(spectra=spectra, \
																    fragment_tol=config.fragment_tol)
	distance_matrix = similarity.similarity_to_distance(similarity_matrix=similarity_matrix)

	# create masked distance matrix for clustering based on precursor mass
	print('Generating mask ...')
	mask = masking.generate_mask(spectra=spectra, mz_tolerance=config.precursor_tol)
	masked_distance_matrix = similarity.similarity_to_distance(\
		similarity_matrix=np.multiply(similarity_matrix, mask))
	# deal with floating point inaccuracy 
	# np.clip results in "ValueError: Linkage 'Z' uses the same cluster more than once." when plotting dendrogram
	masked_distance_matrix = np.where(masked_distance_matrix>0, masked_distance_matrix, 0)

	# cluster spectra and plot dendrogram
	print('Clustering...')
	cluster = clustering.generate_clusters(dist_matrix=masked_distance_matrix, 
										   linkage=config.linkage,
										   distance_threshold=config.max_cluster_dist)
	clustering.plot_dendrogram(clustering=cluster, labels=cluster.labels_)

	# plot molecular network before and after clustering
	network.network_from_distance_matrix(spectra=spectra, dist_matrix=distance_matrix, 
									     max_edges=config.max_edges, max_edge_dist=config.max_edge_dist)
	# get cluster medoids
	medoids = clustering.get_medoids(dist_matrix=masked_distance_matrix, clustering=cluster)
	network.network_from_clusters(spectra=spectra, medoids=medoids, dist_matrix=distance_matrix, 
							      max_edges=config.max_edges, max_edge_dist=config.max_edge_dist)

	# evaluate clustering
	print('Cluster evaluation...')
	eval.evaluate_clustering(filename=anno_filename, clustering=cluster, \
							 spec_map=scan_idx_list)

	# IO
	clustering.clusters_to_csv(clustering=cluster, spec_map=scan_idx_list)

	plt.show() # keep figures alive

	return 0


if __name__ == '__main__':
	sys.exit(main())