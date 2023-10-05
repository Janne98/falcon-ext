import time

from typing import List, Tuple

from spectrum_utils.spectrum import MsmsSpectrum

from matchms import Spectrum
from matchms.similarity import ModifiedCosine

import numpy as np
import itertools as it
import scipy.sparse as ss

from multiprocessing.pool import ThreadPool

from . import cosine


def create_mod_cos_similarity_matrix(
        spectra: List[MsmsSpectrum], 
        fragment_tol: float,
        n_threads: int = 8) -> np.ndarray:
    """
    Create a distance matrix containing the pairwise modified cosine similarity 
        for the input spectra.

    Parameters
    ----------
    spectra : List[MsmsSpectrum]
        list of spectra.
    n_threads : int
        number of threads to use for similarity calculation.

    Returns
    -------
    np.ndarray
        Squarre matrix containing all pairwise modified cosine distances.
    """
    mod_cosine_args = []
    spec_combos = it.combinations(spectra, 2)
    for combo in spec_combos:
        mod_cosine_args.append(combo + (fragment_tol,))

    with ThreadPool(n_threads) as pool:
        similarity_list = pool.starmap(
            _get_modified_cosine_similarity, 
            mod_cosine_args
            )

    distance_matrix = _list_to_matrix(similarity_list, len(spectra))
    
    assert np.allclose(distance_matrix, distance_matrix.T, rtol=1e-05, atol=1e-08), \
        f"Distance matrix not symmetric"
    
    return distance_matrix


def _get_modified_cosine_similarity(
        spec1: MsmsSpectrum,
        spec2: MsmsSpectrum,
        fragment_tol: float) -> float:
    return cosine.modified_cosine(spec1, spec2, fragment_tol).score


def _list_to_matrix(dist_list: List[float], n_spectra: int) -> np.ndarray:
    """
    Create a distance matrix from a list only containing the distances 
        above the diagonal, similarity metric must be symmetric.

    Parameters
    ----------
    dist_list : List[float]
        list of distances.
    n_spectra : int
        number of spectra in distance matrix

    Returns
    -------
    np.ndarray
        Squarre matrix containing all pairwise distances.
    """
    dist_matrix = np.zeros((n_spectra, n_spectra))

    loc_u = np.triu_indices(n_spectra, k=1)
    loc_l = np.tril_indices(n_spectra, k=-1)
    loc_d = np.diag_indices(n_spectra)

    dist_matrix[loc_u] = dist_list
    dist_matrix[loc_l] = dist_matrix.T[loc_l]
    dist_matrix[loc_d] = 1

    return dist_matrix


def similarity_to_distance(similarity_matrix: np.ndarray) -> np.ndarray:
    """
    Convert similarity matrix to distance matrix (distance = 1 - similarity).

    Parameters
    ----------
    similarity_matrix : np.ndarray
        Square 2D array of pairwise similarities.
    
    Returns
    -------
    np.ndarray
        Squarre matrix containing all pairwise distances.
    """
    return 1 - similarity_matrix

def save_matrix(matrix:np.ndarray, filename: str) -> None:
    """
    Save matrix as npz file.

    Parameters
    ----------
    matrix : np.ndarray
        Matrix to save.
    filename : str
        Name of the npz file.
    """
    np.savez(filename, matrix=matrix)

def load_matrix(filename: str) -> np.ndarray:
    """
    Load matrix from npz file.

    Parameters
    ----------
    filename : str
        Name of the file containing a numpy array.
    
    Returns
    -------
    np.ndarray
        Matrix loaded from file.
    """
    matrix = np.load(filename)
    return matrix['arr_0']