from typing import List, Tuple

from spectrum_utils.spectrum import MsmsSpectrum

from matchms import Spectrum
from matchms.similarity import ModifiedCosine

import numpy as np

from multiprocessing.pool import ThreadPool

import itertools as it


def create_mod_cos_dist_matrix(
        spectra: List[MsmsSpectrum], 
        n_threads: int = 512) -> np.ndarray:
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
        Squarre matrix containing all pairwise modified cosine similarities.
    """
    with ThreadPool(n_threads) as pool:
        dist_list = pool.starmap(
            get_modified_cosine_similarity, 
            it.combinations(spectra, 2)
            )

    dist_matrix = create_dist_matrix(dist_list, len(spectra))

    print(dist_matrix)

    assert np.allclose(dist_matrix, dist_matrix.T, rtol=1e-05, atol=1e-08), \
        f"Distance matrix not symmetric"
    
    return dist_matrix


def get_modified_cosine_similarity(
        spec1: MsmsSpectrum, 
        spec2: MsmsSpectrum) -> float:
    """
    Get the modified cosine similarity of the input spectra

    Parameters
    ----------
    spec1 : MsmsSpectrum
        MS/MS spectrum.
    spec2 : MsmsSpectrum
        MS/MS spectrum.

    Returns
    -------
    float
        Modified cosine similarity of the input spectra.
    """
    return modified_cosine_similarity(spec1, spec2).item()[0]


def create_dist_matrix(dist_list: List[float], n_spectra: int) -> np.ndarray:
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


def modified_cosine_similarity(
    spectrum1: MsmsSpectrum, 
    spectrum2: MsmsSpectrum) -> Tuple[float, int]:
    """
    Calculate the modified cosine similarty for the given spectra.

    Parameters
    ----------
    spectrum1 : MsmsSpectrum
        input MS/MS spectrum.
    spectrum2 : MsmsSpectrum
        input MS/MS spectrum.

    Returns
    -------
    Tuple[float, int]
        modified cosine similarity and number of matched peaks.
    """
    spec1 = _convert_to_spectrum(spectrum1)
    spec2 = _convert_to_spectrum(spectrum2)

    mod_cosine = ModifiedCosine()

    return mod_cosine.pair(spec1, spec2)


def _convert_to_spectrum(msms_spectrum : MsmsSpectrum) -> Spectrum:
    """
    Convert MsmsSpectrum to matchms Spectrum.

    Parameters
    ----------
    msms_spectrum : MsmsSpectrum
        input MS/MS spectrum.

    Returns
    -------
    Spectrum
        MS/MS spectrum as matchms Spectrum.
    """
    return Spectrum(
        mz=msms_spectrum.mz.astype(float),
        intensities=msms_spectrum.intensity.astype(float),
        metadata={
            "id": msms_spectrum.identifier,
            "precursor_mz": msms_spectrum.precursor_mz
        }
    )