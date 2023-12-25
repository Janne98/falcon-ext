import numpy as np

from typing import List

from spectrum_utils.spectrum import MsmsSpectrum


def generate_mask(spectra: List[MsmsSpectrum], mz_tol: float, tol_mode: str) -> np.ndarray:
    """
    Generate a mask indicating which spectrum pairs have a near-identical precursor m/z.

    Parameters
    ----------
    spectra : List[MsmsSpectrum]
        list of MS/MS spectra.
    mz_tol : float
        tolerance on precursor m/z to be considered near-identical

    Returns
    -------
    np.ndarray
        Squarre matrix (mask). 
        1 if near-identical precursor m/z, 0 otherwise.
    """
    mask = np.zeros((len(spectra), len(spectra)))

    for i, spec in enumerate(spectra):
        j = i
        while j < len(spectra) and _check_precursor_mz_tolerance(spec, spectra[j], mz_tol, tol_mode):
            mask[i][j] = 1
            mask[j][i] = 1
            j += 1

    return mask


def _check_precursor_mz_tolerance(spec1: MsmsSpectrum, spec2: MsmsSpectrum, mz_tol: float, tol_mode: str):
    """
    Check if the precursor m/z of two spectra are within the given tolerance.

    Parameters
    ----------
    spec1 : MsmsSpectrum
        first spectrum.
    spec2 : MsmsSpectrum
        second spectrum.
    mz_tol : float
        tolerance on precursor m/z to be considered near-identical
    tol_mode: str
        'ppm' or 'Da'

    Returns
    -------
    bool
        True if precursor m/z are within tolerance, False otherwise.
    """
    if tol_mode in ['ppm', 'Da']:
        if tol_mode == 'ppm':
            mz_tol = spec1.precursor_mz * mz_tol / 10**6
    else: 
        raise ValueError('Unknown tolerance mode')
    
    return abs(spec1.precursor_mz - spec2.precursor_mz) < mz_tol
