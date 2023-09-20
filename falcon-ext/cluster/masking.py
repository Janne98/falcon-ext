import numpy as np

from typing import List

from spectrum_utils.spectrum import MsmsSpectrum


def generate_mask(spectra: List[MsmsSpectrum], mz_tolerance: float) -> np.ndarray:
    """
    Generate a mask indicating which spectrum pairs have a near-identical precursor m/z.

    Parameters
    ----------
    spectra : List[MsmsSpectrum]
        list of MS/MS spectra.
    mz_tolerance : float
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
        while j < len(spectra) and spec.precursor_mz + mz_tolerance \
            > spectra[j].precursor_mz:
            mask[i][j] = 1
            mask[j][i] = 1
            j += 1

    return mask
