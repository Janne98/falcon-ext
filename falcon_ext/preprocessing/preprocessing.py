import numpy as np
import collections

from typing import Iterator, List, Optional, Tuple

from spectrum_utils.spectrum import MsmsSpectrum

def process_all_spectra(spectra: Iterator[MsmsSpectrum],
                        min_peaks: int, min_mz_range: float, 
                        mz_min: Optional[float] = None, 
                        mz_max: Optional[float] = None, 
                        remove_precursor_tolerance: Optional[float] = None, 
                        min_intensity: Optional[float] = None, 
                        max_peaks_used: Optional[float] = None, 
                        scaling: Optional[str] = None) -> Iterator[MsmsSpectrum]:
    """
    Process all spectra.

    Parameters
    ----------
    spectra: Iterator[MsmsSpectrum]
        All spectra to be processed.
    min_peaks : int
        Minimum number of peaks the cluster has to contain to be valid.
    min_mz_range : float
        Minimum m/z range the cluster's peaks need to cover to be valid.
    mz_min : Optional[float], optional
        Minimum m/z (inclusive). If not set no minimal m/z restriction will
        occur.
    mz_max : Optional[float], optional
        Maximum m/z (inclusive). If not set no maximal m/z restriction will
        occur.
    remove_precursor_tolerance : Optional[float], optional
        Fragment mass tolerance (in Dalton) around the precursor mass to remove
        the precursor peak. If not set, the precursor peak will not be removed.
    min_intensity : Optional[float], optional
        Remove peaks whose intensity is below `min_intensity` percentage
        of the base peak intensity. If None, no minimum intensity filter will
        be applied.
    max_peaks_used : Optional[int], optional
        Only retain the `max_peaks_used` most intense peaks. If None, all peaks
        are retained.
    scaling : {'root', 'log', 'rank'}, optional
        Method to scale the peak intensities. Potential transformation options
        are:
        - 'root': Square root-transform the peak intensities.
        - 'log':  Log2-transform (after summing the intensities with 1 to avoid
          negative values after the transformation) the peak intensities.
        - 'rank': Rank-transform the peak intensities with maximum rank
          `max_peaks_used`.
        - None: No scaling is performed.

    Returns
    -------
    Iterator[MsmsSpectrum]
        The processed spectra.
    """
    for spectrum in spectra:
        yield process_spectrum(spectrum, min_peaks, min_mz_range, mz_min, mz_max, 
                               remove_precursor_tolerance, min_intensity, max_peaks_used,
                               scaling)


def process_spectrum(spectrum: MsmsSpectrum, 
                     min_peaks: int, min_mz_range: float, 
                     mz_min: Optional[float] = None, 
                     mz_max: Optional[float] = None, 
                     remove_precursor_tolerance: Optional[float] = None, 
                     min_intensity: Optional[float] = None, 
                     max_peaks_used: Optional[float] = None, 
                     scaling: Optional[str] = None) -> Optional[MsmsSpectrum]:
    """
    Process a spectrum.

    Parameters
    ----------
    spectrum: MsmsSpectrum
        Spectrum to be processed.
    min_peaks : int
        Minimum number of peaks the cluster has to contain to be valid.
    min_mz_range : float
        Minimum m/z range the cluster's peaks need to cover to be valid.
    mz_min : Optional[float], optional
        Minimum m/z (inclusive). If not set no minimal m/z restriction will
        occur.
    mz_max : Optional[float], optional
        Maximum m/z (inclusive). If not set no maximal m/z restriction will
        occur.
    remove_precursor_tolerance : Optional[float], optional
        Fragment mass tolerance (in Dalton) around the precursor mass to remove
        the precursor peak. If not set, the precursor peak will not be removed.
    min_intensity : Optional[float], optional
        Remove peaks whose intensity is below `min_intensity` percentage
        of the base peak intensity. If None, no minimum intensity filter will
        be applied.
    max_peaks_used : Optional[int], optional
        Only retain the `max_peaks_used` most intense peaks. If None, all peaks
        are retained.
    scaling : {'root', 'log', 'rank'}, optional
        Method to scale the peak intensities. Potential transformation options
        are:
        - 'root': Square root-transform the peak intensities.
        - 'log':  Log2-transform (after summing the intensities with 1 to avoid
          negative values after the transformation) the peak intensities.
        - 'rank': Rank-transform the peak intensities with maximum rank
          `max_peaks_used`.
        - None: No scaling is performed.

    Returns
    -------
    MsmsSpectrum
        The processed spectrum.
    """
    # set mz range
    spectrum = spectrum.set_mz_range(mz_min, mz_max)
    if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
        return None

    # remove peaks close to precursor m/z
    if remove_precursor_tolerance is not None:
        spectrum = spectrum.remove_precursor_peak(remove_precursor_tolerance, 'Da', 0)
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            return None
        
    # remove low-intensity fragment peaks
    if min_intensity is not None or max_peaks_used is not None:
        min_intensity = 0. if min_intensity is None else min_intensity
        spectrum = spectrum.filter_intensity(min_intensity, max_peaks_used)
        if not _check_spectrum_valid(spectrum.mz, min_peaks, min_mz_range):
            return None

    # scale peak intensity
    spectrum = spectrum.scale_intensity(scaling, max_rank=max_peaks_used)
    spectrum.intensity = _norm_intensity(spectrum.intensity)

    return spectrum


def _check_spectrum_valid(spectrum_mz: np.ndarray, 
                          min_peaks: int, 
                          min_mz_range: float) -> bool:
    """
    Check whether a spectrum is of good enough quality to be used.

    Parameters
    ----------
    spectrum_mz : np.ndarray
        M/z peaks of the spectrum whose quality is checked.
    min_peaks : int
        Minimum number of peaks the spectrum has to contain.
    min_mz_range : float
        Minimum m/z range the spectrum's peaks need to cover.

    Returns
    -------
    bool
        True if the spectrum has enough peaks covering a wide enough mass
        range, False otherwise.
    """
    return (len(spectrum_mz) >= min_peaks and 
            spectrum_mz[-1] - spectrum_mz[0] >= min_mz_range)


def _norm_intensity(spectrum_intensity: np.ndarray) -> np.ndarray:
    """
    Normalize cluster peak intensities by their vector norm.

    Parameters
    ----------
    spectrum_intensity : np.ndarray
        The cluster peak intensities to be normalized.

    Returns
    -------
    np.ndarray
        The normalized peak intensities.
    """
    return spectrum_intensity / np.linalg.norm(spectrum_intensity)