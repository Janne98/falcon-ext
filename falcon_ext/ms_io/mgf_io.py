from typing import Dict, IO, Iterable, Union

import pyteomics.mgf
import spectrum_utils.spectrum as sus


def get_spectra(source: Union[IO, str]) -> Iterable[sus.MsmsSpectrum]:
    """
    Get the MS/MS spectra from the given MGF file.

    Parameters
    ----------
    source : Union[IO, str]
        The MGF source (file name or open file object) from which the spectra
        are read.

    Returns
    -------
    Iterator[MsmsSpectrum]
        An iterator over the spectra in the given file.
    """
    with pyteomics.mgf.MGF(source) as f_in:
        for spectrum_i, spectrum_dict in enumerate(f_in):
            try:
                yield _parse_spectrum(spectrum_dict)
            except (ValueError, KeyError):
                pass


def _parse_spectrum(spectrum_dict: Dict) -> sus.MsmsSpectrum:
    """
    Parse the Pyteomics cluster dict.

    Parameters
    ----------
    spectrum_dict : Dict
        The Pyteomics cluster dict to be parsed.

    Returns
    -------
    MsmsSpectrum
        The parsed cluster.
    """
    if 'scans' in spectrum_dict['params']:
        identifier = spectrum_dict['params']['scans']
    else:
        identifier = id(spectrum_dict)

    mz_array = spectrum_dict['m/z array']
    intensity_array = spectrum_dict['intensity array']
    retention_time = float(spectrum_dict['params'].get('rtinseconds', -1))

    precursor_mz = float(spectrum_dict['params']['pepmass'][0])
    if 'charge' in spectrum_dict['params']:
        precursor_charge = int(spectrum_dict['params']['charge'][0])
    else: 
        #assume 1 if not specified
        precursor_charge = 1

    return sus.MsmsSpectrum(identifier, 
                            precursor_mz, 
                            precursor_charge, 
                            mz_array, 
                            intensity_array, 
                            None,
                            retention_time)