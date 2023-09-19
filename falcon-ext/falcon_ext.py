import os 
import sys
import logging

import itertools as it
import numpy as np

from typing import Dict, Iterator, List, Tuple, Union

from ms_io import mgf_io
from cluster import similarity

logger = logging.getLogger('falcon_ext')

def main(args: Union[str, List[str]] = None) -> int:

    # read file
    filename = sys.argv[-1]
    if not os.path.isfile(filename):
        raise ValueError(f'Non-existing peak file (filename)')

    print('Reading MGF file ...')

    spectra = list(mgf_io.get_spectra(filename))
    spectra.sort(key=lambda x: x.precursor_mz)
    spectra = spectra[:10]
    n_spectra = len(spectra)

    # calculate pairwise mod cos similarity
    print('Calculating modified cosine similarity ...')

    distance_matrix = similarity.create_mod_cos_dist_matrix(spectra)

    print(distance_matrix)
    return 0

if __name__ == '__main__':
    sys.exit(main())