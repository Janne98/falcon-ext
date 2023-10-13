import os
import sys

import numpy as np
import pandas as pd
import pubchempy as pcp

from typing import Union, List


def main():

    filename = sys.argv[-1]
    if not os.path.isfile(filename):
        raise ValueError(f'Non-existing annotations file')
    
    # read tsv file
    annotations = pd.read_csv(filename, sep='\t')
    # process tsv file
    annotations = process_annotations(annotations)
    # write tsv file
    annotations.to_csv('data/annotations.tsv', sep='\t')

    return 0


def process_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    
    annotations = _extract_collision_energy(annotations)
    annotations = _compound_name_cleaning(annotations)
    annotations = _inchi_cleaning(annotations)
    annotations = _get_cid(annotations)

    return annotations


def _extract_collision_energy(annotations: pd.DataFrame) -> pd.DataFrame:

    annotations['collision_e'] = annotations['Compound_Name'].str.extract(
        r'(?<= - )(.*)(?= eV)|(?<=CollisionEnergy:)(.*)').stack().droplevel(1)    
    annotations['collision_e'] = pd.to_numeric(annotations['collision_e'], errors='coerce')
    # remove collision energy from compound name
    annotations['Compound_Name'] = annotations['Compound_Name'].str.replace(
        r' - .*eV| CollisionEnergy:.*', '', regex=True)

    return annotations


def _compound_name_cleaning(annotations: pd.DataFrame) -> pd.DataFrame:

    annotations['Compound_Name'] = annotations['Compound_Name'].str.replace(
        'Spectral Match to ', '')
    annotations['Compound_Name'] = annotations['Compound_Name'].str.replace(
        ' from NIST14', '')
    
    annotations['Compound_Name'] = annotations['Compound_Name'].str.replace(
        r'^Massbank:.* ', '', regex=True)
    annotations['Compound_Name'] = annotations['Compound_Name'].str.replace(
        r'[|].*', '', regex=True)
    
    annotations['Compound_Name'] = annotations['Compound_Name'].str.replace(
        r'(.*)_', '', regex=True)
    
    annotations['Compound_Name'] = annotations['Compound_Name'].str.lower()
    annotations['Compound_Name'] = annotations['Compound_Name'].str.replace(' l-', '-l-')

    return annotations


def _inchi_cleaning(annotations: pd.DataFrame) -> pd.DataFrame:

    annotations['INCHI'] = annotations['INCHI'].str.replace(
        '"', '')

    annotations['INCHI'] = annotations['INCHI'].apply(
        lambda s: 'InChI=' + str(s) if 'InChI=' not in str(s) else str(s))
    
    return annotations


def _get_cid(annotations: pd.DataFrame) -> pd.DataFrame:

    # get cid by searching PubChem using InChI
    # if no InChI specified, use compound name
    # if multiple compounds found, use best match
    annotations['cid'] = annotations.apply(
        lambda r: pcp.get_compounds(r['Compound_Name'], 'name')[0].cid \
        if str(r['INCHI']) == 'InChI=nan' 
        else pcp.get_compounds(r['INCHI'], 'inchi')[0].cid, axis=1)
    
    return annotations


if __name__ == '__main__':
	sys.exit(main())