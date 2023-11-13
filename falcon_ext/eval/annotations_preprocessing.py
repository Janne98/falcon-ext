import os
import sys

import re
import numpy as np
import pandas as pd
import pubchempy as pcp

from typing import Union


def main():

    filename = sys.argv[-2]
    save_to = sys.argv[-1]
    if not os.path.isfile(filename):
        raise ValueError(f'Non-existing annotations file')
    
    # read tsv file
    annotations = pd.read_csv(filename, sep='\t')
    # process tsv file
    annotations = annotations.iloc[:100, :] 
    print(annotations.shape)
    annotations = process_annotations(annotations)
    # write tsv file
    # annotations.to_csv(save_to, sep='\t')

    return 0


def process_annotations(annotations: pd.DataFrame) -> pd.DataFrame:
    
    annotations = _extract_collision_energy(annotations)
    annotations = _compound_name_cleaning(annotations)
    annotations = _inchi_cleaning(annotations)
    # annotations = _get_cid(annotations)
    annotations = _get_inchi_key(annotations)
    print('---missing inchikey---')
    print(annotations.loc[annotations['InChIKey'] == '']['Compound_Name'])

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

    # print(annotations['Compound_Name'].unique())
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

    # new
    annotations['Compound_Name'] = annotations['Compound_Name'].str.replace(
        r'.+?\]\+ ', '', regex=True)
    annotations['Compound_Name'] = annotations['Compound_Name'].str.replace('""', '')
    
    
    # annotations['Compound_Name'] = annotations['Compound_Name'].str.lower()
    annotations['Compound_Name'] = annotations['Compound_Name'].str.replace(' l-', '-l-')
    annotations['Compound_Name'] = annotations['Compound_Name'].str.replace('l c', 'lc')
    # print(annotations['Compound_Name'].unique())

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

def _search_inchi_key(row: pd.Series, lookup_table:pd.DataFrame) -> Union[str, np.nan]:
    if pd.isnull(row['InChIKey']):

        if not pd.isnull(row['Smiles']):
            if row['Smiles'] in lookup_table['Smiles'].values:
                return lookup_table.loc[lookup_table['Smiles'] == row['Smiles'], 'InChIKey'].values[0]
            else: 
                try:
                    compound = pcp.get_compounds(row['Smiles'], 'smiles')[0]
                    lookup_table.loc[len(lookup_table.index)] = {'Compound_Name': compound.name, 'Smiles': compound.smiles, 'InChIKey': compound.inchikey}
                    return compound.inchikey
                except:
                    pass
     
        if row['Compound_Name'] in lookup_table['Compound_Name'].values:
            return lookup_table.loc[lookup_table['Compound_Name'] == row['Compound_Name'], 'InChIKey'].values[0]
        else:
            formula_pattern = re.compile(r'^([a-zA-Z]?[0-9]+)+')
            if formula_pattern.match(row['Compound_Name']):
                try:
                    compound = pcp.get_compounds(row['Compound_Name'], 'formula')[0]
                    lookup_table.loc[len(lookup_table.index)] = {'Compound_Name': compound.name, 'Smiles': compound.smiles, 'InChIKey': compound.inchikey}
                    return compound.inchikey
                except:
                    print(row['Compound_Name'])
                    pass
            else:
                try:
                    compound = pcp.get_compounds(row['Compound_Name'], 'name')[0]
                    lookup_table.loc[len(lookup_table.index)] = {'Compound_Name': compound.name, 'Smiles': compound.smiles, 'InChIKey': compound.inchikey}
                    return compound.inchikey
                except:
                    print(row['Compound_Name'])
                    return np.nan

def _get_inchi_key(annotations: pd.DataFrame) -> pd.DataFrame:

    # get cid by searching PubChem using InChI
    # if no InChI specified, use compound name
    # if multiple compounds found, use best match
    # if pubchem lookup, add to local lookup table -> prevent duplicate lookups
    lookup_table = pd.DataFrame(columns=['Compound_Name', 'Smiles', 'InChIKey'])
    annotations['InChIKey'] = annotations.apply( \
        # lambda r: pcp.get_compounds(r['Compound_Name'], 'name')[0].inchikey \
        # if pd.isnull(r['InChIKey']) else r['InChIKey'], axis=1)
        lambda r: _search_inchi_key(r, lookup_table), axis=1)
    
    print(lookup_table)
    # annotations = annotations.dropna(subset=['InChIKey'])
    print(annotations['InChIKey'].unique())
    
    annotations.loc['inchikey_p1'] = annotations['InChIKey'].apply(lambda r: r.split('-')[0])

    return annotations


if __name__ == '__main__':
	sys.exit(main())