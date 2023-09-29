import pandas as pd
import numpy as np

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import AgglomerativeClustering

from typing import List, Dict, Tuple

def evaluate_clustering(
    filename: str, 
    clustering: AgglomerativeClustering,
    spec_map: List[int]) -> None:
    """
    Evaluate the clustering performance.

    Parameters
    ----------
    filename: str
        filename of the tsv-file containing the true labels for the identified spectra.
    clustering: AgglomerativeClustering
        clustering result 
    spec_map: List[int]
        list of scan indices, mapping of spectrum index (in clustering) to scan index.
    """
    annotations = _read_tsv_file(filename)
    pred_labels = clustering.labels_
    # get annotations of identified spectra that are in clustering
    annotations_subset = annotations[annotations['#Scan#'].isin(spec_map)]
    # get the scan idx of all identified spectra in clustering
    identified_spectra = _get_identified_spectra(annotations_subset)
    # get the true labels of all identified spectra in clustering
    true_labels = _get_spectrum_labels(annotations_subset)
    # get predicted cluster labels for all identified spectra in clustering
    pred_labels_identified = [pred_labels[spec_map.index(scan_id)] \
        for scan_id in identified_spectra]

    # calculate the adjusted rand index for the spectra with ground truth
    ari = _adjusted_rand_index(true_labels, pred_labels_identified)
    print("adjusted rand index: " + str(ari))
    # calculate the adjusted mutual information score for all spectra with ground truth
    mis = _mutual_information_score(true_labels, pred_labels_identified)
    print("mutual information score: " + str(mis))

def _read_tsv_file(filename: str) -> pd.DataFrame:
    """
    Read the annotations file (tsv-format) and extract scan index and compound label.

    Parameters
    ----------
    filename: str
        filename of the tsv-file.

    Returns
    -------
    pd.DataFrame
        dataframe containing the scan id and compound label for each identified spectrum.
    """
    df = pd.read_csv(filename, sep='\t')
    # translate compound name to integer
    df["Compound_idx"] = df["Compound_Name"].astype("category").cat.codes

    return df[["#Scan#", "Compound_idx"]]

def _get_identified_spectra(annotations: pd.DataFrame) -> List[int]:
    """
    Extract the scan ids of the identified spectra.

    Parameters
    ----------
    annotations: pd.DataFrame
        dataframe containing the scan id and compound label for each identified spectrum. 

    Returns
    -------
    List[int]
        scan id of each identified spectrum.
    """
    return annotations["#Scan#"].tolist()

def _get_spectrum_labels(annotations: pd.DataFrame) -> List[int]:
    """
    Extract the ground truth labels.

    Parameters
    ----------
    annotations: pd.DataFrame
        dataframe containing the scan id and compound label for each identified spectrum. 

    Returns
    -------
    List[int]
        compound id of each identified spectrum.
    """
    return annotations["Compound_idx"].tolist()

def _adjusted_rand_index(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Calculate the adjusted Rand index.

    Parameters
    ----------
    true_labels: np.ndarray
        array containing the true labels of the spectra.
    pred_labels: np.ndarray
        array containing the predicted labels for the identified spectra. 

    Returns
    -------
    float
        adjusted rand index.
    """
    return adjusted_rand_score(true_labels, pred_labels)

def _mutual_information_score(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    """
    Calculate the adjusted mutual information score.

    Parameters
    ----------
    true_labels: np.ndarray
        array containing the true labels of the spectra.
    pred_labels: np.ndarray
        array containing the predicted labels for the identified spectra. 

    Returns
    -------
    float
        adjusted mutual information score.
    """
    return adjusted_mutual_info_score(true_labels, pred_labels)