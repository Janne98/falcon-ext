import pandas as pd
import numpy as np

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, completeness_score
from sklearn.cluster import AgglomerativeClustering

from typing import List, Dict, Tuple
from collections import Counter

from cluster.clustering import ClusterResult


def evaluate_clustering(
    filename: str, 
    clustering: ClusterResult,
    idx_to_scan_map: List[int]) -> None:
    """
    Evaluate the clustering performance.

    Parameters
    ----------
    filename: str
        filename of the tsv-file containing the true labels for the identified spectra.
    clustering: ClusterResult
        clustering result 
    spec_map: List[int]
        list of scan indices, mapping of spectrum index (in clustering) to scan index.
    """
    annotations = _read_tsv_file(filename)
    pred_labels = clustering.labels # sorted by pepmass
    # get annotations of identified spectra that are in clustering
    annotations_subset = annotations[annotations['#Scan#'].isin(idx_to_scan_map)]
    # get the scan idx of all identified spectra in clustering
    identified_spectra = _get_identified_spectra(annotations_subset)
    # get the true labels of all identified spectra in clustering
    true_labels = _get_spectrum_labels(annotations_subset)
    # get predicted cluster labels for all identified spectra in clustering
    pred_labels_identified = [pred_labels[idx_to_scan_map.index(scan_id)] \
        for scan_id in identified_spectra]

    # calculate the adjusted rand index for the spectra with ground truth
    # ari = adjusted_rand_index(true_labels, pred_labels_identified)
    # print("adjusted rand index: " + str(ari))
    # calculate the adjusted mutual information score for all spectra with ground truth
    # mis = mutual_information_score(true_labels, pred_labels_identified)
    # print("mutual information score: " + str(mis))
    # calculate completeness score for all spectra with ground truth
    completeness = completeness_score(true_labels, pred_labels_identified)
    print('completeness score: ' + str(completeness))
    # calculate fraction of clustered spectra
    clustered_spectra = _clustered_spectra(clustering)
    print('clustered spectra: ' + str(clustered_spectra))
    # calculate fraction of incorrectly clustered spectra for all spectra with ground truth
    incorrectly_clustered_spectra = _incorrectly_clustered_spectra(pred_labels,
                                                                   pred_labels_identified, 
                                                                   true_labels)
    print('incorreclty clustered spectra: ' + str(incorrectly_clustered_spectra))


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


def _count_singletons(labels: np.ndarray) -> int:
    """
    Count singletons (clusters of size 1).

    Parameters
    ----------
    labels: np.ndarray
        array of labels. 

    Returns
    -------
    int
        number of singletons.
    """
    _, cluster_sizes = np.unique(labels, return_counts=True)
    singleton_count = sum(size > 1 for size in cluster_sizes)
    # add functionality to deal with noise samples (label=-1)
    return singleton_count


def _clustered_spectra(clustering: ClusterResult) -> float:
    """
    Calculate fraction of clustered spectra.

    Parameters
    ----------
    clustering: ClusterResult
        result of clustering. 

    Returns
    -------
    float
        fraction of clustered spectra.
    """
    n_spectra = len(clustering.labels)
    return (n_spectra - _count_singletons(clustering.labels)) / n_spectra


def _incorrectly_clustered_spectra(
        pred_labels : np.ndarray,
        pred_labels_identified: np.ndarray, 
        true_labels: List[int]) -> float:
    """
    Calculate fraction of incorrectly clustered spectra (only annotated spectra).

    Parameters
    ----------
    pred_labels: np.ndarray
        array of predicted labels. 
    pred_labels_identified: np.ndarray
        array of predicted labels for spectra with ground truth.
    true_labels: np.ndarray
        array of true labels.

    Returns
    -------
    float
        fraction of incorrectly labeled spectra.
    """
    incorrect_count = 0

    unique_pred_lables = np.unique(pred_labels_identified)
    # count incorrectly clustered spectra in each cluster
    for label in unique_pred_lables:
        # skip singletons
        if label == -1:
            continue
        if np.count_nonzero(pred_labels == label) < 2:
            continue
        # get idx of all spectra in cluster
        cluster_members = [idx for idx, pred_label in enumerate(pred_labels_identified) \
                           if pred_label == label]
        # get the most frequent true label in cluster
        members_true_labels = np.array(true_labels)[cluster_members]
        most_freq_label = Counter(members_true_labels).most_common(1)[0][0]
        # count clusters where true label != most frequent label
        incorrect_count += sum(1 for label in members_true_labels \
                               if label != most_freq_label)
    
    return incorrect_count / len(true_labels)