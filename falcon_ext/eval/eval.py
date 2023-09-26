import pandas as pd
import numpy as np

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score
from sklearn.cluster import AgglomerativeClustering

from typing import List, Dict, Tuple

def evaluate_clustering(filename: str, clustering: AgglomerativeClustering):

    annotations = _read_tsv_file(filename)
    pred_labels = clustering.labels_
    annotations = annotations[annotations['#Scan#'] < len(pred_labels)+1] # scan idx starts from 1

    identified_spectra = _get_identified_spectra(annotations)
    true_labels = annotations["Compound_idx"].tolist()

    # get cluster labels of the identified spectra only
    pred_labels_identified = pred_labels[[i - 1 for i in identified_spectra]]
    # group identified spectry by cluster label
    # pred_clusters_identified = pd.Series(range(len(pred_labels_identified)))\
    #     .groupby(pred_labels_identified, sort=False).apply(list).tolist()

    # calculate the adjusted rand index for the spectra with ground truth
    ari = _adjusted_rand_index(true_labels, pred_labels_identified)
    print("adjusted rand index: " + str(ari))

    mis = _mutual_information_score(true_labels, pred_labels_identified)
    print("mutual information score: " + str(mis))

    return None

def _read_tsv_file(filename: str) -> pd.DataFrame:
    
    df = pd.read_csv(filename, sep='\t')
    # translate compound name to integer
    df["Compound_idx"] = df["Compound_Name"].astype("category").cat.codes

    return df[["#Scan#", "Compound_idx"]]

    # print(df.columns)
    # print(df[['Precursor_MZ', '#Scan#', 'id']].tail())

    # groups = df['#Scan#'].groupby(df['Compound_Name'], sort=False).apply(list).tolist()
    # print(groups)

def _get_identified_spectra(annotations: pd.DataFrame) -> List[int]:
    return annotations["#Scan#"].tolist()

def _adjusted_rand_index(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    return adjusted_rand_score(true_labels, pred_labels)

def _mutual_information_score(true_labels: np.ndarray, pred_labels: np.ndarray) -> float:
    return adjusted_mutual_info_score(true_labels, pred_labels)