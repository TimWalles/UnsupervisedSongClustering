import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn import set_config
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from tqdm.auto import tqdm, trange

from src.utils.analysis_utils import plot_inertia_silhouette, preprocess_data

set_config(transform_output='pandas')
sns.set_theme(style='darkgrid')


def get_kmeans_config(
    df: pd.DataFrame,
    scaler_name: str,
    scaler,
    max_k: int,
    normalize: bool = False,
    pca: bool = False,
    pca_comp: int | float | None = None,
    seed=123,
):
    # Create an empty list to store the inertia scores
    inertias = []
    sil_scores = []

    df, _ = preprocess_data(df, scaler, normalize, pca, pca_comp)

    # Iterate over the range of cluster numbers
    for i in trange(1, max_k, desc=f'Processing data with {scaler_name}'):
        # Create a KMeans object with the specified number of clusters
        myKMeans = KMeans(n_clusters=i, n_init="auto", random_state=seed)

        # Fit the KMeans model to the scaled data
        myKMeans.fit(df)

        # Append the inertia score to the list
        inertias.append(myKMeans.inertia_)

        if i > 1:
            # Get the cluster labels
            labels = myKMeans.labels_

            # Calculate the silhouette score
            score = silhouette_score(df, labels)

            # Append the silhouette score to the list
            sil_scores.append(score)
    plot_inertia_silhouette(
        inertias=inertias,
        sil_scores=sil_scores,
        n_clusters=max_k,
        plot_title=scaler_name,
        normalize=normalize,
    )
