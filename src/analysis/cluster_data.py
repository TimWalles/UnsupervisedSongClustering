import pandas as pd
import plotly.io as pio
from plotly import graph_objects as go
from plotly.subplots import make_subplots
from sklearn.metrics import pairwise_distances

from src.analysis.preprocess_data import preprocess_data
from src.utils.analysis_utils import plot_heatmap, plot_pca_plot, plot_scatter_polar

pio.templates.default = "plotly_dark"


def cluster_data(
    df: pd.DataFrame,
    scaler_name: str,
    scaler,
    cluster_alg,
    normalize: bool = False,
    pca: bool = False,
    pca_comp: int | float | None = None,
    verbose: bool = False,
) -> pd.DataFrame:
    df, pca_ = preprocess_data(df, scaler, normalize, pca, pca_comp)

    # Fit the model to the data
    cluster_alg = cluster_alg.fit(df)

    # Obtain the cluster output
    clusters = cluster_alg.labels_

    # Attach the cluster output to our original DataFrame
    df["cluster"] = clusters

    if verbose:
        # Find the coordinates of each centroid using the cluster_centers_ attribute
        try:
            centroids = cluster_alg.cluster_centers_
        except AttributeError:
            centroids = cluster_alg.centroids_

        # Calculate the euclidean distance between the centroids
        centroid_distances = pairwise_distances(centroids)

        fig = make_subplots(
            rows=1,
            cols=2,
            specs=[[{'type': 'heatmap'}, {'type': 'polar'}]],
        )

        fig.add_trace(plot_heatmap(centroid_distances=centroid_distances, text=centroid_distances), row=1, col=1)
        for cluster in sorted(df['cluster'].unique()):
            fig.add_trace(plot_scatter_polar(filter=cluster, df=df, categories=df.columns[:-1]), row=1, col=2)
        fig.update_layout(height=800, width=2000, title_text=scaler_name)
        fig.show()
        if pca:
            plot_pca_plot(df=df.copy(), pca=pca_)
    return df
