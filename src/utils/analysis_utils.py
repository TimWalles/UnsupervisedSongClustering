import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.io as pio
import seaborn as sns
from plotly import graph_objects as go
from sklearn import set_config
from sklearn.decomposition import PCA

from src.analysis.preprocess_data import preprocess_data

set_config(transform_output='pandas')
sns.set_theme(style='darkgrid')
pio.templates.default = "plotly_dark"


# region data preprocessing
def get_features_contribution(
    df: pd.DataFrame,
    scaler,
    normalize: bool = False,
    pca: bool = False,
    pca_comp: int | float | None = None,
) -> pd.DataFrame:
    # Make the PCA analysis
    _, pca_ = preprocess_data(df, scaler, normalize, pca, pca_comp)

    columns = df.columns.to_list()

    summed_loadings = {feature: 0 for feature in columns}
    # Loop through all principal components
    for component in pca_.components_:
        # Get the absolute loading scores for the ith principal component
        absolute_loadings = np.abs(component)
        # Go through each loading score and add it to the sum for the corresponding feature
        for j, feature in enumerate(columns):
            summed_loadings[feature] += absolute_loadings[j]
    return pd.DataFrame.from_dict(
        summed_loadings,
        orient='index',
        columns=['summed_loadings'],
    ).sort_values(
        by='summed_loadings',
        ascending=False,
    )


# endregion


# region plot config data
def plot_inertia_silhouette(
    inertias: list,
    sil_scores: list,
    n_clusters: int,
    plot_title: str | None = None,
    normalize: bool = False,
):
    # Stating that we want two plots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 8))

    # Create a line plot of the inertia scores
    sns.lineplot(y=inertias, x=range(1, n_clusters), markers=True, ax=ax1).set(xlabel="Number of clusters", ylabel="Inertia score")
    sns.lineplot(y=sil_scores, x=range(2, n_clusters), markers=True, ax=ax2).set(xlabel="Number of clusters", ylabel="Silhouette score")

    # set titles
    if plot_title:
        fig.suptitle(plot_title)
    ax1.set_title(f"Silhouette score from 2 to {n_clusters} clusters normalized" if normalize else f"Silhouette score from 2 to {n_clusters} clusters")
    ax2.set_title(f"Inertia score from 1 to {n_clusters} clusters normlized" if normalize else f"Inertia score from 1 to {n_clusters} clusters")
    plt.show()


def plot_var_explained(
    df: pd.DataFrame,
    plot_title: str | None = None,
    normalize: bool = False,
):
    if plot_title:
        title = (
            f"{plot_title}: Proportion of variance explained by each principal component normalized"
            if normalize
            else f"{plot_title}: Proportion of variance explained by each principal component"
        )
    else:
        title = "Proportion of variance explained by each principal component"

    sns.relplot(
        kind='line',
        data=df,
        x="Principal component index",
        y="Variance explained",
        marker='o',
        aspect=1.3,
    ).set(title=title).set_axis_labels(
        "Principal component number",
        "Proportion of variance",
    )
    plt.show()


# endregion


# region plot clustered data
def plot_heatmap(
    centroid_distances,
    text,
) -> go.Heatmap:
    return go.Heatmap(z=centroid_distances, text=text, texttemplate="%{text:.2f}", showscale=False, colorscale='Blues_r')


def plot_scatter_polar(
    filter,
    df: pd.DataFrame,
    categories: list[str],
) -> go.Scatterpolar:
    return go.Scatterpolar(
        r=[df.loc[df['cluster'] == filter, col_name].mean() for col_name in categories],
        theta=categories,
        fill='toself',
        name=f'Cluster {filter}',
    )


def plot_pca_plot(
    df: pd.DataFrame,
    pca: PCA,
):
    index_vals = df['cluster'].astype('category').cat.codes
    df.drop(columns=['cluster'], inplace=True)

    df.columns = [f"{col}\n({var:.1f}%)" for col, var in zip(df.columns, pca.explained_variance_ratio_ * 100)]
    fig = go.Figure(
        data=go.Splom(
            dimensions=[dict(label=col_name, values=df[col_name]) for col_name in df.columns],
            diagonal_visible=False,
            showupperhalf=False,
            marker=dict(
                color=index_vals,
                showscale=False,
                line_color='white',
                line_width=0.5,
            ),
        )
    )
    fig.update_layout(
        width=2000,
        height=800,
    )
    fig.show()


# endregion
