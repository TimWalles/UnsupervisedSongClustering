import pandas as pd
from sklearn import set_config
from sklearn.decomposition import PCA
from sklearn.preprocessing import Normalizer

set_config(transform_output='pandas')


def preprocess_data(
    df: pd.DataFrame,
    scaler,
    normalize: bool = False,
    pca: bool = False,
    pca_comp: int | float | None = None,
) -> (pd.DataFrame, PCA | None):
    # scale the data
    df = scaler.fit_transform(df)

    if normalize:
        # normalize data to gaussian distribution
        df = Normalizer().fit_transform(df)

    if pca:
        # apply pca on the data
        if pca_comp:
            pca_ = PCA(n_components=pca_comp)
            df = pca_.fit_transform(df)
            return df, pca_
        pca_ = PCA().fit(df)
        df = pca_.transform(df)
        return df, pca_
    return df, None
