import pandas as pd
import seaborn as sns
from sklearn import set_config
from sklearn.decomposition import PCA

from src.analysis.preprocess_data import preprocess_data
from src.utils.analysis_utils import plot_var_explained

set_config(transform_output='pandas')


def get_pca_config(
    df: pd.DataFrame,
    scaler_name: str,
    scaler,
    normalize: bool = False,
):
    df, _ = preprocess_data(df, scaler, normalize, pca=False)

    # fit transform pca
    pca = PCA().fit(df)
    df = pca.transform(df)

    explained_variance_df = pd.DataFrame(
        {
            "Variance explained": pca.explained_variance_ratio_,
            "Principal component index": range(len(pca.explained_variance_ratio_)),
        }
    )
    plot_var_explained(df=explained_variance_df, plot_title=scaler_name, normalize=normalize)
