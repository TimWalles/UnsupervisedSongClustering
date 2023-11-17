import pandas as pd


def categories_songs(df: pd.DataFrame, thresholds: dict) -> pd.DataFrame:
    # Categorize 'instrumentalness' values.
    df['instrumentalness_category'] = [
        'lot_vocals'
        if x <= thresholds['instrumentalness']['lot_vocals']
        else 'mix_vocals'
        if x <= thresholds['instrumentalness']['instrumental']
        else 'instrumental'
        for x in df['instrumentalness']
    ]

    # Categorize 'valence' values.
    df['valence_category'] = [
        'negative' if x <= thresholds['valence']['negative'] else 'neutral' if x <= thresholds['valence']['positive'] else 'positive' for x in df['valence']
    ]

    # Categorize 'acousticness' values.
    df['acousticness_category'] = [
        'not_acoustic'
        if x <= thresholds['acousticness']['not_acoustic']
        else 'moderately_acoustic'
        if x <= thresholds['acousticness']['acoustic']
        else 'acoustic'
        for x in df['acousticness']
    ]

    # Categorize 'danceability' values.
    df['danceability_category'] = [
        'not_danceable'
        if x <= thresholds['danceability']['not_danceable']
        else 'moderately_danceable'
        if x <= thresholds['danceability']['danceable']
        else 'danceable'
        for x in df['danceability']
    ]

    return df.loc[:, ['instrumentalness_category', 'valence_category', 'acousticness_category', 'danceability_category', 'cluster']]


def create_playlist_name(
    df: pd.DataFrame,
    cluster: int,
):
    common_instrumentalness = df.loc[df['cluster'] == cluster, 'instrumentalness_category'].mode()[0]
    common_valence = df.loc[df['cluster'] == cluster, 'valence_category'].mode()[0]
    common_acousticness = df.loc[df['cluster'] == cluster, 'acousticness_category'].mode()[0]
    common_danceability = df.loc[df['cluster'] == cluster, 'danceability_category'].mode()[0]

    # Construct the playlist name using the most common categories.
    return f"C: {cluster} " f"I: {common_instrumentalness} " f"V: {common_valence} " f"A: {common_acousticness} " f"D: {common_danceability}"
