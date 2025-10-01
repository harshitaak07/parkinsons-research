from sklearn.decomposition import PCA

def add_pca_embeddings(df, numeric_cols, n_components=20):
    df_numeric = df[numeric_cols].dropna()
    pca = PCA(n_components=n_components)
    embeddings = pca.fit_transform(df_numeric)
    for i in range(n_components):
        df.loc[df_numeric.index, f'pca_{i+1}'] = embeddings[:, i]
    return df, pca
