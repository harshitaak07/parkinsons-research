from sklearn.decomposition import PCA
import warnings
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np


def add_pca_embeddings(df, numeric_cols, n_components=20):
    if df.empty:
        raise ValueError("Input DataFrame is empty.")
    if not isinstance(numeric_cols, list):
        raise TypeError("numeric_cols must be a list.")
    if not numeric_cols:
        raise ValueError("numeric_cols cannot be empty.")
    available_cols = [col for col in numeric_cols if col in df.columns]
    if not available_cols:
        raise ValueError("None of the numeric columns found in DataFrame.")
    df_numeric = df[available_cols].dropna()
    if df_numeric.empty:
        raise ValueError("No valid numeric data after dropping NaN values.")
    n_features = df_numeric.shape[1]
    if n_components > n_features:
        warnings.warn(f"n_components ({n_components}) > number of features ({n_features}). Setting n_components to {n_features}.")
        n_components = n_features
    if n_components <= 0:
        raise ValueError("n_components must be positive.")
    try:
        pca = PCA(n_components=n_components)
        embeddings = pca.fit_transform(df_numeric)
        for i in range(n_components):
            df.loc[df_numeric.index, f'pca_{i+1}'] = embeddings[:, i]
    except Exception as e:
        raise ValueError(f"Error during PCA: {e}")
    return df, pca

def get_autoencoder_embeddings(X, n_components=20, epochs=50, batch_size=32):
    input_dim = X.shape[1]
    input_layer = keras.Input(shape=(input_dim,))
    encoded = layers.Dense(64, activation='relu')(input_layer)
    bottleneck = layers.Dense(n_components, activation='linear', name='embedding')(encoded)
    decoded = layers.Dense(64, activation='relu')(bottleneck)
    output_layer = layers.Dense(input_dim, activation='linear')(decoded)
    autoencoder = keras.Model(inputs=input_layer, outputs=output_layer)
    autoencoder.compile(optimizer='adam', loss='mse')
    autoencoder.fit(X, X, epochs=epochs, batch_size=batch_size, verbose=0)
    encoder = keras.Model(inputs=input_layer, outputs=bottleneck)
    embeddings = encoder.predict(X)
    return embeddings
