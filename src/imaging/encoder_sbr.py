import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model

autoencoderinput_sbr = np.load('autoencoderinput_sbr.npy')  # shape: patients x phases x features

def build_lstm_encoder(input_shape, latent_dim=8):
    inp = layers.Input(shape=input_shape)
    x = layers.LSTM(32, return_sequences=True)(inp)
    x = layers.LSTM(16)(x)
    out = layers.Dense(latent_dim, activation='relu')(x)
    return Model(inp, out)

input_shape = autoencoderinput_sbr.shape[1:]
encoder = build_lstm_encoder(input_shape)
encoded = encoder.predict(autoencoderinput_sbr)
print("Encoded SBR shape:", encoded.shape)
np.save('encoded_sbr.npy', encoded)
