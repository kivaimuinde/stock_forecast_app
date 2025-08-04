# backend/models/model_lstm.py

import numpy as np
import pandas as pd
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.preprocessing import MinMaxScaler

def preprocess_data_lstm(df, feature='Close', time_steps=60):
    """
    Normalize and prepare sequences for LSTM.
    :param df: DataFrame with historical prices
    :param feature: Column to forecast
    :param time_steps: Number of previous time steps per input
    :return: sequences X, y, scaler
    """
    data = df[[feature]].values
    scaler = MinMaxScaler()
    data_scaled = scaler.fit_transform(data)

    X, y = [], []
    for i in range(time_steps, len(data_scaled)):
        X.append(data_scaled[i-time_steps:i])
        y.append(data_scaled[i])
    return np.array(X), np.array(y), scaler

def build_lstm_model(input_shape):
    """
    Create and compile LSTM model.
    :param input_shape: Shape of input sequence (time_steps, features)
    :return: Compiled model
    """
    model = Sequential([
        Input(shape=input_shape),
        LSTM(50, return_sequences=True),
        Dropout(0.2),
        LSTM(50),
        Dropout(0.2),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model
