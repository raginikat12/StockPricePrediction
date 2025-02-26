import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

def preprocess_data(data):
    """Prepares stock price data for LSTM model."""
    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(data["Close"].values.reshape(-1, 1))
    
    X, y = [], []
    lookback = 60  # Use past 60 days to predict next day
    for i in range(lookback, len(scaled_data)):
        X.append(scaled_data[i - lookback:i, 0])
        y.append(scaled_data[i, 0])

    return np.array(X), np.array(y), scaler

def build_lstm_model():
    """Builds and returns an LSTM model."""
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=(60, 1)),
        Dropout(0.2),
        LSTM(50, return_sequences=False),
        Dropout(0.2),
        Dense(25),
        Dense(1)
    ])
    model.compile(optimizer="adam", loss="mean_squared_error")
    return model

def predict_stock_price(data):
    """Trains LSTM model and predicts future stock prices."""
    if data.shape[0] < 60:  # Ensure sufficient data
        return None
    
    # Preprocess data
    X, y, scaler = preprocess_data(data)

    # Reshape for LSTM
    X = X.reshape((X.shape[0], X.shape[1], 1))

    # Build & Train model
    model = build_lstm_model()
    model.fit(X, y, epochs=10, batch_size=16, verbose=0)

    # Predict future prices
    future_predictions = []
    last_60_days = X[-1].reshape(1, 60, 1)

    for _ in range(10):  # Predict next 10 days
        predicted_price = model.predict(last_60_days)[0][0]
        future_predictions.append(predicted_price)
        last_60_days = np.append(last_60_days[:, 1:, :], [[[predicted_price]]], axis=1)

    return scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
