# backend/models/model_prophet.py

from prophet import Prophet
import pandas as pd

def train_prophet_model(df):
    """
    Train a Prophet model on stock data.
    :param df: DataFrame with columns ['Date', 'Close']
    :return: Trained Prophet model
    """
    print(df.head())
    prophet_df = df.rename(columns={'Date': 'ds', 'Close': 'y'})[['ds', 'y']]
    prophet_df['ds'] = prophet_df['ds'].dt.tz_localize(None)
    model = Prophet()
    model.fit(prophet_df)
    return model

def forecast_prophet(model, periods, freq='D'):
    """
    Make future forecast with the trained model.
    :param model: Trained Prophet model
    :param periods: Number of periods to predict
    :param freq: Frequency of predictions
    :return: DataFrame with forecast
    """
    future = model.make_future_dataframe(periods=periods, freq=freq)
    forecast = model.predict(future)
    return forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']]
