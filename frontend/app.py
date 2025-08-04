# frontend/app.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
from dotenv import load_dotenv
import os

# Load environment variables
load_dotenv()

# Imports from your backend
from backend.scraper.scraper import fetch_stock_data
from backend.data.mock_data import generate_mock_stock_data
from backend.models.model_prophet import train_prophet_model, forecast_prophet
from backend.models.model_lstm import preprocess_data_lstm, build_lstm_model
from backend.scraper.sentiment import get_newsapi_sentiment

# UI config
st.set_page_config(page_title="📈 Stock Forecasting App", layout="wide")
st.title("📈 Stock Price Forecasting App")

# Layout with two columns
col1, col2 = st.columns([1, 1])

with col1:
    # User input
    ticker = st.text_input("Enter stock ticker (e.g., AAPL)", "AAPL")
    interval = st.select_slider("Select forecast interval (days)", options=list(range(1, 31)), value=7)
    model_type = st.radio("Select model", ["Prophet", "LSTM"])

if st.button("Fetch & Forecast"):
    try:
        df = fetch_stock_data(ticker, period="3mo", interval="1d")
        if df.empty:
            raise ValueError("No data")
    except Exception:
        st.warning("⚠️ Falling back to mock data.")
        df = generate_mock_stock_data()

    if not df.empty:
        st.success("✅ Data loaded successfully.")
        df = df.reset_index() if "Date" not in df.columns else df
        df = df[['Date', 'Close']]

        with col2:
            st.subheader("📊 Recent Data")
            st.dataframe(df.tail(), use_container_width=True)

        # Prophet
        if model_type == "Prophet":
            st.subheader("📈 Prophet Forecast")
            model = train_prophet_model(df)
            forecast = forecast_prophet(model, periods=interval)

            fig = go.Figure()
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat'], name='Forecast'))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_upper'], name='Upper Bound', line=dict(dash='dot')))
            fig.add_trace(go.Scatter(x=forecast['ds'], y=forecast['yhat_lower'], name='Lower Bound', line=dict(dash='dot')))
            st.plotly_chart(fig, use_container_width=True)

        # LSTM
        else:
            st.subheader("📈 LSTM Forecast")
            df = df.rename(columns={"Date": "ds"})
            df['ds'] = pd.to_datetime(df['ds'])
            df = df.sort_values('ds')

            time_steps = 60
            X, y, scaler = preprocess_data_lstm(df, time_steps=time_steps)
            model = build_lstm_model(X.shape[1:])
            model.fit(X, y, epochs=5, batch_size=16, verbose=0)

            future_inputs = X[-1]
            future_predictions = []

            for _ in range(interval):
                pred = model.predict(np.expand_dims(future_inputs, axis=0))[0][0]
                future_predictions.append(pred)
                future_inputs = np.append(future_inputs[1:], [[pred]], axis=0)

            forecast = scaler.inverse_transform(np.array(future_predictions).reshape(-1, 1)).flatten()
            future_dates = pd.date_range(start=df['ds'].iloc[-1] + pd.Timedelta(days=1), periods=interval)
            st.line_chart(pd.DataFrame({'Forecast': forecast}, index=future_dates))

        # Sentiment
        with st.expander("📰 News Sentiment Analysis"):
            sentiment = get_newsapi_sentiment(query=ticker)
            st.write(f"Average sentiment polarity (News): **{sentiment:.2f}**")
            if sentiment > 0.2:
                st.success("📈 Market sentiment is positive.")
            elif sentiment < -0.2:
                st.error("📉 Market sentiment is negative.")
            else:
                st.warning("🤝 Market sentiment is neutral.")
    else:
        st.error("❌ No data available. Try another ticker.")
