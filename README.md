# 📈 Stock Price Forecasting App

A full-stack AI-powered web application for forecasting stock prices using machine learning models (Prophet & LSTM), real-time market data, and sentiment analysis from news and Twitter. Built using **Streamlit**, **Prophet**, **TensorFlow**, and integrates data from **Yahoo Finance** and **NewsAPI**.

---

## 🌟 Features

- 📊 Forecast stock prices using:
  - [x] Prophet (additive time-series model)
  - [x] LSTM (Recurrent Neural Network)
- 🔍 Pull stock data from Yahoo Finance or use mock data
- 📰 Sentiment analysis based on real news headlines and tweets
- 📈 Visualize price forecasts with Plotly
- 🔐 Secure API keys stored via `.env` file
- ✅ Responsive Streamlit frontend

---

## 📁 Project Structure

```
stock-forecast-app/
├── backend/
│   ├── data/
│   │   └── mock_data.py               # Generate sample stock data
│   ├── models/
│   │   ├── model_lstm.py              # LSTM model training & forecasting
│   │   └── model_prophet.py           # Prophet model training & forecasting
│   └── scraper/
│       ├── scraper.py                 # Yahoo Finance data fetcher
│       └── sentiment.py               # News & Twitter sentiment analysis
│
├── frontend/
│   └── app.py                         # Streamlit frontend app
│
├── notebooks/
│   ├── 1_prophet_forecast_demo.ipynb # Jupyter demo for Prophet
│   └── 2_lstm_forecast_demo.ipynb    # Jupyter demo for LSTM
│
├── .env                              # API keys (excluded in Git)
├── requirements.txt                  # Python dependencies
└── README.md                         # You're here!
```

---

## ⚙️ Setup Instructions

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/stock-forecast-app.git
cd stock-forecast-app
```

### 2. Create and Activate Virtual Environment

```bash
python3 -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

### 3. Install Requirements

```bash
pip install -r requirements.txt
```

### 4. Create `.env` File

Create a `.env` file in the root directory and add the following:

```
NEWSAPI_KEY=your_newsapi_key_here
TWITTER_API_KEY=your_twitter_api_key
TWITTER_API_SECRET=your_twitter_api_secret
TWITTER_ACCESS_TOKEN=your_twitter_access_token
TWITTER_ACCESS_SECRET=your_twitter_access_secret
```

You can get:
- News API Key from: https://newsapi.org/
- Twitter API credentials from: https://developer.twitter.com/

---

## 🚀 Run the App

```bash
streamlit run frontend/app.py
```

Then visit `http://localhost:8501` in your browser.

---

## 📓 Jupyter Notebooks

Explore the working of each model using the notebooks:

```bash
cd notebooks
jupyter notebook
```

- `1_prophet_forecast_demo.ipynb`: Demonstrates Prophet forecasting
- `2_lstm_forecast_demo.ipynb`: Demonstrates LSTM forecasting

---

## 🧠 ML Models Used

### Prophet
- Additive time-series forecasting model by Facebook
- Handles seasonality, trends, and outliers

### LSTM
- Recurrent Neural Network architecture
- Ideal for sequential/time-series prediction
- Built with TensorFlow & Keras

---

## 🗞 Sentiment Analysis

- **NewsAPI**: Headlines fetched and analyzed using `TextBlob`
- **Twitter**: Tweets about the stock symbol analyzed similarly
- Sentiment polarity score determines:
  - 📈 Positive ( > 0.2 )
  - 🤝 Neutral (between -0.2 and 0.2)
  - 📉 Negative ( < -0.2 )

---

## 🛠 TODOs

- [ ] Add ARIMA model option
- [ ] Save and reload model checkpoints
- [ ] Deploy to Streamlit Cloud or Docker
- [ ] Add user login for saved forecasts

---

## 📜 License

MIT License. See `LICENSE` for details.

---

## 👨‍💻 Author

Developed by **Josphat Kivai Muinde**  
📫 [muindejosphat@gmail.com]  
https://github.com/kivaimuinde/

---

## 🙌 Acknowledgements

- [Facebook Prophet](https://facebook.github.io/prophet/)
- [Yahoo Finance](https://finance.yahoo.com/)
- [NewsAPI](https://newsapi.org/)
- [Tweepy](https://www.tweepy.org/)
- [Streamlit](https://streamlit.io/)