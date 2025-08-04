# backend/scraper/scraper.py

import yfinance as yf

def fetch_stock_data(ticker, period="1mo", interval="1d"):
    """
    Fetch historical stock data using yfinance.
    :param ticker: Stock ticker symbol, e.g. 'AAPL'
    :param period: Data period, e.g. '1mo', '3mo', '1y'
    :param interval: Data interval, e.g. '1d', '1h', '5m'
    :return: DataFrame with historical data
    """
    stock = yf.Ticker(ticker)
    df = stock.history(period=period, interval=interval)
    return df
