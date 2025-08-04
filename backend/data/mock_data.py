# backend/data/mock_data.py

import pandas as pd
import numpy as np
from datetime import datetime, timedelta

def generate_mock_stock_data(days=90):
    """
    Generate synthetic stock data for testing (default: 90 days).
    Returns a DataFrame with Date and Close columns.
    """
    base_date = datetime.today()
    dates = [base_date - timedelta(days=i) for i in range(days)]
    dates.reverse()

    np.random.seed(42)  # reproducibility
    prices = np.cumsum(np.random.randn(days) * 2 + 0.5) + 100  # random walk
    df = pd.DataFrame({'Date': dates, 'Close': prices})
    return df
