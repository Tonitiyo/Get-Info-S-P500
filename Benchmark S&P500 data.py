#Benchmark S&P500 data

import yfinance as yf
import datetime as date
import pandas as pd
import numpy as np

def get_data(tickers):
    # Download historical data for S&P 500
    data = yf.download(tickers, start='2000-01-01', end=date.datetime.today())

    # Dowload risk free rate data (3-Month Treasury Bill and 10-Year Treasury Note)
    risk_free_3m = yf.download('^IRX', start='2000-01-01', end=date.datetime.today())
    risk_free_10 = yf.download('^TNX', start='2000-01-01', end=date.datetime.today())

    # Closing prices
    data = data['Close']

   # If it's a Series (single ticker), make it a DataFrame with a nice name
    if isinstance(data, pd.Series):
        data = data.to_frame(name="Close")

    data = data.dropna()


    # Daily returns
    data['Return'] = data['Close'].pct_change()
    data = data.dropna()
    # Log returns
    data['LogReturn'] = np.log(data['Close'] / data['Close'].shift(1))
    data = data.dropna()
    #Daily volatility
    data['Daily_vol'] = data['LogReturn'].rolling(window=21).std()
    # Annualized volatility
    data['Volatility'] = data['LogReturn'].rolling(window=21).std() * np.sqrt(252)  
    data = data.dropna()
    # Cumulative returns
    data['CumulativeReturn'] = (1 + data['Return']).cumprod() - 1
    data = data.dropna()
    # Cumulative log returns
    data['CumulativeLogReturn'] = data['LogReturn'].cumsum()
    data = data.dropna() 

    # risk free rate
    data['RiskFree_3M'] = risk_free_3m['Close'] / 100
    data['RiskFree_10Y'] = risk_free_10['Close'] / 100
    data = data.dropna()

    return data 

print
