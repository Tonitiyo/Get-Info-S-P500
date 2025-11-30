### S&P 500 constituents data

import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
import yfinance as yf
import datetime as date
import scipy.stats as stats
import requests # To fetch the Wikipedia page

# Retrieve ticker symbols for S&P 500 constituents from Wikipedia
def get_stock_data():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    ### instal pip install lxml if you don't have it
    # Wikipedia blocks pandas.read_html by default, so we use requests to get the page content
    # We are using a user-agent to mimic a browser visit
    headers = {'User-Agent': 'Mozilla/5.0'}
    #Install requests if you don't have it and import it
    response = requests.get(url, headers=headers)
    tables = pd.read_html(response.text)
    sp500_tickers = tables[0]["Symbol"].tolist()
    # Some tickers have dots in them which yfinance does not recognize, replace them with hyphens
    data = yf.download(sp500_tickers, start='2000-01-01', end=date.datetime.today(), group_by='ticker')
    data = data.dropna(how='all')  # Drop rows where all elements are NaN
    return data

# test
# SP500_constituents = get_stock_data()
# print(SP500_constituents["MSFT"].loc["2025"].head())

# Calculate returns 
def calculate_returns(data):
    returns = pd.DataFrame()
    for ticker in data.columns.levels[0]:
        returns[ticker] = data[ticker]['Close'].pct_change()
    returns = returns.dropna()
    return returns

# calculate volatility
def calculate_volatility(returns):
    ## Daily volatility
    ### Rolling window is set to 21 days (approximately one month - can be 3 months or 1 year: window=252)
    Daily_vol = returns.rolling(window=21).std()
    Ann_vol = returns.rolling(window=21).std() * np.sqrt(252)  # Annualized volatility
    return Daily_vol, Ann_vol

# Calculate covariance matrix
def calculate_covariance(returns):
    covariance_matrix = returns.cov() * 252  # Annualize the covariance matrix
    return covariance_matrix

# Build equally-weighted portfolio returns
def build_equal_weight_portfolio(returns):
    """
    Construct an equally-weighted portfolio from individual stock returns.
    """
    n_assets = returns.shape[1]
    weights = np.ones(n_assets) / n_assets  # 1/N
    # Matrix multiplication: each row (date) * weights
    portfolio_returns = returns.dot(weights)
    portfolio_returns.name = "Portfolio_Return"
    return portfolio_returns

def compute_performance_metrics(portfolio_returns, rf_annual=0.02):
    """
    Compute key risk & performance metrics for a portfolio time series of daily returns.
    
    rf_annual : annual risk-free rate (e.g. 2% = 0.02)
    """
    # Convert risk-free to daily
    rf_daily = (1 + rf_annual) ** (1/252) - 1

    # Excess returns
    excess_returns = portfolio_returns - rf_daily

    # Annualized return & volatility
    avg_daily_ret = portfolio_returns.mean()
    ann_ret = (1 + avg_daily_ret) ** 252 - 1
    ann_vol = portfolio_returns.std() * np.sqrt(252)

    # Sharpe ratio (annualized)
    sharpe = (excess_returns.mean() * 252) / (portfolio_returns.std() * np.sqrt(252))

    # Max drawdown
    cum_returns = (1 + portfolio_returns).cumprod()
    running_max = cum_returns.cummax()
    drawdown = (cum_returns / running_max) - 1
    max_dd = drawdown.min()

    # Historical VaR (95%) on daily returns
    var_95 = np.percentile(portfolio_returns, 5)  # 5% quantile (left tail)

    metrics = {
        "Annual Return": ann_ret,
        "Annual Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Daily 95% VaR": var_95
    }
    return metrics, cum_returns, drawdown


def plot_portfolio_performance(cum_returns, drawdown):
    """
    Plot cumulative performance and drawdown of the portfolio.
    """
    fig, axes = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Cumulative performance
    axes[0].plot(cum_returns.index, cum_returns.values)
    axes[0].set_title("Cumulative Portfolio Performance")
    axes[0].set_ylabel("Cumulative Value (base = 1)")
    axes[0].grid(True)

    # Drawdown
    axes[1].plot(drawdown.index, drawdown.values)
    axes[1].set_title("Portfolio Drawdown")
    axes[1].set_ylabel("Drawdown")
    axes[1].set_xlabel("Date")
    axes[1].grid(True)

    plt.tight_layout()
    plt.show()


def plot_rolling_sharpe(portfolio_returns, rf_annual=0.02, window=252):
    """
    Plot rolling annualized Sharpe ratio.
    window : rolling window in days (252 ≈ 1 year)
    """
    rf_daily = (1 + rf_annual) ** (1/252) - 1
    excess_returns = portfolio_returns - rf_daily

    rolling_vol = excess_returns.rolling(window).std() * np.sqrt(252)
    rolling_mean = excess_returns.rolling(window).mean() * 252
    rolling_sharpe = rolling_mean / rolling_vol

    plt.figure(figsize=(10,4))
    plt.plot(rolling_sharpe.index, rolling_sharpe.values)
    plt.title(f"Rolling Sharpe Ratio (window = {window} days)")
    plt.ylabel("Sharpe")
    plt.xlabel("Date")
    plt.grid(True)
    plt.show()
