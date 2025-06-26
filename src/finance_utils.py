import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

def load_price_data(filepath):
    df = pd.read_csv(filepath, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def add_moving_averages(df, short_window=20, long_window=50):
    df['MA_Short'] = df['Close'].rolling(window=short_window).mean()
    df['MA_Long'] = df['Close'].rolling(window=long_window).mean()
    return df

def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    
    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()
    
    rs = avg_gain / avg_loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

def add_rsi(df, period=14):
    df['RSI'] = compute_rsi(df['Close'], period)
    return df

def add_macd(df):
    exp1 = df['Close'].ewm(span=12, adjust=False).mean()
    exp2 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = exp1 - exp2
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    df['MACD_Hist'] = df['MACD'] - df['MACD_Signal']
    return df

def plot_price_with_indicators(df, stock_name):
    plt.figure(figsize=(15, 10))

    plt.subplot(3, 1, 1)
    plt.plot(df['Date'], df['Close'], label='Close Price', color='black')
    plt.plot(df['Date'], df['MA_Short'], label='MA Short', color='blue')
    plt.plot(df['Date'], df['MA_Long'], label='MA Long', color='red')
    plt.title(f'{stock_name} - Price & Moving Averages')
    plt.legend()

    plt.subplot(3, 1, 2)
    plt.plot(df['Date'], df['RSI'], label='RSI', color='purple')
    plt.axhline(70, color='red', linestyle='--', label='Overbought')
    plt.axhline(30, color='green', linestyle='--', label='Oversold')
    plt.title('Relative Strength Index (RSI)')
    plt.legend()

    plt.subplot(3, 1, 3)
    plt.plot(df['Date'], df['MACD'], label='MACD', color='orange')
    plt.plot(df['Date'], df['MACD_Signal'], label='Signal Line', color='blue')
    plt.bar(df['Date'], df['MACD_Hist'], label='MACD Histogram', color='grey')
    plt.title('MACD')
    plt.legend()

    plt.tight_layout()
    plt.show()

def plot_volume(df, stock_name):
    plt.figure(figsize=(12, 4))
    plt.bar(df['Date'], df['Volume'], color='skyblue')
    plt.title(f'{stock_name} - Trading Volume Over Time')
    plt.xlabel('Date')
    plt.ylabel('Volume')
    plt.show()

def plot_candlestick(df, stock_name):
    try:
        import mplfinance as mpf
    except ImportError:
        print("Please install mplfinance to use candlestick plotting: pip install mplfinance")
        return

    df_candle = df.set_index('Date')[['Open', 'High', 'Low', 'Close', 'Volume']]
    mpf.plot(df_candle, type='candle', volume=True, title=f"{stock_name} Candlestick Chart")

def plot_correlation_heatmap(df, stock_name):
    corr_cols = ['Open', 'High', 'Low', 'Close', 'Volume']
    corr = df[corr_cols].corr()

    plt.figure(figsize=(8, 6))
    sns.heatmap(corr, annot=True, cmap='coolwarm')
    plt.title(f'{stock_name} - Correlation Heatmap of Price & Volume')
    plt.show()
