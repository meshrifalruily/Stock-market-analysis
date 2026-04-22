import pandas as pd
import numpy as np
import os

def calculate_technical_indicators(df):
    """Calculates common technical indicators."""
    
    # Simple Moving Averages
    df['SMA_20'] = df['Close'].rolling(window=20).mean()
    df['SMA_50'] = df['Close'].rolling(window=50).mean()
    df['SMA_200'] = df['Close'].rolling(window=200).mean()
    
    # Exponential Moving Average
    df['EMA_20'] = df['Close'].ewm(span=20, adjust=False).mean()
    
    # RSI (Relative Strength Index)
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # MACD
    ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD'] = ema_12 - ema_26
    df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # ATR (Average True Range) for exit points
    high_low = df['High'] - df['Low']
    high_close = np.abs(df['High'] - df['Close'].shift())
    low_close = np.abs(df['Low'] - df['Close'].shift())
    ranges = pd.concat([high_low, high_close, low_close], axis=1)
    true_range = np.max(ranges, axis=1)
    df['ATR'] = true_range.rolling(14).mean()
    
    # Bollinger Bands (Institutional Volatility measure)
    df['BB_Mid'] = df['Close'].rolling(window=20).mean()
    df['BB_Std'] = df['Close'].rolling(window=20).std()
    df['BB_Upper'] = df['BB_Mid'] + (df['BB_Std'] * 2)
    df['BB_Lower'] = df['BB_Mid'] - (df['BB_Std'] * 2)
    
    # Support & Resistance (Recent Swing High/Low)
    df['Support'] = df['Low'].rolling(window=50).min()
    df['Resistance'] = df['High'].rolling(window=50).max()
    
    # Trend Strength (Simple ADX approximation)
    up_move = df['High'].diff()
    down_move = df['Low'].diff()
    df['Plus_DI'] = 100 * (up_move.where((up_move > down_move) & (up_move > 0), 0).rolling(14).mean() / df['ATR'])
    df['Minus_DI'] = 100 * (down_move.where((down_move > up_move) & (down_move > 0), 0).rolling(14).mean() / df['ATR'])
    df['ADX'] = abs((df['Plus_DI'] - df['Minus_DI']) / (df['Plus_DI'] + df['Minus_DI'])) * 100
    
    # Returns
    df['Daily_Return'] = df['Close'].pct_change()
    df['Weekly_Return'] = df['Close'].pct_change(5)
    df['Monthly_Return'] = df['Close'].pct_change(21)
    
    # Target: Next Day Return (Shifted)
    df['Target_Next_Day_Return'] = df['Close'].pct_change().shift(-1)
    
    # Clean up (remove first few rows with NaN due to rolling windows)
    # Ensure Company Name and Symbol are preserved
    return df.dropna(subset=df.columns.difference(['Company Name', 'Symbol']))

def preprocess_all_data(combined_file_path, output_file_path):
    if not os.path.exists(combined_file_path):
        print(f"Error: {combined_file_path} not found.")
        return
    
    df = pd.read_csv(combined_file_path)
    # Convert 'Date' to datetime and set as index (if applicable)
    df['Date'] = pd.to_datetime(df['Date'], utc=True)
    df = df.sort_values(['Symbol', 'Date'])
    
    processed_dfs = []
    for symbol, group in df.groupby('Symbol'):
        print(f"Processing indicators for {symbol}...")
        processed_group = calculate_technical_indicators(group.copy())
        processed_dfs.append(processed_group)
    
    final_df = pd.concat(processed_dfs)
    final_df.to_csv(output_file_path, index=False)
    print(f"Preprocessed data saved to {output_file_path}")

if __name__ == "__main__":
    preprocess_all_data("data/tasi_combined.csv", "data/tasi_processed.csv")
