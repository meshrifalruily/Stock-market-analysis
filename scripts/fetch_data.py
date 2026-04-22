import yfinance as yf
import pandas as pd
import os
from datetime import datetime, timedelta

# Major TASI Symbols and their names (or fetch dynamically)
TASI_SYMBOLS = {
    "1120.SR": "Al Rajhi Bank",
    "1180.SR": "SNB (Saudi National Bank)",
    "2222.SR": "Saudi Aramco",
    "2010.SR": "SABIC",
    "7010.SR": "STC (Saudi Telecom)",
    "1150.SR": "Alinma Bank",
    "2350.SR": "Saudi Kayan",
    "2020.SR": "SABIC Agri-Nutrients",
    "4003.SR": "Extra",
    "1010.SR": "Riyad Bank"
}

DATA_DIR = "data/raw"

def fetch_stock_data(symbol, name, period="2y", interval="1d"):
    print(f"Fetching data for {name} ({symbol})...")
    try:
        # yf.download is often more reliable for latest data
        df = yf.download(symbol, period=period, interval=interval, progress=False)

        if df.empty:
            print(f"No data found for {symbol}")
            return None

        # Flatten MultiIndex columns if present (common in recent yfinance versions)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.sort_index()
        df['Company Name'] = name
        return df

    except Exception as e:
        print(f"Error fetching {symbol}: {e}")
        return None

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    all_data = []
    
    for symbol, name in TASI_SYMBOLS.items():
        df = fetch_stock_data(symbol, name)
        if df is not None:
            # Save individual file
            df.to_csv(f"{DATA_DIR}/{symbol}.csv")
            df['Symbol'] = symbol
            all_data.append(df)
    
    if all_data:
        combined_df = pd.concat(all_data)
        combined_df.to_csv("data/tasi_combined.csv")
        print(f"Saved combined data for {len(TASI_SYMBOLS)} symbols.")

if __name__ == "__main__":
    main()
