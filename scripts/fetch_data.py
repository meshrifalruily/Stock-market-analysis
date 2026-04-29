import yfinance as yf
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta

# رموز تاسي الرئيسية بأسماء عربية
TASI_SYMBOLS = {
    "1120.SR": "مصرف الراجحي",
    "1180.SR": "البنك الأهلي السعودي",
    "2222.SR": "أرامكو السعودية",
    "2010.SR": "سابك",
    "7010.SR": "إس تي سي (الاتصالات)",
    "1150.SR": "مصرف الإنماء",
    "2350.SR": "كيان السعودية",
    "2020.SR": "سابك للمغذيات الزراعية",
    "4003.SR": "إكسترا",
    "1010.SR": "بنك الرياض"
}

MACRO_SYMBOLS = {"BZ=F": "Brent Oil"}
DATA_DIR = "data/raw"

def fetch_stock_data(symbol, name, is_macro=False):
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    start_date = today - timedelta(days=365*3) # 3 سنوات لضمان استقرار المؤشرات
    
    print(f"جاري سحب أحدث بيانات لـ {name} ({symbol})...")
    try:
        # جلب البيانات بأعلى دقة متوفرة
        df = yf.download(
            symbol, 
            start=start_date.strftime('%Y-%m-%d'),
            end=tomorrow.strftime('%Y-%m-%d'),
            interval="1d",
            progress=False,
            auto_adjust=True
        )
        
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)

        if not is_macro:
            ticker = yf.Ticker(symbol)
            # جلب البيانات الأساسية (Fundamentals)
            info = ticker.info
            df['PE_Ratio'] = info.get('trailingPE', np.nan)
            df['Div_Yield'] = info.get('dividendYield', 0.0)
            df['Market_Cap'] = info.get('marketCap', np.nan)
            df['Symbol'] = symbol
            df['Company Name'] = name
            
        return df
    except Exception as e:
        print(f"خطأ في سحب بيانات {symbol}: {e}")
        return None

def main():
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    
    macro_dfs = {}
    for symbol, name in MACRO_SYMBOLS.items():
        df = fetch_stock_data(symbol, name, is_macro=True)
        if df is not None:
            clean_name = symbol.replace('=', '_')
            df.to_csv(f"{DATA_DIR}/{clean_name}.csv")
            macro_dfs[symbol] = df['Close'].rename(f"Macro_{clean_name}")

    market_proxy_symbol = "1120.SR"
    market_proxy_df = None
    stocks_data = {}

    for symbol, name in TASI_SYMBOLS.items():
        df = fetch_stock_data(symbol, name)
        if df is not None:
            if symbol == market_proxy_symbol:
                market_proxy_df = df['Close'].rename("Macro_TASI_PROXY")
            stocks_data[symbol] = df

    if stocks_data:
        final_list = []
        for symbol, df in stocks_data.items():
            for m_sym, m_series in macro_dfs.items():
                m_clean = m_sym.replace('=', '_')
                df = df.join(m_series, how='left')
                df[f"Macro_{m_clean}"] = df[f"Macro_{m_clean}"].ffill().bfill()
            
            if market_proxy_df is not None:
                df = df.join(market_proxy_df, how='left')
                df["Macro_TASI_PROXY"] = df["Macro_TASI_PROXY"].ffill().bfill()
            
            df.to_csv(f"{DATA_DIR}/{symbol}.csv")
            final_list.append(df)
            
        combined_df = pd.concat(final_list)
        combined_df.to_csv("data/tasi_combined.csv")
        
        print(f"--- نجاح جلب البيانات ---")
        print(f"تم تحديث البيانات لـ {len(final_list)} شركة.")
        print(f"أحدث تاريخ متوفر: {combined_df.index.max().strftime('%Y-%m-%d')}")

if __name__ == "__main__":
    main()
