import yfinance as yf
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta

# رموز تاسي الرئيسية
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

# رموز المؤشرات الاقتصادية
MACRO_SYMBOLS = {
    "BZ=F": "Brent Oil"
}

DATA_DIR = "data/raw"

def fetch_stock_data(symbol, name, is_macro=False, period="5y", interval="1d"):
    print(f"جاري جلب البيانات لـ {name} ({symbol})...")
    try:
        ticker = yf.Ticker(symbol)
        df = ticker.history(period=period, interval=interval)
        
        if df.empty:
            print(f"لم يتم العثور على بيانات لـ {symbol}")
            return None

        # إضافة البيانات الأساسية إذا كان سهماً
        if not is_macro:
            info = ticker.info
            df['PE_Ratio'] = info.get('trailingPE', np.nan)
            df['Div_Yield'] = info.get('dividendYield', 0.0)
            df['Market_Cap'] = info.get('marketCap', np.nan)
            df['Symbol'] = symbol
            df['Company Name'] = name
            
        return df
    except Exception as e:
        print(f"خطأ في جلب {symbol}: {e}")
        return None

def main():
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)
    
    all_data = []
    
    # جلب البيانات الاقتصادية الكلية أولاً
    macro_dfs = {}
    for symbol, name in MACRO_SYMBOLS.items():
        df = fetch_stock_data(symbol, name, is_macro=True)
        if df is not None:
            clean_name = symbol.replace('=', '_').replace('^', '')
            df.to_csv(f"{DATA_DIR}/{clean_name}.csv")
            macro_dfs[symbol] = df['Close'].rename(f"Macro_{clean_name}")

    # استخدام الراجحي كبديل للمؤشر إذا لم يتوفر TASI
    market_proxy_symbol = "1120.SR"
    market_proxy_df = None

    for symbol, name in TASI_SYMBOLS.items():
        df = fetch_stock_data(symbol, name)
        if df is not None:
            if symbol == market_proxy_symbol:
                market_proxy_df = df['Close'].rename("Macro_TASI_PROXY")
            
            # دمج مع البيانات الاقتصادية
            for m_sym, m_series in macro_dfs.items():
                m_clean = m_sym.replace('=', '_').replace('^', '')
                df = df.join(m_series, how='left')
                df[f"Macro_{m_clean}"] = df[f"Macro_{m_clean}"].ffill()
            
            df['Symbol'] = symbol
            all_data.append(df)
    
    if all_data:
        final_list = []
        for df in all_data:
            if market_proxy_df is not None:
                df = df.join(market_proxy_df, how='left')
                df["Macro_TASI_PROXY"] = df["Macro_TASI_PROXY"].ffill()
            
            df.to_csv(f"{DATA_DIR}/{df['Symbol'].iloc[0]}.csv")
            final_list.append(df)
            
        combined_df = pd.concat(final_list)
        combined_df.to_csv("data/tasi_combined.csv")
        print(f"تم حفظ البيانات المدمجة مع الميزات الاقتصادية وبديل المؤشر لـ {len(TASI_SYMBOLS)} شركة.")

if __name__ == "__main__":
    main()
