import yfinance as yf
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta

# رموز تاسي الرئيسية
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
            # إزالة المنطقة الزمنية لتسهيل الدمج
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
            df.to_csv(f"{DATA_DIR}/{clean_name}.csv")
            macro_dfs[symbol] = df['Close'].rename(f"Macro_{clean_name}")

    # استخدام الراجحي كبديل للمؤشر إذا لم يتوفر TASI
    market_proxy_symbol = "1120.SR"
    market_proxy_df = None

    for symbol, name in TASI_SYMBOLS.items():
        df = fetch_stock_data(symbol, name)
        if df is not None:
            # إزالة المنطقة الزمنية للتوافق
            if df.index.tz is not None:
                df.index = df.index.tz_localize(None)
                
            if symbol == market_proxy_symbol:
                market_proxy_df = df['Close'].rename("Macro_TASI_PROXY")
            
            # دمج مع البيانات الاقتصادية
            for m_sym, m_series in macro_dfs.items():
                m_clean = m_sym.replace('=', '_').replace('^', '')
                df = df.join(m_series, how='left')
                # تعبئة الفراغات (مهم جداً للنفط لأنه يغلق في أيام تختلف عن تاسي)
                df[f"Macro_{m_clean}"] = df[f"Macro_{m_clean}"].ffill().bfill()
            
            df['Symbol'] = symbol
            all_data.append(df)
    
    if all_data:
        final_list = []
        for df in all_data:
            if market_proxy_df is not None:
                df = df.join(market_proxy_df, how='left')
                df["Macro_TASI_PROXY"] = df["Macro_TASI_PROXY"].ffill().bfill()
            
            # حفظ الملف الفردي
            df.to_csv(f"{DATA_DIR}/{symbol}.csv")
            final_list.append(df)
            
        combined_df = pd.concat(final_list)
        combined_df.to_csv("data/tasi_combined.csv")
        
        last_date = combined_df.index.max().strftime('%Y-%m-%d')
        print(f"تم جلب ومعالجة البيانات بنجاح لـ {len(TASI_SYMBOLS)} شركة.")
        print(f"أحدث تاريخ بيانات متوفر من ياهو فاينانس هو: {last_date}")

if __name__ == "__main__":
    main()
