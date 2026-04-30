import yfinance as yf
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta

# قائمة موسعة لشركات تاسي (20 شركة كبرى لعمل Heatmap)
TASI_SYMBOLS = {
    "1120.SR": {"name": "مصرف الراجحي", "sector": "البنوك"},
    "1180.SR": {"name": "الأهلي السعودي", "sector": "البنوك"},
    "2222.SR": {"name": "أرامكو السعودية", "sector": "الطاقة"},
    "2010.SR": {"name": "سابك", "sector": "المواد الأساسية"},
    "7010.SR": {"name": "إس تي سي", "sector": "الاتصالات"},
    "1150.SR": {"name": "مصرف الإنماء", "sector": "البنوك"},
    "2350.SR": {"name": "كيان السعودية", "sector": "المواد الأساسية"},
    "2020.SR": {"name": "سابك للمغذيات", "sector": "المواد الأساسية"},
    "4003.SR": {"name": "إكسترا", "sector": "التجزئة"},
    "1010.SR": {"name": "بنك الرياض", "sector": "البنوك"},
    "1111.SR": {"name": "تداول السعودية", "sector": "الخدمات المالية"},
    "7020.SR": {"name": "اتحاد اتصالات", "sector": "الاتصالات"},
    "5110.SR": {"name": "كهرباء السعودية", "sector": "المرافق العامة"},
    "2080.SR": {"name": "مجموعة تداول", "sector": "الخدمات المالية"},
    "1211.SR": {"name": "معادن", "sector": "المواد الأساسية"},
    "4260.SR": {"name": "بدجت السعودية", "sector": "النقل"},
    "4030.SR": {"name": "البحري", "sector": "النقل"},
    "2280.SR": {"name": "المراعي", "sector": "الأغذية"},
    "4190.SR": {"name": "جرير", "sector": "التجزئة"},
    "1060.SR": {"name": "البنك السعودي الفرنسي", "sector": "البنوك"}
}

MACRO_SYMBOLS = {"BZ=F": "Brent Oil"}
DATA_DIR = "data/raw"

def get_sentiment_score(symbol):
    """محاكاة لتحليل المشاعر بناءً على حجم التداول والعوائد الأخيرة."""
    # في نسخة متقدمة، يمكن هنا كشط أخبار 'أرقام' أو 'تويتر'
    # حالياً سنستخدم منطقاً يعتمد على تدفق السيولة (Volume Flow)
    return np.random.uniform(0.1, 0.9) # قيمة تجريبية سيتم استبدالها بتحليل حقيقي لاحقاً

def fetch_stock_data(symbol, info_dict, is_macro=False):
    today = datetime.now()
    tomorrow = today + timedelta(days=1)
    start_date = today - timedelta(days=365*3)
    
    name = info_dict if is_macro else info_dict['name']
    print(f"جاري سحب بيانات {name} ({symbol})...")
    
    try:
        df = yf.download(symbol, start=start_date.strftime('%Y-%m-%d'), end=tomorrow.strftime('%Y-%m-%d'), interval="1d", progress=False, auto_adjust=False)
        
        if df.empty: return None
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)

        if not is_macro:
            ticker = yf.Ticker(symbol)
            info = ticker.info
            df['PE_Ratio'] = info.get('trailingPE', np.nan)
            df['Div_Yield'] = info.get('dividendYield', 0.0)
            df['Market_Cap'] = info.get('marketCap', np.nan)
            df['Symbol'] = symbol
            df['Company Name'] = name
            df['Sector'] = info_dict['sector']
            # إضافة درجة المشاعر
            df['Sentiment'] = get_sentiment_score(symbol)
            
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

    for symbol, info in TASI_SYMBOLS.items():
        df = fetch_stock_data(symbol, info)
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
        print(f"تم تحديث البيانات لـ {len(final_list)} شركة مع إضافة المشاعر والقطاعات.")

if __name__ == "__main__":
    main()
