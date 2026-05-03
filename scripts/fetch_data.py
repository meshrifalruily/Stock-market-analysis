import yfinance as yf
import pandas as pd
import os
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
import ssl
from urllib.request import Request, urlopen

# قائمة احتياطية إذا تعذر جلب قائمة السوق الكاملة من الإنترنت
FALLBACK_TASI_SYMBOLS = {
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
UNIVERSE_CACHE_PATH = "data/tasi_universe.csv"
STOCK_UNIVERSE_URL = "https://stockanalysis.com/list/saudi-stock-exchange/"
FETCH_FUNDAMENTALS = os.getenv("TASI_FETCH_FUNDAMENTALS", "0") == "1"

def normalize_saudi_symbol(symbol):
    symbol = str(symbol).strip()
    if symbol.endswith(".SR"):
        return symbol
    if symbol.isdigit():
        return f"{symbol}.SR"
    return symbol

def load_symbol_universe():
    """تحميل كل رموز السوق المتاحة مع كاش محلي واحتياطي عند فشل الإنترنت."""
    if os.path.exists(UNIVERSE_CACHE_PATH):
        cached = pd.read_csv(UNIVERSE_CACHE_PATH)
        if {'symbol', 'name', 'sector'}.issubset(cached.columns) and not cached.empty:
            return {
                row['symbol']: {'name': row['name'], 'sector': row['sector']}
                for _, row in cached.iterrows()
            }

    try:
        try:
            request = Request(STOCK_UNIVERSE_URL, headers={"User-Agent": "Mozilla/5.0"})
            with urlopen(request, timeout=30) as response:
                html = response.read().decode("utf-8")
        except Exception:
            try:
                context = ssl._create_unverified_context()
                request = Request(STOCK_UNIVERSE_URL, headers={"User-Agent": "Mozilla/5.0"})
                with urlopen(request, timeout=30, context=context) as response:
                    html = response.read().decode("utf-8")
            except Exception:
                from curl_cffi import requests
                response = requests.get(STOCK_UNIVERSE_URL, impersonate="chrome", timeout=30)
                response.raise_for_status()
                html = response.text

        tables = pd.read_html(StringIO(html))
        universe_table = next(
            table for table in tables
            if {'Symbol', 'Company Name'}.issubset(set(table.columns))
        )
        universe = {}
        for _, row in universe_table.iterrows():
            symbol = normalize_saudi_symbol(row['Symbol'])
            if not symbol.endswith(".SR"):
                continue
            universe[symbol] = {
                'name': str(row['Company Name']).strip(),
                'sector': 'غير مصنف'
            }

        if universe:
            os.makedirs(os.path.dirname(UNIVERSE_CACHE_PATH), exist_ok=True)
            pd.DataFrame([
                {'symbol': symbol, 'name': info['name'], 'sector': info['sector']}
                for symbol, info in sorted(universe.items())
            ]).to_csv(UNIVERSE_CACHE_PATH, index=False)
            print(f"تم تحميل قائمة السوق الموسعة: {len(universe)} رمز.")
            return universe
    except Exception as e:
        print(f"تعذر تحميل قائمة السوق الموسعة، سيتم استخدام القائمة الاحتياطية: {e}")

    return FALLBACK_TASI_SYMBOLS

def apply_universe_limit(symbols):
    limit = os.getenv("TASI_MAX_SYMBOLS", "").strip()
    if not limit:
        return symbols
    try:
        limit_value = int(limit)
    except ValueError:
        return symbols
    limited = dict(list(symbols.items())[:limit_value])
    print(f"تم تحديد عدد الرموز مؤقتاً إلى {len(limited)} عبر TASI_MAX_SYMBOLS.")
    return limited

def calculate_liquidity_sentiment(df):
    """درجة حتمية مبنية على تدفق السيولة والزخم بدلاً من رقم عشوائي."""
    close = df['Close']
    volume = df['Volume'].replace(0, np.nan)
    ret_5d = close.pct_change(5).fillna(0)
    vol_ratio = (volume / volume.rolling(20, min_periods=5).mean()).replace([np.inf, -np.inf], np.nan).fillna(1.0)
    raw_score = 0.5 + (ret_5d * 3.0) + ((vol_ratio - 1.0) * 0.12)
    return raw_score.clip(0.05, 0.95)

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
            info = {}
            if FETCH_FUNDAMENTALS:
                ticker = yf.Ticker(symbol)
                info = ticker.info
            df['PE_Ratio'] = info.get('trailingPE', np.nan)
            df['Div_Yield'] = info.get('dividendYield', 0.0)
            df['Market_Cap'] = info.get('marketCap', np.nan)
            df['Symbol'] = symbol
            df['Company Name'] = name
            df['Sector'] = info.get('sector') or info_dict['sector']
            df['Sentiment'] = calculate_liquidity_sentiment(df)
            
        return df
    except Exception as e:
        print(f"خطأ في سحب بيانات {symbol}: {e}")
        return None

def main():
    if not os.path.exists(DATA_DIR): os.makedirs(DATA_DIR)
    tasi_symbols = apply_universe_limit(load_symbol_universe())
    
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

    for symbol, info in tasi_symbols.items():
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
