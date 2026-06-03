import yfinance as yf
import pandas as pd
import os
import sys
import numpy as np
from datetime import datetime, timedelta
from io import StringIO
import ssl
from urllib.request import Request, urlopen

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.company_names import get_arabic_company_name

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
TARGET_MARKET = os.getenv("TASI_TARGET_MARKET", "main").strip().lower()
REFRESH_LATEST_PRICE = os.getenv("TASI_REFRESH_LATEST_PRICE", "1") == "1"
MARKET_CLOSE_HOUR = int(os.getenv("TASI_MARKET_CLOSE_HOUR", "15"))
MARKET_CLOSE_MINUTE = int(os.getenv("TASI_MARKET_CLOSE_MINUTE", "20"))

NOMU_PREFIXES = ("95", "96")

def classify_market(symbol):
    code = str(symbol).replace(".SR", "").strip()
    if code.startswith(NOMU_PREFIXES):
        return "نمو"
    return "السوق الرئيسي"

def filter_target_market(universe):
    if TARGET_MARKET in {"all", "كل", "all_markets"}:
        return universe

    filtered = {
        symbol: info
        for symbol, info in universe.items()
        if info.get("market", classify_market(symbol)) == "السوق الرئيسي"
    }
    removed = len(universe) - len(filtered)
    if removed:
        print(f"تم استبعاد {removed} رمزاً من نمو/الأسواق غير المستهدفة. الكون الحالي: {len(filtered)} رمز من السوق الرئيسي.")
    return filtered

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
            if 'market' not in cached.columns:
                cached['market'] = cached['symbol'].apply(classify_market)
            universe = {
                row['symbol']: {'name': row['name'], 'sector': row['sector'], 'market': row['market']}
                for _, row in cached.iterrows()
            }
            return filter_target_market(universe)

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
                'sector': 'غير مصنف',
                'market': classify_market(symbol)
            }

        if universe:
            os.makedirs(os.path.dirname(UNIVERSE_CACHE_PATH), exist_ok=True)
            pd.DataFrame([
                {'symbol': symbol, 'name': info['name'], 'sector': info['sector'], 'market': info['market']}
                for symbol, info in sorted(universe.items())
            ]).to_csv(UNIVERSE_CACHE_PATH, index=False)
            print(f"تم تحميل قائمة السوق الموسعة: {len(universe)} رمز.")
            return filter_target_market(universe)
    except Exception as e:
        print(f"تعذر تحميل قائمة السوق الموسعة، سيتم استخدام القائمة الاحتياطية: {e}")

    fallback = {
        symbol: {**info, 'market': classify_market(symbol)}
        for symbol, info in FALLBACK_TASI_SYMBOLS.items()
    }
    return filter_target_market(fallback)

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

def load_cached_symbol_data(symbol):
    cache_path = os.path.join(DATA_DIR, f"{symbol}.csv")
    if not os.path.exists(cache_path):
        return None
    try:
        cached = pd.read_csv(cache_path, index_col=0, parse_dates=True)
        if cached.empty or 'Close' not in cached.columns:
            return None
        cached.index = pd.to_datetime(cached.index).tz_localize(None)
        print(f"استخدام آخر نسخة محفوظة لـ {symbol} بسبب تعذر الجلب من Yahoo.")
        return cached
    except Exception as e:
        print(f"تعذر تحميل الكاش المحلي لـ {symbol}: {e}")
        return None

def latest_completed_saudi_trading_day(now=None):
    current = now or datetime.now()
    trade_day = pd.Timestamp(current).normalize()
    close_time = current.replace(hour=MARKET_CLOSE_HOUR, minute=MARKET_CLOSE_MINUTE, second=0, microsecond=0)
    if current < close_time:
        trade_day -= pd.Timedelta(days=1)

    while trade_day.weekday() in (4, 5):  # Friday, Saturday
        trade_day -= pd.Timedelta(days=1)
    return trade_day

def _scalar(value, default=np.nan):
    if value is None:
        return default
    try:
        if isinstance(value, pd.Series):
            value = value.dropna().iloc[-1] if not value.dropna().empty else default
        if isinstance(value, (list, tuple, np.ndarray)):
            value = value[-1] if len(value) else default
        return float(value)
    except (TypeError, ValueError):
        return default

def _latest_intraday_snapshot(ticker):
    try:
        intraday = ticker.history(period="5d", interval="1m", auto_adjust=False, prepost=False)
    except Exception:
        return None

    if intraday is None or intraday.empty or 'Close' not in intraday.columns:
        return None
    if isinstance(intraday.columns, pd.MultiIndex):
        intraday.columns = intraday.columns.get_level_values(0)
    if intraday.index.tz is not None:
        intraday.index = intraday.index.tz_convert(None)

    close = intraday['Close'].dropna()
    if close.empty:
        return None

    latest_ts = close.index[-1]
    latest_date = pd.Timestamp(latest_ts).normalize()
    day_rows = intraday[intraday.index.normalize() == latest_date].copy()
    if day_rows.empty:
        day_rows = intraday.tail(1).copy()

    price = _scalar(day_rows['Close'].dropna().iloc[-1] if 'Close' in day_rows else np.nan)
    if not np.isfinite(price) or price <= 0:
        return None

    return {
        "date": latest_date,
        "close": price,
        "open": _scalar(day_rows['Open'].dropna().iloc[0] if 'Open' in day_rows and not day_rows['Open'].dropna().empty else price, price),
        "high": _scalar(day_rows['High'].max() if 'High' in day_rows else price, price),
        "low": _scalar(day_rows['Low'].min() if 'Low' in day_rows else price, price),
        "volume": _scalar(day_rows['Volume'].sum() if 'Volume' in day_rows else 0, 0),
    }

def _latest_fast_info_snapshot(ticker, quote_date=None):
    try:
        fast_info = ticker.fast_info
    except Exception:
        fast_info = {}

    price = _scalar(
        getattr(fast_info, "last_price", None)
        or (fast_info.get("last_price") if hasattr(fast_info, "get") else None)
        or (fast_info.get("lastPrice") if hasattr(fast_info, "get") else None)
    )
    if not np.isfinite(price) or price <= 0:
        return None

    return {
        "date": pd.Timestamp(quote_date if quote_date is not None else datetime.now()).normalize(),
        "close": price,
        "open": _scalar(
            getattr(fast_info, "open", None)
            or (fast_info.get("open") if hasattr(fast_info, "get") else None),
            price
        ),
        "high": _scalar(
            getattr(fast_info, "day_high", None)
            or (fast_info.get("day_high") if hasattr(fast_info, "get") else None)
            or (fast_info.get("dayHigh") if hasattr(fast_info, "get") else None),
            price
        ),
        "low": _scalar(
            getattr(fast_info, "day_low", None)
            or (fast_info.get("day_low") if hasattr(fast_info, "get") else None)
            or (fast_info.get("dayLow") if hasattr(fast_info, "get") else None),
            price
        ),
        "volume": _scalar(
            getattr(fast_info, "last_volume", None)
            or (fast_info.get("last_volume") if hasattr(fast_info, "get") else None)
            or (fast_info.get("lastVolume") if hasattr(fast_info, "get") else None),
            0
        ),
    }

def apply_latest_price_snapshot(df, ticker, symbol):
    if not REFRESH_LATEST_PRICE or df.empty:
        return df

    df = df.copy()
    if df.index.tz is not None:
        df.index = df.index.tz_localize(None)
    df.index = pd.to_datetime(df.index).normalize()
    if 'Close' in df.columns:
        df = df[df['Close'].notna()].copy()
    if df.empty:
        return df

    latest_data_date = df.index.max()
    completed_trade_day = max(latest_data_date, latest_completed_saudi_trading_day())
    snapshot = _latest_fast_info_snapshot(ticker, completed_trade_day) or _latest_intraday_snapshot(ticker)
    if not snapshot:
        return df

    quote_date = snapshot["date"]
    current_close = _scalar(snapshot["close"])
    previous_close = _scalar(df['Close'].dropna().iloc[-1])

    if quote_date < latest_data_date or not np.isfinite(current_close) or current_close <= 0:
        return df

    if quote_date in df.index:
        idx = quote_date
        df.loc[idx, 'Close'] = current_close
        if 'Adj Close' in df.columns:
            df.loc[idx, 'Adj Close'] = current_close
        df.loc[idx, 'High'] = max(_scalar(df.loc[idx, 'High'], current_close), snapshot["high"], current_close)
        df.loc[idx, 'Low'] = min(_scalar(df.loc[idx, 'Low'], current_close), snapshot["low"], current_close)
        df.loc[idx, 'Open'] = _scalar(df.loc[idx, 'Open'], snapshot["open"])
        df.loc[idx, 'Volume'] = max(_scalar(df.loc[idx, 'Volume'], 0), snapshot["volume"])
    else:
        new_row = df.loc[latest_data_date].copy()
        new_row['Open'] = snapshot["open"] if np.isfinite(snapshot["open"]) else previous_close
        new_row['High'] = max(snapshot["high"], current_close, new_row['Open'])
        new_row['Low'] = min(snapshot["low"], current_close, new_row['Open'])
        new_row['Close'] = current_close
        if 'Adj Close' in df.columns:
            new_row['Adj Close'] = current_close
        new_row['Volume'] = snapshot["volume"]
        df.loc[quote_date] = new_row
        df = df.sort_index()

    if not np.isclose(previous_close, current_close, rtol=0, atol=0.0001):
        print(f"تم تحديث آخر سعر لـ {symbol}: {previous_close:.2f} -> {current_close:.2f} بتاريخ {quote_date.date()}")
    return df

def fetch_stock_data(symbol, info_dict, is_macro=False):
    latest_trade_day = latest_completed_saudi_trading_day()
    start_date = latest_trade_day - pd.Timedelta(days=365*3)
    end_date = latest_trade_day + pd.Timedelta(days=1)  # yfinance end is exclusive.
    
    name = info_dict if is_macro else info_dict['name']
    print(f"جاري سحب بيانات {name} ({symbol}) حتى إغلاق {latest_trade_day.date()}...")
    
    try:
        ticker = yf.Ticker(symbol)
        df = yf.download(
            symbol,
            start=start_date.strftime('%Y-%m-%d'),
            end=end_date.strftime('%Y-%m-%d'),
            interval="1d",
            progress=False,
            auto_adjust=False,
        )
        
        if df.empty:
            return load_cached_symbol_data(symbol)
        if isinstance(df.columns, pd.MultiIndex): df.columns = df.columns.get_level_values(0)
        if df.index.tz is not None: df.index = df.index.tz_localize(None)
        df = apply_latest_price_snapshot(df, ticker, symbol)

        if not is_macro:
            info = {}
            if FETCH_FUNDAMENTALS:
                info = ticker.info
            df['PE_Ratio'] = info.get('trailingPE', np.nan)
            df['Div_Yield'] = info.get('dividendYield', 0.0)
            df['Market_Cap'] = info.get('marketCap', np.nan)
            df['Symbol'] = symbol
            df['Company Name'] = name
            df['Company Name Arabic'] = get_arabic_company_name(symbol, name)
            df['Sector'] = info.get('sector') or info_dict['sector']
            df['Market'] = info_dict.get('market', classify_market(symbol))
            df['Sentiment'] = calculate_liquidity_sentiment(df)
            
        return df
    except Exception as e:
        print(f"خطأ في سحب بيانات {symbol}: {e}")
        return load_cached_symbol_data(symbol)

def attach_macro_series(df, series):
    column = series.name
    if column in df.columns:
        df = df.drop(columns=[column])
    df = df.join(series, how='left')
    df[column] = df[column].ffill().bfill()
    return df

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
                df = attach_macro_series(df, m_series)
            
            if market_proxy_df is not None:
                df = attach_macro_series(df, market_proxy_df)
            
            df.to_csv(f"{DATA_DIR}/{symbol}.csv")
            final_list.append(df)
            
        combined_df = pd.concat(final_list)
        combined_df.to_csv("data/tasi_combined.csv")
        print(f"تم تحديث البيانات لـ {len(final_list)} شركة مع إضافة المشاعر والقطاعات.")

if __name__ == "__main__":
    main()
