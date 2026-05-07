import pandas as pd
import numpy as np
import pandas_ta as ta
import os
import sys
import warnings

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.company_names import get_arabic_company_name

# تجاهل التحذيرات الرياضية المتوقعة عند التعامل مع التباين الصفري
warnings.filterwarnings('ignore', category=RuntimeWarning)

NOMU_PREFIXES = ("95", "96")

def is_main_market_symbol(symbol):
    code = str(symbol).replace(".SR", "").strip()
    return not code.startswith(NOMU_PREFIXES)

def indicator_or_nan(values, index):
    if values is None:
        return pd.Series(np.nan, index=index)
    return values

def calculate_advanced_metrics(df):
    """محاكاة Pine Script وحساب المؤشرات الفنية والمقاييس الأسبوعية."""
    
    # توحيد أسماء الأعمدة (أحرف صغيرة)
    df_ta = df.copy()
    df_ta.columns = [c.lower() for c in df_ta.columns]
    
    # 1. مؤشرات الاتجاه (Trend)
    df['sma_20'] = indicator_or_nan(ta.sma(df_ta['close'], length=20), df.index)
    df['sma_50'] = indicator_or_nan(ta.sma(df_ta['close'], length=50), df.index)
    df['sma_200'] = indicator_or_nan(ta.sma(df_ta['close'], length=200), df.index)
    df['ema_20'] = indicator_or_nan(ta.ema(df_ta['close'], length=20), df.index)
    
    # 2. مؤشرات الزخم (Momentum)
    df['rsi'] = indicator_or_nan(ta.rsi(df_ta['close'], length=14), df.index)
    macd = ta.macd(df_ta['close'])
    if macd is not None:
        df = pd.concat([df, macd], axis=1)
        
    # 3. قوة الاتجاه (ADX)
    adx_df = ta.adx(df_ta['high'], df_ta['low'], df_ta['close'], length=14)
    if adx_df is not None:
        df['tv_adx'] = adx_df['ADX_14']
    else:
        df['tv_adx'] = 0
    
    # 4. مؤشرات التقلب (Volatility)
    df['atr'] = indicator_or_nan(ta.atr(df_ta['high'], df_ta['low'], df_ta['close'], length=14), df.index)
    bbands = ta.bbands(df_ta['close'], length=20)
    if bbands is not None:
        df = pd.concat([df, bbands], axis=1)
    
    # 5. محاكاة إشارة TradingView
    df['tv_signal'] = 0
    buy_cond = (df_ta['close'] > df['ema_20']) & (df['rsi'] > 50) & (df['tv_adx'] > 20)
    strong_buy = buy_cond & (df_ta['close'] > df['sma_50']) & (df['rsi'] > 60)
    df.loc[buy_cond, 'tv_signal'] = 1
    df.loc[strong_buy, 'tv_signal'] = 2
    
    # 6. مقاييس المخاطر والأداء
    df['daily_return'] = df_ta['close'].pct_change()
    for lag in [1, 2, 3, 5, 10, 20]:
        df[f'return_{lag}d'] = df_ta['close'].pct_change(lag)
    df['weekly_return_hist'] = df_ta['close'].pct_change(5) 
    df['volatility_10d'] = df['daily_return'].rolling(10).std()
    df['volatility_20d'] = df['daily_return'].rolling(20).std()
    df['traded_value'] = df_ta['close'] * df_ta['volume']
    df['avg_traded_value_20d'] = df['traded_value'].rolling(20).mean()
    df['volume_ratio_20d'] = df_ta['volume'] / df_ta['volume'].rolling(20).mean().replace(0, 1)
    df['volume_ratio_5d'] = df_ta['volume'] / df_ta['volume'].rolling(5).mean().replace(0, 1)
    df['atr_pct'] = df['atr'] / df_ta['close'].replace(0, 1)
    df['price_vs_sma_20'] = (df_ta['close'] / df['sma_20'].replace(0, 1)) - 1
    df['price_vs_sma_50'] = (df_ta['close'] / df['sma_50'].replace(0, 1)) - 1
    df['price_vs_sma_200'] = (df_ta['close'] / df['sma_200'].replace(0, 1)) - 1
    
    # 6.5 Momentum Features
    df['momentum_20d'] = df_ta['close'].pct_change(20)
    df['momentum_10d'] = df_ta['close'].pct_change(10)
    df['rsi_slope'] = df['rsi'].diff(3)
    
    # 6.6 Z-Score Normalization
    for col in ['volume', 'traded_value', 'volatility_20d']:
        if col in df.columns:
            rolling_mean = df[col].rolling(window=60).mean()
            rolling_std = df[col].rolling(window=60).std().replace(0, 0.001)
            df[f'{col}_zscore'] = (df[col] - rolling_mean) / rolling_std

    # 6.7 Alpha Features
    df['relative_volume'] = df_ta['volume'] / df_ta['volume'].rolling(60).mean().replace(0, 1)
    df['vol_shock'] = df['volume_ratio_5d'] / df['volume_ratio_20d'].replace(0, 1)
    df['price_velocity'] = df['momentum_10d'] / df['volatility_20d'].replace(0, 0.001)
    df['distance_from_high_20d'] = (df_ta['close'] / df_ta['high'].rolling(20).max().replace(0, 1)) - 1
    
    # Relative RSI (placeholder)
    df['rsi_relative'] = df['rsi'] 

    # Stochastic RSI
    stoch_rsi = ta.stochrsi(df_ta['close'], length=14, rsi_length=14, k=3, d=3)
    if stoch_rsi is not None:
        df['stoch_rsi_k'] = stoch_rsi.iloc[:, 0]
        df['stoch_rsi_d'] = stoch_rsi.iloc[:, 1]
    else:
        df['stoch_rsi_k'] = 0
        df['stoch_rsi_d'] = 0
        
    df['atr_slope'] = df['atr'].diff(5)
    
    if 'macdh_12_26_9' in df.columns:
        df['macd_histogram_slope'] = macd.iloc[:, 2].diff(3) if macd is not None else 0
    else:
        df['macd_histogram_slope'] = 0
        
    if 'bbb_20_2.0_2.0' in df.columns:
        df['bb_width'] = df['bbb_20_2.0_2.0']
    else:
        df['bb_width'] = 0
        
    # 6.8 Breakout Features
    df['is_breakout_20d'] = (df_ta['close'] > df_ta['high'].shift(1).rolling(20).max().fillna(999999)).astype(int)
    df['is_breakout_50d'] = (df_ta['close'] > df_ta['high'].shift(1).rolling(50).max().fillna(999999)).astype(int)

    df['sharpe_ratio_rolling'] = (df['daily_return'].rolling(window=20).mean() / 
                                  df['daily_return'].rolling(window=20).std().replace(0, 0.001)) * np.sqrt(252)
    
    tasi_col = 'Macro_TASI_PROXY'
    if tasi_col in df.columns:
        df['market_return'] = df[tasi_col].pct_change()
        df['relative_return_1d'] = df['daily_return'] - df['market_return']
        df['relative_return_5d'] = df['return_5d'] - df[tasi_col].pct_change(5)
        covariance = df['daily_return'].rolling(60).cov(df['market_return'])
        variance = df['market_return'].rolling(60).var().replace(0, 0.0001)
        df['beta'] = covariance / variance
    
    oil_col = 'Macro_BZ_F'
    if oil_col in df.columns:
        df['oil_return'] = df[oil_col].pct_change()
        df['oil_correlation'] = df['daily_return'].rolling(60).corr(df['oil_return'])

    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'])
        df['day_of_week'] = dates.dt.dayofweek
        df['month'] = dates.dt.month

    # 7. الأهداف المستقبلية
    df['target_next_day_return'] = df['daily_return'].shift(-1)
    df['target_next_week_return'] = df_ta['close'].pct_change(5).shift(-5)
    
    df.columns = [c.lower() for c in df.columns]
    rename_dict = {
        'macd_12_26_9': 'macd',
        'macds_12_26_9': 'macd_signal',
        'macdh_12_26_9': 'macd_hist'
    }
    df = df.rename(columns=rename_dict)
    
    target_cols = ['target_next_day_return', 'target_next_week_return', 'target_outperformer_1d', 'target_alpha_weekly', 'target_daily_bin', 'target_weekly_bin']
    feature_cols = [c for c in df.columns if c not in target_cols]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)
    return df


def preprocess_all_data(combined_file_path, output_file_path):
    if not os.path.exists(combined_file_path):
        print(f"خطأ: {combined_file_path} غير موجود.")
        return
    
    df = pd.read_csv(combined_file_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    before_symbols = df['Symbol'].nunique()
    df = df[df['Symbol'].apply(is_main_market_symbol)].copy()
    if 'Market' in df.columns:
        df = df[df['Market'].fillna("السوق الرئيسي") == "السوق الرئيسي"].copy()
    after_symbols = df['Symbol'].nunique()
    print(f"نطاق التحليل: السوق السعودي الرئيسي فقط ({after_symbols} من أصل {before_symbols} رمز).")
    df = df.sort_values(['Symbol', 'Date'])
    
    processed_dfs = []
    for symbol, group in df.groupby('Symbol'):
        processed_group = calculate_advanced_metrics(group.copy())
        if 'sector' not in processed_group.columns:
            processed_group['sector'] = group['Sector'].iloc[0] if 'Sector' in group.columns else "عام"
        if 'company name arabic' not in processed_group.columns:
            fallback_name = group['Company Name'].iloc[0] if 'Company Name' in group.columns else symbol
            processed_group['company name arabic'] = get_arabic_company_name(symbol, fallback_name)
        if 'market' not in processed_group.columns:
            processed_group['market'] = group['Market'].iloc[0] if 'Market' in group.columns else "السوق الرئيسي"
        if 'sentiment' not in processed_group.columns:
            processed_group['sentiment'] = 0.5
        processed_dfs.append(processed_group)
    
    final_df = pd.concat(processed_dfs)
    final_df = final_df.sort_values(['date', 'symbol'])
    
    print("حساب أهداف الـ Alpha والمقاييس العابرة للشركات...")
    
    # ميزات نسبية
    market_median_rsi = final_df.groupby('date')['rsi'].transform('median')
    final_df['rsi_relative'] = final_df['rsi'] - market_median_rsi

    # 1. هدف التفوق اليومي
    market_median_1d = final_df.groupby('date')['target_next_day_return'].transform('median')
    final_df['target_outperformer_1d'] = ((final_df['target_next_day_return'] > market_median_1d) & (final_df['target_next_day_return'] > 0)).astype(int)
    
    # 2. هدف الـ Alpha الأسبوعي
    def get_v6_winners(x):
        threshold = x.quantile(0.75)
        return ((x > threshold) & (x > 0.01)).astype(int)
    
    final_df['target_alpha_weekly'] = final_df.groupby('date')['target_next_week_return'].transform(get_v6_winners)
    
    # 3. هدف المدى المتوسط (أسبوعين)
    final_df['target_return_10d'] = final_df.groupby('symbol')['daily_return'].transform(lambda x: x.shift(-10).rolling(10).sum()) # تقريبي للعائد بعد أسبوعين
    final_df['target_alpha_medium'] = final_df.groupby('date')['target_return_10d'].transform(get_v6_winners)

    if {'date', 'close', 'sma_20', 'sma_50'}.issubset(final_df.columns):
        breadth = pd.DataFrame({
            'market_breadth_sma20': final_df.assign(above=final_df['close'] > final_df['sma_20'].replace(0, 999999)).groupby('date')['above'].mean(),
            'market_breadth_sma50': final_df.assign(above=final_df['close'] > final_df['sma_50'].replace(0, 999999)).groupby('date')['above'].mean(),
        })
        final_df = final_df.merge(breadth, left_on='date', right_index=True, how='left')
    
    if {'date', 'sector', 'daily_return', 'market_return'}.issubset(final_df.columns):
        sector_stats = final_df.groupby(['date', 'sector'])['daily_return'].mean().reset_index()
        sector_stats = sector_stats.sort_values(['sector', 'date'])
        sector_stats['sector_return_1d'] = sector_stats['daily_return']
        sector_stats['sector_momentum_20d'] = sector_stats.groupby('sector')['daily_return'].transform(lambda x: x.rolling(20).mean())
        
        final_df = final_df.merge(sector_stats[['date', 'sector', 'sector_return_1d', 'sector_momentum_20d']], on=['date', 'sector'], how='left')
        final_df['sector_relative_return_1d'] = final_df['sector_return_1d'] - final_df['market_return']
        final_df['relative_sector_alpha'] = final_df['daily_return'] - final_df['sector_return_1d']
        
        cols_to_fill = ['sector_return_1d', 'sector_momentum_20d', 'sector_relative_return_1d', 'relative_sector_alpha']
        final_df[cols_to_fill] = final_df[cols_to_fill].fillna(0)
        
    final_df.to_csv(output_file_path, index=False)
    print(f"تم حفظ البيانات المعالجة الشاملة في {output_file_path}")

if __name__ == "__main__":
    preprocess_all_data("data/tasi_combined.csv", "data/tasi_processed.csv")
