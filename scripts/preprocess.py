import pandas as pd
import numpy as np
import pandas_ta as ta
import os

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
    df['weekly_return_hist'] = df_ta['close'].pct_change(5) # العائد في الـ 5 أيام الماضية
    df['volatility_10d'] = df['daily_return'].rolling(10).std()
    df['volatility_20d'] = df['daily_return'].rolling(20).std()
    df['traded_value'] = df_ta['close'] * df_ta['volume']
    df['avg_traded_value_20d'] = df['traded_value'].rolling(20).mean()
    df['volume_ratio_20d'] = df_ta['volume'] / df_ta['volume'].rolling(20).mean()
    df['atr_pct'] = df['atr'] / df_ta['close']
    df['price_vs_sma_20'] = (df_ta['close'] / df['sma_20']) - 1
    df['price_vs_sma_50'] = (df_ta['close'] / df['sma_50']) - 1
    df['price_vs_sma_200'] = (df_ta['close'] / df['sma_200']) - 1
    
    df['sharpe_ratio_rolling'] = (df['daily_return'].rolling(window=20).mean() / 
                                  df['daily_return'].rolling(window=20).std()) * np.sqrt(252)
    
    tasi_col = 'Macro_TASI_PROXY'
    if tasi_col in df.columns:
        df['market_return'] = df[tasi_col].pct_change()
        df['relative_return_1d'] = df['daily_return'] - df['market_return']
        df['relative_return_5d'] = df['return_5d'] - df[tasi_col].pct_change(5)
        covariance = df['daily_return'].rolling(60).cov(df['market_return'])
        variance = df['market_return'].rolling(60).var()
        df['beta'] = covariance / variance
    
    oil_col = 'Macro_BZ_F'
    if oil_col in df.columns:
        df['oil_return'] = df[oil_col].pct_change()
        df['oil_correlation'] = df['daily_return'].rolling(60).corr(df['oil_return'])

    if 'Date' in df.columns:
        dates = pd.to_datetime(df['Date'])
        df['day_of_week'] = dates.dt.dayofweek
        df['month'] = dates.dt.month

    # 7. الأهداف المستقبلية (Targets)
    df['target_next_day_return'] = df['daily_return'].shift(-1)
    df['target_next_week_return'] = df_ta['close'].pct_change(5).shift(-5) # العائد المتوقع بعد أسبوع
    # تنظيف وتوحيد الأسماء
    df.columns = [c.lower() for c in df.columns]
    rename_dict = {
        'macd_12_26_9': 'macd',
        'macds_12_26_9': 'macd_signal',
        'macdh_12_26_9': 'macd_hist'
    }
    df = df.rename(columns=rename_dict)

    # لا نملأ الأهداف المستقبلية حتى لا يتسرب المستقبل إلى التدريب أو الاختبار.
    target_cols = ['target_next_day_return', 'target_next_week_return']
    feature_cols = [c for c in df.columns if c not in target_cols]
    df[feature_cols] = df[feature_cols].replace([np.inf, -np.inf], np.nan).ffill().fillna(0)

    # الاحتفاظ بالأعمدة النصية الهامة
    return df


def preprocess_all_data(combined_file_path, output_file_path):
    if not os.path.exists(combined_file_path):
        print(f"خطأ: {combined_file_path} غير موجود.")
        return
    
    df = pd.read_csv(combined_file_path)
    df['Date'] = pd.to_datetime(df['Date']).dt.tz_localize(None)
    df = df.sort_values(['Symbol', 'Date'])
    
    processed_dfs = []
    for symbol, group in df.groupby('Symbol'):
        print(f"جاري معالجة البيانات المتقدمة لـ {symbol}...")
        processed_group = calculate_advanced_metrics(group.copy())
        
        # الحفاظ على الأعمدة الوصفية
        if 'sector' not in processed_group.columns:
            processed_group['sector'] = group['Sector'].iloc[0] if 'Sector' in group.columns else "عام"
        if 'sentiment' not in processed_group.columns:
            processed_group['sentiment'] = 0.5
        
        processed_dfs.append(processed_group)
    
    final_df = pd.concat(processed_dfs)
    final_df = final_df.sort_values(['date', 'symbol'])
    if {'date', 'close', 'sma_20', 'sma_50'}.issubset(final_df.columns):
        breadth = pd.DataFrame({
            'market_breadth_sma20': final_df.assign(above=final_df['close'] > final_df['sma_20']).groupby('date')['above'].mean(),
            'market_breadth_sma50': final_df.assign(above=final_df['close'] > final_df['sma_50']).groupby('date')['above'].mean(),
        })
        final_df = final_df.merge(breadth, left_on='date', right_index=True, how='left')
    if {'date', 'sector', 'daily_return', 'market_return'}.issubset(final_df.columns):
        sector_return = final_df.groupby(['date', 'sector'])['daily_return'].mean().rename('sector_return_1d')
        final_df = final_df.merge(sector_return, left_on=['date', 'sector'], right_index=True, how='left')
        final_df['sector_relative_return_1d'] = final_df['sector_return_1d'] - final_df['market_return']
        final_df[['sector_return_1d', 'sector_relative_return_1d']] = (
            final_df[['sector_return_1d', 'sector_relative_return_1d']].fillna(0)
        )
    final_df.to_csv(output_file_path, index=False)
    print(f"تم حفظ البيانات المعالجة الشاملة في {output_file_path}")

if __name__ == "__main__":
    preprocess_all_data("data/tasi_combined.csv", "data/tasi_processed.csv")
