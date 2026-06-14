"""بحث معامل المعلومات (IC) للعوامل — اكتشاف الإشارة التنبؤية الحقيقية.

المنهج (Factor/Alpha IC Research) هو ما يفعله باحثو الكم قبل بناء أي نموذج:
لكل عامل (feature) ولكل أفق زمني، نحسب ارتباط Spearman المقطعي (cross-sectional
rank IC) بين قيمة العامل اليوم والعائد المستقبلي. ثم نلخّص عبر الزمن:

  - mean_IC      : متوسط الإشارة (الإشارة + = زخم، − = انعكاس).
  - IC_IR        : Information Ratio = mean_IC / std_IC (ثبات الإشارة).
  - t_stat       : دلالة إحصائية = IC_IR * sqrt(عدد الفترات). |t|>2 ذو دلالة.
  - hit_ratio    : نسبة الأيام ذات IC بنفس الاتجاه السائد.

عامل بـ |IC_IR| كبير و|t|>2 يحمل إشارة حقيقية مستقرة. الإشارة السالبة قيّمة
بقدر الموجبة — تعني أن العامل تنبؤي لكن باتجاه معكوس (نعكس إشارته).

لا تسريب: العائد المستقبلي محسوب بـ shift سالب لكل رمز، والعامل كما هو في تاريخه.
"""

import os
import sys
import json
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.strategy_rules import filter_main_market

DATA_PATH = os.path.join(ROOT_DIR, "data/tasi_processed.csv")
REPORT_PATH = os.path.join(ROOT_DIR, "reports/signal_research.json")

HORIZONS = [1, 3, 5, 10, 20, 60]
MIN_STOCKS_PER_DAY = 20  # حد أدنى لعينة مقطعية ذات معنى

# العوامل المرشّحة — مزيج زخم/انعكاس/تقلب/حجم/microstructure/قيمة.
CANDIDATE_FEATURES = [
    # زخم/اتجاه
    "return_1d", "return_2d", "return_3d", "return_5d", "return_10d", "return_20d",
    "momentum_10d", "momentum_20d", "rsi", "rsi_slope", "macd_histogram_slope",
    "price_vs_sma_20", "price_vs_sma_50", "price_vs_sma_200", "tv_adx",
    "relative_return_1d", "relative_return_5d", "relative_sector_alpha",
    "is_breakout_20d", "distance_from_high_20d", "price_velocity",
    # تقلب
    "volatility_10d", "volatility_20d", "atr_pct",
    # حجم/سيولة
    "volume_ratio_20d", "volume_ratio_5d", "relative_volume", "vol_shock",
    "traded_value_zscore",
    # microstructure
    "close_location_value", "gap_open", "intraday_range_pct", "close_vs_open",
    "clv_mean_5d", "upper_shadow_pct", "lower_shadow_pct",
    # قيمة/أساسي
    "pe_ratio", "div_yield", "beta", "oil_correlation",
    # مشاعر
    "sentiment",
]


def compute_forward_returns(df, horizons):
    """عائد مستقبلي لكل رمز عبر آفاق متعددة (بلا تسريب)."""
    df = df.sort_values(["symbol", "date"]).copy()
    grp = df.groupby("symbol")["close"]
    for h in horizons:
        df[f"fwd_ret_{h}"] = grp.shift(-h) / df["close"] - 1.0
    return df


def daily_rank_ic(df, feature, fwd_col, min_stocks=MIN_STOCKS_PER_DAY):
    """سلسلة الـ IC المقطعي اليومي (Spearman) لعامل مقابل عائد مستقبلي."""
    sub = df[["date", feature, fwd_col]].replace([np.inf, -np.inf], np.nan).dropna()
    if sub.empty:
        return pd.Series(dtype=float)

    def _ic(g):
        if len(g) < min_stocks or g[feature].nunique() < 5 or g[fwd_col].nunique() < 2:
            return np.nan
        return g[feature].corr(g[fwd_col], method="spearman")

    return sub.groupby("date", group_keys=False).apply(_ic).dropna()


def summarize_ic(ic_series):
    """تلخيص إحصائي لسلسلة IC يومية."""
    n = len(ic_series)
    if n < 20:
        return None
    mean_ic = float(ic_series.mean())
    std_ic = float(ic_series.std())
    ic_ir = float(mean_ic / std_ic) if std_ic > 0 else 0.0
    t_stat = float(ic_ir * np.sqrt(n))
    # نسبة الأيام بنفس اتجاه الإشارة السائدة
    direction = np.sign(mean_ic) if mean_ic != 0 else 1
    hit_ratio = float((np.sign(ic_series) == direction).mean())
    return {
        "mean_ic": round(mean_ic, 5),
        "ic_ir": round(ic_ir, 4),
        "t_stat": round(t_stat, 3),
        "hit_ratio": round(hit_ratio, 3),
        "n_periods": int(n),
    }


def long_short_decile_return(df, feature, fwd_col, sign, decile=0.1, min_stocks=MIN_STOCKS_PER_DAY):
    """عائد محفظة long-short عشيرية يومياً على عامل واحد.

    sign=+1: long أعلى العامل / short أدناه. sign=-1: معكوس (للإشارة الانعكاسية).
    نُرجع متوسط العائد اليومي للمحفظة و شارب سنوي تقريبي.
    """
    sub = df[["date", feature, fwd_col]].replace([np.inf, -np.inf], np.nan).dropna()
    rets = []
    for _, g in sub.groupby("date"):
        if len(g) < min_stocks:
            continue
        k = max(1, int(len(g) * decile))
        ranked = g.sort_values(feature, ascending=False)
        top = ranked.head(k)[fwd_col].mean()
        bottom = ranked.tail(k)[fwd_col].mean()
        rets.append(sign * (top - bottom))
    if len(rets) < 20:
        return None
    rets = pd.Series(rets)
    # نطبّع العائد لكل يوم (آفاق متداخلة، فهذا تقريبي للمقارنة فقط)
    mean_r = float(rets.mean())
    sharpe = float((rets.mean() / rets.std()) * np.sqrt(252)) if rets.std() > 0 else 0.0
    return {"ls_mean_per_period": round(mean_r, 5), "ls_sharpe_approx": round(sharpe, 2), "n": len(rets)}


def run_research(data_path=DATA_PATH):
    if not os.path.exists(data_path):
        print(f"خطأ: {data_path} غير موجود.")
        return

    print("تحميل البيانات...")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = filter_main_market(df)
    df = compute_forward_returns(df, HORIZONS)
    print(f"البيانات: {len(df):,} صف، {df['symbol'].nunique()} رمز، "
          f"{df['date'].nunique()} يوم تداول.")

    features = [f for f in CANDIDATE_FEATURES if f in df.columns]
    print(f"عوامل قيد الاختبار: {len(features)} | آفاق: {HORIZONS}\n")

    results = []
    for feat in features:
        for h in HORIZONS:
            fwd = f"fwd_ret_{h}"
            ic_series = daily_rank_ic(df, feat, fwd)
            summary = summarize_ic(ic_series)
            if summary is None:
                continue
            row = {"feature": feat, "horizon": h, **summary}
            results.append(row)

    res_df = pd.DataFrame(results)
    if res_df.empty:
        print("لا نتائج.")
        return

    # ترتيب حسب الدلالة الإحصائية المطلقة
    res_df["abs_t"] = res_df["t_stat"].abs()
    res_df = res_df.sort_values("abs_t", ascending=False).reset_index(drop=True)

    print("=" * 78)
    print("أقوى 20 إشارة (حسب |t-stat|):")
    print("=" * 78)
    print(f"{'العامل':<24}{'أفق':>5}{'mean_IC':>10}{'IC_IR':>8}{'t':>8}{'hit':>7}")
    print("-" * 78)
    for _, r in res_df.head(20).iterrows():
        direction = "زخم" if r["mean_ic"] > 0 else "انعكاس"
        print(f"{r['feature']:<24}{int(r['horizon']):>5}{r['mean_ic']:>10.4f}"
              f"{r['ic_ir']:>8.3f}{r['t_stat']:>8.2f}{r['hit_ratio']:>7.2f}  {direction}")

    # اختبار long-short لأقوى 5 إشارات ذات دلالة
    print("\n" + "=" * 78)
    print("اختبار محفظة Long-Short العشيرية لأقوى الإشارات ذات الدلالة (|t|>2):")
    print("=" * 78)
    ls_tests = []
    significant = res_df[res_df["abs_t"] > 2.0].head(8)
    for _, r in significant.iterrows():
        sign = 1 if r["mean_ic"] > 0 else -1
        ls = long_short_decile_return(df, r["feature"], f"fwd_ret_{int(r['horizon'])}", sign)
        if ls:
            tag = "زخم" if sign > 0 else "انعكاس(معكوس)"
            print(f"{r['feature']:<24} أفق {int(r['horizon']):>2}يوم [{tag:<14}] "
                  f"عائد/فترة={ls['ls_mean_per_period']:>8.4f}  شارب≈{ls['ls_sharpe_approx']:>6.2f}")
            ls_tests.append({**r.to_dict(), "sign": sign, **ls})

    # حفظ التقرير
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    report = {
        "n_features": len(features),
        "horizons": HORIZONS,
        "top_signals": res_df.head(30).to_dict(orient="records"),
        "long_short_tests": ls_tests,
    }
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"\nتم حفظ التقرير الكامل في {REPORT_PATH}")

    return res_df


if __name__ == "__main__":
    run_research()
