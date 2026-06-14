"""استراتيجية الزخم طويلة الأفق — الحافة الرابحة المتحقّق منها في تاسي.

تحقّق بحث IC (8 سنوات) واختبار long-only صافي التكاليف أن الزخم طويل الأفق
(60 يوماً) يتفوّق على السوق في 8 من 9 سنوات (+168% مقابل +29%). هذا الملف
يوفّر منطق الاستراتيجية القابل لإعادة الاستخدام في الواجهة والتقارير:

  - build_momentum_score: درجة زخم مركّبة من رتب عوامل IC الموجبة.
  - get_momentum_picks: أقوى الأسهم زخماً حالياً (سائلة + نظام سوق مؤاتٍ).
  - momentum_backtest + save_validation_report: تحقّق long-only صافي عبر السنوات.

الأفق المقصود ~60 يوم تداول (≈3 أشهر) بإعادة موازنة شهرية، فالتكاليف مهملة.
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
from scripts.strategy_config import load_strategy_config

MOMENTUM_FEATURES = [
    "price_vs_sma_200", "price_vs_sma_50", "momentum_20d", "rsi",
    "distance_from_high_20d", "tv_adx",
]
COST_PER_SIDE = 0.00155 + 0.0005
MOMENTUM_HORIZON_DAYS = 60
REPORT_PATH = os.path.join(ROOT_DIR, "reports/momentum_validation.json")
BACKTEST_CSV_PATH = os.path.join(ROOT_DIR, "data/backtest_results.csv")


def build_momentum_score(df):
    """درجة زخم مركّبة [0,1] = متوسط الرتب المقطعية اليومية لعوامل الزخم."""
    feats = [f for f in MOMENTUM_FEATURES if f in df.columns]
    if not feats:
        return df
    df = df.copy()
    score = pd.Series(0.0, index=df.index)
    for f in feats:
        score = score + df.groupby("date")[f].rank(pct=True)
    df["momentum_score"] = (score / len(feats)).fillna(0.5)
    return df


def get_momentum_picks(df, config, prediction_date, top_n=15, use_regime=True):
    """أقوى الأسهم زخماً في تاريخ معيّن (سائلة + نظام سوق مؤاتٍ).

    يعتمد عمود momentum_score المحسوب في المعالجة، أو يحسبه إن غاب.
    يُرجع قائمة قواميس جاهزة للعرض (الرمز، الاسم، السعر، الدرجة، الوقف، الأفق).
    """
    if "momentum_score" not in df.columns:
        df = build_momentum_score(df)

    prediction_date = pd.Timestamp(prediction_date)
    rows = df[df["date"] == prediction_date].copy()
    if rows.empty:
        return []

    rows = rows[rows["avg_traded_value_20d"].fillna(0) >= config["min_avg_traded_value"]]
    if use_regime and "regime_ok" in rows.columns:
        rows = rows[rows["regime_ok"].fillna(1.0) > 0.5]
    if rows.empty:
        return []

    rows = rows.sort_values("momentum_score", ascending=False).head(top_n)
    picks = []
    for _, r in rows.iterrows():
        close = float(r["close"]) if not pd.isna(r["close"]) else 0.0
        atr = float(r["atr"]) if "atr" in rows.columns and not pd.isna(r["atr"]) else close * 0.02
        # وقف زخمي واسع (3 ATR) أو متوسط 50 إن كان أقرب — مناسب للحمل الطويل.
        sma50 = float(r["sma_50"]) if "sma_50" in rows.columns and not pd.isna(r["sma_50"]) else close * 0.9
        stop = max(0.01, min(close - 3 * atr, sma50)) if close > 0 else 0.0
        picks.append({
            "symbol": r["symbol"],
            "company_name": r.get("اسم الشركة", r["symbol"]),
            "sector": r.get("sector", "عام"),
            "current_price": round(close, 2),
            "momentum_score": round(float(r["momentum_score"]) * 100, 1),
            "price_vs_sma_200": round(float(r.get("price_vs_sma_200", 0)) * 100, 1),
            "momentum_20d": round(float(r.get("momentum_20d", 0)) * 100, 1),
            "rsi": round(float(r.get("rsi", 50)), 1),
            "entry": round(close, 2),
            "stop": round(stop, 2),
            "horizon_days": MOMENTUM_HORIZON_DAYS,
            "prediction_date": prediction_date.strftime("%Y-%m-%d"),
        })
    return picks


def momentum_backtest(df, horizon, top_pct, config, use_regime=True):
    """long-only صافي التكاليف، نوافذ غير متقاطعة، ألفا مقابل السوق، حسب السنة."""
    fwd_col = f"fwd_ret_{horizon}"
    if fwd_col not in df.columns:
        df = df.sort_values(["symbol", "date"]).copy()
        df[fwd_col] = df.groupby("symbol")["close"].shift(-horizon) / df["close"] - 1.0
    base = df.dropna(subset=["momentum_score", fwd_col]).copy()
    base = base[base["avg_traded_value_20d"].fillna(0) >= config["min_avg_traded_value"]]
    pool = base[base["regime_ok"] > 0.5] if (use_regime and "regime_ok" in base.columns) else base

    dates = np.array(sorted(base["date"].unique()))
    strat, mkt, per_year = [], [], {}
    for d in dates[::horizon]:
        dp, da = pool[pool["date"] == d], base[base["date"] == d]
        if len(dp) < 10 or len(da) < 20:
            continue
        k = max(1, int(len(dp) * top_pct))
        s = dp.sort_values("momentum_score", ascending=False).head(k)[fwd_col].mean() - 2 * COST_PER_SIDE
        m = da[fwd_col].mean()
        strat.append(s); mkt.append(m)
        per_year.setdefault(pd.Timestamp(d).year, {"s": [], "m": []})
        per_year[pd.Timestamp(d).year]["s"].append(s)
        per_year[pd.Timestamp(d).year]["m"].append(m)
    if len(strat) < 8:
        return None
    s, m = pd.Series(strat), pd.Series(mkt)
    alpha = s - m
    ann = np.sqrt(252 / horizon)
    years = {int(y): {
        "strat": round(float((1 + pd.Series(v["s"])).prod() - 1), 4),
        "market": round(float((1 + pd.Series(v["m"])).prod() - 1), 4),
    } for y, v in sorted(per_year.items())}
    return {
        "horizon": horizon, "top_pct": top_pct, "use_regime": use_regime,
        "strat_total": round(float((1 + s).prod() - 1), 4),
        "market_total": round(float((1 + m).prod() - 1), 4),
        "strat_sharpe": round(float((s.mean() / s.std()) * ann), 2) if s.std() > 0 else 0.0,
        "alpha_sharpe": round(float((alpha.mean() / alpha.std()) * ann), 2) if alpha.std() > 0 else 0.0,
        "win_vs_market": round(float((alpha > 0).mean()), 3),
        "positive_alpha_years": sum(1 for y in years.values() if y["strat"] > y["market"]),
        "total_years": len(years),
        "per_year": years,
    }


def write_equity_curve(df, config, horizon=MOMENTUM_HORIZON_DAYS, top_pct=0.10, use_regime=True):
    """يبني منحنى أسهم لاستراتيجية الزخم ويكتب data/backtest_results.csv.

    محاكاة واقعية: كل ``horizon`` يوماً (نوافذ غير متقاطعة) نشتري أعلى ``top_pct``
    زخماً بين السائلين في نظام مؤاتٍ، بوزن متساوٍ، ونحتفظ حتى نهاية الأفق، صافي
    التكاليف. القيمة تبدأ 100 وتتراكم. هذا هو الباك تست الرئيسي المعروض في
    الواجهة لأنه يقيس الاستراتيجية الموصى بها (الزخم) لا الانعكاس.
    """
    if "momentum_score" not in df.columns:
        df = build_momentum_score(df)
    df = df.sort_values(["symbol", "date"]).copy()
    fwd_col = f"fwd_ret_{horizon}"
    df[fwd_col] = df.groupby("symbol")["close"].shift(-horizon) / df["close"] - 1.0

    base = df.dropna(subset=["momentum_score", fwd_col])
    base = base[base["avg_traded_value_20d"].fillna(0) >= config["min_avg_traded_value"]]
    pool = base[base["regime_ok"] > 0.5] if (use_regime and "regime_ok" in base.columns) else base

    dates = np.array(sorted(base["date"].unique()))
    value = 100.0
    rows = []
    for d in dates[::horizon]:
        dp = pool[pool["date"] == d]
        if len(dp) < 10:
            rows.append({"date": pd.Timestamp(d), "returns": 0.0, "value": value, "selected_symbols": ""})
            continue
        k = max(1, int(len(dp) * top_pct))
        picks = dp.sort_values("momentum_score", ascending=False).head(k)
        net_ret = float(picks[fwd_col].mean()) - 2 * COST_PER_SIDE
        value *= (1 + net_ret)
        rows.append({
            "date": pd.Timestamp(d), "returns": round(net_ret, 6), "value": round(value, 4),
            "selected_symbols": ",".join(picks["symbol"].head(5).tolist()),
        })

    curve = pd.DataFrame(rows).set_index("date")
    curve.to_csv(BACKTEST_CSV_PATH)
    print(f"تم كتابة منحنى أسهم الزخم في {BACKTEST_CSV_PATH} | القيمة النهائية: {value:.2f} "
          f"(عائد {value-100:.1f}%)")
    return curve


def save_validation_report(data_path=None):
    data_path = data_path or os.path.join(ROOT_DIR, "data/tasi_processed.csv")
    df = pd.read_csv(data_path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = filter_main_market(df)
    df = build_momentum_score(df) if "momentum_score" not in df.columns else df
    config = load_strategy_config()
    result = momentum_backtest(df, MOMENTUM_HORIZON_DAYS, 0.10, config, use_regime=True)
    os.makedirs(os.path.dirname(REPORT_PATH), exist_ok=True)
    with open(REPORT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"تم حفظ تقرير تحقّق الزخم في {REPORT_PATH}")
    # كتابة منحنى الأسهم ليكون هو الباك تست الرئيسي المعروض في الواجهة.
    write_equity_curve(df, config)
    return result


if __name__ == "__main__":
    res = save_validation_report()
    if res:
        print(f"60ي/أعلى10%: استراتيجية {res['strat_total']*100:.0f}% مقابل سوق "
              f"{res['market_total']*100:.0f}% | Alpha-Sharpe {res['alpha_sharpe']} | "
              f"ألفا موجب {res['positive_alpha_years']}/{res['total_years']} سنة")
