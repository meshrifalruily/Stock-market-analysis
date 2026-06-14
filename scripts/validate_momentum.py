"""التحقق من استراتيجية الزخم طويلة الأفق (long-only، صافي التكاليف، عبر السنوات).

كشف بحث IC على 8 سنوات أن أقوى إشارة في تاسي هي الزخم طويل الأفق (price_vs_sma_200
على 60 يوماً، t=+27). هنا نتحقق من النسخة القابلة للتطبيق فعلياً:
  - long-only (شراء الأقوى زخماً) — بلا بيع مكشوف.
  - أفق 20 و60 يوماً (تدوير منخفض → التكاليف شبه مهملة، عكس الانعكاس اليومي).
  - في الأسهم السائلة فقط، نوافذ غير متقاطعة، صافي عمولة + انزلاق.
  - مقسّماً حسب السنة (8 سنوات/أنظمة متعددة) ومقابل متوسط السوق (ألفا).

البوابة: إن تفوّق صافياً على السوق في غالبية السنوات → الإشارة قابلة للاستغلال
ونبنيها في خط الأنابيب. وإلا نكون صادقين كما في الانعكاس.
"""

import os
import sys
import numpy as np
import pandas as pd

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.strategy_rules import filter_main_market
from scripts.strategy_config import load_strategy_config

DATA_PATH = os.path.join(ROOT_DIR, "data/tasi_processed.csv")

# عوامل الزخم ذات الـ IC الموجب طويل الأفق (من نتائج بحث 8 سنوات).
MOMENTUM_FEATURES = [
    "price_vs_sma_200", "price_vs_sma_50", "momentum_20d", "rsi",
    "distance_from_high_20d", "tv_adx",
]
COST_PER_SIDE = 0.00155 + 0.0005  # ≈ 0.2% لكل جهة


def build_momentum_score(df):
    """درجة زخم مركّبة = متوسط الرتب المقطعية اليومية لعوامل الزخم (موجبة)."""
    feats = [f for f in MOMENTUM_FEATURES if f in df.columns]
    df = df.copy()
    score = pd.Series(0.0, index=df.index)
    for f in feats:
        score = score + df.groupby("date")[f].rank(pct=True)  # الأعلى = الأقوى زخماً
    df["momentum_score"] = score / max(1, len(feats))
    return df


def momentum_backtest(df, horizon, top_pct, config, use_regime=False):
    """long-only: كل horizon يوماً نشتري أعلى درجة زخم بين السائلين، صافي التكاليف.

    نقيس مقابل متوسط السوق (equal-weight) على نفس النوافذ غير المتقاطعة.
    """
    fwd_col = f"fwd_ret_{horizon}"
    base = df.dropna(subset=["momentum_score", fwd_col]).copy()
    base = base[base["avg_traded_value_20d"].fillna(0) >= config["min_avg_traded_value"]]
    if use_regime and "regime_ok" in base.columns:
        pool = base[base["regime_ok"] > 0.5]
    else:
        pool = base

    all_dates = np.array(sorted(base["date"].unique()))
    entry_dates = all_dates[::horizon]

    strat_rets, mkt_rets, per_year = [], [], {}
    for d in entry_dates:
        day_pool = pool[pool["date"] == d]
        day_all = base[base["date"] == d]
        if len(day_pool) < 10 or len(day_all) < 20:
            continue
        k = max(1, int(len(day_pool) * top_pct))
        picks = day_pool.sort_values("momentum_score", ascending=False).head(k)
        strat_net = picks[fwd_col].mean() - 2 * COST_PER_SIDE
        mkt = day_all[fwd_col].mean()
        strat_rets.append(strat_net)
        mkt_rets.append(mkt)
        y = pd.Timestamp(d).year
        per_year.setdefault(y, {"s": [], "m": []})
        per_year[y]["s"].append(strat_net)
        per_year[y]["m"].append(mkt)

    if len(strat_rets) < 8:
        return None
    s, m = pd.Series(strat_rets), pd.Series(mkt_rets)
    alpha = s - m
    ann = np.sqrt(252 / horizon)
    year_stats = {}
    for y, v in sorted(per_year.items()):
        sv, mv = pd.Series(v["s"]), pd.Series(v["m"])
        year_stats[int(y)] = {
            "strat": round(float((1 + sv).prod() - 1), 4),
            "market": round(float((1 + mv).prod() - 1), 4),
            "alpha": round(float((1 + sv).prod() - (1 + mv).prod()), 4),
            "periods": len(sv),
        }
    return {
        "horizon": horizon, "top_pct": top_pct, "use_regime": use_regime, "n_periods": len(s),
        "strat_total": round(float((1 + s).prod() - 1), 4),
        "market_total": round(float((1 + m).prod() - 1), 4),
        "strat_sharpe": round(float((s.mean() / s.std()) * ann), 2) if s.std() > 0 else 0.0,
        "alpha_sharpe": round(float((alpha.mean() / alpha.std()) * ann), 2) if alpha.std() > 0 else 0.0,
        "win_vs_market": round(float((alpha > 0).mean()), 3),
        "per_year": year_stats,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = filter_main_market(df)
    df = df.sort_values(["symbol", "date"])
    grp = df.groupby("symbol")["close"]
    for h in [20, 60]:
        df[f"fwd_ret_{h}"] = grp.shift(-h) / df["close"] - 1.0
    df = build_momentum_score(df)
    config = load_strategy_config()

    print("=" * 80)
    print("التحقق من استراتيجية الزخم طويلة الأفق (long-only، صافي، مقابل السوق)")
    print(f"السائلون فقط | عوامل الزخم: {len(MOMENTUM_FEATURES)} | "
          f"الفترة: {df['date'].min().date()} → {df['date'].max().date()}")
    print("=" * 80)

    for horizon in [20, 60]:
        for top_pct in [0.1, 0.2]:
            for use_regime in [False, True]:
                res = momentum_backtest(df, horizon, top_pct, config, use_regime)
                if not res:
                    continue
                reg = "مع فلتر نظام" if use_regime else "بلا فلتر   "
                print(f"\n▶ أفق {horizon}ي | أعلى {int(top_pct*100)}% | {reg} | {res['n_periods']} فترة")
                print(f"  استراتيجية={res['strat_total']*100:>7.1f}% | سوق={res['market_total']*100:>7.1f}% | "
                      f"شارب={res['strat_sharpe']:>5.2f} | ألفا-شارب={res['alpha_sharpe']:>5.2f} | "
                      f"تفوّق {res['win_vs_market']*100:.0f}%")
                pos = sum(1 for s in res["per_year"].values() if s["alpha"] > 0)
                tot = len(res["per_year"])
                print(f"  ألفا موجب في {pos}/{tot} سنة:", end=" ")
                print("  ".join(f"{y}:{s['alpha']*100:+.0f}%" for y, s in res["per_year"].items()))


if __name__ == "__main__":
    main()
