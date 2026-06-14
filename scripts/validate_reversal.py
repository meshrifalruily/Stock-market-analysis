"""التحقق من قابلية تطبيق إشارة الانعكاس (long-only، صافي التكاليف، عبر السنوات).

بعد أن كشف signal_research.py أن الإشارة انعكاسية، نتحقق هنا من النسخة القابلة
للتطبيق فعلياً في تاسي:
  - long-only (شراء المتشبّع بيعياً) — لأن البيع المكشوف صعب في السوق.
  - في الأسهم السائلة فقط.
  - بأفق 3 و5 أيام (تدوير أقل من اليوم الواحد لتحمّل العمولات).
  - صافي تكاليف واقعية (عمولة + انزلاق).
  - مقسّماً حسب السنة (walk-forward) للتأكد من ثبات الإشارة عبر الزمن.

نبني "درجة انعكاس" مركّبة = متوسط الرتب المقطعية لأقوى عوامل الانعكاس (منفية،
بحيث الأعلى = الأكثر تشبّعاً بيعياً = المرشّح للارتداد).
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

# أقوى عوامل الانعكاس من نتائج البحث (تُنفى لأن الإشارة سالبة).
REVERSAL_FEATURES = [
    "close_location_value", "lower_shadow_pct", "return_1d", "return_2d",
    "rsi_slope", "close_vs_open", "relative_return_1d",
]
COST_PER_SIDE = 0.00155 + 0.0005  # عمولة + انزلاق ≈ 0.2% لكل جهة


def build_reversal_score(df):
    """درجة انعكاس مركّبة من رتب مقطعية يومية لعوامل الانعكاس (منفية)."""
    feats = [f for f in REVERSAL_FEATURES if f in df.columns]
    df = df.copy()
    score = pd.Series(0.0, index=df.index)
    for f in feats:
        # رتبة مقطعية [0,1] داخل كل يوم؛ ننفيها (الأقل قيمة = الأكثر تشبّعاً = درجة أعلى).
        ranks = df.groupby("date")[f].rank(pct=True)
        score = score + (1.0 - ranks)
    df["reversal_score"] = score / max(1, len(feats))
    return df


def long_only_backtest(df, horizon, top_pct, config, hold_days=None):
    """باك تست long-only: كل يوم نشتري أعلى درجة انعكاس بين السائلين، نحتفظ horizon.

    نموذج مبسّط لا متداخل: ندخل كل ``hold_days`` (=horizon افتراضياً) ونحتفظ
    حتى نهاية الأفق، فيكون العائد صافياً بعد تكلفة دخول وخروج واحدة.
    """
    hold_days = hold_days or horizon
    fwd_col = f"fwd_ret_{horizon}"
    liquid = df[df["avg_traded_value_20d"] >= config["min_avg_traded_value"]].copy()
    liquid = liquid.dropna(subset=["reversal_score", fwd_col])

    all_dates = np.array(sorted(liquid["date"].unique()))
    # ندخل كل hold_days يوماً لتجنّب التداخل (نوافذ غير متقاطعة).
    entry_dates = all_dates[::hold_days]

    period_rets, per_year = [], {}
    for d in entry_dates:
        g = liquid[liquid["date"] == d]
        if len(g) < 20:
            continue
        k = max(1, int(len(g) * top_pct))
        picks = g.sort_values("reversal_score", ascending=False).head(k)
        gross = picks[fwd_col].mean()
        net = gross - 2 * COST_PER_SIDE  # دخول + خروج
        period_rets.append(net)
        per_year.setdefault(pd.Timestamp(d).year, []).append(net)

    if len(period_rets) < 10:
        return None

    rets = pd.Series(period_rets)
    periods_per_year = 252 / hold_days
    ann_factor = np.sqrt(periods_per_year)
    sharpe = float((rets.mean() / rets.std()) * ann_factor) if rets.std() > 0 else 0.0
    total = float((1 + rets).prod() - 1)
    win_rate = float((rets > 0).mean())

    year_stats = {}
    for y, rs in sorted(per_year.items()):
        rs = pd.Series(rs)
        year_stats[int(y)] = {
            "periods": len(rs),
            "mean_net": round(float(rs.mean()), 4),
            "total": round(float((1 + rs).prod() - 1), 4),
            "win_rate": round(float((rs > 0).mean()), 2),
        }

    return {
        "horizon": horizon,
        "top_pct": top_pct,
        "n_periods": len(rets),
        "mean_net_per_period": round(float(rets.mean()), 5),
        "sharpe_annual": round(sharpe, 2),
        "total_return": round(total, 4),
        "win_rate": round(win_rate, 3),
        "per_year": year_stats,
    }


def pullback_in_uptrend_backtest(df, horizon, top_pct, config):
    """الاستراتيجية المركّبة: زخم متوسط المدى + توقيت دخول انعكاسي قصير المدى.

    من بين السائلين في اتجاه صاعد (close>sma_50 و momentum_20d>0)، نشتري الأكثر
    تراجعاً قصير المدى (أعلى درجة انعكاس = شراء الانخفاض). نقيس الألفا مقابل
    متوسط السوق (equal-weight) على نفس النوافذ غير المتقاطعة، صافي التكاليف.
    """
    fwd_col = f"fwd_ret_{horizon}"
    base = df.dropna(subset=["reversal_score", fwd_col, "sma_50", "momentum_20d"]).copy()
    base = base[base["avg_traded_value_20d"] >= config["min_avg_traded_value"]]
    uptrend = base[(base["close"] > base["sma_50"]) & (base["momentum_20d"] > 0)]

    all_dates = np.array(sorted(base["date"].unique()))
    entry_dates = all_dates[::horizon]

    strat_rets, mkt_rets, per_year = [], [], {}
    for d in entry_dates:
        day_up = uptrend[uptrend["date"] == d]
        day_all = base[base["date"] == d]
        if len(day_up) < 5 or len(day_all) < 20:
            continue
        k = max(1, int(len(day_up) * top_pct))
        picks = day_up.sort_values("reversal_score", ascending=False).head(k)
        strat_net = picks[fwd_col].mean() - 2 * COST_PER_SIDE
        mkt = day_all[fwd_col].mean()  # متوسط السوق (لا تكلفة — مرجع)
        strat_rets.append(strat_net)
        mkt_rets.append(mkt)
        per_year.setdefault(pd.Timestamp(d).year, {"s": [], "m": []})
        per_year[pd.Timestamp(d).year]["s"].append(strat_net)
        per_year[pd.Timestamp(d).year]["m"].append(mkt)

    if len(strat_rets) < 10:
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
        "horizon": horizon, "top_pct": top_pct, "n_periods": len(s),
        "strat_total": round(float((1 + s).prod() - 1), 4),
        "market_total": round(float((1 + m).prod() - 1), 4),
        "alpha_mean_per_period": round(float(alpha.mean()), 5),
        "alpha_sharpe": round(float((alpha.mean() / alpha.std()) * ann), 2) if alpha.std() > 0 else 0.0,
        "strat_sharpe": round(float((s.mean() / s.std()) * ann), 2) if s.std() > 0 else 0.0,
        "win_vs_market": round(float((alpha > 0).mean()), 3),
        "per_year": year_stats,
    }


def main():
    df = pd.read_csv(DATA_PATH)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = filter_main_market(df)
    df = df.sort_values(["symbol", "date"])
    grp = df.groupby("symbol")["close"]
    for h in [3, 5, 10]:
        df[f"fwd_ret_{h}"] = grp.shift(-h) / df["close"] - 1.0
    df = build_reversal_score(df)
    config = load_strategy_config()

    print("=" * 78)
    print("[1] إشارة انعكاس صرفة (long-only، صافي التكاليف) — هل تربح مطلقاً؟")
    print("=" * 78)
    for horizon in [3, 5, 10]:
        res = long_only_backtest(df, horizon, 0.1, config)
        if res:
            print(f"  أفق {horizon:>2}ي | إجمالي صافٍ={res['total_return']*100:>6.1f}% | "
                  f"شارب={res['sharpe_annual']:>5.2f} | win={res['win_rate']*100:.0f}%")

    print("\n" + "=" * 78)
    print("[2] الاستراتيجية المركّبة: زخم متوسط + دخول انعكاسي (شراء الانخفاض في صعود)")
    print("    القياس: ألفا مقابل متوسط السوق، صافي التكاليف، غير متقاطع")
    print("=" * 78)
    for horizon in [5, 10]:
        for top_pct in [0.1, 0.2]:
            res = pullback_in_uptrend_backtest(df, horizon, top_pct, config)
            if not res:
                continue
            print(f"\n▶ أفق {horizon} أيام | أعلى {int(top_pct*100)}% | {res['n_periods']} فترة")
            print(f"  الاستراتيجية={res['strat_total']*100:>6.1f}% | السوق={res['market_total']*100:>6.1f}% | "
                  f"ألفا شارب={res['alpha_sharpe']:>5.2f} | تفوّق على السوق {res['win_vs_market']*100:.0f}% من الفترات")
            for y, s in res["per_year"].items():
                flag = "✓" if s["alpha"] > 0 else "✗"
                print(f"    {y}: استراتيجية={s['strat']*100:>6.1f}%  سوق={s['market']*100:>6.1f}%  "
                      f"ألفا={s['alpha']*100:>6.1f}% {flag}  (فترات={s['periods']})")


if __name__ == "__main__":
    main()
