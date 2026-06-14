import numpy as np
import pandas as pd

NOMU_PREFIXES = ("95", "96")

def is_main_market_symbol(symbol):
    code = str(symbol).replace(".SR", "").strip()
    return not code.startswith(NOMU_PREFIXES)

def filter_main_market(df, symbol_col="symbol"):
    filtered = df[df[symbol_col].apply(is_main_market_symbol)].copy()
    market_col = "market" if "market" in filtered.columns else "Market" if "Market" in filtered.columns else None
    if market_col:
        filtered = filtered[filtered[market_col].fillna("السوق الرئيسي") == "السوق الرئيسي"].copy()
    return filtered

def add_hybrid_scores(day_data):
    """ترتيب الفرص وفق منطق "شراء الانخفاض في اتجاه صاعد".

    اكتشف بحث IC للعوامل أن سوق تاسي قصير المدى انعكاسي (mean-reversion)، لذا:
      - التوقيت يعتمد على درجة الانعكاس (الأكثر تشبّعاً بيعياً = أولوية للارتداد).
      - الاتجاه المتوسط (price_vs_sma_50 الموجب) عامل مساعد لا مطارد للقمم.
    أُزيلت مكوّنات الزخم القديمة (return_5d، prob_win) لأنها ذات IC سالب.
    """
    day_data = day_data.copy()
    # توقيت الدخول الانعكاسي (الإشارة الأقوى).
    if "reversal_score" in day_data.columns:
        reversal_timing = day_data["reversal_score"].fillna(0.5)
    else:
        # رجوع آمن إن لم تُحسب الدرجة: ننفي عائد 5 أيام (المتراجع مرشّح للارتداد).
        reversal_timing = 0.5 - day_data.get("return_5d", pd.Series(0.0, index=day_data.index)).fillna(0)
    medium_trend = day_data.get("price_vs_sma_50", pd.Series(0.0, index=day_data.index)).fillna(0).clip(lower=0)
    risk_penalty = day_data.get("volatility_20d", pd.Series(0.0, index=day_data.index)).fillna(0).abs()
    sector_alpha = day_data.get("sector_relative_return_1d", pd.Series(0.0, index=day_data.index)).fillna(0)

    day_data["hybrid_score"] = (
        1.0 * reversal_timing
        + 0.5 * medium_trend
        + 0.3 * sector_alpha
        - 0.5 * risk_penalty
    )
    day_data["rank"] = day_data["hybrid_score"].rank(ascending=False, method="first")
    return day_data

def entry_candidates(day_data, current_symbols, config):
    """ترشيح الدخول وفق "شراء الانخفاض في اتجاه صاعد" (مبني على بحث IC).

    منطق مُصحَّح: نملك السهم في اتجاه متوسط صاعد (close>sma_50, momentum_20d>0)
    لكن ندخل عند تراجع قصير المدى (أعلى نصف تشبّعاً بيعياً) بدل مطاردة القمم.
    أُزيلت بوابة prob_win الصارمة لأن نماذج الزخم ذات IC سالب، ولأن المعايرة
    ضغطت الاحتمالات دون العتبة القديمة فكانت تُفرّغ الترشيح بالكامل.
    """
    idx = day_data.index
    next_return = day_data.get("predicted_next_return", pd.Series(0.0, index=idx))
    sma50 = day_data.get("sma_50", day_data["close"])
    mask = (
        (~day_data["symbol"].isin(current_symbols)) &
        (day_data["rank"] <= config["max_entry_rank"]) &
        (day_data["close"] > sma50) &                                   # اتجاه متوسط صاعد
        (day_data["return_5d"] <= config["max_entry_return_5d"]) &      # لا نطارد القمم
        (next_return >= config.get("min_predicted_next_return", 0.0)) &
        (day_data["rsi"] <= config["max_entry_rsi"])                    # نتجنّب التشبّع الشرائي فقط
    )
    # توقيت انعكاسي: نشترط أن يكون السهم في النصف الأكثر تشبّعاً بيعياً اليوم.
    if "reversal_score" in day_data.columns:
        mask = mask & (day_data["reversal_score"] >= day_data["reversal_score"].median())
    # زخم متوسط المدى موجب (يُملك الرابح، يُشترى عند انخفاضه).
    if "momentum_20d" in day_data.columns:
        mask = mask & (day_data["momentum_20d"].fillna(0) > 0)
    # فلتر نظام السوق: لا ندخل صفقات انعكاسية في هبوط قوي (سكين هابط).
    if "regime_ok" in day_data.columns:
        mask = mask & (day_data["regime_ok"].fillna(1.0) > 0.5)
    return day_data[mask].sort_values("rank")

def entry_diagnostics(row, config, market_breadth=None):
    """أسباب رفض الدخول وفق منطق "شراء الانخفاض في اتجاه صاعد"."""
    reasons = []
    if market_breadth is not None and market_breadth < config["min_entry_market_breadth"]:
        reasons.append("صحة السوق أقل من شرط الدخول")
    if row.get("avg_traded_value_20d", 0) < config["min_avg_traded_value"]:
        reasons.append("السيولة أقل من الحد المطلوب")
    diagnostic_rank = row.get("entry_rank", row.get("rank", 999))
    if diagnostic_rank > config["max_entry_rank"]:
        reasons.append("ترتيب الفرصة الانعكاسية خارج الأفضل")
    if row.get("close", 0) <= row.get("sma_50", 0):
        reasons.append("ليس في اتجاه متوسط صاعد (دون متوسط 50)")
    if row.get("momentum_20d", 0) <= 0:
        reasons.append("زخم 20 يوم غير موجب")
    if row.get("return_5d", 0) > config["max_entry_return_5d"]:
        reasons.append("ارتفع كثيراً مؤخراً (مطاردة قمة)")
    if row.get("rsi", 50) > config["max_entry_rsi"]:
        reasons.append("تشبّع شرائي (RSI مرتفع)")
    if row.get("predicted_next_return", 0) < config.get("min_predicted_next_return", 0.0):
        reasons.append("توقع الغد سلبي")
    return "، ".join(reasons[:3]) if reasons else "مستوفي لشروط الدخول (انخفاض في صعود)"

def summarize_returns(returns):
    returns = pd.Series(returns).replace([np.inf, -np.inf], np.nan).fillna(0)
    total_return = float((1 + returns).prod() - 1)
    volatility = float(returns.std())
    sharpe = float((returns.mean() / volatility) * np.sqrt(252)) if volatility > 0 else 0.0
    equity = (1 + returns).cumprod()
    drawdown = (equity / equity.cummax()) - 1
    max_drawdown = float(drawdown.min()) if not drawdown.empty else 0.0
    active_days = int((returns != 0).sum())
    win_rate = float((returns[returns != 0] > 0).mean()) if active_days else 0.0
    return {
        "total_return": total_return,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "active_days": active_days,
        "win_rate": win_rate,
    }
