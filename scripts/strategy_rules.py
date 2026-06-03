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
    day_data = day_data.copy()
    trend_score = (
        day_data.get("price_vs_sma_50", 0).fillna(0)
        + day_data.get("return_5d", 0).fillna(0)
        + (0.01 * day_data.get("tv_adx", 0).fillna(0))
    )
    risk_penalty = day_data.get("volatility_20d", 0).fillna(0).abs()
    sector_alpha = day_data.get("sector_relative_return_1d", 0).fillna(0)
    prob_component = day_data.get("prob_win", 0.5).fillna(0.5) - 0.5
    day_data["hybrid_score"] = trend_score + sector_alpha - risk_penalty + (0.15 * prob_component)
    day_data["rank"] = day_data["hybrid_score"].rank(ascending=False, method="first")
    return day_data

def entry_candidates(day_data, current_symbols, config):
    next_return = day_data.get("predicted_next_return", pd.Series(0.0, index=day_data.index))
    return day_data[
        (~day_data["symbol"].isin(current_symbols)) &
        (day_data["prob_win"] >= config["buy_prob_threshold"]) &
        (day_data["rank"] <= config["max_entry_rank"]) &
        (day_data["return_5d"].between(config["min_entry_return_5d"], config["max_entry_return_5d"])) &
        (next_return >= config.get("min_predicted_next_return", 0.0)) &
        (day_data["rsi"].between(config["min_entry_rsi"], config["max_entry_rsi"])) &
        (day_data["close"] > day_data["sma_20"]) &
        (day_data["tv_adx"] >= config["min_entry_adx"])
    ].sort_values("rank")

def entry_diagnostics(row, config, market_breadth=None):
    reasons = []
    if market_breadth is not None and market_breadth < config["min_entry_market_breadth"]:
        reasons.append("صحة السوق أقل من شرط الدخول")
    if row.get("avg_traded_value_20d", 0) < config["min_avg_traded_value"]:
        reasons.append("السيولة أقل من الحد المطلوب")
    if row.get("prob_win", 0) < config["buy_prob_threshold"]:
        reasons.append("احتمالية التفوق الأسبوعي غير كافية")
    if row.get("rank", 999) > config["max_entry_rank"]:
        reasons.append("ترتيب الاستراتيجية خارج أفضل الفرص")
    if not (config["min_entry_return_5d"] <= row.get("return_5d", 0) <= config["max_entry_return_5d"]):
        reasons.append("حركة آخر 5 أيام خارج النطاق المقبول")
    if row.get("predicted_next_return", 0) < config.get("min_predicted_next_return", 0.0):
        reasons.append("توقع الغد سلبي")
    if not (config["min_entry_rsi"] <= row.get("rsi", 50) <= config["max_entry_rsi"]):
        reasons.append("RSI خارج نطاق الدخول")
    if row.get("close", 0) <= row.get("sma_20", 0):
        reasons.append("السعر دون متوسط 20 يوم")
    if row.get("tv_adx", 0) < config["min_entry_adx"]:
        reasons.append("قوة الاتجاه ADX غير كافية")
    return "، ".join(reasons[:3]) if reasons else "مستوفي لشروط الدخول"

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
