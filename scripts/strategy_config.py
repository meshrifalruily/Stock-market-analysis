import json
import os

CONFIG_PATH = os.path.join("models", "strategy_config.json")

DEFAULT_STRATEGY_CONFIG = {
    "min_avg_traded_value": 1_000_000,
    "buy_prob_threshold": 0.45,
    "min_entry_market_breadth": 0.50,
    "max_entry_rank": 15,
    "sell_prob_threshold": 0.54,
    "max_hold_days": 3,
    "max_entry_return_5d": 0.07,
    "min_entry_return_5d": -0.03,
    "min_entry_rsi": 45,
    "max_entry_rsi": 72,
    "min_entry_adx": 12,
    "stop_loss_atr_mult": 1.0,
    "take_profit_atr_mult": 4.0,
    "market_breadth_exit": 0.25,
    "max_positions": 3,
    "risk_per_position": 0.01,
}

ENV_MAP = {
    "TASI_MIN_AVG_TRADED_VALUE": ("min_avg_traded_value", float),
    "TASI_BUY_PROB_THRESHOLD": ("buy_prob_threshold", float),
    "TASI_MIN_ENTRY_MARKET_BREADTH": ("min_entry_market_breadth", float),
    "TASI_MAX_ENTRY_RANK": ("max_entry_rank", int),
    "TASI_SELL_PROB_THRESHOLD": ("sell_prob_threshold", float),
    "TASI_MAX_HOLD_DAYS": ("max_hold_days", int),
    "TASI_MAX_ENTRY_RETURN_5D": ("max_entry_return_5d", float),
    "TASI_MIN_ENTRY_RETURN_5D": ("min_entry_return_5d", float),
    "TASI_MIN_ENTRY_RSI": ("min_entry_rsi", float),
    "TASI_MAX_ENTRY_RSI": ("max_entry_rsi", float),
    "TASI_MIN_ENTRY_ADX": ("min_entry_adx", float),
}

def load_strategy_config(path=CONFIG_PATH):
    config = DEFAULT_STRATEGY_CONFIG.copy()
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            config.update(json.load(f))

    for env_name, (key, caster) in ENV_MAP.items():
        value = os.getenv(env_name)
        if value not in (None, ""):
            config[key] = caster(value)

    return config

def save_strategy_config(config, path=CONFIG_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    cleaned = DEFAULT_STRATEGY_CONFIG.copy()
    cleaned.update(config)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, ensure_ascii=False, indent=2)
    return cleaned

