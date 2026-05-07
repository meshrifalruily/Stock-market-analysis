import itertools
import json
import os
import sys

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

from scripts.strategy_config import DEFAULT_STRATEGY_CONFIG, save_strategy_config
from scripts.strategy_rules import add_hybrid_scores, entry_candidates, filter_main_market, summarize_returns

COMMISSION = 0.00155
SLIPPAGE = 0.0005
TARGET = "target_alpha_weekly"

def prepare_features(df, features, medians):
    X = df[features].replace([np.inf, -np.inf], np.nan)
    return X.fillna(medians).fillna(0)

def train_model(train_df, features):
    train_df = train_df.dropna(subset=[TARGET]).copy()
    X_raw = train_df[features].replace([np.inf, -np.inf], np.nan)
    medians = X_raw.median(numeric_only=True).fillna(0)
    X = X_raw.fillna(medians).fillna(0)
    y = train_df[TARGET]
    model = HistGradientBoostingClassifier(
        max_iter=120,
        learning_rate=0.03,
        max_leaf_nodes=63,
        l2_regularization=1.5,
        random_state=42,
        class_weight="balanced",
    )
    model.fit(X, y)
    return model, medians

def simulate_with_probabilities(test_df, dates, config):
    portfolio_value = 100.0
    holdings = {}
    returns = []
    selected = []

    for i in range(len(dates) - 1):
        current_date = dates[i]
        next_date = dates[i + 1]
        day_data = test_df[test_df["date"] == current_date].copy()
        next_day = test_df[test_df["date"] == next_date]

        if "avg_traded_value_20d" in day_data.columns:
            day_data = day_data[day_data["avg_traded_value_20d"] >= config["min_avg_traded_value"]]
        if day_data.empty:
            returns.append(0)
            selected.append("")
            continue

        day_data = add_hybrid_scores(day_data)
        market_breadth = day_data["market_breadth_sma50"].iloc[0] if "market_breadth_sma50" in day_data else 0.5
        daily_return = 0.0
        new_holdings = {}

        for symbol, info in holdings.items():
            current_stock = day_data[day_data["symbol"] == symbol]
            next_stock = next_day[next_day["symbol"] == symbol]
            weight = info["weight"]
            if current_stock.empty or next_stock.empty:
                daily_return -= (COMMISSION + SLIPPAGE) * weight
                continue

            current_price = current_stock["close"].iloc[0]
            atr = current_stock["atr"].iloc[0] if "atr" in current_stock else 0
            next_ret = next_stock["daily_return"].iloc[0]
            new_stop = max(info["stop_loss"], current_price - (config["stop_loss_atr_mult"] * atr))
            if current_price >= info["entry_price"] * 1.012:
                new_stop = max(new_stop, info["entry_price"])

            daily_return += next_ret * weight
            days_held = info["days_held"] + 1
            rank_value = current_stock["rank"].iloc[0]
            prob_value = current_stock["prob_win"].iloc[0]

            should_sell = (
                current_price < info["stop_loss"]
                or current_price > info["take_profit"]
                or market_breadth < config["market_breadth_exit"]
                or days_held >= config["max_hold_days"]
                or (days_held >= 2 and (rank_value > config["max_entry_rank"] or prob_value < config["sell_prob_threshold"]))
            )

            if should_sell:
                daily_return -= (COMMISSION + SLIPPAGE) * weight
            else:
                new_holdings[symbol] = {
                    **info,
                    "stop_loss": new_stop,
                    "days_held": days_held,
                }

        available_slots = config["max_positions"] - len(new_holdings)
        if available_slots > 0 and market_breadth > config["min_entry_market_breadth"]:
            buys = entry_candidates(day_data, new_holdings.keys(), config).head(available_slots)
            for _, row in buys.iterrows():
                next_stock = next_day[next_day["symbol"] == row["symbol"]]
                if next_stock.empty:
                    continue
                atr = row["atr"] if "atr" in row and row["atr"] > 0 else row["close"] * 0.02
                atr_pct = atr / row["close"] if row["close"] > 0 else 0.02
                weight = min(1 / config["max_positions"], config["risk_per_position"] / max(0.001, atr_pct))
                daily_return -= (COMMISSION + SLIPPAGE) * weight
                daily_return += next_stock["daily_return"].iloc[0] * weight
                new_holdings[row["symbol"]] = {
                    "entry_price": row["close"],
                    "stop_loss": row["close"] - (config["stop_loss_atr_mult"] * atr),
                    "take_profit": row["close"] + (config["take_profit_atr_mult"] * config["stop_loss_atr_mult"] * atr),
                    "weight": weight,
                    "days_held": 0,
                }

        portfolio_value *= 1 + daily_return
        holdings = new_holdings
        returns.append(daily_return)
        selected.append(",".join(holdings.keys()))

    summary = summarize_returns(returns)
    summary["portfolio_value"] = portfolio_value
    summary["selected_days"] = int(sum(bool(x) for x in selected))
    return summary

def candidate_configs():
    base = DEFAULT_STRATEGY_CONFIG.copy()
    grid = {
        "buy_prob_threshold": [0.40, 0.45, 0.50, 0.55],
        "min_entry_market_breadth": [0.40, 0.45, 0.50, 0.55],
        "max_entry_rank": [10, 15, 20],
        "max_hold_days": [2, 3, 5],
        "max_entry_return_5d": [0.05, 0.07],
        "sell_prob_threshold": [0.50, 0.54],
    }
    keys = list(grid)
    for values in itertools.product(*(grid[key] for key in keys)):
        config = base.copy()
        config.update(dict(zip(keys, values)))
        yield config

def optimize_strategy(processed_file_path="data/tasi_processed.csv", features_path="models/feature_names.joblib"):
    df = pd.read_csv(processed_file_path)
    df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
    df = filter_main_market(df)
    features = joblib.load(features_path)

    max_date = df["date"].max()
    start_date = max_date - pd.Timedelta(days=180)
    train_df = df[df["date"] < start_date].copy()
    test_df = df[df["date"] >= start_date].copy()
    dates = sorted(test_df["date"].unique())

    model, medians = train_model(train_df, features)
    X = prepare_features(test_df, features, medians)
    test_df = test_df.copy()
    test_df["prob_win"] = model.predict_proba(X)[:, 1]

    results = []
    for config in candidate_configs():
        summary = simulate_with_probabilities(test_df, dates, config)
        score = (
            summary["total_return"] * 100
            + summary["sharpe"] * 1.5
            + summary["max_drawdown"] * 50
            - (0.03 if summary["active_days"] < 8 else 0)
        )
        results.append({"score": score, "summary": summary, "config": config})

    results = sorted(results, key=lambda x: x["score"], reverse=True)
    best = results[0]
    saved_config = save_strategy_config(best["config"])

    os.makedirs("reports", exist_ok=True)
    with open("reports/strategy_optimization.json", "w", encoding="utf-8") as f:
        json.dump(results[:25], f, ensure_ascii=False, indent=2)

    print("--- أفضل إعداد للاستراتيجية ---")
    print(json.dumps(saved_config, ensure_ascii=False, indent=2))
    print("--- ملخص الأداء ---")
    print(json.dumps(best["summary"], ensure_ascii=False, indent=2))
    print("تم حفظ الإعداد في models/strategy_config.json")
    return best

if __name__ == "__main__":
    optimize_strategy()
