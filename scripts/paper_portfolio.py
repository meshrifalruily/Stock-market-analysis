import json
import math
import os
from datetime import datetime

COMMISSION = 0.00155
SLIPPAGE = 0.0005
DEFAULT_INITIAL_CAPITAL = 1000.0
DEFAULT_STATE_PATH = os.path.join("data", "paper_portfolio.json")


def _empty_state(initial_capital=DEFAULT_INITIAL_CAPITAL):
    return {
        "initial_capital": float(initial_capital),
        "cash": float(initial_capital),
        "positions": {},
        "trades": [],
        "last_signal_date": None,
        "updated_at": None,
    }


def load_portfolio_state(path=DEFAULT_STATE_PATH, initial_capital=DEFAULT_INITIAL_CAPITAL):
    if not os.path.exists(path):
        return _empty_state(initial_capital)
    try:
        with open(path, "r", encoding="utf-8") as f:
            state = json.load(f)
    except (json.JSONDecodeError, OSError):
        return _empty_state(initial_capital)

    state.setdefault("initial_capital", float(initial_capital))
    state.setdefault("cash", float(initial_capital))
    state.setdefault("positions", {})
    state.setdefault("trades", [])
    state.setdefault("last_signal_date", None)
    state.setdefault("updated_at", None)
    return state


def save_portfolio_state(state, path=DEFAULT_STATE_PATH):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(state, f, ensure_ascii=False, indent=2)
    return state


def reset_portfolio(path=DEFAULT_STATE_PATH, initial_capital=DEFAULT_INITIAL_CAPITAL):
    state = _empty_state(initial_capital)
    save_portfolio_state(state, path)
    return state


def _recommendation_map(recommendations):
    return {item["symbol"]: item for item in recommendations}


def _mark_positions(state, rec_map):
    for symbol, position in state["positions"].items():
        rec = rec_map.get(symbol)
        if rec:
            position["company_name"] = rec.get("company_name", position.get("company_name", symbol))
            position["last_price"] = float(rec.get("current_price") or position.get("last_price") or position["avg_price"])
            position["stop"] = float(rec.get("stop") or position.get("stop") or 0)
            if rec.get("target_weekly"):
                position["target"] = float(rec["target_weekly"])


def _portfolio_totals(state):
    positions = []
    invested_value = 0.0
    unrealized_pnl = 0.0

    for symbol, position in state["positions"].items():
        shares = int(position.get("shares", 0))
        avg_price = float(position.get("avg_price", 0))
        last_price = float(position.get("last_price", avg_price))
        market_value = shares * last_price
        cost_basis = shares * avg_price
        pnl = market_value - cost_basis
        pnl_pct = (pnl / cost_basis * 100) if cost_basis else 0.0
        invested_value += market_value
        unrealized_pnl += pnl
        positions.append({
            **position,
            "symbol": symbol,
            "shares": shares,
            "market_value": round(market_value, 2),
            "unrealized_pnl": round(pnl, 2),
            "unrealized_pnl_pct": round(pnl_pct, 2),
        })

    total_value = float(state.get("cash", 0)) + invested_value
    initial_capital = float(state.get("initial_capital", DEFAULT_INITIAL_CAPITAL))
    total_return = total_value - initial_capital
    total_return_pct = (total_return / initial_capital * 100) if initial_capital else 0.0

    return {
        "initial_capital": round(initial_capital, 2),
        "cash": round(float(state.get("cash", 0)), 2),
        "invested_value": round(invested_value, 2),
        "total_value": round(total_value, 2),
        "total_return": round(total_return, 2),
        "total_return_pct": round(total_return_pct, 2),
        "unrealized_pnl": round(unrealized_pnl, 2),
        "positions": sorted(positions, key=lambda item: item.get("market_value", 0), reverse=True),
        "trades": state.get("trades", [])[-20:][::-1],
        "last_signal_date": state.get("last_signal_date"),
        "updated_at": state.get("updated_at"),
    }


def _record_trade(state, trade):
    state["trades"].append({
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        **trade,
    })
    state["trades"] = state["trades"][-200:]


def _sell_position(state, symbol, rec, data_date, reason):
    position = state["positions"].pop(symbol)
    shares = int(position.get("shares", 0))
    price = float(rec.get("current_price") if rec else position.get("last_price", position.get("avg_price", 0)))
    proceeds = shares * price * (1 - COMMISSION - SLIPPAGE)
    pnl = proceeds - (shares * float(position["avg_price"]))
    state["cash"] += proceeds
    _record_trade(state, {
        "date": data_date,
        "side": "بيع",
        "symbol": symbol,
        "company_name": position.get("company_name", symbol),
        "shares": shares,
        "price": round(price, 2),
        "amount": round(proceeds, 2),
        "pnl": round(pnl, 2),
        "reason": reason,
    })


def _buy_position(state, rec, data_date, budget):
    price = float(rec.get("current_price") or 0)
    if price <= 0:
        return False

    effective_price = price * (1 + COMMISSION + SLIPPAGE)
    shares = math.floor(budget / effective_price)
    if shares < 1:
        return False

    cost = shares * effective_price
    if cost > state["cash"]:
        shares = math.floor(state["cash"] / effective_price)
        cost = shares * effective_price
    if shares < 1:
        return False

    state["cash"] -= cost
    state["positions"][rec["symbol"]] = {
        "company_name": rec.get("company_name", rec["symbol"]),
        "shares": int(shares),
        "avg_price": round(price, 2),
        "entry_date": data_date,
        "last_price": round(price, 2),
        "stop": float(rec.get("stop") or 0),
        "target": float(rec.get("target_weekly") or rec.get("target_medium") or 0),
        "action": rec.get("action", "شراء"),
    }
    _record_trade(state, {
        "date": data_date,
        "side": "شراء",
        "symbol": rec["symbol"],
        "company_name": rec.get("company_name", rec["symbol"]),
        "shares": int(shares),
        "price": round(price, 2),
        "amount": round(cost, 2),
        "pnl": 0.0,
        "reason": rec.get("action", "توصية شراء"),
    })
    return True


def sync_portfolio_with_recommendations(recommendations, data_date, config, path=DEFAULT_STATE_PATH, initial_capital=DEFAULT_INITIAL_CAPITAL):
    state = load_portfolio_state(path, initial_capital)
    data_date = str(data_date)
    rec_map = _recommendation_map(recommendations)
    _mark_positions(state, rec_map)

    if state.get("last_signal_date") == data_date:
        state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
        save_portfolio_state(state, path)
        return _portfolio_totals(state)

    for symbol in list(state["positions"].keys()):
        position = state["positions"][symbol]
        rec = rec_map.get(symbol)
        current_price = float(rec.get("current_price") if rec else position.get("last_price", position["avg_price"]))
        stop = float(position.get("stop") or 0)
        target = float(position.get("target") or 0)
        market_breadth = float(rec.get("market_breadth", 0.5)) if rec else 0.0

        sell_reason = None
        if current_price <= stop:
            sell_reason = "تفعيل وقف الخسارة"
        elif target > 0 and current_price >= target:
            sell_reason = "تحقق الهدف"
        elif market_breadth < config["market_breadth_exit"]:
            sell_reason = "ضعف عام في السوق"
        elif not rec or rec.get("action_class") == "avoid":
            sell_reason = "خروج التوصية من شروط الشراء"

        if sell_reason:
            _sell_position(state, symbol, rec, data_date, sell_reason)

    max_positions = int(config.get("max_positions", 3))
    available_slots = max_positions - len(state["positions"])
    if available_slots > 0:
        current_symbols = set(state["positions"])
        buys = [
            rec for rec in recommendations
            if rec.get("is_actionable") and rec["symbol"] not in current_symbols
        ][:available_slots]

        total_value = _portfolio_totals(state)["total_value"]
        target_budget = total_value / max_positions if max_positions else state["cash"]
        for rec in buys:
            if state["cash"] <= 0:
                break
            _buy_position(state, rec, data_date, min(state["cash"], target_budget))

    state["last_signal_date"] = data_date
    state["updated_at"] = datetime.now().strftime("%Y-%m-%d %H:%M")
    save_portfolio_state(state, path)
    return _portfolio_totals(state)
