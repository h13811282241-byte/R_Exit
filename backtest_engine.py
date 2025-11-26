#!/usr/bin/env python3
"""
Generic backtest engine: simulate trades given signals and SL/TP.
"""

from typing import List, Dict, Optional
import math
import pandas as pd
import numpy as np


def summarize_trades(trades: List[Dict], key: str = "net_R") -> Dict:
    if not trades:
        return {
            "num_trades": 0,
            "win_rate": 0.0,
            "avg_raw_R": 0.0,
            "avg_R": 0.0,
            "median_R": 0.0,
            "max_R": 0.0,
            "min_R": 0.0,
            "avg_win_R": 0.0,
            "avg_loss_R": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_R": 0.0,
        }
    def get_R(t, k):
        if k in t and t[k] is not None and not math.isnan(t[k]):
            return float(t[k])
        if "net_R" in t and t["net_R"] is not None and not math.isnan(t["net_R"]):
            return float(t["net_R"])
        if "R" in t and t["R"] is not None and not math.isnan(t["R"]):
            return float(t["R"])
        return math.nan

    R_vals = [get_R(t, key) for t in trades]
    R_vals = [r for r in R_vals if not math.isnan(r)]
    if not R_vals:
        return {
            "num_trades": len(trades),
            "win_rate": 0.0,
            "avg_raw_R": 0.0,
            "avg_R": 0.0,
            "median_R": 0.0,
            "max_R": 0.0,
            "min_R": 0.0,
            "avg_win_R": 0.0,
            "avg_loss_R": 0.0,
            "profit_factor": 0.0,
            "max_drawdown_R": 0.0,
        }
    wins = [r for r in R_vals if r > 0]
    losses = [r for r in R_vals if r < 0]
    cum = np.array(R_vals).cumsum()
    peak = np.maximum.accumulate(cum)
    dd = peak - cum
    max_dd = dd.max() if len(dd) else 0.0
    profit_factor = (sum(wins) / abs(sum(losses))) if losses else float('inf') if wins else 0.0
    return {
        "num_trades": len(R_vals),
        "win_rate": len(wins) / len(R_vals),
        "avg_raw_R": float(np.mean(R_vals)),
        "avg_R": float(np.mean(R_vals)),
        "median_R": float(np.median(R_vals)),
        "max_R": float(np.max(R_vals)),
        "min_R": float(np.min(R_vals)),
        "avg_win_R": float(np.mean(wins)) if wins else 0.0,
        "avg_loss_R": float(np.mean(losses)) if losses else 0.0,
        "profit_factor": profit_factor,
        "max_drawdown_R": float(max_dd),
    }


def simulate_basic(
    df: pd.DataFrame,
    signals: List[Dict],
    fee_side_pct: float = 0.000248,
    max_holding_bars: Optional[int] = None,
    lower_df: Optional[pd.DataFrame] = None,
    upper_interval_sec: int = 0,
    lower_interval_sec: int = 0,
    lower_fetch=None,
    entry_slip_pct: float = 0.0,
    sl_buffer_pct: float = 0.0,
    min_risk_pct: float = 0.001,
) -> (List[Dict], Dict[str, int]):
    """
    signals: list of dict with idx, side, entry, sl, tp (optional), exit_idx (optional)
    Optional keys:
      - entry_mode: "retest" -> 等待价格回踩/反弹触发 entry_trigger，超时则放弃
      - entry_trigger: 触发价格（不写则用 entry）
      - entry_expire: 触发有效期（根数）
    """
    required = {"timestamp", "high", "low", "close"}
    if not required.issubset(df.columns):
        raise ValueError(f"缺少必需列: {required - set(df.columns)}")
    prices = df[["high", "low", "close"]].to_numpy()
    ts_upper = pd.to_datetime(df["timestamp"], utc=True).to_numpy()
    lower_data = None
    if lower_df is not None:
        lower_data = {
            "ts": pd.to_datetime(lower_df["timestamp"], utc=True).to_numpy(),
            "high": lower_df["high"].to_numpy(),
            "low": lower_df["low"].to_numpy(),
        }
    lower_cache_built = lower_data is not None

    fee_round = fee_side_pct * 2.0
    trades: List[Dict] = []
    last_index = len(df) - 1

    def resolve_conflict(side: str, sl: float, tp: float, start_ts, end_ts):
        nonlocal lower_data, lower_cache_built
        if lower_data is None and lower_fetch and not lower_cache_built:
            try:
                df_lower = lower_fetch()
                lower_data = {
                    "ts": pd.to_datetime(df_lower["timestamp"], utc=True).to_numpy(),
                    "high": df_lower["high"].to_numpy(),
                    "low": df_lower["low"].to_numpy(),
                }
            except Exception:
                lower_data = None
            lower_cache_built = True
        if lower_data is None:
            return "sl", sl
        mask = (lower_data["ts"] >= start_ts) & (lower_data["ts"] < end_ts)
        if not mask.any():
            return "sl", sl
        for h, l in zip(lower_data["high"][mask], lower_data["low"][mask]):
            if side == "long":
                hit_sl = l <= sl
                hit_tp = h >= tp
            else:
                hit_sl = h >= sl
                hit_tp = l <= tp
            if hit_sl and not hit_tp:
                return "sl", sl
            if hit_tp and not hit_sl:
                return "tp", tp
            if hit_sl and hit_tp:
                return "sl", sl
        return "sl", sl

    skipped_small_risk = 0
    skipped_not_trigger = 0

    for sig in signals:
        idx = sig["idx"]
        if idx >= last_index:
            continue
        side = sig["side"]
        entry_raw = sig["entry"]
        entry = entry_raw * (1 + entry_slip_pct if side == "long" else 1 - entry_slip_pct)
        sl = sig["sl"]
        tp = sig.get("tp")
        if sl_buffer_pct != 0:
            if side == "long":
                sl = sl - entry * sl_buffer_pct
            else:
                sl = sl + entry * sl_buffer_pct
        risk = entry - sl if side == "long" else sl - entry
        if risk <= 0:
            continue
        risk_pct = risk / entry
        if risk_pct < min_risk_pct:
            skipped_small_risk += 1
            continue
        exit_idx = None
        exit_price = None
        outcome = None
        end_idx = last_index if max_holding_bars is None else min(last_index, idx + max_holding_bars)

        # 处理回踩触发
        entry_mode = sig.get("entry_mode")
        trigger_price = sig.get("entry_trigger", entry_raw)
        entry_expire = sig.get("entry_expire")
        entry_hit_idx = idx + 1
        entry_price_used = entry
        if entry_mode == "retest":
            triggered = False
            max_bar = end_idx if entry_expire is None else min(end_idx, idx + entry_expire)
            for bar in range(idx + 1, max_bar + 1):
                high_b, low_b, close_b = prices[bar]
                if side == "long":
                    hit = low_b <= trigger_price <= high_b
                else:
                    hit = low_b <= trigger_price <= high_b
                if hit:
                    entry_hit_idx = bar
                    entry_price_used = trigger_price * (1 + entry_slip_pct if side == "long" else 1 - entry_slip_pct)
                    triggered = True
                    break
            if not triggered:
                skipped_not_trigger += 1
                continue
            # 触发后重新计算风险
            risk = entry_price_used - sl if side == "long" else sl - entry_price_used
            if risk <= 0:
                continue
            risk_pct = risk / entry_price_used
            if risk_pct < min_risk_pct:
                skipped_small_risk += 1
                continue
            entry = entry_price_used

        # 分批止盈配置
        tp_multipliers = sig.get("tp_multipliers")
        tp_fractions = sig.get("tp_fractions")
        partial_targets = []
        if tp_multipliers:
            fracs = tp_fractions or [0.33, 0.33, 0.17, 0.09, 0.08]
            if len(fracs) < len(tp_multipliers):
                # 补齐平均
                remain = len(tp_multipliers) - len(fracs)
                fracs += [0.0] * remain
            # 归一化
            total_frac = sum(fracs)
            fracs = [f / total_frac for f in fracs]
            for mult, frac in zip(tp_multipliers, fracs):
                price = entry + mult * risk if side == "long" else entry - mult * risk
                partial_targets.append({"price": price, "mult": mult, "frac": frac, "hit": False})
        filled_R = 0.0
        remaining_frac = 1.0

        # 执行后续持仓模拟
        for bar in range(entry_hit_idx, end_idx + 1):
            high_b, low_b, close_b = prices[bar]
            # 处理分批止盈
            if partial_targets:
                for t in partial_targets:
                    if t["hit"]:
                        continue
                    if side == "long" and high_b >= t["price"]:
                        t["hit"] = True
                        filled_R += t["mult"] * t["frac"]
                        remaining_frac -= t["frac"]
                    elif side == "short" and low_b <= t["price"]:
                        t["hit"] = True
                        filled_R += t["mult"] * t["frac"]
                        remaining_frac -= t["frac"]
            if side == "long":
                hit_sl = low_b <= sl
                hit_tp = (tp is not None) and (high_b >= tp) and not partial_targets
            else:
                hit_sl = high_b >= sl
                hit_tp = (tp is not None) and (low_b <= tp) and not partial_targets
            if hit_sl and hit_tp:
                start_ts = ts_upper[idx]
                end_ts = start_ts + pd.Timedelta(seconds=upper_interval_sec) if upper_interval_sec else ts_upper[bar]
                sub_outcome, sub_price = resolve_conflict(side, sl, tp, start_ts, end_ts)
                outcome = sub_outcome
                exit_price = sub_price
                exit_idx = bar
                break
            if hit_sl:
                outcome = "sl"
                exit_price = sl
                exit_idx = bar
                break
            if hit_tp:
                outcome = "tp"
                exit_price = tp
                exit_idx = bar
                break
            # 当所有分批目标命中完毕
            if partial_targets and all(t["hit"] for t in partial_targets):
                outcome = "tp_partial"
                exit_price = partial_targets[-1]["price"]
                exit_idx = bar
                break
        if exit_idx is None:
            exit_idx = end_idx
            exit_price = prices[end_idx][2]
            outcome = "timeout"

        pnl = (exit_price - entry) if side == "long" else (entry - exit_price)
        raw_R = pnl / risk
        if partial_targets:
            # 剩余仓位按最终价格结算
            raw_R = filled_R + remaining_frac * (pnl / risk)
        # risk_pct 已计算
        fee_R = fee_round / risk_pct if risk_pct > 0 else 0.0
        net_R = raw_R - fee_R

        trades.append(
            {
                "entry_time": df.loc[idx, "timestamp"],
                "exit_time": df.loc[exit_idx, "timestamp"],
                "side": side,
                "entry": entry,
                "sl": sl,
                "tp": tp,
                "exit_price": exit_price,
                "outcome": outcome,
                "raw_R": raw_R,
                "fee_R": fee_R,
                "net_R": net_R,
                "risk_pct": risk_pct,
                "risk_pct_abs": risk_pct,
                "entry_idx": entry_hit_idx if entry_mode == "retest" else idx,
                "signal_idx": idx,
                "exit_idx": exit_idx,
            }
        )
    return trades, {"skipped_small_risk": skipped_small_risk, "skipped_not_trigger": skipped_not_trigger}


def equity_curve(trades: List[Dict], key: str = "net_R") -> pd.DataFrame:
    """返回累计 R 曲线 DataFrame"""
    cum = []
    total = 0.0
    for i, t in enumerate(trades, start=1):
        val = t.get(key, t.get("net_R", t.get("R", 0.0)))
        if val is None or math.isnan(val):
            val = 0.0
        total += val
        cum.append({"trade_index": i, "cumulative_R": total})
    return pd.DataFrame(cum)


def compound_stats(trades: List[Dict], initial_capital: float = 100.0, risk_per_trade: float = 0.01, key: str = "net_R") -> Dict:
    """
    以 R 为基础，假设每笔风险 = 账户权益 * risk_per_trade，计算逐笔复利。
    """
    equity = initial_capital
    peak = equity
    max_dd = 0.0
    for t in trades:
        R = t.get(key, t.get("net_R", 0.0))
        if R is None or math.isnan(R):
            R = 0.0
        risk_amount = equity * risk_per_trade
        pnl = R * risk_amount
        equity += pnl
        peak = max(peak, equity)
        dd = (peak - equity) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return {
        "final_equity": equity,
        "max_drawdown_pct": max_dd,
        "return_pct": (equity - initial_capital) / initial_capital if initial_capital > 0 else 0.0,
    }


def compound_curve(
    trades: List[Dict],
    initial_capital: float = 100.0,
    risk_per_trade: float = 0.01,
    key: str = "net_R",
) -> pd.DataFrame:
    equity = initial_capital
    rows = []
    for i, t in enumerate(trades, start=1):
        R = t.get(key, t.get("net_R", 0.0))
        if R is None or math.isnan(R):
            R = 0.0
        risk_amount = equity * risk_per_trade
        pnl = R * risk_amount
        equity += pnl
        rows.append({"trade_index": i, "equity": equity})
    return pd.DataFrame(rows)
