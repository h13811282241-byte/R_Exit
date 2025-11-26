#!/usr/bin/env python3
"""
Streamlit Web UI for RSI 背离 + Alligator 策略回测。
"""

import os
import tempfile
import random
from typing import Optional

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import numpy as np
import hashlib
import hmac
import time
import requests
from datetime import datetime, time as dtime, timedelta
import pytz

from data_loader import load_csv, download_binance_klines, ensure_ohlcv_df
from indicators import atr, alligator
from market_pattern_detector import PatternClassifier, PatternConfig, pattern_counts, pattern_performance
from strategy_rsi_divergence import detect_rsi_divergence_signals
from strategy_alligator import detect_alligator_signals, prepare_sl_tp
from breakout_strategy import detect_breakouts, simulate_trades as simulate_breakout
from backtest_engine import simulate_basic, summarize_trades, compound_stats, compound_curve
from strategy_volume_ma_reversal import detect_volume_ma_signals
from strategy_ema220_breakout import detect_signals as detect_ema220_signals, simulate_trades as simulate_ema220_trades
from strategy_ct_divergence import detect_ct_signals, CTDivergenceConfig
from strategy_three_push_ema220 import (
    ThreePushParams,
    generate_signals as generate_three_push_signals,
    simulate_trades as simulate_three_push_trades,
    find_swings as tp_find_swings,
    atr as tp_atr,
    plot_random_three_push_samples,
)
from strategy_box_reversion import detect_signals as detect_box_signals, simulate_trades as simulate_box_trades
from strategy_bear_combo import detect_bear_combo_signals, BearComboConfig


def plot_price(df: pd.DataFrame, trades: pd.DataFrame):
    fig = go.Figure(data=[go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"])])
    if not trades.empty and "side" in trades.columns:
        # 补全缺失的 entry_time
        if "entry_time" not in trades.columns and "entry_idx" in trades.columns:
            trades = trades.copy()
            trades["entry_time"] = trades["entry_idx"].apply(lambda i: df.loc[i, "timestamp"] if 0 <= i < len(df) else None)
        longs = trades[trades["side"] == "long"]
        shorts = trades[trades["side"] == "short"]
        fig.add_trace(go.Scatter(x=longs["entry_time"], y=longs["entry"], mode="markers", marker=dict(color="green", symbol="triangle-up", size=10), name="Long entry"))
        fig.add_trace(go.Scatter(x=shorts["entry_time"], y=shorts["entry"], mode="markers", marker=dict(color="red", symbol="triangle-down", size=10), name="Short entry"))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


def plot_pattern_chart(
    df: pd.DataFrame,
    title: str,
    pattern_filter: Optional[str] = None,
    window: int = 200,
    target_idx: Optional[int] = None,
    show_overlays: bool = True,
    show_arrows: bool = False,
    entry_line: Optional[float] = None,
    entry_idx: Optional[int] = None,
    entry_point: Optional[tuple[int, float, str]] = None,
):
    if df.empty:
        st.info("数据为空，无法绘图")
        return
    # 默认定位：匹配形态时取最后一个匹配的索引；否则尾部
    if target_idx is None:
        target_idx = df.index[-1]
        if pattern_filter and "pattern" in df.columns:
            idx_match = df[df["pattern"] == pattern_filter].index
            if len(idx_match) > 0:
                target_idx = idx_match[-1]
            else:
                st.info(f"未找到 {pattern_filter}，展示尾部数据")

    # 将 target_idx 视为位置索引，保证 RangeIndex 和普通索引都能取到窗口
    if isinstance(target_idx, (int, np.integer)):
        pos = int(target_idx)
        # 让形态尽量靠左：把窗口起点前移一半窗口
        start_pos = max(0, pos - int(window * 0.6))
        end_pos = min(len(df) - 1, start_pos + window - 1)
        segment = df.iloc[start_pos : end_pos + 1]
    else:
        start_label = target_idx
        # 非整数索引时退回 label 切片，可能无 window 限制
        segment = df.loc[:target_idx].tail(window)

    fig = go.Figure(
        data=[
            go.Candlestick(
                x=segment["timestamp"],
                open=segment["open"],
                high=segment["high"],
                low=segment["low"],
                close=segment["close"],
            )
        ]
    )
    x_vals = segment["timestamp"]
    annotations = []

    if show_overlays:
        if "range_high" in segment and segment["range_high"].notna().any():
            fig.add_trace(go.Scatter(x=x_vals, y=segment["range_high"], mode="lines", name="range_high", line=dict(color="blue", dash="dash")))
            fig.add_trace(go.Scatter(x=x_vals, y=segment["range_low"], mode="lines", name="range_low", line=dict(color="blue", dash="dash")))
        if "channel_upper" in segment and segment["channel_upper"].notna().any():
            fig.add_trace(go.Scatter(x=x_vals, y=segment["channel_upper"], mode="lines", name="channel_upper", line=dict(color="purple")))
            fig.add_trace(go.Scatter(x=x_vals, y=segment["channel_lower"], mode="lines", name="channel_lower", line=dict(color="purple")))
        if "triangle_upper" in segment and segment["triangle_upper"].notna().any():
            fig.add_trace(go.Scatter(x=x_vals, y=segment["triangle_upper"], mode="lines", name="triangle_upper", line=dict(color="orange")))
            fig.add_trace(go.Scatter(x=x_vals, y=segment["triangle_lower"], mode="lines", name="triangle_lower", line=dict(color="orange")))
        if "wedge_upper" in segment and segment["wedge_upper"].notna().any():
            fig.add_trace(go.Scatter(x=x_vals, y=segment["wedge_upper"], mode="lines", name="wedge_upper", line=dict(color="brown")))
            fig.add_trace(go.Scatter(x=x_vals, y=segment["wedge_lower"], mode="lines", name="wedge_lower", line=dict(color="brown")))

    # 标记三推高低点（无论是否显示其他覆盖线）
    if "wedge_push_highs" in df.columns and "wedge_push_lows" in df.columns:
        row_idx = segment.index[-1]
        # 找到当前位置之前最近的三推点序列（兼容列表存储）
        highs_idx = None
        lows_idx = None
        try:
            sub_df = df.loc[:row_idx]
        except Exception:
            try:
                pos = df.index.get_loc(row_idx)
                sub_df = df.iloc[: pos + 1]
            except Exception:
                sub_df = None
        if sub_df is not None:
            if sub_df["wedge_push_highs"].notna().any():
                highs_idx = sub_df["wedge_push_highs"].dropna().iloc[-1]
            if sub_df["wedge_push_lows"].notna().any():
                lows_idx = sub_df["wedge_push_lows"].dropna().iloc[-1]

        if highs_idx:
            xs = []
            ys = []
            for j, idx in enumerate(highs_idx[:3], start=1):
                if idx in df.index:
                    ts = df.loc[idx, "timestamp"]
                    price = df.loc[idx, "high"]
                    if ts in x_vals.values:
                        xs.append(ts)
                        ys.append(price)
                        annotations.append(
                            dict(
                                x=ts,
                                y=price,
                                xref="x",
                                yref="y",
                                text=str(j),
                                showarrow=True,
                                arrowhead=1,
                                ax=0,
                                ay=-25,
                                font=dict(color="red"),
                            )
                        )
            if xs:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color="red", size=10, symbol="triangle-up"), name="三推高点"))
        if lows_idx:
            xs = []
            ys = []
            for j, idx in enumerate(lows_idx[:3], start=1):
                if idx in df.index:
                    ts = df.loc[idx, "timestamp"]
                    price = df.loc[idx, "low"]
                    if ts in x_vals.values:
                        xs.append(ts)
                        ys.append(price)
                        annotations.append(
                            dict(
                                x=ts,
                                y=price,
                                xref="x",
                                yref="y",
                                text=str(j),
                                showarrow=True,
                                arrowhead=1,
                                ax=0,
                                ay=25,
                                font=dict(color="green"),
                            )
                        )
            if xs:
                fig.add_trace(go.Scatter(x=xs, y=ys, mode="markers", marker=dict(color="green", size=10, symbol="triangle-down"), name="三推低点"))

    if show_arrows:
        high_max = float(segment["high"].max())
        x_start = segment["timestamp"].iloc[0]
        x_end = segment["timestamp"].iloc[-1]
        annotations.append(
            dict(
                x=x_start,
                y=high_max,
                xref="x",
                yref="y",
                text="Start",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
            )
        )
        annotations.append(
            dict(
                x=x_end,
                y=high_max,
                xref="x",
                yref="y",
                text="End",
                showarrow=True,
                arrowhead=2,
                ax=0,
                ay=-40,
            )
        )

    if annotations:
        fig.update_layout(annotations=annotations)

    if entry_line is not None:
        fig.add_hline(y=entry_line, line=dict(color="orange", dash="dot"), annotation_text="Entry", annotation_position="top left")
    if entry_idx is not None and isinstance(entry_idx, (int, np.integer)) and 0 <= entry_idx < len(df):
        ts_entry = df["timestamp"].iloc[entry_idx]
        if ts_entry in x_vals.values:
            fig.add_vline(x=ts_entry, line=dict(color="orange", dash="dot"))
    if entry_point is not None:
        ep_idx, ep_price, ep_side = entry_point
        if ep_idx in df.index:
            ts_ep = df.loc[ep_idx, "timestamp"]
            if ts_ep in x_vals.values:
                color = "blue" if ep_side == "long" else "purple"
                fig.add_trace(
                    go.Scatter(
                        x=[ts_ep],
                        y=[ep_price],
                        mode="markers+text",
                        marker=dict(color=color, size=12, symbol="star"),
                        text=["Entry"],
                        textposition="top center",
                        name="EntryPoint",
                    )
                )

    fig.update_layout(height=520, title=title, xaxis_rangeslider_visible=False)

    st.plotly_chart(fig, use_container_width=True)


def plot_equity(trades: pd.DataFrame):
    if trades.empty:
        st.info("无交易")
        return
    if "net_R" not in trades.columns:
        if "raw_R" in trades.columns:
            trades = trades.copy()
            trades["net_R"] = trades["raw_R"]
        else:
            st.info("无 net_R 列，无法绘制权益曲线")
            return
    trades["cum_net_R"] = trades["net_R"].cumsum()
    fig = go.Figure(data=[go.Scatter(x=trades.index, y=trades["cum_net_R"], mode="lines", name="Cumulative net R")])
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def plot_compound(trades: pd.DataFrame, initial_capital: float = 100.0, risk_per_trade: float = 0.02):
    if trades.empty:
        st.info("无交易，无法绘制复利曲线")
        return
    curve = compound_curve(trades.to_dict("records"), initial_capital=initial_capital, risk_per_trade=risk_per_trade)
    if curve.empty:
        st.info("无曲线数据")
        return
    fig = go.Figure(data=[go.Scatter(x=curve["trade_index"], y=curve["equity"], mode="lines", name="Equity")])
    fig.update_layout(height=300, title=f"复利曲线(初始{initial_capital}, 每笔风险 {risk_per_trade*100:.1f}%)")
    st.plotly_chart(fig, use_container_width=True)


def plot_trade_zoom(df: pd.DataFrame, trade: pd.Series, window: int = 120):
    if df.empty or trade.empty:
        st.info("无可用数据绘制单笔交易")
        return
    entry_idx = trade.get("entry_idx")
    exit_idx = trade.get("exit_idx")
    signal_idx = trade.get("signal_idx", entry_idx)
    prev_signal_idx = trade.get("prev_signal_idx")
    prev_entry_price = trade.get("prev_entry_price")
    # 如果没有索引，用时间匹配
    if entry_idx is None and "entry_time" in trade and "timestamp" in df.columns:
        match = df.index[df["timestamp"] == trade["entry_time"]]
        if len(match) > 0:
            entry_idx = int(match[0])
    if exit_idx is None and "exit_time" in trade and "timestamp" in df.columns:
        match = df.index[df["timestamp"] == trade["exit_time"]]
        if len(match) > 0:
            exit_idx = int(match[0])
    if entry_idx is None:
        st.info("缺少 entry_idx，无法绘制放大图")
        return
    entry_idx = int(entry_idx)
    start = max(0, entry_idx - window // 2)
    end = min(len(df) - 1, start + window - 1)
    segment = df.iloc[start : end + 1]
    fig = go.Figure(
        data=[
            go.Candlestick(
                x=segment["timestamp"],
                open=segment["open"],
                high=segment["high"],
                low=segment["low"],
                close=segment["close"],
            )
        ]
    )
    ts_entry = df.loc[entry_idx, "timestamp"] if entry_idx in df.index else None
    if ts_entry is not None:
        fig.add_vline(x=ts_entry, line=dict(color="orange", dash="dot"))
        fig.add_trace(
            go.Scatter(
                x=[ts_entry],
                y=[trade.get("entry", segment["close"].iloc[0])],
                mode="markers+text",
                marker=dict(color="orange", size=10, symbol="star"),
                text=["Entry"],
                textposition="top center",
                name="Entry",
            )
        )
    if signal_idx is not None and signal_idx in df.index:
        ts_sig = df.loc[signal_idx, "timestamp"]
        fig.add_trace(
            go.Scatter(
                x=[ts_sig],
                y=[df.loc[signal_idx, "close"]],
                mode="markers+text",
                marker=dict(color="blue", size=10, symbol="diamond"),
                text=["Signal"],
                textposition="bottom center",
                name="Signal",
            )
        )
    if prev_signal_idx is not None and prev_signal_idx in df.index and prev_entry_price is not None:
        ts_prev = df.loc[prev_signal_idx, "timestamp"]
        fig.add_trace(
            go.Scatter(
                x=[ts_prev],
                y=[prev_entry_price],
                mode="markers+text",
                marker=dict(color="gray", size=10, symbol="diamond"),
                text=["Prev Signal"],
                textposition="top center",
                name="PrevSignal",
            )
        )
    if exit_idx is not None and exit_idx in df.index:
        ts_exit = df.loc[exit_idx, "timestamp"]
        fig.add_vline(x=ts_exit, line=dict(color="red", dash="dot"))
        fig.add_trace(
            go.Scatter(
                x=[ts_exit],
                y=[trade.get("exit_price", trade.get("exit", segment["close"].iloc[-1]))],
                mode="markers+text",
                marker=dict(color="red", size=10, symbol="triangle-down"),
                text=["Exit"],
                textposition="bottom center",
                name="Exit",
            )
        )
    if "sl" in trade and trade["sl"] is not None:
        fig.add_hline(y=trade["sl"], line=dict(color="gray", dash="dash"), annotation_text="SL", annotation_position="bottom left")
    if "tp" in trade and trade["tp"] is not None:
        fig.add_hline(y=trade["tp"], line=dict(color="green", dash="dash"), annotation_text="TP", annotation_position="top left")
    fig.update_layout(height=420, title="单笔交易放大图", xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


def analyze_ny_open_first_hit(
    df: pd.DataFrame,
    session_end: str = "16:00",
    target_mult: float = 2.0,
    window_minutes: int = 90,
):
    """
    对称双目标 + 首触方向统计：
    9:30-11:00 计算 O/H90/L90/C90/R；方向= C90 vs O；
    目标：up_target=O+2R, down_target=O-2R（可调倍数）；
    11:00 之后首触 up/down 记录 first_hit。
    """
    if df.empty:
        return pd.DataFrame(), {}
    ts_utc = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.copy()
    df["ts_utc"] = ts_utc
    df["ts_ny"] = ts_utc.dt.tz_convert("America/New_York")
    df = df.dropna(subset=["ts_ny"])

    try:
        se_hour, se_min = map(int, session_end.split(":"))
    except Exception:
        se_hour, se_min = 16, 0

    tz_ny = pytz.timezone("America/New_York")
    days = sorted({t.date() for t in df["ts_ny"] if t.weekday() < 5})

    up_dir_total = up_dir_trend_win = up_dir_revert_win = up_no_touch = 0
    down_dir_total = down_dir_trend_win = down_dir_revert_win = down_no_touch = 0
    records = []

    for day in days:
        start_ny = tz_ny.localize(datetime.combine(day, dtime(9, 30)))
        end90_ny = start_ny + timedelta(minutes=window_minutes)
        session_end_ny = tz_ny.localize(datetime.combine(day, dtime(se_hour, se_min)))

        win_mask = (df["ts_ny"] >= start_ny) & (df["ts_ny"] < end90_ny)
        after_mask = (df["ts_ny"] >= end90_ny) & (df["ts_ny"] <= session_end_ny)
        df_win = df[win_mask]
        df_after = df[after_mask]
        if df_win.empty or df_after.empty:
            continue

        O = df_win.iloc[0]["open"]
        H90 = df_win["high"].max()
        L90 = df_win["low"].min()
        C90 = df_win.iloc[-1]["close"]
        R = H90 - L90
        if R <= 0:
            continue

        if C90 > O:
            direction = "UP"
        elif C90 < O:
            direction = "DOWN"
        else:
            continue

        up_target = O + target_mult * R
        down_target = O - target_mult * R
        first_hit = "NONE"
        for _, row in df_after.iterrows():
            h, l = row["high"], row["low"]
            if h >= up_target:
                first_hit = "UP"
                break
            if l <= down_target:
                first_hit = "DOWN"
                break

        if direction == "UP":
            up_dir_total += 1
            if first_hit == "UP":
                up_dir_trend_win += 1
            elif first_hit == "DOWN":
                up_dir_revert_win += 1
            else:
                up_no_touch += 1
        else:  # DOWN
            down_dir_total += 1
            if first_hit == "DOWN":
                down_dir_trend_win += 1
            elif first_hit == "UP":
                down_dir_revert_win += 1
            else:
                down_no_touch += 1

        records.append(
            {
                "date_ny": day.isoformat(),
                "direction": direction,
                "O": O,
                "H90": H90,
                "L90": L90,
                "C90": C90,
                "R": R,
                "up_target": up_target,
                "down_target": down_target,
                "first_hit": first_hit,
            }
        )

    daily_df = pd.DataFrame(records)
    summary = {
        "up_dir_total": up_dir_total,
        "up_dir_trend_win": up_dir_trend_win,
        "up_dir_revert_win": up_dir_revert_win,
        "up_no_touch": up_no_touch,
        "trend_win_up": (up_dir_trend_win / up_dir_total) if up_dir_total else 0.0,
        "revert_win_up": (up_dir_revert_win / up_dir_total) if up_dir_total else 0.0,
        "down_dir_total": down_dir_total,
        "down_dir_trend_win": down_dir_trend_win,
        "down_dir_revert_win": down_dir_revert_win,
        "down_no_touch": down_no_touch,
        "trend_win_down": (down_dir_trend_win / down_dir_total) if down_dir_total else 0.0,
        "revert_win_down": (down_dir_revert_win / down_dir_total) if down_dir_total else 0.0,
    }
    return daily_df, summary


def analyze_ny_open_k_scan(
    df: pd.DataFrame,
    k_list,
    session_end: str = "16:00",
    window_minutes: int = 90,
):
    """对称双目标，扫描多个 k；返回每个 k 的顺/反向首触胜率表。"""
    if df.empty:
        return pd.DataFrame()
    ts_utc = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.copy()
    df["ts_ny"] = ts_utc.dt.tz_convert("America/New_York")
    df = df.dropna(subset=["ts_ny"])
    try:
        se_hour, se_min = map(int, session_end.split(":"))
    except Exception:
        se_hour, se_min = 16, 0
    tz_ny = pytz.timezone("America/New_York")
    days = sorted({t.date() for t in df["ts_ny"] if t.weekday() < 5})

    stats = {k: {"up_total": 0, "up_trend": 0, "up_revert": 0, "up_none": 0,
                 "down_total": 0, "down_trend": 0, "down_revert": 0, "down_none": 0} for k in k_list}

    for day in days:
        start_ny = tz_ny.localize(datetime.combine(day, dtime(9, 30)))
        end90_ny = start_ny + timedelta(minutes=window_minutes)
        session_end_ny = tz_ny.localize(datetime.combine(day, dtime(se_hour, se_min)))
        win_mask = (df["ts_ny"] >= start_ny) & (df["ts_ny"] < end90_ny)
        after_mask = (df["ts_ny"] >= end90_ny) & (df["ts_ny"] <= session_end_ny)
        df_win = df[win_mask]
        df_after = df[after_mask]
        if df_win.empty or df_after.empty:
            continue
        O = df_win.iloc[0]["open"]
        H90 = df_win["high"].max()
        L90 = df_win["low"].min()
        C90 = df_win.iloc[-1]["close"]
        R = H90 - L90
        if R <= 0:
            continue
        if C90 > O:
            direction = "UP"
        elif C90 < O:
            direction = "DOWN"
        else:
            continue

        highs_after = df_after["high"].to_numpy()
        lows_after = df_after["low"].to_numpy()
        for k in k_list:
            up_target = O + k * R
            down_target = O - k * R
            first_hit = "NONE"
            for h, l in zip(highs_after, lows_after):
                if h >= up_target:
                    first_hit = "UP"
                    break
                if l <= down_target:
                    first_hit = "DOWN"
                    break
            st_k = stats[k]
            if direction == "UP":
                st_k["up_total"] += 1
                if first_hit == "UP":
                    st_k["up_trend"] += 1
                elif first_hit == "DOWN":
                    st_k["up_revert"] += 1
                else:
                    st_k["up_none"] += 1
            else:
                st_k["down_total"] += 1
                if first_hit == "DOWN":
                    st_k["down_trend"] += 1
                elif first_hit == "UP":
                    st_k["down_revert"] += 1
                else:
                    st_k["down_none"] += 1

    rows = []
    for k in k_list:
        st_k = stats[k]
        up_total = st_k["up_total"]
        down_total = st_k["down_total"]
        rows.append(
            {
                "k": k,
                "up_total": up_total,
                "up_trend": st_k["up_trend"],
                "up_revert": st_k["up_revert"],
                "up_none": st_k["up_none"],
                "trend_winrate_up": st_k["up_trend"] / up_total if up_total else 0.0,
                "revert_winrate_up": st_k["up_revert"] / up_total if up_total else 0.0,
                "down_total": down_total,
                "down_trend": st_k["down_trend"],
                "down_revert": st_k["down_revert"],
                "down_none": st_k["down_none"],
                "trend_winrate_down": st_k["down_trend"] / down_total if down_total else 0.0,
                "revert_winrate_down": st_k["down_revert"] / down_total if down_total else 0.0,
            }
        )
    return pd.DataFrame(rows)


def analyze_R90_vs_later_move(df: pd.DataFrame, session_end: str = "16:00", window_minutes: int = 90, bucket_count: int = 4):
    """
    不看方向，按 R90 大小分桶，统计 11:00 后最大上行/下行相对 O 的倍数。
    """
    if df.empty:
        return pd.DataFrame()
    ts_utc = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.copy()
    df["ts_ny"] = ts_utc.dt.tz_convert("America/New_York")
    df = df.dropna(subset=["ts_ny"])
    try:
        se_hour, se_min = map(int, session_end.split(":"))
    except Exception:
        se_hour, se_min = 16, 0
    tz_ny = pytz.timezone("America/New_York")
    days = sorted({t.date() for t in df["ts_ny"] if t.weekday() < 5})
    rows = []
    for day in days:
        start_ny = tz_ny.localize(datetime.combine(day, dtime(9, 30)))
        end90_ny = start_ny + timedelta(minutes=window_minutes)
        session_end_ny = tz_ny.localize(datetime.combine(day, dtime(se_hour, se_min)))
        win_mask = (df["ts_ny"] >= start_ny) & (df["ts_ny"] < end90_ny)
        after_mask = (df["ts_ny"] >= end90_ny) & (df["ts_ny"] <= session_end_ny)
        df_win = df[win_mask]
        df_after = df[after_mask]
        if df_win.empty or df_after.empty:
            continue
        O = df_win.iloc[0]["open"]
        H90 = df_win["high"].max()
        L90 = df_win["low"].min()
        C90 = df_win.iloc[-1]["close"]
        R = H90 - L90
        if R <= 0:
            continue
        max_up = (df_after["high"] - O).max()
        max_down = (df_after["low"] - O).min()
        rows.append(
            {
                "date_ny": day.isoformat(),
                "O": O,
                "H90": H90,
                "L90": L90,
                "C90": C90,
                "R90": R,
                "max_up_move": max_up,
                "max_down_move": max_down,
                "direction": "UP" if C90 > O else "DOWN" if C90 < O else "FLAT",
                "max_up_R": max_up / R if R else np.nan,
                "max_down_R": abs(max_down) / R if R else np.nan,
            }
        )
    if not rows:
        return pd.DataFrame()
    df_days = pd.DataFrame(rows)
    # 分桶：按 R90 量化 bucket_count 段
    try:
        quantiles = np.linspace(0, 1, bucket_count + 1)
        bins = df_days["R90"].quantile(quantiles).to_numpy()
        bins[0] = -np.inf
        bins[-1] = np.inf
    except Exception:
        return df_days
    bucket_labels = [f"Q{i}" for i in range(1, bucket_count + 1)]
    df_days["bucket"] = pd.cut(df_days["R90"], bins=bins, labels=bucket_labels, include_lowest=True)
    summary_rows = []
    for b in bucket_labels:
        sub = df_days[df_days["bucket"] == b]
        if sub.empty:
            continue
        sub_up = sub[sub["direction"] == "UP"]
        sub_down = sub[sub["direction"] == "DOWN"]
        summary_rows.append(
            {
                "bucket": b,
                "count": len(sub),
                "R90_min": sub["R90"].min(),
                "R90_max": sub["R90"].max(),
                "avg_max_up_R": sub["max_up_R"].mean(),
                "avg_max_down_R": sub["max_down_R"].mean(),
                "avg_max_up_R_updays": sub_up["max_up_R"].mean() if not sub_up.empty else np.nan,
                "avg_max_up_R_downdays": sub_down["max_up_R"].mean() if not sub_down.empty else np.nan,
                "avg_max_down_R_updays": sub_up["max_down_R"].mean() if not sub_up.empty else np.nan,
                "avg_max_down_R_downdays": sub_down["max_down_R"].mean() if not sub_down.empty else np.nan,
            }
        )
    return pd.DataFrame(summary_rows), df_days


def analyze_ny_open_expectation(
    df: pd.DataFrame,
    session_end: str = "16:00",
    window_minutes: int = 90,
    k: float = 1.0,
    fee_round: float = 0.0009,
):
    """按日模拟：9:30 O 入场，TP/SL = O ± k*R（顺/逆对称），收盘平仓；返回日明细和汇总。"""
    if df.empty:
        return pd.DataFrame(), {}
    ts_utc = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.copy()
    df["ts_ny"] = ts_utc.dt.tz_convert("America/New_York")
    df = df.dropna(subset=["ts_ny"])
    try:
        se_hour, se_min = map(int, session_end.split(":"))
    except Exception:
        se_hour, se_min = 16, 0
    tz_ny = pytz.timezone("America/New_York")
    days = sorted({t.date() for t in df["ts_ny"] if t.weekday() < 5})

    rows = []
    for day in days:
        start_ny = tz_ny.localize(datetime.combine(day, dtime(9, 30)))
        end90_ny = start_ny + timedelta(minutes=window_minutes)
        session_end_ny = tz_ny.localize(datetime.combine(day, dtime(se_hour, se_min)))
        win_mask = (df["ts_ny"] >= start_ny) & (df["ts_ny"] < end90_ny)
        after_mask = (df["ts_ny"] >= end90_ny) & (df["ts_ny"] <= session_end_ny)
        df_win = df[win_mask]
        df_after = df[after_mask]
        if df_win.empty or df_after.empty:
            continue
        O = df_win.iloc[0]["open"]
        H90 = df_win["high"].max()
        L90 = df_win["low"].min()
        C90 = df_win.iloc[-1]["close"]
        R = H90 - L90
        if R <= 0:
            continue
        if C90 > O:
            direction = "UP"
            tp = O + k * R
            sl = O - k * R
            label = "none"
            exit_price = df_after.iloc[-1]["close"]
            for _, r in df_after.iterrows():
                if r["high"] >= tp:
                    exit_price = tp
                    label = "tp"
                    break
                if r["low"] <= sl:
                    exit_price = sl
                    label = "sl"
                    break
            pnl_R = (exit_price - O) / R
        elif C90 < O:
            direction = "DOWN"
            tp = O - k * R
            sl = O + k * R
            label = "none"
            exit_price = df_after.iloc[-1]["close"]
            for _, r in df_after.iterrows():
                if r["low"] <= tp:
                    exit_price = tp
                    label = "tp"
                    break
                if r["high"] >= sl:
                    exit_price = sl
                    label = "sl"
                    break
            pnl_R = (O - exit_price) / R
        else:
            continue
        risk_pct = R / O if O else 0
        fee_R = fee_round / risk_pct if risk_pct else 0.0
        net_pnl_R = pnl_R - fee_R
        rows.append(
            {
                "date_ny": day.isoformat(),
                "O": O,
                "H90": H90,
                "L90": L90,
                "C90": C90,
                "R": R,
                "direction": direction,
                "tp": tp,
                "sl": sl,
                "pnl_R": pnl_R,
                "fee_R": fee_R,
                "net_pnl_R": net_pnl_R,
                "hit": label,
            }
        )
    if not rows:
        return pd.DataFrame(), {}
    df_days = pd.DataFrame(rows)
    def summarize(sub):
        if sub.empty:
            return {"count": 0, "avg": 0.0, "avg_net": 0.0, "median_net": 0.0, "max_dd": 0.0, "max_loss_streak": 0}
        cum = sub["net_pnl_R"].cumsum()
        peak = cum.cummax()
        dd = peak - cum
        max_dd = dd.max() if len(dd) else 0.0
        # 最大连亏
        max_streak = 0
        cur = 0
        for v in sub["net_pnl_R"]:
            if v < 0:
                cur += 1
                max_streak = max(max_streak, cur)
            else:
                cur = 0
        return {
            "count": len(sub),
            "avg": sub["pnl_R"].mean(),
            "avg_net": sub["net_pnl_R"].mean(),
            "median_net": sub["net_pnl_R"].median(),
            "max_dd": max_dd,
            "max_loss_streak": max_streak,
        }
    overall = summarize(df_days)
    up_summary = summarize(df_days[df_days["direction"] == "UP"])
    down_summary = summarize(df_days[df_days["direction"] == "DOWN"])
    summary = {"overall": overall, "up": up_summary, "down": down_summary}
    return df_days, summary


def _summarize_pnl(df_days: pd.DataFrame):
    if df_days.empty:
        return {"count": 0, "avg": 0.0, "avg_net": 0.0, "median_net": 0.0, "max_dd": 0.0, "max_loss_streak": 0}
    cum = df_days["net_pnl_R"].cumsum()
    peak = cum.cummax()
    dd = peak - cum
    max_dd = dd.max() if len(dd) else 0.0
    max_streak = 0
    cur = 0
    for v in df_days["net_pnl_R"]:
        if v < 0:
            cur += 1
            max_streak = max(max_streak, cur)
        else:
            cur = 0
    return {
        "count": len(df_days),
        "avg": df_days["pnl_R"].mean(),
        "avg_net": df_days["net_pnl_R"].mean(),
        "median_net": df_days["net_pnl_R"].median(),
        "max_dd": max_dd,
        "max_loss_streak": max_streak,
    }


def analyze_ny_no_future(
    df: pd.DataFrame,
    session_end: str = "16:00",
    window_minutes: int = 90,
    k: float = 1.0,
    fee_round: float = 0.0009,
    version: str = "A",
    tolerance_pct: float = 0.0005,
    bucket_mode: str = "all",
):
    """
    无未来函数版本：
    - 9:30-11:00 计算 O/H/L/C/R 和方向
    - 11:00 收盘后决定方向
    - 版本A：11:00 收盘价进场；TP/SL = C90 ± k*R（按方向）
    - 版本B：11:00 后回踩/反弹到 O(+/-tolerance) 再进场；TP/SL 基于 O±kR；没进场则无交易
    - 收盘强平，含手续费
    - bucket_mode: all / Q1_Q2 / Q3_Q4 根据 R 分位过滤
    """
    if df.empty:
        return pd.DataFrame(), {}
    ts_utc = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    df = df.copy()
    df["ts_ny"] = ts_utc.dt.tz_convert("America/New_York")
    df = df.dropna(subset=["ts_ny"])
    try:
        se_hour, se_min = map(int, session_end.split(":"))
    except Exception:
        se_hour, se_min = 16, 0
    tz_ny = pytz.timezone("America/New_York")
    days = sorted({t.date() for t in df["ts_ny"] if t.weekday() < 5})

    # 预先算 R 分位用于过滤
    r_list = []
    for day in days:
        start_ny = tz_ny.localize(datetime.combine(day, dtime(9, 30)))
        end90_ny = start_ny + timedelta(minutes=window_minutes)
        win_mask = (df["ts_ny"] >= start_ny) & (df["ts_ny"] < end90_ny)
        df_win = df[win_mask]
        if df_win.empty:
            continue
        R = df_win["high"].max() - df_win["low"].min()
        if R > 0:
            r_list.append(R)
    q1 = q2 = q3 = None
    if r_list:
        q1 = np.percentile(r_list, 25)
        q3 = np.percentile(r_list, 75)

    rows = []
    for day in days:
        start_ny = tz_ny.localize(datetime.combine(day, dtime(9, 30)))
        end90_ny = start_ny + timedelta(minutes=window_minutes)
        session_end_ny = tz_ny.localize(datetime.combine(day, dtime(se_hour, se_min)))
        win_mask = (df["ts_ny"] >= start_ny) & (df["ts_ny"] < end90_ny)
        after_mask = (df["ts_ny"] >= end90_ny) & (df["ts_ny"] <= session_end_ny)
        df_win = df[win_mask]
        df_after = df[after_mask]
        if df_win.empty or df_after.empty:
            continue
        O = df_win.iloc[0]["open"]
        H90 = df_win["high"].max()
        L90 = df_win["low"].min()
        C90 = df_win.iloc[-1]["close"]
        R = H90 - L90
        if R <= 0:
            continue
        # bucket filter
        if bucket_mode != "all" and q1 is not None and q3 is not None:
            if bucket_mode == "Q1_Q2" and R > q3:
                continue
            if bucket_mode == "Q3_Q4" and R < q1:
                continue
        if C90 > O:
            direction = "UP"
        elif C90 < O:
            direction = "DOWN"
        else:
            continue

        entry_price = None
        entry_ts_idx = None
        if version == "A":
            entry_price = C90
            entry_ts_idx = df_after.index[0]
            tp = C90 + k * R if direction == "UP" else C90 - k * R
            sl = C90 - k * R if direction == "UP" else C90 + k * R
        else:  # version B
            tp = O + k * R if direction == "UP" else O - k * R
            sl = O - k * R if direction == "UP" else O + k * R
            tol = tolerance_pct
            for idx, r in df_after.iterrows():
                if abs(r["close"] - O) / O <= tol:
                    entry_price = r["close"]
                    entry_ts_idx = idx
                    break
            if entry_price is None:
                # 当天未进场
                continue
        label = "none"
        exit_price = df_after.loc[entry_ts_idx, "close"]
        after_run = df_after.loc[df_after.index >= entry_ts_idx]
        for _, r in after_run.iterrows():
            if direction == "UP":
                hit_sl = r["low"] <= sl
                hit_tp = r["high"] >= tp
            else:
                hit_sl = r["high"] >= sl
                hit_tp = r["low"] <= tp
            if hit_tp and hit_sl:
                # 保守：先止损
                label = "sl"
                exit_price = sl
                break
            if hit_tp:
                label = "tp"
                exit_price = tp
                break
            if hit_sl:
                label = "sl"
                exit_price = sl
                break
        else:
            # 未触达，收盘价
            exit_price = after_run.iloc[-1]["close"]
            label = "none"

        pnl_R = (exit_price - entry_price) / R if direction == "UP" else (entry_price - exit_price) / R
        risk_pct = R / entry_price if entry_price else 0
        fee_R = fee_round / risk_pct if risk_pct else 0.0
        net_pnl_R = pnl_R - fee_R

        rows.append(
            {
                "date_ny": day.isoformat(),
                "O": O,
                "H90": H90,
                "L90": L90,
                "C90": C90,
                "R": R,
                "direction": direction,
                "entry": entry_price,
                "tp": tp,
                "sl": sl,
                "hit": label,
                "pnl_R": pnl_R,
                "fee_R": fee_R,
                "net_pnl_R": net_pnl_R,
            }
        )
    if not rows:
        return pd.DataFrame(), {}
    df_days = pd.DataFrame(rows)
    overall = _summarize_pnl(df_days)
    up_summary = _summarize_pnl(df_days[df_days["direction"] == "UP"])
    down_summary = _summarize_pnl(df_days[df_days["direction"] == "DOWN"])
    summary = {"overall": overall, "up": up_summary, "down": down_summary, "total_days": len(days), "traded_days": len(df_days)}
    return df_days, summary


def load_data_ui():
    for key in ["data_df", "trades_df", "summary", "strategy_name"]:
        st.session_state.setdefault(key, None)
    source = st.sidebar.radio("数据来源", ["上传 CSV", "币安下载"], key="data_source")
    df = st.session_state.get("data_df")
    if source == "上传 CSV":
        file = st.sidebar.file_uploader(
            "上传 CSV (需包含 timestamp,open,high,low,close,volume)", type=["csv"], key="upload_csv"
        )
        if file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                df = load_csv(tmp.name)
                st.session_state["data_df"] = df
                st.session_state["current_symbol"] = st.session_state.get("current_symbol", "未知")
                st.session_state["current_interval"] = st.session_state.get("current_interval", "")
    else:
        symbols = [
            "ETHUSDT",
            "DOGEUSDT",
            "BTCUSDT",
            "XRPUSDT",
            "LINKUSDT",
            "BNBUSDT",
            "TRXUSDT",
            "ADAUSDT",
            "SOLUSDT",
            "BCHUSDT",
            "ZECUSDT",
            "HYPEUSDT",
            "LEOUSDT",
            "XLMUSDT",
            "XMRUSDT",
            "LTCUSDT",
            "AVAXUSDT",
            "SUIUSDT",
            "SHIBUSDT",
            "PEPEUSDT",
        ]
        symbol_choice = st.sidebar.selectbox("常用交易对", symbols, index=0, key="dl_symbol_choice")
        symbol_custom = st.sidebar.text_input("自定义交易对", symbol_choice, key="dl_symbol_custom")
        symbol = symbol_custom.strip().upper() or symbol_choice
        st.sidebar.caption("示例：ETHUSDT，可下拉选择或直接输入任何支持的交易对")
        interval = st.sidebar.text_input("周期(如 5m/15m/1h/4h)", "1h", key="dl_interval")
        start = st.sidebar.text_input("开始时间 UTC", "2024-01-01 00:00:00", key="dl_start")
        end = st.sidebar.text_input("结束时间 UTC", "2024-02-01 00:00:00", key="dl_end")
        if st.sidebar.button("下载", key="dl_button"):
            try:
                df = download_binance_klines(symbol, interval, start, end, market_type="spot")
                st.session_state["data_df"] = df
                st.session_state["current_symbol"] = symbol
                st.session_state["current_interval"] = interval
                st.success(f"下载完成，{len(df)} 行")
            except Exception as e:
                st.error(f"下载失败: {e}")
    return st.session_state.get("data_df")


def parse_interval_seconds(interval: str) -> int:
    unit = interval[-1].lower()
    value = int(interval[:-1])
    if unit == "m":
        return value * 60
    if unit == "h":
        return value * 3600
    if unit == "d":
        return value * 86400
    raise ValueError(f"不支持的周期 {interval}")


def run_rsi_divergence(
    df: pd.DataFrame,
    lower_df=None,
    upper_interval_sec: int = 0,
    lower_interval_sec: int = 0,
    lower_fetch=None,
):
    st.header("RSI 背离策略")
    with st.sidebar.form(key="rsi_form"):
        rsi_period = st.number_input("RSI period", 5, 50, 14, key="rsi_period")
        overbought = st.number_input("Overbought", 50.0, 90.0, 70.0, key="rsi_overbought")
        oversold = st.number_input("Oversold", 10.0, 50.0, 30.0, key="rsi_oversold")
        lookback_bars = st.number_input("Lookback bars", 5, 100, 20, key="rsi_lookback")
        pivot_left = st.number_input("Pivot left", 1, 5, 2, key="rsi_pivot_left")
        min_rsi_diff = st.number_input("最小RSI差值", 0.0, 20.0, 3.0, key="rsi_min_diff")
        sl_mode = st.selectbox("SL模式", ["swing", "atr"], key="rsi_sl_mode")
        atr_period = st.number_input("ATR period", 5, 50, 14, key="rsi_atr_period")
        k_sl = st.number_input("k_sl (ATR倍数)", 0.5, 5.0, 1.5, key="rsi_k_sl")
        tp_R = st.number_input("TP R 倍数", 0.5, 10.0, 2.0, key="rsi_tp_R")
        fee_side = st.number_input("单边手续费(比例)", 0.0, 0.01, 0.00045, format="%.6f", key="rsi_fee_side")
        entry_slip_pct = st.number_input("开仓滑点比例(如0.0005=0.05%)", 0.0, 0.01, 0.0, format="%.5f", key="rsi_entry_slip")
        sl_buffer_pct = st.number_input("止损缓冲比例(如0.002=0.2%)", 0.0, 0.05, 0.0, format="%.4f", key="rsi_sl_buffer")
        wait_retest = st.checkbox("背离价位回踩触发", value=False, key="rsi_wait_retest")
        retest_expire = st.number_input("回踩有效期(根)", 1, 100, 10, key="rsi_retest_expire")
        wait_break_trigger = st.checkbox("第二次背离突破前一触发价后进场", value=False, key="rsi_wait_break")
        max_break_gap = st.number_input("触发价有效距离(根)", 1, 200, 50, key="rsi_break_gap")
        run = st.form_submit_button("运行回测", use_container_width=True)
    if not run:
        return None, None
    signals = detect_rsi_divergence_signals(
        df,
        rsi_period=rsi_period,
        overbought=overbought,
        oversold=oversold,
        lookback_bars=lookback_bars,
        pivot_left=pivot_left,
        pivot_right=0,
        min_rsi_diff=min_rsi_diff,
        sl_mode=sl_mode,
        atr_period=atr_period,
        k_sl=k_sl,
        tp_R=tp_R,
        wait_retest=wait_retest,
        retest_expire=retest_expire,
        wait_break_trigger=wait_break_trigger,
        max_break_gap=max_break_gap,
    )
    trades, _stats = simulate_basic(
        df,
        signals,
        fee_side_pct=fee_side,
        lower_df=lower_df,
        upper_interval_sec=upper_interval_sec,
        lower_interval_sec=lower_interval_sec,
        lower_fetch=lower_fetch,
        entry_slip_pct=entry_slip_pct,
        sl_buffer_pct=sl_buffer_pct,
    )
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades)
    return trades_df, summary


def run_alligator(
    df: pd.DataFrame,
    lower_df=None,
    upper_interval_sec: int = 0,
    lower_interval_sec: int = 0,
    lower_fetch=None,
):
    st.header("Alligator 趋势策略")
    with st.sidebar.form(key="allig_form"):
        jaw_period = st.number_input("Jaw period", 5, 30, 13, key="allig_jaw")
        teeth_period = st.number_input("Teeth period", 5, 20, 8, key="allig_teeth")
        lips_period = st.number_input("Lips period", 3, 15, 5, key="allig_lips")
        trend_confirm_bars = st.number_input("趋势确认根数", 1, 10, 3, key="allig_trend_confirm")
        entry_fresh_bars = st.number_input("入场新鲜度(根数)", 1, 20, 5, key="allig_entry_fresh")
        sl_mode = st.selectbox("SL模式", ["atr", "swing"], key="allig_sl_mode")
        atr_period = st.number_input("ATR period", 5, 50, 14, key="allig_atr_period")
        k_sl = st.number_input("k_sl (ATR倍数)", 0.5, 5.0, 1.5, key="allig_k_sl")
        tp_R = st.number_input("TP R 倍数", 0.5, 10.0, 2.0, key="allig_tp_R")
        fee_side = st.number_input("单边手续费(比例)", 0.0, 0.01, 0.00045, format="%.6f", key="allig_fee_side")
        run = st.form_submit_button("运行回测", use_container_width=True)
    if not run:
        return None, None
    sig_raw = detect_alligator_signals(
        df,
        jaw_period=jaw_period,
        teeth_period=teeth_period,
        lips_period=lips_period,
        trend_confirm_bars=trend_confirm_bars,
        entry_fresh_bars=entry_fresh_bars,
    )
    signals = prepare_sl_tp(
        df,
        sig_raw,
        sl_mode=sl_mode,
        atr_period=atr_period,
        k_sl=k_sl,
        tp_R=tp_R,
    )
    trades, _stats = simulate_basic(
        df,
        signals,
        fee_side_pct=fee_side,
        lower_df=lower_df,
        upper_interval_sec=upper_interval_sec,
        lower_interval_sec=lower_interval_sec,
        lower_fetch=lower_fetch,
    )
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades)
    return trades_df, summary


def run_breakout(df: pd.DataFrame, lower_fetch=None):
    st.header("趋势突破策略")
    with st.sidebar.form(key="breakout_form"):
        ema_span = st.number_input("EMA span", 10, 500, 100, key="br_ema")
        donchian_n = st.number_input("Donchian N", 5, 200, 24, key="br_donchian")
        atr_period = st.number_input("ATR period", 5, 100, 20, key="br_atr")
        k_buffer = st.number_input("ATR buffer倍数", 0.0, 5.0, 0.1, key="br_buf")
        vol_lookback = st.number_input("量均值窗口", 5, 200, 20, key="br_vol_lb")
        vol_mult = st.number_input("量倍数", 0.5, 5.0, 1.5, key="br_vol_mult")
        atr_median_lookback = st.number_input("ATR中位窗口", 10, 500, 100, key="br_atr_med")
        k_sl = st.number_input("k_sl(ATR倍数)", 0.1, 5.0, 1.5, key="br_k_sl")
        R_target = st.number_input("R_target", 0.5, 10.0, 3.0, key="br_r_target")
        k_trail = st.number_input("k_trail(ATR倍数)", 0.5, 5.0, 2.0, key="br_k_trail")
        fee_side = st.number_input("单边手续费(比例)", 0.0, 0.01, 0.00045, format="%.6f", key="br_fee")
        stop_loss_streak = st.number_input("连亏触发笔数", 0, 50, 0, key="br_streak")
        stop_duration_days = st.number_input("休息天数", 0, 365, 0, key="br_stop_days")
        run = st.form_submit_button("运行回测", use_container_width=True)
        entry_slip_pct = st.number_input("开仓滑点比例(如0.0005=0.05%)", 0.0, 0.01, 0.0, format="%.5f", key="br_entry_slip")
        sl_buffer_pct = st.number_input("止损缓冲比例(如0.002=0.2%)", 0.0, 0.05, 0.0, format="%.4f", key="br_sl_buffer")
    if not run:
        return None, None
    min_risk_pct = 0.001  # 固定的最小风险过滤

    signals = detect_breakouts(
        df,
        ema_span=ema_span,
        donchian_n=donchian_n,
        atr_period=atr_period,
        k_buffer=k_buffer,
        vol_lookback=vol_lookback,
        vol_mult=vol_mult,
        atr_median_lookback=atr_median_lookback,
    )
    trades = simulate_breakout(
        df,
        signals,
        k_sl=k_sl,
        R_target=R_target,
        k_trail=k_trail,
        fee_side=fee_side,
        lower_df=None,
        upper_interval_sec=0,
        lower_interval_sec=60,
        stop_loss_streak=stop_loss_streak,
        stop_duration_days=stop_duration_days,
        lower_fetch=lower_fetch,
        entry_slip_pct=entry_slip_pct,
        sl_buffer_pct=sl_buffer_pct,
        min_risk_pct=min_risk_pct,
    )
    if isinstance(trades, tuple):
        trades, _ = trades
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades, key="net_R")
    return trades_df, summary


def run_volume_ma_reversal(df: pd.DataFrame):
    st.header("5m 量20MA放量反转")
    with st.sidebar.form(key="volma_form"):
        vol_ma = st.number_input("量均线周期", 5, 200, 20, key="volma_ma")
        vol_mult = st.number_input("放量倍数", 0.1, 10.0, 2.0, key="volma_mult")
        shadow_ratio = st.number_input("影线占比上限(实体倍数)", 0.0, 5.0, 0.66, key="volma_shadow", format="%.2f")
        side_mode = st.selectbox("方向模式", ["reversal", "follow"], index=0, key="volma_side")
        exclude_us = st.checkbox("排除美股盘中(9:30-16:00 NY)", value=True, key="volma_exclude_us")
        fee_side = st.number_input("单边手续费(比例)", 0.0, 0.01, 0.00045, format="%.6f", key="volma_fee")
        run = st.form_submit_button("运行回测", use_container_width=True)
    if not run:
        return None, None
    signals = detect_volume_ma_signals(
        df,
        vol_ma=vol_ma,
        vol_mult=vol_mult,
        shadow_ratio=shadow_ratio,
        side_mode=side_mode,
        exclude_us_session=exclude_us,
    )
    trades, _stats = simulate_basic(df, signals, fee_side_pct=fee_side)
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades)
    return trades_df, summary


def run_ema220_breakout(df: pd.DataFrame):
    st.header("1h EMA220 趋势突破（多空）")
    with st.sidebar.form(key="ema220_form"):
        ema_len = st.number_input("EMA长度", 50, 400, 220, key="ema220_len")
        atr_len = st.number_input("ATR长度", 5, 50, 14, key="ema220_atr")
        lookback_box = st.number_input("箱体窗口", 20, 200, 80, key="ema220_box")
        vol_ma_len = st.number_input("量均线周期", 5, 200, 20, key="ema220_volma")
        volume_factor = st.number_input("放量倍数", 0.1, 5.0, 1.0, key="ema220_volfact")
        touch_atr = st.number_input("箱体边缘ATR倍数", 0.0, 2.0, 0.1, key="ema220_touch")
        near_high_frac = st.number_input("收盘靠近高点阈值", 0.05, 1.0, 0.2, key="ema220_near_high")
        dist_min = st.number_input("与EMA距离下限(ATR倍)", 0.0, 10.0, 0.0, key="ema220_dist_min")
        dist_max = st.number_input("与EMA距离上限(ATR倍)", 0.0, 10.0, 10.0, key="ema220_dist_max")
        slope_lookback = st.number_input("EMA斜率回看根数", 1, 200, 30, key="ema220_slope_look")
        slope_thresh = st.number_input("EMA斜率阈值(比例)", 0.0, 0.01, 0.0000, format="%.4f", key="ema220_slope_thr")
        env_window = st.number_input("黏均线窗口", 10, 200, 50, key="ema220_env_win")
        env_band_atr = st.number_input("黏均线带宽(ATR倍)", 0.0, 1.0, 0.5, key="ema220_env_band")
        env_ratio_max = st.number_input("黏均线比例上限", 0.0, 1.0, 0.7, key="ema220_env_ratio")
        fee_side = st.number_input("单边手续费(比例)", 0.0, 0.01, 0.00045, format="%.6f", key="ema220_fee")
        trailing_mode = st.selectbox("尾仓跟踪", ["ema50", "swing"], index=0, key="ema220_trail")
        swing_lookback = st.number_input("Swing跟踪回看", 2, 20, 5, key="ema220_swing")
        enable_short = st.checkbox("启用空头", value=True, key="ema220_short")
        trend_filter_mode = st.selectbox("趋势过滤", ["none", "ema_gate", "no_cross"], index=1, key="ema220_trend_mode")
        ema_gate_len = st.number_input("EMA门限长度", 50, 400, 220, key="ema220_gate_len")
        cross_lookback = st.number_input("交叉检查回看", 5, 100, 30, key="ema220_cross_look")
        run = st.form_submit_button("运行回测", use_container_width=True)
    if not run:
        return None, None
    signals = detect_ema220_signals(
        df,
        ema_len=ema_len,
        atr_len=atr_len,
        lookback_box=lookback_box,
        vol_ma_len=vol_ma_len,
        volume_factor=volume_factor,
        touch_atr=touch_atr,
        near_high_frac=near_high_frac,
        dist_min=dist_min,
        dist_max=dist_max,
        slope_lookback=slope_lookback,
        slope_thresh=slope_thresh,
        env_window=env_window,
        env_band_atr=env_band_atr,
        env_ratio_max=env_ratio_max,
        enable_short=enable_short,
        trend_filter_mode=trend_filter_mode,
        ema_gate_len=ema_gate_len,
        cross_lookback=cross_lookback,
    )
    trades = simulate_ema220_trades(
        df,
        signals,
        fee_side_pct=fee_side,
        trailing_mode=trailing_mode,
        swing_lookback=swing_lookback,
    )
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades, key="net_R")
    return trades_df, summary


def run_three_push(df: pd.DataFrame):
    st.header("5m 三推反转 + 4H EMA220 趋势过滤")
    with st.sidebar.form(key="tp_form"):
        atr_period = st.number_input("ATR周期", 5, 50, 14, key="tp_atr")
        swing_lookback = st.number_input("摆动识别左右根数", 1, 5, 2, key="tp_swing")
        three_push_max_lookback = st.number_input("三推最大跨度(根)", 50, 400, 200, key="tp_maxlook")
        max_push_slope_atr = st.number_input("三推斜率上限(ATR/根)", 0.01, 2.0, 0.5, key="tp_slope_limit")
        buffer_atr = st.number_input("触发/SL 缓冲ATR倍数", 0.0, 1.0, 0.1, key="tp_buffer")
        break_scan_forward = st.number_input("突破扫描前进(根)", 10, 200, 50, key="tp_scan")
        trend_lookback_4h = st.number_input("4H趋势回看根数", 1, 50, 5, key="tp_trend_look")
        trend_slope_threshold = st.number_input("4H趋势斜率阈值", 0.0, 0.01, 0.001, format="%.4f", key="tp_trend_slope")
        allow_trend_none = st.checkbox("允许趋势=none交易", value=False, key="tp_allow_none")
        enable_countertrend = st.checkbox("允许逆势", value=True, key="tp_counter")
        counter_factor = st.number_input("逆势仓位系数", 0.1, 1.0, 0.5, key="tp_counter_fac")
        avoid_chop = st.checkbox("震荡过滤", value=True, key="tp_chop_flag")
        chop_lookback = st.number_input("震荡窗口(根)", 5, 100, 20, key="tp_chop_win")
        chop_atr_factor = st.number_input("震荡阈值(ATR倍)", 0.5, 5.0, 1.5, key="tp_chop_fac")
        lookahead_bars = st.number_input("入场后观察根数", 5, 50, 20, key="tp_lookahead")
        impulse_R = st.number_input("direct_reversal阈值(R倍)", 0.5, 5.0, 2.0, key="tp_impulse")
        range_factor = st.number_input("range_chop阈值(R倍)", 0.5, 5.0, 1.5, key="tp_range")
        fee_side = st.number_input("单边手续费(比例)", 0.0, 0.01, 0.00045, format="%.6f", key="tp_fee")
        run = st.form_submit_button("运行回测", use_container_width=True)
    if not run:
        return None, None

    # 计算 5m ATR、摆动
    df5 = tp_find_swings(df.copy(), swing_lookback)
    df5["atr"] = tp_atr(df5, atr_period)

    # 生成 4H 数据：如果用户未提供，则尝试由 5m 重采样
    df5_ts = pd.to_datetime(df5["timestamp"], utc=True)
    df5_rs = df5.copy()
    df5_rs["timestamp"] = df5_ts
    df4h = (
        df5_rs.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        .resample("4H")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    params = ThreePushParams(
        atr_period=atr_period,
        swing_lookback=swing_lookback,
        three_push_max_lookback=three_push_max_lookback,
        buffer_atr=buffer_atr,
        break_scan_forward=break_scan_forward,
        trend_lookback_4h=trend_lookback_4h,
        trend_slope_threshold=trend_slope_threshold,
        allow_trend_none=allow_trend_none,
        enable_countertrend=enable_countertrend,
        countertrend_position_factor=counter_factor,
        avoid_trading_in_chop=avoid_chop,
        chop_lookback_bars=chop_lookback,
        chop_atr_factor=chop_atr_factor,
        lookahead_bars=lookahead_bars,
        impulse_R=impulse_R,
        range_factor=range_factor,
        fee_side=fee_side,
        max_push_slope_atr=max_push_slope_atr,
    )
    signals = generate_three_push_signals(df5, df4h, params)
    trades = simulate_three_push_trades(df5, signals, params)
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades, key="net_R")
    st.session_state["tp_trades"] = trades_df
    return trades_df, summary


def run_box_reversion(df: pd.DataFrame):
    st.header("5m 盒子震荡高抛低吸")
    with st.sidebar.form(key="box_form"):
        n_box = st.number_input("盒子窗口(根)", 20, 200, 80, key="box_n")
        atr_mult_box = st.number_input("盒子宽度上限(ATR倍)", 1.0, 10.0, 3.0, key="box_atr_mult")
        rsi_low = st.number_input("RSI低阈值", 10.0, 50.0, 35.0, key="box_rsi_low")
        rsi_high = st.number_input("RSI高阈值", 50.0, 90.0, 65.0, key="box_rsi_high")
        wick_body_ratio = st.number_input("影线/实体最小倍数", 0.5, 5.0, 1.2, key="box_wick_ratio")
        vol_mult = st.number_input("放量倍数上限", 0.1, 5.0, 2.0, key="box_vol_mult")
        adx_thresh = st.number_input("4H ADX阈值", 5.0, 40.0, 20.0, key="box_adx")
        ema_dev = st.number_input("4H偏离阈值(比例)", 0.0, 0.1, 0.03, format="%.3f", key="box_ema_dev")
        swing_lookback = st.number_input("摆动左右根数", 1, 5, 2, key="box_swing")
        run = st.form_submit_button("运行回测", use_container_width=True)
    if not run:
        return None, None
    # 需要4H数据：若未提供单独的4H，这里用5m重采样生成
    df5 = df.copy()
    df5["timestamp"] = pd.to_datetime(df5["timestamp"], utc=True)
    df4h = (
        df5.set_index("timestamp")[["open", "high", "low", "close", "volume"]]
        .resample("4H")
        .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
        .dropna()
        .reset_index()
    )
    signals = detect_box_signals(
        df5,
        df4h,
        n_box=n_box,
        atr_mult_box=atr_mult_box,
        rsi_low=rsi_low,
        rsi_high=rsi_high,
        wick_body_ratio=wick_body_ratio,
        vol_mult=vol_mult,
        adx_thresh=adx_thresh,
        ema_dev=ema_dev,
        swing_lookback=swing_lookback,
    )
    trades = simulate_box_trades(df5, signals)
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades, key="net_R")
    return trades_df, summary


def run_ct_divergence(df: pd.DataFrame):
    st.header("4H 下跌段底背离对冲（B1/B2 + 6根站上确认）")
    with st.sidebar.form(key="ct_form"):
        rsi_period = st.number_input("RSI周期", 5, 50, 14, key="ct_rsi")
        oversold = st.number_input("RSI超卖", 5.0, 60.0, 30.0, key="ct_os")
        lookback_bars = st.number_input("背离 lookback(根)", 10, 200, 50, key="ct_lb")
        pivot_left = st.number_input("Pivot 左", 1, 5, 2, key="ct_pivot_left")
        min_rsi_diff = st.number_input("最小 RSI 差值", 0.0, 20.0, 3.0, key="ct_rsi_diff")
        atr_period = st.number_input("ATR周期", 5, 50, 14, key="ct_atr")
        k_sl = st.number_input("SL ATR倍数", 0.5, 5.0, 1.5, key="ct_k_sl")
        tp_R = st.number_input("TP R倍数", 0.5, 5.0, 1.5, key="ct_tpR")
        confirm_bars = st.number_input("站上确认根数", 1, 20, 6, key="ct_confirm_bars")
        confirm_min_pct = st.number_input("确认上方最小偏移(比例)", 0.0, 0.01, 0.001, format="%.4f", key="ct_confirm_pct")
        downleg_gap = st.number_input("同段最大间隔(根，5m)", 10, 500, 168, key="ct_down_gap")
        retest_tol = st.number_input("回踩触发容忍(比例)", 0.0, 0.01, 0.001, format="%.4f", key="ct_retest_tol")
        fee_side = st.number_input("单边手续费(比例)", 0.0, 0.01, 0.00045, format="%.6f", key="ct_fee")
        run = st.form_submit_button("运行回测", use_container_width=True)
    if not run:
        return None, None

    cfg = CTDivergenceConfig(
        rsi_period=rsi_period,
        oversold=oversold,
        lookback_bars=lookback_bars,
        pivot_left=pivot_left,
        pivot_right=0,
        min_rsi_diff=min_rsi_diff,
        atr_period=atr_period,
        k_sl=k_sl,
        tp_R=tp_R,
        confirm_bars=confirm_bars,
        confirm_min_pct=confirm_min_pct,
        downleg_gap_bars_5m=downleg_gap,
        retest_tol_pct=retest_tol,
    )
    signals = detect_ct_signals(df, cfg)
    trades, _stats = simulate_basic(df, signals, fee_side_pct=fee_side)
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades, key="net_R")
    return trades_df, summary


def run_bear_combo(df: pd.DataFrame):
    st.header("熊市顺势空 + 对冲多（B1/B2）")
    with st.sidebar.form(key="bear_form"):
        ema_fast = st.number_input("4H EMA快", 5, 200, 20, key="bear_ema_fast")
        ema_slow = st.number_input("4H EMA慢", 10, 400, 60, key="bear_ema_slow")
        ema_bias = st.number_input("空头判定偏移", 0.95, 1.0, 0.998, format="%.4f", key="bear_ema_bias")
        trend_timeframe = st.selectbox("趋势级别", ["1H", "4H"], index=0, key="bear_trend_tf")
        trend_method = st.selectbox("趋势判定", ["ema", "donchian", "regression", "swing"], index=0, key="bear_trend_method")
        trend_price_tol = st.number_input("趋势价差容差(比例)", 0.0, 0.01, 0.002, format="%.4f", key="bear_trend_tol")
        trend_pivot_lb = st.number_input("趋势枢轴左右根(结构法用)", 1, 5, 2, key="bear_trend_pivot")
        allow_non_bear = st.checkbox("非bear趋势也运行", value=True, key="bear_allow_non_bear")
        rsi_period = st.number_input("RSI周期", 5, 50, 14, key="bear_rsi")
        swing_left = st.number_input("Swing 左右根", 1, 5, 2, key="bear_swing_left")
        div_price_tol = st.number_input("顶背离价差(比例)", 0.0, 0.01, 0.001, format="%.4f", key="bear_price_tol")
        div_rsi_diff = st.number_input("顶背离RSI差值", 0.0, 10.0, 2.0, key="bear_rsi_diff")
        lookback_div = st.number_input("背离最大跨度(根)", 20, 400, 200, key="bear_div_lb")
        short_sl_buf = st.number_input("空单SL缓冲(比例)", 0.0, 0.01, 0.001, format="%.4f", key="bear_sl_buf")
        short_tp_R = st.number_input("空单TP R倍数", 0.1, 5.0, 2.0, key="bear_tpR")
        enable_fallback = st.checkbox("开启无背离破前低做空", value=True, key="bear_fallback")
        fee_side = st.number_input("单边手续费(比例)", 0.0, 0.01, 0.00045, format="%.6f", key="bear_fee")
        # 对冲参数（简化版）
        hedge_tpR = st.number_input("对冲多 TP R倍数", 0.1, 5.0, 1.5, key="hedge_tpR")
        hedge_confirm = st.number_input("站上确认根数", 1, 20, 6, key="hedge_confirm")
        hedge_confirm_pct = st.number_input("站上最小偏移(比例)", 0.0, 0.01, 0.001, format="%.4f", key="hedge_confirm_pct")
        hedge_gap = st.number_input("同段间隔(根,5m)", 10, 500, 168, key="hedge_gap")
        run = st.form_submit_button("运行回测", use_container_width=True)
    if not run:
        return None, None, None

    cfg = BearComboConfig(
        ema_fast=ema_fast,
        ema_slow=ema_slow,
        ema_bias_bear=ema_bias,
        rsi_period=rsi_period,
        swing_left=swing_left,
        swing_right=0,
        div_price_tolerance=div_price_tol,
        div_rsi_diff=div_rsi_diff,
        lookback_div=lookback_div,
        short_sl_buffer_pct=short_sl_buf,
        short_tp_R=short_tp_R,
        enable_fallback_break=enable_fallback,
        trend_timeframe=trend_timeframe,
        trend_price_tol=trend_price_tol,
        trend_pivot_lookback=trend_pivot_lb,
        trend_method=trend_method,
    )
    hedge_cfg = CTDivergenceConfig(
        tp_R=hedge_tpR,
        confirm_bars=hedge_confirm,
        confirm_min_pct=hedge_confirm_pct,
        downleg_gap_bars_5m=hedge_gap,
    )
    signals, stats = detect_bear_combo_signals(df, cfg, hedge_cfg)
    if stats.get("trend_latest") != "bear" and not allow_non_bear:
        st.info(f"当前 {trend_timeframe} 趋势={stats.get('trend_latest')}，暂不运行熊市空+对冲策略")
        return None, None, stats
    trades, _stats = simulate_basic(df, signals, fee_side_pct=fee_side)
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades, key="net_R")
    summary.update(stats)
    return trades_df, summary, stats


def detect_turtle_signals(
    df: pd.DataFrame,
    n_entry: int = 20,
    n_exit: int = 10,
    atr_period: int = 20,
    risk_atr: float = 2.0,
    reverse: bool = False,
):
    hh = df["high"].rolling(n_entry, min_periods=n_entry).max().shift(1)
    ll = df["low"].rolling(n_entry, min_periods=n_entry).min().shift(1)
    atr_series = atr(df, atr_period)
    signals = []
    for i in range(len(df)):
        if i == 0:
            continue
        price = df.at[i, "close"]
        dist = risk_atr * atr_series.iloc[i]
        if dist <= 0 or pd.isna(dist):
            continue
        if not pd.isna(hh.iloc[i]) and price > hh.iloc[i]:
            if reverse:
                sl = price - dist  # 反向：原 SL 变 TP，SL/TP 对调
                tp = price + dist
                signals.append({"idx": i, "side": "short", "entry": price, "sl": sl, "tp": tp})
            else:
                sl = price - dist
                tp = price + dist
                signals.append({"idx": i, "side": "long", "entry": price, "sl": sl, "tp": tp})
        elif not pd.isna(ll.iloc[i]) and price < ll.iloc[i]:
            if reverse:
                sl = price + dist
                tp = price - dist
                signals.append({"idx": i, "side": "long", "entry": price, "sl": sl, "tp": tp})
            else:
                sl = price + dist
                tp = price - dist
                signals.append({"idx": i, "side": "short", "entry": price, "sl": sl, "tp": tp})
    return signals


def run_turtle(df: pd.DataFrame):
    st.header("海龟策略（唐奇安突破）")
    with st.sidebar.form(key="turtle_form"):
        n_entry = st.number_input("突破窗口 N1", 5, 200, 20, key="turtle_n1")
        n_exit = st.number_input("退出窗口 N2", 5, 200, 10, key="turtle_n2")
        atr_period = st.number_input("ATR 周期", 5, 100, 20, key="turtle_atr")
        risk_atr = st.number_input("SL ATR 倍数", 0.1, 10.0, 2.0, key="turtle_risk_atr")
        fee_side = st.number_input("单边手续费(比例)", 0.0, 0.01, 0.00045, format="%.6f", key="turtle_fee")
        entry_slip_pct = st.number_input("开仓滑点比例", 0.0, 0.01, 0.0, format="%.5f", key="turtle_slip")
        sl_buffer_pct = st.number_input("止损缓冲比例", 0.0, 0.05, 0.0, format="%.4f", key="turtle_slbuf")
        reverse = st.checkbox("反向交易（突破做反向）", value=False, key="turtle_reverse")
        run = st.form_submit_button("运行回测", use_container_width=True)
    if not run:
        return None, None
    signals = detect_turtle_signals(
        df,
        n_entry=n_entry,
        n_exit=n_exit,
        atr_period=atr_period,
        risk_atr=risk_atr,
        reverse=reverse,
    )
    trades, _stats = simulate_basic(
        df,
        signals,
        fee_side_pct=fee_side,
        entry_slip_pct=entry_slip_pct,
        sl_buffer_pct=sl_buffer_pct,
    )
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades, key="net_R")
    return trades_df, summary


def run_pattern_detection(df: pd.DataFrame):
    st.header("市场形态识别（窄/宽震荡、通道、楔形、三角、突破）")
    with st.sidebar.form(key="pattern_form"):
        window_range = st.number_input("震荡窗口", 20, 200, 50, key="pat_range_win")
        window_channel = st.number_input("通道窗口", 20, 200, 40, key="pat_channel_win")
        window_pattern = st.number_input("楔形/三角窗口", 20, 300, 60, key="pat_pattern_win")
        range_max_slope_atr = st.number_input("震荡斜率上限(ATR/根)", 0.01, 1.0, 0.1, format="%.3f", key="pat_range_slope")
        range_mid_band_ratio = st.number_input("震荡中轴带比例", 0.05, 0.8, 0.3, format="%.2f", key="pat_range_band")
        range_mid_band_min_ratio = st.number_input("中轴命中比例下限", 0.1, 1.0, 0.6, format="%.2f", key="pat_mid_ratio")
        range_narrow_max_span_atr = st.number_input("窄震荡最大高度(ATR)", 0.2, 5.0, 2.0, key="pat_range_narrow")
        range_wide_min_span_atr = st.number_input("宽震荡最小高度(ATR)", 0.2, 6.0, 2.0, key="pat_range_wide_min")
        range_wide_max_span_atr = st.number_input("宽震荡最大高度(ATR)", 0.5, 10.0, 5.0, key="pat_range_wide_max")
        channel_min_slope_atr = st.number_input("通道最小斜率(ATR/根)", 0.01, 1.0, 0.15, format="%.3f", key="pat_chan_slope")
        channel_narrow_width_atr = st.number_input("窄通道宽度上限(ATR)", 0.5, 6.0, 2.0, key="pat_chan_narrow")
        channel_wide_width_atr = st.number_input("宽通道宽度上限(ATR)", 1.0, 10.0, 4.0, key="pat_chan_wide")
        channel_min_inliers_ratio = st.number_input("通道内点比例", 0.1, 1.0, 0.7, format="%.2f", key="pat_chan_inlier")
        breakout_buffer_atr = st.number_input("突破缓冲(ATR倍)", 0.0, 2.0, 0.3, key="pat_br_buf")
        breakout_confirm_bars = st.number_input("突破确认根数", 1, 5, 1, key="pat_br_confirm")
        breakout_min_body_atr = st.number_input("突破实体下限(ATR)", 0.0, 3.0, 0.5, key="pat_br_body")
        breakout_volume_factor = st.number_input("突破放量倍数", 0.5, 5.0, 1.5, key="pat_br_vol")
        swing_lookback = st.number_input("摆动左右根数", 1, 10, 3, key="pat_swing")
        pattern_min_swings = st.number_input("楔/三角最少拐点数", 2, 10, 4, key="pat_min_swings")
        convergence_min_ratio = st.number_input("收敛比例(终距/起距下降至少)", 0.05, 0.9, 0.3, format="%.2f", key="pat_converge")
        pattern_min_length = st.number_input("形态最短持续(根)", 10, 200, 20, key="pat_min_len")
        smoothing_window = st.number_input("标签平滑窗口(仅用过去)", 1, 10, 1, key="pat_smooth")
        perf_horizon = st.number_input("未来收益评估窗口(根)", 1, 50, 5, key="pat_perf_hor")
        run = st.form_submit_button("运行形态识别", use_container_width=True)
    if not run and "pattern_df" not in st.session_state:
        st.info("调整参数后点击“运行形态识别”")
        return

    if run:
        cfg = PatternConfig(
            window_range=window_range,
            window_channel=window_channel,
            window_pattern=window_pattern,
            range_max_slope_atr=range_max_slope_atr,
            range_mid_band_ratio=range_mid_band_ratio,
            range_mid_band_min_ratio=range_mid_band_min_ratio,
            range_narrow_max_span_atr=range_narrow_max_span_atr,
            range_wide_min_span_atr=range_wide_min_span_atr,
            range_wide_max_span_atr=range_wide_max_span_atr,
            channel_min_slope_atr=channel_min_slope_atr,
            channel_narrow_width_atr=channel_narrow_width_atr,
            channel_wide_width_atr=channel_wide_width_atr,
            channel_min_inliers_ratio=channel_min_inliers_ratio,
            breakout_buffer_atr=breakout_buffer_atr,
            breakout_confirm_bars=breakout_confirm_bars,
            breakout_min_body_atr=breakout_min_body_atr,
            breakout_volume_factor=breakout_volume_factor,
            swing_lookback=swing_lookback,
            pattern_min_swings=pattern_min_swings,
            convergence_min_ratio=convergence_min_ratio,
            pattern_min_length=pattern_min_length,
            smoothing_window=smoothing_window,
        )
        classifier = PatternClassifier(cfg)
        pattern_df = classifier.classify(df.copy())
        st.session_state["pattern_df"] = pattern_df
        st.session_state["pattern_cfg"] = cfg
    else:
        pattern_df = st.session_state.get("pattern_df")
        cfg = st.session_state.get("pattern_cfg")
        if pattern_df is None:
            st.info("暂无形态结果，请先运行")
            return

    st.subheader("形态分布")
    counts = pattern_counts(pattern_df)
    st.dataframe(counts.rename("count"))

    st.subheader(f"未来 {perf_horizon} 根收益（简单收盘涨跌）")
    perf_df = pattern_performance(pattern_df, horizon=perf_horizon)
    st.dataframe(perf_df)

    st.subheader("最近形态标签（尾部50行）")
    st.dataframe(pattern_df[["timestamp", "pattern"]].tail(50))

    pattern_options = ["全部"] + counts.index.tolist()
    sel_pattern = st.selectbox("选择形态查看示例", pattern_options, key="pat_select")
    window_show = st.number_input("展示窗口(根)", 50, 400, 200, key="pat_win_show")
    filt = None if sel_pattern == "全部" else sel_pattern
    plot_pattern_chart(pattern_df, f"{sel_pattern} 示例", pattern_filter=filt, window=window_show, show_overlays=True, show_arrows=False)

    if sel_pattern != "全部":
        sample_key = f"pat_sample_used_{sel_pattern}"
        last_samples_key = f"pat_last_samples_{sel_pattern}"
        candidates = [i for i, v in enumerate(pattern_df["pattern"]) if v == sel_pattern]
        if not candidates:
            st.info(f"未找到 {sel_pattern} 可供取样")
            return
        if st.button("随机取样5个（不重复）", key=f"btn_sample_{sel_pattern}"):
            used = set(st.session_state.get(sample_key, []))
            available = [i for i in candidates if i not in used]
            if len(available) < 5:
                # 重置使用记录，重新开始
                used = set()
                available = candidates
            sample_count = min(5, len(available))
            picked = random.sample(available, sample_count)
            used.update(picked)
            st.session_state[sample_key] = list(used)
            st.session_state[last_samples_key] = picked

        samples = st.session_state.get(last_samples_key, [])
        if samples:
            st.subheader(f"{sel_pattern} 随机样本（{len(samples)}个）")
            for idx in samples:
                ep_side = pattern_df.at[idx, "pattern"]
                price_ep = pattern_df.at[idx, "close"] if "close" in pattern_df.columns else None
                plot_pattern_chart(
                    pattern_df,
                    f"{sel_pattern} 样本 idx={idx}",
                    window=window_show,
                    target_idx=idx,
                    show_overlays=False,
                    show_arrows=True,
                    entry_point=(idx, price_ep, "long" if "UP" in ep_side or "BULL" in ep_side else "short"),
                )
        else:
            st.caption("点击上方按钮生成随机样本")

    st.markdown("---")
    st.subheader("三推 -> 二推突破 取样（反转测试）")
    lookahead = st.number_input("向后扫描根数", 5, 200, 30, key="pat_second_break_look")
    sample_n = st.number_input("取样数量", 1, 20, 5, key="pat_second_break_n")
    if st.button("生成二推突破样本", key="btn_second_break"):
        candidates = []
        idx_array = pattern_df.index.to_list()
        for pos, idx in enumerate(idx_array):
            pat = pattern_df.at[idx, "pattern"]
            if pat not in ["THREE_PUSH_RISING_WEDGE", "THREE_PUSH_FALLING_WEDGE"]:
                continue
            highs_list = pattern_df.at[idx, "wedge_push_highs"]
            lows_list = pattern_df.at[idx, "wedge_push_lows"]
            if not highs_list or not lows_list:
                continue
            if pat == "THREE_PUSH_RISING_WEDGE":
                if len(lows_list) < 2:
                    continue
                entry_level = pattern_df.iloc[lows_list[1]]["low"] if lows_list[1] < len(pattern_df) else None
                if entry_level is None:
                    continue
                trigger_idx = None
                for j in range(pos + 1, min(len(idx_array), pos + 1 + lookahead)):
                    if pattern_df.iloc[j]["close"] < entry_level:
                        trigger_idx = idx_array[j]
                        break
                if trigger_idx is None:
                    continue
                candidates.append((trigger_idx, entry_level, idx, "down"))
            else:  # FALLING
                if len(highs_list) < 2:
                    continue
                entry_level = pattern_df.iloc[highs_list[1]]["high"] if highs_list[1] < len(pattern_df) else None
                if entry_level is None:
                    continue
                trigger_idx = None
                for j in range(pos + 1, min(len(idx_array), pos + 1 + lookahead)):
                    if pattern_df.iloc[j]["close"] > entry_level:
                        trigger_idx = idx_array[j]
                        break
                if trigger_idx is None:
                    continue
                candidates.append((trigger_idx, entry_level, idx, "up"))
        if not candidates:
            st.info("未找到满足条件的样本")
        else:
            picks = random.sample(candidates, k=min(sample_n, len(candidates)))
            st.session_state["second_break_samples"] = picks

    if "second_break_samples" in st.session_state:
        samples = st.session_state["second_break_samples"]
        if samples:
            st.subheader("二推突破样本")
            for trig_idx, entry_level, wedge_idx, direction in samples:
                plot_pattern_chart(
                    pattern_df,
                    f"Trigger idx={trig_idx} (from wedge {wedge_idx})",
                    window=window_show,
                    target_idx=trig_idx,
                    show_overlays=False,
                    show_arrows=True,
                    entry_line=entry_level,
                    entry_idx=pattern_df.index.get_loc(trig_idx),
                    entry_point=(trig_idx, entry_level, "long" if direction == "up" else "short"),
                )

    st.markdown("---")
    st.subheader("窄幅震荡 -> 突破 取样")
    nb_lookahead = st.number_input("向后扫描根数(窄震荡后找突破)", 5, 200, 40, key="pat_narrow_break_look")
    nb_sample_n = st.number_input("取样数量", 1, 20, 5, key="pat_narrow_break_n")
    if st.button("生成窄震荡突破样本", key="btn_narrow_break"):
        candidates = []
        idx_array = pattern_df.index.to_list()
        for pos, idx in enumerate(idx_array):
            if pattern_df.at[idx, "pattern"] != "RANGE_NARROW":
                continue
            rh = pattern_df.at[idx, "range_high"] if "range_high" in pattern_df.columns else None
            rl = pattern_df.at[idx, "range_low"] if "range_low" in pattern_df.columns else None
            for j in range(pos + 1, min(len(idx_array), pos + 1 + nb_lookahead)):
                p = pattern_df.at[idx_array[j], "pattern"]
                if p in ["BULL_BREAKOUT", "THREE_PUSH_BULL_BREAKOUT"]:
                    candidates.append((idx_array[j], "up", rh, rl, idx))
                    break
                if p in ["BEAR_BREAKOUT", "THREE_PUSH_BEAR_BREAKOUT"]:
                    candidates.append((idx_array[j], "down", rh, rl, idx))
                    break
        if not candidates:
            st.info("未找到满足条件的样本")
        else:
            picks = random.sample(candidates, k=min(nb_sample_n, len(candidates)))
            st.session_state["narrow_break_samples"] = picks

    if "narrow_break_samples" in st.session_state:
        samples = st.session_state["narrow_break_samples"]
        if samples:
            st.subheader("窄震荡突破样本")
            for b_idx, direction, rh, rl, r_idx in samples:
                entry_price = pattern_df.at[b_idx, "close"] if "close" in pattern_df.columns else None
                entry_line = rh if direction == "up" else rl
                plot_pattern_chart(
                    pattern_df,
                    f"Break idx={b_idx} (from range {r_idx})",
                    window=window_show,
                    target_idx=b_idx,
                    show_overlays=True,
                    show_arrows=True,
                    entry_line=entry_line,
                    entry_point=(b_idx, entry_price, "long" if direction == "up" else "short"),
                )


def sign_params(params, secret):
    qs = "&".join(f"{k}={params[k]}" for k in sorted(params))
    sig = hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params


def binance_request(method, path, api_key, api_secret, base_url, params=None, signed=False):
    params = params or {}
    if signed:
        params["timestamp"] = int(time.time() * 1000)
        params = sign_params(params, api_secret)
    headers = {"X-MBX-APIKEY": api_key}
    url = base_url + path
    resp = requests.request(method, url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def run_binance_panel():
    st.header("Binance 永续下单（默认测试网）")
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        st.error("请先在环境变量设置 BINANCE_API_KEY / BINANCE_API_SECRET")
        return
    with st.sidebar.form(key="binance_form"):
        symbol = st.text_input("交易对", "BTCUSDT", key="bn_symbol")
        action = st.selectbox("操作", ["price", "balance", "order", "cancel"], key="bn_action")
        side = st.selectbox("方向", ["BUY", "SELL"], key="bn_side")
        order_type = st.selectbox("类型", ["MARKET", "LIMIT"], key="bn_type")
        qty = st.number_input("数量", min_value=0.0, value=0.001, format="%.6f", key="bn_qty")
        price = st.number_input("限价(限价单必填)", min_value=0.0, value=0.0, format="%.2f", key="bn_price")
        live = st.checkbox("使用主网(谨慎)", value=False, key="bn_live")
        submitted = st.form_submit_button("执行", use_container_width=True)
    if not submitted:
        return
    base_url = "https://fapi.binance.com" if live else "https://testnet.binancefuture.com"
    try:
        if action == "price":
            data = binance_request("GET", "/fapi/v1/ticker/price", api_key, api_secret, base_url, params={"symbol": symbol})
        elif action == "balance":
            data = binance_request("GET", "/fapi/v2/balance", api_key, api_secret, base_url, signed=True)
        elif action == "order":
            payload = {
                "symbol": symbol,
                "side": side,
                "type": order_type,
                "quantity": qty,
                "newClientOrderId": f"web_{int(time.time()*1000)}",
            }
            if order_type == "LIMIT":
                if price <= 0:
                    st.error("限价单需填写 price")
                    return
                payload.update({"price": price, "timeInForce": "GTC"})
            data = binance_request("POST", "/fapi/v1/order", api_key, api_secret, base_url, params=payload, signed=True)
        else:  # cancel
            data = binance_request("DELETE", "/fapi/v1/allOpenOrders", api_key, api_secret, base_url, params={"symbol": symbol}, signed=True)
        st.success("执行成功")
        st.json(data)
    except Exception as e:
        st.error(f"执行失败: {e}")


def main():
    st.set_page_config(page_title="RSI 背离 + Alligator 回测", layout="wide")
    df = load_data_ui()
    if df is None or df.empty:
        st.info("请先加载数据")
        return
    df = ensure_ohlcv_df(df)
    # 通用下行周期设置
    lower_interval = st.sidebar.text_input("下行周期(如1m，可空)", "", key="lower_global")
    auto_lower = st.sidebar.checkbox("无下行周期时自动下载1m判顺序", value=False, key="auto_lower_global")
    lower_market = st.sidebar.selectbox("下行数据市场类型", ["spot", "usdt_perp", "coin_perp", "usdc_perp"], index=1, key="lower_market")
    current_symbol = st.session_state.get("current_symbol", "ETHUSDT")
    follow_symbol = st.sidebar.checkbox("下行数据交易对跟随上方交易对", value=True, key="lower_follow")
    lower_symbol_default = current_symbol if follow_symbol else current_symbol
    lower_symbol = st.sidebar.text_input("下行数据交易对", lower_symbol_default, key="lower_symbol")

    def build_lower_fetch():
        if not (lower_interval or auto_lower):
            return None
        start_ts = pd.to_datetime(df["timestamp"].iloc[0])
        end_ts = pd.to_datetime(df["timestamp"].iloc[-1])
        li = lower_interval if lower_interval else "1m"

        def _fetch():
            return download_binance_klines(
                lower_symbol,
                li,
                start_ts.strftime("%Y-%m-%d %H:%M:%S"),
                end_ts.strftime("%Y-%m-%d %H:%M:%S"),
                market_type=lower_market,
            )

        return _fetch

    lower_fetch = build_lower_fetch()
    lower_df = None  # 不预下载，只在冲突时由 lower_fetch 拉取

    # 估计主周期秒数
    ts_main = pd.to_datetime(df["timestamp"])
    diffs_main = ts_main.diff().dropna().dt.total_seconds()
    upper_interval_sec = int(diffs_main.median()) if not diffs_main.empty else 0
    # 推断周期标签
    interval_label = st.session_state.get("current_interval", "")
    if not interval_label:
        if upper_interval_sec % 3600 == 0 and upper_interval_sec > 0:
            interval_label = f"{int(upper_interval_sec/3600)}h"
        elif upper_interval_sec % 60 == 0 and upper_interval_sec > 0:
            interval_label = f"{int(upper_interval_sec/60)}m"
        elif upper_interval_sec > 0:
            interval_label = f"{upper_interval_sec}s"
        else:
            interval_label = "未知"

    strategy = st.sidebar.radio(
        "策略",
        [
            "RSI 背离",
            "Alligator",
            "Breakout",
            "形态识别",
            "4H 底背离对冲",
            "熊市顺势空+对冲",
            "海龟策略",
            "NY开盘2R统计",
            "量MA反转",
            "EMA220突破",
            "三推EMA220",
            "盒子震荡",
            "Binance下单",
        ],
        key="strategy_choice",
    )
    trades_df = None
    summary = None
    daily_df = None
    if strategy == "RSI 背离":
        trades_df, summary = run_rsi_divergence(
            df,
            lower_df=lower_df,
            upper_interval_sec=upper_interval_sec,
            lower_interval_sec=int(pd.to_datetime(lower_df["timestamp"]).diff().dt.total_seconds().median()) if lower_df is not None else 0,
            lower_fetch=lower_fetch,
        )
    elif strategy == "Alligator":
        trades_df, summary = run_alligator(
            df,
            lower_df=lower_df,
            upper_interval_sec=upper_interval_sec,
            lower_interval_sec=int(pd.to_datetime(lower_df["timestamp"]).diff().dt.total_seconds().median()) if lower_df is not None else 0,
            lower_fetch=lower_fetch,
        )
    elif strategy == "Breakout":
        trades_df, summary = run_breakout(df, lower_fetch=lower_fetch)
    elif strategy == "海龟策略":
        trades_df, summary = run_turtle(df)
    elif strategy == "4H 底背离对冲":
        trades_df, summary = run_ct_divergence(df)
    elif strategy == "熊市顺势空+对冲":
        trades_df, summary, stats_extra = run_bear_combo(df)
        if trades_df is not None and summary is not None:
            st.info(f"空单信号: {summary.get('short_signals', 0)} / 对冲多信号: {summary.get('hedge_long_signals', 0)}")
    elif strategy == "形态识别":
        run_pattern_detection(df)
        return
    elif strategy == "量MA反转":
        trades_df, summary = run_volume_ma_reversal(df)
    elif strategy == "EMA220突破":
        trades_df, summary = run_ema220_breakout(df)
    elif strategy == "三推EMA220":
        trades_df, summary = run_three_push(df)
    elif strategy == "盒子震荡":
        trades_df, summary = run_box_reversion(df)
    elif strategy == "Binance下单":
        run_binance_panel()
        return
    else:
        st.header("NY 开盘 90 分钟 2R 首触统计")
        session_end = st.sidebar.text_input("美股收盘时间(纽约时区，HH:MM)", "16:00", key="ny_session_end")
        target_mult = st.sidebar.number_input("目标倍数(默认2R)", 0.5, 5.0, 2.0, key="ny_target_mult")
        run_ny = st.sidebar.button("运行统计", key="ny_run")
        if run_ny:
            daily_df, summary = analyze_ny_open_first_hit(df, session_end=session_end, target_mult=target_mult)
            st.session_state["ny_daily_df"] = daily_df
            st.session_state["ny_summary"] = summary
        else:
            daily_df = st.session_state.get("ny_daily_df")
            summary = st.session_state.get("ny_summary")

    # 过滤净R（已移除）
    if trades_df is not None and summary is not None:
        st.session_state["trades_df"] = trades_df
        st.session_state["summary"] = summary
        st.session_state["strategy_name"] = strategy
    # 展示
    if strategy == "NY开盘2R统计":
        if daily_df is not None and summary is not None:
            st.subheader("首触方向胜率概览")
            st.write(
                {
                    "UP日总数": summary.get("up_dir_total", 0),
                    "UP日顺势首触": summary.get("up_dir_trend_win", 0),
                    "UP日反向首触": summary.get("up_dir_revert_win", 0),
                    "UP日无触达": summary.get("up_no_touch", 0),
                    "UP日顺势胜率": f"{summary.get('trend_win_up',0)*100:.2f}%",
                    "UP日反向胜率": f"{summary.get('revert_win_up',0)*100:.2f}%",
                    "DOWN日总数": summary.get("down_dir_total", 0),
                    "DOWN日顺势首触": summary.get("down_dir_trend_win", 0),
                    "DOWN日反向首触": summary.get("down_dir_revert_win", 0),
                    "DOWN日无触达": summary.get("down_no_touch", 0),
                    "DOWN日顺势胜率": f"{summary.get('trend_win_down',0)*100:.2f}%",
                    "DOWN日反向胜率": f"{summary.get('revert_win_down',0)*100:.2f}%",
                }
            )
            st.subheader("按日明细")
            st.dataframe(daily_df)
        else:
            st.info("点击“运行统计”查看结果")

        # k 扫描
        k_text = st.sidebar.text_input("k 列表(逗号分隔)", "0.5,1,1.5,2,2.5", key="ny_k_list")
        bucket_count = st.sidebar.number_input("R90 分桶数", 2, 10, 4, key="ny_bucket_count")
        k_for_pnl = st.sidebar.number_input("P&L 模拟 k (TP/SL 对称倍数)", 0.1, 5.0, 1.0, key="ny_k_pnl")
        bucket_mode = st.sidebar.selectbox("R90 过滤", ["all", "Q1_Q2", "Q3_Q4"], index=0, key="ny_bucket_mode")
        tolerance_pct = st.sidebar.number_input("版本B回踩容忍度(如0.0005=0.05%)", 0.0, 0.01, 0.0005, format="%.4f", key="ny_tol")
        version_sel = st.sidebar.selectbox("无未来回测版本", ["A_11点收盘入场", "B_回踩O入场"], index=0, key="ny_version_sel")
        run_k = st.sidebar.button("运行 k 扫描/分桶", key="ny_run_k")
        k_df = None
        bucket_df = None
        pnl_df = None
        pnl_summary = None
        if run_k:
            try:
                k_list = [float(x.strip()) for x in k_text.split(",") if x.strip()]
            except Exception:
                k_list = [2.0]
            k_df = analyze_ny_open_k_scan(df, k_list=k_list, session_end=session_end)
            bucket_df, r_daily = analyze_R90_vs_later_move(df, session_end=session_end, bucket_count=bucket_count)
            ver = "A" if version_sel.startswith("A") else "B"
            pnl_df, pnl_summary = analyze_ny_no_future(
                df,
                session_end=session_end,
                k=k_for_pnl,
                version=ver,
                tolerance_pct=tolerance_pct,
                bucket_mode=bucket_mode,
            )
            st.session_state["ny_k_df"] = k_df
            st.session_state["ny_bucket_df"] = bucket_df
            st.session_state["ny_bucket_daily"] = r_daily
            st.session_state["ny_pnl_df"] = pnl_df
            st.session_state["ny_pnl_summary"] = pnl_summary
        else:
            k_df = st.session_state.get("ny_k_df")
            bucket_df = st.session_state.get("ny_bucket_df")
            r_daily = st.session_state.get("ny_bucket_daily")
            pnl_df = st.session_state.get("ny_pnl_df")
            pnl_summary = st.session_state.get("ny_pnl_summary")

        if k_df is not None and not k_df.empty:
            st.subheader("k 扫描顺/反向首触胜率")
            st.dataframe(k_df)
        if bucket_df is not None and isinstance(bucket_df, pd.DataFrame) and not bucket_df.empty:
            st.subheader("R90 分桶与后续波动统计")
            st.dataframe(bucket_df)
        if pnl_summary:
            st.subheader("P&L 模拟汇总（TP/SL 对称，收盘平仓，含手续费）")
            st.write(
                {
                    "总天数(有交易)": pnl_summary["overall"]["count"],
                    "覆盖工作日": pnl_summary.get("total_days", 0),
                    "平均R": round(pnl_summary["overall"]["avg"], 4),
                    "平均净R": round(pnl_summary["overall"]["avg_net"], 4),
                    "中位净R": round(pnl_summary["overall"]["median_net"], 4),
                    "最大回撤R": round(pnl_summary["overall"]["max_dd"], 4),
                    "最大连亏": pnl_summary["overall"]["max_loss_streak"],
                    "UP日平均净R": round(pnl_summary["up"]["avg_net"], 4),
                    "DOWN日平均净R": round(pnl_summary["down"]["avg_net"], 4),
                }
            )
        if pnl_df is not None and not pnl_df.empty:
            st.subheader("按日 P&L 明细（前 200）")
            st.dataframe(pnl_df.head(200))
        return

    trades_df = st.session_state.get("trades_df")
    summary = st.session_state.get("summary")
    if trades_df is not None and summary is not None:
        if "net_R" not in trades_df.columns and "raw_R" in trades_df.columns:
            trades_df = trades_df.copy()
            trades_df["net_R"] = trades_df["raw_R"]
        if "raw_R" not in trades_df.columns and "net_R" in trades_df.columns:
            trades_df = trades_df.copy()
            trades_df["raw_R"] = trades_df["net_R"]
        symbol_label = st.session_state.get("current_symbol", "未知")
        st.subheader(f"统计概览（{st.session_state.get('strategy_name','')} | {symbol_label} | {interval_label}）")
        st.write(
            {
                "交易笔数": summary["num_trades"],
                "胜率": f"{summary['win_rate']*100:.2f}%",
                "平均净R": round(summary["avg_R"], 3),
                "盈亏比": round((summary["avg_win_R"] / abs(summary["avg_loss_R"])) if summary["avg_loss_R"] else 0.0, 3),
                "ProfitFactor": round(summary["profit_factor"], 3),
                "最大回撤R": round(summary["max_drawdown_R"], 3),
                "交易对": symbol_label,
                "周期": interval_label,
            }
        )
        st.subheader("权益曲线")
        plot_equity(trades_df)
        st.subheader("复利曲线（初始100，单笔风险2%）")
        comp_info = compound_stats(trades_df.to_dict("records"), initial_capital=100.0, risk_per_trade=0.02)
        st.write(
            {
                "期末权益": round(comp_info["final_equity"], 2),
                "总收益%": f"{comp_info['return_pct']*100:.2f}%",
                "最大回撤%": f"{comp_info['max_drawdown_pct']*100:.2f}%",
            }
        )
        plot_compound(trades_df, initial_capital=100.0, risk_per_trade=0.02)
        st.subheader("价格+信号")
        plot_price(df, trades_df)
        st.subheader("交易明细（前 200）")
        view_df = trades_df.copy()
        if "exit_price" not in view_df.columns and "exit" in view_df.columns:
            view_df["exit_price"] = view_df["exit"]
        # 调整常见列顺序以方便查看
        cols_pref = [c for c in ["entry_time", "exit_time", "side", "entry", "exit_price", "sl", "tp", "tp1", "tp2", "raw_R", "net_R"] if c in view_df.columns]
        other_cols = [c for c in view_df.columns if c not in cols_pref]
        ordered_cols = cols_pref + other_cols
        view_df = view_df[ordered_cols]
        st.dataframe(view_df.head(200))
        st.subheader("单笔放大图（随机样本）")
        if not trades_df.empty:
            sample_count = min(5, len(view_df))
            sample_idxs = view_df.sample(sample_count, random_state=42).index.tolist()
            sel_idx = st.selectbox("选择交易行", sample_idxs, index=0)
            sel_trade = view_df.loc[sel_idx]
            plot_trade_zoom(df, sel_trade, window=120)
        # 三推样本图
        if st.session_state.get("strategy_name") == "三推EMA220" and not trades_df.empty:
            import tempfile, os, glob
            if st.button("生成三推随机样本图（5个）"):
                tmpdir = tempfile.mkdtemp(prefix="three_push_samples_")
                plot_random_three_push_samples(df, trades_df.to_dict("records"), n_samples=5, save_dir=tmpdir)
                imgs = sorted(glob.glob(os.path.join(tmpdir, "*.png")))
                if imgs:
                    st.subheader("随机样本图")
                    for img in imgs:
                        st.image(img, caption=os.path.basename(img), use_column_width=True)
                else:
                    st.info("未生成样本图")
    else:
        st.info("点击“运行回测”查看结果")


if __name__ == "__main__":
    main()
