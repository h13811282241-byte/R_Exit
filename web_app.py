#!/usr/bin/env python3
"""
Streamlit Web UI for RSI 背离 + Alligator 策略回测。
"""

import os
import tempfile
import streamlit as st
import pandas as pd
import plotly.graph_objects as go

from data_loader import load_csv, download_binance_klines, ensure_ohlcv_df
from indicators import atr, alligator
from strategy_rsi_divergence import detect_rsi_divergence_signals
from strategy_alligator import detect_alligator_signals, prepare_sl_tp
from backtest_engine import simulate_basic, summarize_trades


def plot_price(df: pd.DataFrame, trades: pd.DataFrame):
    fig = go.Figure(data=[go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"])])
    if not trades.empty:
        longs = trades[trades["side"] == "long"]
        shorts = trades[trades["side"] == "short"]
        fig.add_trace(go.Scatter(x=longs["entry_time"], y=longs["entry"], mode="markers", marker=dict(color="green", symbol="triangle-up", size=10), name="Long entry"))
        fig.add_trace(go.Scatter(x=shorts["entry_time"], y=shorts["entry"], mode="markers", marker=dict(color="red", symbol="triangle-down", size=10), name="Short entry"))
    fig.update_layout(height=500, xaxis_rangeslider_visible=False)
    st.plotly_chart(fig, use_container_width=True)


def plot_equity(trades: pd.DataFrame):
    if trades.empty:
        st.info("无交易")
        return
    trades["cum_net_R"] = trades["net_R"].cumsum()
    fig = go.Figure(data=[go.Scatter(x=trades.index, y=trades["cum_net_R"], mode="lines", name="Cumulative net R")])
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


def load_data_ui():
    source = st.sidebar.radio("数据来源", ["上传 CSV", "币安下载"])
    df = None
    if source == "上传 CSV":
        file = st.sidebar.file_uploader("上传 CSV (需包含 timestamp,open,high,low,close,volume)", type=["csv"])
        if file:
            with tempfile.NamedTemporaryFile(delete=False) as tmp:
                tmp.write(file.read())
                df = load_csv(tmp.name)
    else:
        symbol = st.sidebar.text_input("交易对", "ETHUSDT")
        interval = st.sidebar.selectbox("周期", ["5m", "15m", "1h", "4h"], index=2)
        start = st.sidebar.text_input("开始时间 UTC", "2024-01-01 00:00:00")
        end = st.sidebar.text_input("结束时间 UTC", "2024-02-01 00:00:00")
        if st.sidebar.button("下载"):
            try:
                df = download_binance_klines(symbol, interval, start, end, market_type="spot")
                st.success(f"下载完成，{len(df)} 行")
            except Exception as e:
                st.error(f"下载失败: {e}")
    return df


def run_rsi_divergence(df: pd.DataFrame):
    st.header("RSI 背离策略")
    with st.sidebar:
        rsi_period = st.number_input("RSI period", 5, 50, 14)
        overbought = st.number_input("Overbought", 50.0, 90.0, 70.0)
        oversold = st.number_input("Oversold", 10.0, 50.0, 30.0)
        lookback_bars = st.number_input("Lookback bars", 5, 100, 20)
        pivot_left = st.number_input("Pivot left", 1, 5, 2)
        pivot_right = st.number_input("Pivot right", 1, 5, 2)
        min_rsi_diff = st.number_input("最小RSI差值", 0.0, 20.0, 3.0)
        sl_mode = st.selectbox("SL模式", ["swing", "atr"])
        atr_period = st.number_input("ATR period", 5, 50, 14)
        k_sl = st.number_input("k_sl (ATR倍数)", 0.5, 5.0, 1.5)
        tp_R = st.number_input("TP R 倍数", 0.5, 5.0, 2.0)
        fee_side = st.number_input("单边手续费(比例)", 0.0, 0.01, 0.000248, format="%.6f")
    signals = detect_rsi_divergence_signals(
        df,
        rsi_period=rsi_period,
        overbought=overbought,
        oversold=oversold,
        lookback_bars=lookback_bars,
        pivot_left=pivot_left,
        pivot_right=pivot_right,
        min_rsi_diff=min_rsi_diff,
        sl_mode=sl_mode,
        atr_period=atr_period,
        k_sl=k_sl,
        tp_R=tp_R,
    )
    trades = simulate_basic(df, signals, fee_side_pct=fee_side)
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades)
    return trades_df, summary


def run_alligator(df: pd.DataFrame):
    st.header("Alligator 趋势策略")
    with st.sidebar:
        jaw_period = st.number_input("Jaw period", 5, 30, 13)
        teeth_period = st.number_input("Teeth period", 5, 20, 8)
        lips_period = st.number_input("Lips period", 3, 15, 5)
        trend_confirm_bars = st.number_input("趋势确认根数", 1, 10, 3)
        entry_fresh_bars = st.number_input("入场新鲜度(根数)", 1, 20, 5)
        sl_mode = st.selectbox("SL模式", ["atr", "swing"])
        atr_period = st.number_input("ATR period", 5, 50, 14)
        k_sl = st.number_input("k_sl (ATR倍数)", 0.5, 5.0, 1.5)
        tp_R = st.number_input("TP R 倍数", 0.5, 5.0, 2.0)
        fee_side = st.number_input("单边手续费(比例)", 0.0, 0.01, 0.000248, format="%.6f")
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
    trades = simulate_basic(df, signals, fee_side_pct=fee_side)
    trades_df = pd.DataFrame(trades)
    summary = summarize_trades(trades)
    return trades_df, summary


def main():
    st.set_page_config(page_title="RSI 背离 + Alligator 回测", layout="wide")
    df = load_data_ui()
    if df is None or df.empty:
        st.info("请先加载数据")
        return
    df = ensure_ohlcv_df(df)

    strategy = st.sidebar.radio("策略", ["RSI 背离", "Alligator"])
    if st.sidebar.button("运行回测"):
        if strategy == "RSI 背离":
            trades_df, summary = run_rsi_divergence(df)
        else:
            trades_df, summary = run_alligator(df)

        st.subheader("统计概览")
        st.write(
            {
                "交易笔数": summary["num_trades"],
                "胜率": f"{summary['win_rate']*100:.2f}%",
                "平均净R": round(summary["avg_R"], 3),
                "ProfitFactor": round(summary["profit_factor"], 3),
                "最大回撤R": round(summary["max_drawdown_R"], 3),
            }
        )
        st.subheader("权益曲线")
        plot_equity(trades_df)
        st.subheader("价格+信号")
        plot_price(df, trades_df)
        st.subheader("交易明细（前 200）")
        st.dataframe(trades_df.head(200))


if __name__ == "__main__":
    main()
