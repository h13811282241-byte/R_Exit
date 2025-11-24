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
from breakout_strategy import detect_breakouts, simulate_trades as simulate_breakout
from backtest_engine import simulate_basic, summarize_trades


def plot_price(df: pd.DataFrame, trades: pd.DataFrame):
    fig = go.Figure(data=[go.Candlestick(x=df["timestamp"], open=df["open"], high=df["high"], low=df["low"], close=df["close"])])
    if not trades.empty:
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


def plot_equity(trades: pd.DataFrame):
    if trades.empty:
        st.info("无交易")
        return
    trades["cum_net_R"] = trades["net_R"].cumsum()
    fig = go.Figure(data=[go.Scatter(x=trades.index, y=trades["cum_net_R"], mode="lines", name="Cumulative net R")])
    fig.update_layout(height=300)
    st.plotly_chart(fig, use_container_width=True)


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
    else:
        symbol = st.sidebar.text_input("交易对", "ETHUSDT", key="dl_symbol")
        interval = st.sidebar.text_input("周期(如 5m/15m/1h/4h)", "1h", key="dl_interval")
        start = st.sidebar.text_input("开始时间 UTC", "2024-01-01 00:00:00", key="dl_start")
        end = st.sidebar.text_input("结束时间 UTC", "2024-02-01 00:00:00", key="dl_end")
        if st.sidebar.button("下载", key="dl_button"):
            try:
                df = download_binance_klines(symbol, interval, start, end, market_type="spot")
                st.session_state["data_df"] = df
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
        pivot_right = st.number_input("Pivot right", 1, 5, 2, key="rsi_pivot_right")
        min_rsi_diff = st.number_input("最小RSI差值", 0.0, 20.0, 3.0, key="rsi_min_diff")
        sl_mode = st.selectbox("SL模式", ["swing", "atr"], key="rsi_sl_mode")
        atr_period = st.number_input("ATR period", 5, 50, 14, key="rsi_atr_period")
        k_sl = st.number_input("k_sl (ATR倍数)", 0.5, 5.0, 1.5, key="rsi_k_sl")
        tp_R = st.number_input("TP R 倍数", 0.5, 10.0, 2.0, key="rsi_tp_R")
        fee_side = st.number_input("单边手续费(比例)", 0.0, 0.01, 0.00045, format="%.6f", key="rsi_fee_side")
        entry_slip_pct = st.number_input("开仓滑点比例(如0.0005=0.05%)", 0.0, 0.01, 0.0, format="%.5f", key="rsi_entry_slip")
        sl_buffer_pct = st.number_input("止损缓冲比例(如0.002=0.2%)", 0.0, 0.05, 0.0, format="%.4f", key="rsi_sl_buffer")
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
        pivot_right=pivot_right,
        min_rsi_diff=min_rsi_diff,
        sl_mode=sl_mode,
        atr_period=atr_period,
        k_sl=k_sl,
        tp_R=tp_R,
    )
    trades = simulate_basic(
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
    trades = simulate_basic(
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
    lower_symbol = st.sidebar.text_input("下行数据交易对", "ETHUSDT", key="lower_symbol")

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

    strategy = st.sidebar.radio("策略", ["RSI 背离", "Alligator", "Breakout"], key="strategy_choice")
    trades_df = None
    summary = None
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
    else:
        trades_df, summary = run_breakout(df, lower_fetch=lower_fetch)

    # 过滤净R（已移除）
    if trades_df is not None and summary is not None:
        st.session_state["trades_df"] = trades_df
        st.session_state["summary"] = summary
        st.session_state["strategy_name"] = strategy

    trades_df = st.session_state.get("trades_df")
    summary = st.session_state.get("summary")
    if trades_df is not None and summary is not None:
        st.subheader(f"统计概览（{st.session_state.get('strategy_name','')}）")
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
    else:
        st.info("点击“运行回测”查看结果")


if __name__ == "__main__":
    main()
