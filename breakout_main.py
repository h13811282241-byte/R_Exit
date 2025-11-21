#!/usr/bin/env python3
"""
趋势突破策略回测入口（1h 主周期，EMA 趋势过滤 + Donchian 突破 + ATR 止损/拖尾）
保留最近 5 次开仓时间的输出。
"""

import argparse
from datetime import time
from pathlib import Path

import pandas as pd

from download_klines import download_klines, save_klines_to_csv
from breakout_strategy import detect_breakouts, simulate_trades
from backtest_engine import summarize_trades, equity_curve
from visualization import plot_price_with_signals, plot_equity_curve


def parse_args():
    p = argparse.ArgumentParser(description="趋势突破策略回测")
    p.add_argument("--symbol", required=True, help="交易对，例如 ETHUSDT")
    p.add_argument("--interval", default="1h", help="主周期，默认 1h")
    p.add_argument("--start", required=True, help="开始时间 UTC，例如 2025-01-01 00:00:00")
    p.add_argument("--end", required=True, help="结束时间 UTC")
    p.add_argument("--market_type", default="usdt_perp", choices=["spot", "usdt_perp", "coin_perp", "usdc_perp"])
    p.add_argument("--use_local_csv", help="本地 CSV，若提供则不下载")
    p.add_argument("--save_csv", help="下载后保存的路径，可选")
    p.add_argument("--lower_interval", default="", help="下行周期用于同根 TP/SL 先后判定，如 1m，留空禁用")
    p.add_argument(
        "--us_session_mode",
        choices=["all", "us_only", "non_us"],
        default="all",
        help="美股时段过滤：all 不过滤；us_only 09:30-16:00 ET；non_us 其余。",
    )

    # 策略参数
    p.add_argument("--ema_span", type=int, default=100)
    p.add_argument("--donchian_n", type=int, default=24)
    p.add_argument("--atr_period", type=int, default=20)
    p.add_argument("--k_buffer", type=float, default=0.1)
    p.add_argument("--vol_lookback", type=int, default=20)
    p.add_argument("--vol_mult", type=float, default=1.5)
    p.add_argument("--atr_median_lookback", type=int, default=100)
    p.add_argument("--k_sl", type=float, default=1.5)
    p.add_argument("--R_target", type=float, default=3.0)
    p.add_argument("--k_trail", type=float, default=2.0)
    p.add_argument("--fee_side", type=float, default=0.000248, help="单边手续费比例，默认 0.0248%")
    p.add_argument("--stop_loss_streak", type=int, default=0, help="连亏达到此笔数后停止开仓，0 表示不启用")
    p.add_argument("--stop_duration_days", type=int, default=0, help="连亏触发后休息的天数")
    p.add_argument("--initial_capital", type=float, default=7000.0, help="复利计算初始资金，默认7000")
    p.add_argument("--risk_perc", type=float, default=0.02, help="每笔风险占用资金比例，默认2%")

    # 绘图
    p.add_argument("--plot", action="store_true", help="生成图表")
    p.add_argument("--plot_dir", default="plots_breakout")
    return p.parse_args()


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


def filter_us_session(df: pd.DataFrame, mode: str) -> pd.DataFrame:
    if mode == "all":
        return df
    ts = pd.to_datetime(df["timestamp"], utc=True)
    if ts.dt.tz is None:
        ts = ts.dt.tz_localize("UTC")
    ny = ts.dt.tz_convert("America/New_York")
    mask = (ny.dt.time >= time(9, 30)) & (ny.dt.time < time(16, 0))
    if mode == "us_only":
        return df.loc[mask].reset_index(drop=True)
    if mode == "non_us":
        return df.loc[~mask].reset_index(drop=True)
    return df


def compound_equity(trades, initial_capital: float, risk_perc: float) -> dict:
    capital = initial_capital
    peak = capital
    max_dd = 0.0
    for t in trades:
        r = t.get("net_R", t.get("R", 0.0))
        if r is None or pd.isna(r):
            r = 0.0
        capital *= (1 + risk_perc * r)
        peak = max(peak, capital)
        dd = (peak - capital) / peak if peak > 0 else 0.0
        max_dd = max(max_dd, dd)
    return {"final": capital, "max_drawdown": max_dd}


def load_klines(args, interval: str) -> pd.DataFrame:
    if args.use_local_csv:
        df = pd.read_csv(args.use_local_csv)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        if not required.issubset(df.columns):
            raise ValueError(f"本地 CSV 缺少必需列: {required - set(df.columns)}")
        return df
    df = download_klines(
        symbol=args.symbol,
        interval=interval,
        start_time=args.start,
        end_time=args.end,
        market_type=args.market_type,
    )
    if args.save_csv:
        save_klines_to_csv(df, args.save_csv)
    return df


def main():
    args = parse_args()
    df = load_klines(args, args.interval)
    lower_df = None
    if args.lower_interval:
        lower_df = load_klines(args, args.lower_interval)
    if args.us_session_mode != "all":
        df = filter_us_session(df, args.us_session_mode)
        if lower_df is not None:
            lower_df = filter_us_session(lower_df, args.us_session_mode)

    signals = detect_breakouts(
        df,
        ema_span=args.ema_span,
        donchian_n=args.donchian_n,
        atr_period=args.atr_period,
        k_buffer=args.k_buffer,
        vol_lookback=args.vol_lookback,
        vol_mult=args.vol_mult,
        atr_median_lookback=args.atr_median_lookback,
    )
    trades = simulate_trades(
        df,
        signals,
        k_sl=args.k_sl,
        R_target=args.R_target,
        k_trail=args.k_trail,
        fee_side=args.fee_side,
        lower_df=lower_df,
        upper_interval_sec=parse_interval_seconds(args.interval),
        lower_interval_sec=parse_interval_seconds(args.lower_interval) if args.lower_interval else 60,
        stop_loss_streak=args.stop_loss_streak,
        stop_duration_days=args.stop_duration_days,
    )
    summary = summarize_trades(trades)
    eq = equity_curve(trades)
    comp = compound_equity(trades, args.initial_capital, args.risk_perc)

    def loss_streak_info(trades, key="net_R"):
        max_len = 0
        cur_len = 0
        worst_sum = 0.0
        cur_sum = 0.0
        max_range = (None, None)
        cur_start = None
        for i, t in enumerate(trades):
            val = t.get(key) if key in t else t.get("R")
            if val is None or pd.isna(val):
                val = 0.0
            if val < 0:
                if cur_len == 0:
                    cur_start = i
                cur_len += 1
                cur_sum += val
                if cur_sum < worst_sum:
                    worst_sum = cur_sum
                    max_range = (cur_start, i)
                if cur_len > max_len:
                    max_len = cur_len
            else:
                cur_len = 0
                cur_sum = 0.0
                cur_start = None
        return max_len, worst_sum, max_range

    max_loss_len, worst_loss_sum, loss_range = loss_streak_info(trades, key="net_R")
    loss_range_time = None
    if loss_range[0] is not None:
        start_idx = trades[loss_range[0]].get("entry_idx", 0)
        end_idx = trades[loss_range[1]].get("entry_idx", 0)
        if 0 <= start_idx < len(df) and 0 <= end_idx < len(df):
            t1 = pd.to_datetime(df.loc[start_idx, "timestamp"], utc=True).tz_convert("Asia/Shanghai")
            t2 = pd.to_datetime(df.loc[end_idx, "timestamp"], utc=True).tz_convert("Asia/Shanghai")
            loss_range_time = f"{t1.strftime('%Y-%m-%d %H:%M:%S %Z')} ~ {t2.strftime('%Y-%m-%d %H:%M:%S %Z')}"

    last_entries = []
    if trades:
        for t in trades[-5:]:
            idx = t.get("entry_idx", 0)
            if 0 <= idx < len(df):
                ts = pd.to_datetime(df.loc[idx, "timestamp"], utc=True).tz_convert("Asia/Shanghai")
                last_entries.append(ts.strftime("%Y-%m-%d %H:%M:%S %Z"))

    print("==== 趋势突破回测结果 ====")
    print(f"交易笔数: {summary['num_trades']}")
    print(f"胜率: {summary['win_rate']*100:.2f}%")
    print(f"平均净R: {summary['avg_R']:.3f}, 中位R: {summary['median_R']:.3f}")
    print(f"最大R: {summary['max_R']:.3f}, 最小R: {summary['min_R']:.3f}")
    print(f"平均盈利R: {summary['avg_win_R']:.3f}, 平均亏损R: {summary['avg_loss_R']:.3f}")
    print(f"盈亏比: {summary['win_loss_ratio']:.3f}")
    if max_loss_len > 0:
        print(f"最长连亏: {max_loss_len} 笔，净R合计 {worst_loss_sum:.3f}"
              + (f"，时间段: {loss_range_time}" if loss_range_time else ""))
    print(f"复利最终资金: {comp['final']:.2f} (初始 {args.initial_capital}, 每笔风险 {args.risk_perc*100:.2f}% ), 最大回撤: {comp['max_drawdown']*100:.2f}%")
    if last_entries:
        print("最近5次开仓时间(北京):", "; ".join(last_entries))

    if args.plot:
        out_dir = Path(args.plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        price_plot = out_dir / "breakout_price_signals.png"
        eq_plot = out_dir / "breakout_equity_curve.png"
        plot_price_with_signals(df, trades, out_file=str(price_plot))
        plot_equity_curve(trades, out_file=str(eq_plot))
        print(f"图表已保存到: {price_plot}, {eq_plot}")


if __name__ == "__main__":
    main()
