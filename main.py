#!/usr/bin/env python3
"""
放量大K 50% 反转策略回测入口。
"""

import argparse
from pathlib import Path
from datetime import time

import pandas as pd

from download_klines import download_klines, save_klines_to_csv
from strategy_volume_reversal import detect_signals, simulate_trades
from backtest_engine import summarize_trades, equity_curve
from visualization import plot_price_with_signals, plot_equity_curve


def parse_args():
    p = argparse.ArgumentParser(description="放量大K 50% 反转策略回测")
    p.add_argument("--symbol", help="交易对，例如 ETHUSDT")
    p.add_argument("--interval", default="5m")
    p.add_argument("--start", help="开始时间 UTC，如 2025-10-01 00:00:00")
    p.add_argument("--end", help="结束时间 UTC")
    p.add_argument("--market_type", default="spot", choices=["spot", "usdt_perp", "coin_perp", "usdc_perp"])
    p.add_argument("--use_local_csv", help="本地 CSV，若提供则不下载")
    p.add_argument("--save_csv", help="下载后保存的路径，可选")

    p.add_argument("--quiet_lookback", type=int, default=20)
    p.add_argument("--vol_spike_mult", type=float, default=2.0)
    p.add_argument("--quiet_max_mult", type=float, default=0.0, help="安静期最大量倍数，<=0 表示不限制，默认不限制")
    p.add_argument("--body_mult", type=float, default=1.5)
    p.add_argument("--tp_ratio", type=float, default=0.5, help="止盈比例：0-1，0.5 表示 K 线中位价")
    p.add_argument("--sl_offset_ratio", type=float, default=1.0, help="止损偏移倍数，1.0=K 高度，0.5=半个K高度")
    p.add_argument("--max_holding_bars", type=int, default=20)
    p.add_argument("--cooldown_bars", type=int, default=20)
    p.add_argument("--tp_ref", choices=["signal", "prev"], default="prev", help="止盈距离参考的K线：signal 当前信号K，prev 前一根K（默认）")
    p.add_argument("--sl_ref", choices=["signal", "prev"], default="prev", help="止损距离参考的K线：signal 当前信号K，prev 前一根K（默认）")
    p.add_argument("--cooldown_mode", choices=["bars", "vol"], default="bars", help="冷静期模式：bars 固定根数，vol 需降温")
    p.add_argument("--cooldown_vol_mult", type=float, default=1.0, help="冷静期降温阈值，vol<=均量*该倍数才算降温")
    p.add_argument("--cooldown_quiet_bars", type=int, default=3, help="降温需连续满足的根数 (cooldown_mode=vol)")
    p.add_argument("--invert_side", action="store_true", help="反转方向：原本做多改做空，原本做空改做多")

    p.add_argument("--plot", action="store_true", help="生成图表")
    p.add_argument("--plot_dir", default="plots_out")
    p.add_argument("--lower_interval", default="", help="下行周期用于判定同根TP/SL先后，例如 1m，留空禁用")
    p.add_argument(
        "--us_session_mode",
        choices=["all", "us_only", "non_us"],
        default="all",
        help="美股交易时段过滤：all 不过滤；us_only 仅09:30-16:00(ET)；non_us 仅盘前/盘后",
    )
    return p.parse_args()


def load_klines(args) -> pd.DataFrame:
    if args.use_local_csv:
        df = pd.read_csv(args.use_local_csv)
        required = {"timestamp", "open", "high", "low", "close", "volume"}
        has_required = required.issubset(set(df.columns))
        if not has_required:
            raise ValueError(f"本地 CSV 缺少必需列: {required}, 实际列: {df.columns.tolist()}")
        return df
    if not args.symbol or not args.start or not args.end:
        raise ValueError("未提供 symbol/start/end，且未指定 use_local_csv")
    df = download_klines(
        symbol=args.symbol,
        interval=args.interval,
        start_time=args.start,
        end_time=args.end,
        market_type=args.market_type,
    )
    if args.save_csv:
        save_klines_to_csv(df, args.save_csv)
    return df


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


def main():
    args = parse_args()
    df = load_klines(args)
    lower_df = None
    if args.lower_interval:
        lower_df = download_klines(
            symbol=args.symbol,
            interval=args.lower_interval,
            start_time=args.start,
            end_time=args.end,
            market_type=args.market_type,
        )
    if args.us_session_mode != "all":
        df = filter_us_session(df, args.us_session_mode)
        if lower_df is not None:
            lower_df = filter_us_session(lower_df, args.us_session_mode)

    signals = detect_signals(
        df,
        quiet_lookback=args.quiet_lookback,
        vol_spike_mult=args.vol_spike_mult,
        quiet_max_mult=args.quiet_max_mult,
        body_mult=args.body_mult,
        sl_mode="outer_bar",
        tp_ratio=args.tp_ratio,
        sl_offset_ratio=args.sl_offset_ratio,
        tp_ref=args.tp_ref,
        sl_ref=args.sl_ref,
        invert_side=args.invert_side,
    )
    trades = simulate_trades(
        df,
        signals,
        max_holding_bars=args.max_holding_bars,
        cooldown_bars=args.cooldown_bars,
        cooldown_mode=args.cooldown_mode,
        cooldown_vol_mult=args.cooldown_vol_mult,
        cooldown_quiet_bars=args.cooldown_quiet_bars,
        quiet_lookback=args.quiet_lookback,
        lower_df=lower_df,
        upper_interval_sec=parse_interval_seconds(args.interval),
        lower_interval_sec=parse_interval_seconds(args.lower_interval) if args.lower_interval else 60,
    )
    summary = summarize_trades(trades)
    eq = equity_curve(trades)

    print("==== 回测结果 ====")
    print(f"交易笔数: {summary['num_trades']}")
    print(f"胜率: {summary['win_rate']*100:.2f}%")
    print(f"平均R: {summary['avg_R']:.3f}, 中位R: {summary['median_R']:.3f}")
    print(f"最大R: {summary['max_R']:.3f}, 最小R: {summary['min_R']:.3f}")
    print(f"平均盈利R: {summary['avg_win_R']:.3f}, 平均亏损R: {summary['avg_loss_R']:.3f}")
    print(f"盈亏比: {summary['win_loss_ratio']:.3f}")

    if args.plot:
        out_dir = Path(args.plot_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        price_plot = out_dir / "price_signals.png"
        eq_plot = out_dir / "equity_curve.png"
        plot_price_with_signals(df, trades, out_file=str(price_plot))
        plot_equity_curve(trades, out_file=str(eq_plot))
        print(f"图表已保存到: {price_plot}, {eq_plot}")


if __name__ == "__main__":
    main()
