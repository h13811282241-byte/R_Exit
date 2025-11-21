#!/usr/bin/env python3
"""
放量大K 50% 反转策略回测入口。
"""

import argparse
from pathlib import Path

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
    p.add_argument("--max_holding_bars", type=int, default=20)
    p.add_argument("--cooldown_bars", type=int, default=20)

    p.add_argument("--plot", action="store_true", help="生成图表")
    p.add_argument("--plot_dir", default="plots_out")
    return p.parse_args()


def load_klines(args) -> pd.DataFrame:
    if args.use_local_csv:
        return pd.read_csv(args.use_local_csv)
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


def main():
    args = parse_args()
    df = load_klines(args)

    # 调试：输出前 5 行，便于确认数据获取正常
    print("前 5 行 K 线数据:")
    print(df.head())

    signals = detect_signals(
        df,
        quiet_lookback=args.quiet_lookback,
        vol_spike_mult=args.vol_spike_mult,
        quiet_max_mult=args.quiet_max_mult,
        body_mult=args.body_mult,
        sl_mode="outer_bar",
    )
    trades = simulate_trades(
        df,
        signals,
        max_holding_bars=args.max_holding_bars,
        cooldown_bars=args.cooldown_bars,
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
