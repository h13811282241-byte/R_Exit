#!/usr/bin/env python3
"""
批量回测：多交易对、逐年、5m、RSI 背离，TP=4R，手续费单边 0.00045。
运行方式：
    source .venv/bin/activate
    python ti2.py
结果会打印表格，并生成 rsi_yearly_5m_4R.csv
"""

import time
import pandas as pd

from data_loader import download_binance_klines
from strategy_rsi_divergence import detect_rsi_divergence_signals
from backtest_engine import simulate_basic, summarize_trades


def load_symbols():
    """
    固定只跑关注的 6 个交易对：ZEC/XMR/SOL/DOGE/XLM/ADA（USDT 本位）。
    不再读取 symbols_top50.txt。
    """
    return ["ZECUSDT", "XMRUSDT", "SOLUSDT", "DOGEUSDT", "XLMUSDT", "ADAUSDT"]


def main():
    symbols = load_symbols()
    years = [2021, 2022, 2023, 2024]
    interval = "5m"
    market_type = "usdt_perp"  # 如需现货改为 "spot"

    results = []
    trades_records = []
    for sym in symbols:
        for y in years:
            start = f"{y}-01-01 00:00:00"
            end = f"{y+1}-01-01 00:00:00"
            try:
                df = download_binance_klines(sym, interval, start, end, market_type=market_type)
                if df.empty:
                    print(f"[skip] {sym} {y} 无数据")
                    continue
                signals = detect_rsi_divergence_signals(
                    df,
                    rsi_period=14,
                    overbought=70,
                    oversold=30,
                    lookback_bars=20,
                    pivot_left=2,
                    pivot_right=2,
                    min_rsi_diff=3,
                    sl_mode="atr",
                    atr_period=14,
                    k_sl=1.5,
                    tp_R=4.0,  # 4:1 盈亏比
                )
                trades, _stats = simulate_basic(df, signals, fee_side_pct=0.00045)
                summary = summarize_trades(trades)
                results.append(
                    {
                        "symbol": sym,
                        "year": y,
                        "trades": summary["num_trades"],
                        "win_rate": summary["win_rate"],
                        "avg_R": summary["avg_R"],
                    }
                )
                if trades:
                    tdf = pd.DataFrame(trades)
                    tdf["symbol"] = sym
                    tdf["year"] = y
                    trades_records.append(tdf)
                time.sleep(1.0)  # 降低频率，避免 418
            except Exception as e:
                print(f"[error] {sym} {y}: {e}")

    df_out = pd.DataFrame(results)
    if not df_out.empty:
        df_out["win_rate(%)"] = (df_out["win_rate"] * 100).round(2)
        df_out["avg_R"] = df_out["avg_R"].round(3)
        cols = ["symbol", "year", "trades", "win_rate(%)", "avg_R"]
        print(df_out[cols].to_string(index=False))
        df_out[cols].to_csv("rsi_yearly_5m_4R.csv", index=False)
    else:
        print("无结果")

    # 保存所有交易记录（按 symbol/year 打标签）
    if trades_records:
        trades_all = pd.concat(trades_records, ignore_index=True)
        trades_all.to_csv("rsi_yearly_5m_4R_trades.csv", index=False)
        print(f"已保存交易明细到 rsi_yearly_5m_4R_trades.csv，共 {len(trades_all)} 笔")
    else:
        print("无交易明细可保存")


if __name__ == "__main__":
    main()
