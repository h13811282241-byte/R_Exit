#!/usr/bin/env python3
"""
Binance Kline downloader with pagination.
"""

import argparse
import os
from datetime import datetime, timezone
from typing import Optional, List, Dict

import pandas as pd
import requests


BASE_ENDPOINTS: Dict[str, str] = {
    "spot": "https://api.binance.com/api/v3/klines",
    "usdt_perp": "https://fapi.binance.com/fapi/v1/klines",
    "coin_perp": "https://dapi.binance.com/dapi/v1/klines",
    # USDC 永续也使用 USD-M 端点，若有专用端点可在此调整
    "usdc_perp": "https://fapi.binance.com/fapi/v1/klines",
}


def _require_api_keys() -> str:
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("请在环境变量中设置 BINANCE_API_KEY 和 BINANCE_API_SECRET")
    return api_key


def _to_millis(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _to_timestamp_str(ms: int, tz: str = "UTC") -> str:
    dt = datetime.fromtimestamp(ms / 1000, tz=timezone.utc)
    if tz and tz.upper() in {"CST", "UTC+8", "BEIJING"}:
        dt = dt.astimezone(datetime.now().astimezone().tzinfo)
    return dt.strftime("%Y-%m-%d %H:%M:%S")


def download_klines(
    symbol: str,
    interval: str,
    start_time: str,
    end_time: str,
    market_type: str = "spot",
) -> pd.DataFrame:
    """
    下载 K 线数据（自动翻页，直到 end_time 或无更多数据）
    """
    api_key = _require_api_keys()
    if market_type not in BASE_ENDPOINTS:
        raise ValueError(f"不支持的 market_type: {market_type}")

    url = BASE_ENDPOINTS[market_type]
    headers = {"X-MBX-APIKEY": api_key}
    start_ms = _to_millis(start_time)
    end_ms = _to_millis(end_time)

    all_rows: List[List] = []
    while True:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = requests.get(url, params=params, headers=headers, timeout=15)
        if resp.status_code != 200:
            raise RuntimeError(f"请求失败: {resp.status_code} {resp.text}")
        batch = resp.json()
        if not batch:
            break
        all_rows.extend(batch)
        last_close = batch[-1][6]  # closeTime
        next_start = last_close + 1
        if next_start > end_ms or len(batch) < 1000:
            break
        start_ms = next_start

    if not all_rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    records = []
    for row in all_rows:
        records.append(
            {
                "timestamp": _to_timestamp_str(row[0], tz="UTC"),
                "open": float(row[1]),
                "high": float(row[2]),
                "low": float(row[3]),
                "close": float(row[4]),
                "volume": float(row[5]),
            }
        )
    df = pd.DataFrame(records)
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def save_klines_to_csv(df: pd.DataFrame, outfile: str) -> None:
    df.to_csv(outfile, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="下载 Binance K 线数据")
    parser.add_argument("--symbol", required=True, help="交易对，例如 ETHUSDT")
    parser.add_argument("--interval", default="5m", help="K 线周期，默认 5m")
    parser.add_argument("--start", required=True, help="开始时间，例如 2024-01-01 00:00:00 (UTC)")
    parser.add_argument("--end", required=True, help="结束时间，例如 2024-02-01 00:00:00 (UTC)")
    parser.add_argument("--outfile", required=True, help="输出 CSV 路径")
    parser.add_argument(
        "--market_type",
        default="spot",
        choices=list(BASE_ENDPOINTS.keys()),
        help="市场类型：spot/usdt_perp/coin_perp/usdc_perp",
    )
    args = parser.parse_args()

    df = download_klines(
        symbol=args.symbol,
        interval=args.interval,
        start_time=args.start,
        end_time=args.end,
        market_type=args.market_type,
    )
    save_klines_to_csv(df, args.outfile)
    print(f"已保存 {len(df)} 根 K 线到 {args.outfile}")


if __name__ == "__main__":
    main()
