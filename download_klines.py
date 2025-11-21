#!/usr/bin/env python3
"""
Binance Kline downloader with pagination.
"""

import argparse
import os
from datetime import datetime, timezone
from typing import List, Dict

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BASE_ENDPOINTS: Dict[str, str] = {
    "spot": "https://api.binance.com/api/v3/klines",
    "usdt_perp": "https://fapi.binance.com/fapi/v1/klines",
    "coin_perp": "https://dapi.binance.com/dapi/v1/klines",
    "usdc_perp": "https://fapi.binance.com/fapi/v1/klines",
}


def _require_api_key() -> str:
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("请在环境变量中设置 BINANCE_API_KEY 和 BINANCE_API_SECRET")
    return api_key


def _to_ms(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ts_to_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _build_session() -> requests.Session:
    session = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    session.mount("https://", adapter)
    session.mount("http://", adapter)
    return session


def download_klines(
    symbol: str, interval: str, start_time: str, end_time: str, market_type: str = "spot", timeout: int = 30
) -> pd.DataFrame:
    api_key = _require_api_key()
    if market_type not in BASE_ENDPOINTS:
        raise ValueError(f"不支持的市场类型: {market_type}")
    url = BASE_ENDPOINTS[market_type]

    headers = {"X-MBX-APIKEY": api_key}
    start_ms = _to_ms(start_time)
    end_ms = _to_ms(end_time)

    rows: List[List] = []
    session = _build_session()
    while True:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = session.get(url, params=params, headers=headers, timeout=timeout)
        if resp.status_code != 200:
            raise RuntimeError(f"请求失败: {resp.status_code} {resp.text}")
        batch = resp.json()
        if not batch:
            break
        rows.extend(batch)
        last_close = batch[-1][6]
        next_start = last_close + 1
        if next_start > end_ms or len(batch) < 1000:
            break
        start_ms = next_start

    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    records = []
    for r in rows:
        records.append(
            {
                "timestamp": _ts_to_str(r[0]),
                "open": float(r[1]),
                "high": float(r[2]),
                "low": float(r[3]),
                "close": float(r[4]),
                "volume": float(r[5]),
            }
        )
    df = pd.DataFrame(records).sort_values("timestamp").reset_index(drop=True)
    return df


def save_klines_to_csv(df: pd.DataFrame, outfile: str) -> None:
    df.to_csv(outfile, index=False, encoding="utf-8-sig")


def main():
    parser = argparse.ArgumentParser(description="下载 Binance K 线数据")
    parser.add_argument("--symbol", required=True)
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--start", required=True, help="UTC 时间，例如 2024-01-01 00:00:00")
    parser.add_argument("--end", required=True)
    parser.add_argument("--outfile", required=True)
    parser.add_argument("--market_type", default="spot", choices=list(BASE_ENDPOINTS.keys()))
    parser.add_argument("--timeout", type=int, default=30, help="请求超时秒数，默认30")
    args = parser.parse_args()

    df = download_klines(args.symbol, args.interval, args.start, args.end, market_type=args.market_type)
    save_klines_to_csv(df, args.outfile)
    print(f"已保存 {len(df)} 根 K 线到 {args.outfile}")


if __name__ == "__main__":
    main()
