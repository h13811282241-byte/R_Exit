#!/usr/bin/env python3
"""
Data loading utilities: from local CSV or Binance klines.
"""

import os
from datetime import datetime, timezone
from typing import Optional, List, Dict

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry


BINANCE_ENDPOINTS: Dict[str, str] = {
    "spot": "https://api.binance.com/api/v3/klines",
    "usdt_perp": "https://fapi.binance.com/fapi/v1/klines",
    "coin_perp": "https://dapi.binance.com/dapi/v1/klines",
    "usdc_perp": "https://fapi.binance.com/fapi/v1/klines",
}


def _to_ms(dt_str: str) -> int:
    dt = datetime.strptime(dt_str, "%Y-%m-%d %H:%M:%S").replace(tzinfo=timezone.utc)
    return int(dt.timestamp() * 1000)


def _ts_to_str(ms: int) -> str:
    return datetime.fromtimestamp(ms / 1000, tz=timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def _build_session() -> requests.Session:
    sess = requests.Session()
    retries = Retry(
        total=3,
        backoff_factor=1,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retries)
    sess.mount("https://", adapter)
    sess.mount("http://", adapter)
    return sess


def download_binance_klines(
    symbol: str,
    interval: str,
    start_time: str,
    end_time: str,
    market_type: str = "spot",
    timeout: int = 20,
) -> pd.DataFrame:
    if market_type not in BINANCE_ENDPOINTS:
        raise ValueError(f"Unsupported market_type: {market_type}")
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise RuntimeError("请在环境变量中设置 BINANCE_API_KEY 和 BINANCE_API_SECRET")

    url = BINANCE_ENDPOINTS[market_type]
    headers = {"X-MBX-APIKEY": api_key}
    start_ms = _to_ms(start_time)
    end_ms = _to_ms(end_time)

    sess = _build_session()
    rows: List[List] = []
    while True:
        params = {
            "symbol": symbol.upper(),
            "interval": interval,
            "startTime": start_ms,
            "endTime": end_ms,
            "limit": 1000,
        }
        resp = sess.get(url, params=params, headers=headers, timeout=timeout)
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


def load_csv(file_path: str) -> pd.DataFrame:
    df = pd.read_csv(file_path)
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"CSV 缺少必需列: {required - set(df.columns)}")
    df = df.sort_values("timestamp").reset_index(drop=True)
    return df


def ensure_ohlcv_df(df: pd.DataFrame) -> pd.DataFrame:
    required = {"timestamp", "open", "high", "low", "close", "volume"}
    if not required.issubset(df.columns):
        raise ValueError(f"DataFrame 缺少必需列: {required - set(df.columns)}")
    return df.sort_values("timestamp").reset_index(drop=True)
