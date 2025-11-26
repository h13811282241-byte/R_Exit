#!/usr/bin/env python3
print("bot started")

"""
基于 1h EMA220 突破策略的简易“模拟盘”脚本，默认连币安永续测试网：
- 拉取 1h K 线，按 strategy_ema220_breakout 生成最新信号
- 若最新收盘 K 触发信号，则在测试网下市价单，并挂 SL/TP（reduceOnly）

使用前：
  export BINANCE_API_KEY=...
  export BINANCE_API_SECRET=...

示例：
  python ema220_bot.py --symbol BTCUSDT --qty 0.001

参数：
  --symbol    交易对，如 BTCUSDT
  --limit     拉取 K 线根数（默认 400）
  --live      使用主网（默认测试网，谨慎）
  --qty       下单数量（必填）
  --price_buffer   为 SL/TP 预留的触发价微调比例，默认 0.001
"""

import argparse
import hashlib
import hmac
import os
import time
from typing import Dict, Any, Tuple

import pandas as pd
import requests

from strategy_ema220_breakout import detect_signals


TESTNET = "https://testnet.binancefuture.com"
MAINNET = "https://fapi.binance.com"


def sign(params: Dict[str, Any], secret: str) -> Dict[str, Any]:
    qs = "&".join(f"{k}={params[k]}" for k in sorted(params))
    sig = hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params


def send(method: str, path: str, params: Dict[str, Any], api_key: str, api_secret: str, base: str, signed=False):
    params = params or {}
    if signed:
        params["timestamp"] = int(time.time() * 1000)
        params = sign(params, api_secret)
    headers = {"X-MBX-APIKEY": api_key}
    url = base + path
    resp = requests.request(method, url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def fetch_klines(symbol: str, interval: str, limit: int, api_key: str, api_secret: str, base: str) -> pd.DataFrame:
    data = send("GET", "/fapi/v1/klines", {"symbol": symbol, "interval": interval, "limit": limit}, api_key, api_secret, base)
    cols = [
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "close_time",
        "quote_asset_volume",
        "number_of_trades",
        "taker_buy_base_asset_volume",
        "taker_buy_quote_asset_volume",
        "ignore",
    ]
    df = pd.DataFrame(data, columns=cols)
    df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", utc=True)
    df[["open", "high", "low", "close", "volume"]] = df[["open", "high", "low", "close", "volume"]].astype(float)
    return df[["timestamp", "open", "high", "low", "close", "volume"]]


def fetch_exchange_info(symbol: str, api_key: str, api_secret: str, base: str) -> Tuple[float, float]:
    info = send("GET", "/fapi/v1/exchangeInfo", {"symbol": symbol}, api_key, api_secret, base)
    filters = info["symbols"][0]["filters"]
    step = 0.0
    min_qty = 0.0
    for f in filters:
        if f["filterType"] == "LOT_SIZE":
            step = float(f["stepSize"])
            min_qty = float(f["minQty"])
    return step, min_qty


def round_qty(qty: float, step: float) -> float:
    if step <= 0:
        return qty
    return max(0.0, (qty // step) * step)


def fetch_equity(api_key: str, api_secret: str, base: str) -> float:
    data = send("GET", "/fapi/v2/account", {}, api_key, api_secret, base, signed=True)
    for asset in data.get("assets", []):
        if asset.get("asset") == "USDT":
            return float(asset.get("walletBalance", 0.0))
    return 0.0


def has_position(symbol: str, api_key: str, api_secret: str, base: str) -> bool:
    data = send("GET", "/fapi/v2/positionRisk", {"symbol": symbol}, api_key, api_secret, base, signed=True)
    for pos in data:
        if pos.get("symbol") == symbol and abs(float(pos.get("positionAmt", 0))) > 0:
            return True
    return False


def place_orders(symbol: str, side: str, qty: float, entry: float, sl: float, tp1: float, tp2: float, base: str, key: str, secret: str, buffer: float):
    """
    下市价单 + SL/TP 触发单（reduceOnly），tp1/tp2 作为两级止盈。
    """
    order = {
        "symbol": symbol,
        "side": side,
        "type": "MARKET",
        "quantity": qty,
        "newClientOrderId": f"ema220_{int(time.time()*1000)}",
    }
    resp_main = send("POST", "/fapi/v1/order", order, key, secret, base, signed=True)

    sl_order = {
        "symbol": symbol,
        "side": "SELL" if side == "BUY" else "BUY",
        "type": "STOP_MARKET",
        "stopPrice": round(sl * (1 - buffer) if side == "BUY" else sl * (1 + buffer), 4),
        "closePosition": "true",
    }
    send("POST", "/fapi/v1/order", sl_order, key, secret, base, signed=True)

    for tp in [tp1, tp2]:
        tp_order = {
            "symbol": symbol,
            "side": "SELL" if side == "BUY" else "BUY",
            "type": "TAKE_PROFIT_MARKET",
            "stopPrice": round(tp * (1 + buffer) if side == "BUY" else tp * (1 - buffer), 4),
            "closePosition": "false",
            "quantity": qty / 2,
            "reduceOnly": "true",
        }
        send("POST", "/fapi/v1/order", tp_order, key, secret, base, signed=True)
    return resp_main


def main():
    ap = argparse.ArgumentParser(description="1h EMA220 突破策略 - 测试网批量扫描下单脚本")
    ap.add_argument("--symbols", default="BTCUSDT,ETHUSDT,BNBUSDT,ADAUSDT,LINKUSDT,SUIUSDT,DOGEUSDT", help="逗号分隔交易对列表")
    ap.add_argument("--limit", type=int, default=400)
    ap.add_argument("--risk_pct", type=float, default=0.06, help="单笔风险占权益比例，默认6%（测试网）")
    ap.add_argument("--live", action="store_true", help="使用主网（默认测试网，谨慎）")
    ap.add_argument("--price_buffer", type=float, default=0.001, help="SL/TP 触发价微调比例")
    ap.add_argument("--interval_sec", type=int, default=300, help="轮询间隔秒数，默认5分钟")
    args = ap.parse_args()

    key = os.environ.get("BINANCE_API_KEY")
    secret = os.environ.get("BINANCE_API_SECRET")
    if not key or not secret:
        raise SystemExit("请先设置 BINANCE_API_KEY / BINANCE_API_SECRET")
    base = MAINNET_URL if args.live else TESTNET

    symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()]
    print(f"运行于 {'主网' if args.live else '测试网'}，轮询 {args.interval_sec}s，标的: {symbols}")

    while True:
        loop_start = time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())
        try:
            equity = fetch_equity(key, secret, base)
            if equity <= 0:
                print(f"[{loop_start}] 无权益，跳过本轮")
                time.sleep(args.interval_sec)
                continue
            for sym in symbols:
                print(f"[{loop_start}] === 扫描 {sym} ===")
                try:
                    if has_position(sym, key, secret, base):
                        print("已有持仓，跳过")
                        continue
                    df = fetch_klines(sym, "1h", args.limit, key, secret, base)
                    sigs = detect_signals(df)
                    if not sigs or sigs[-1]["idx"] != len(df) - 1:
                        print("最新K无信号")
                        continue
                    sig = sigs[-1]
                    risk_price = abs(sig["entry"] - sig["sl"])
                    if risk_price <= 0:
                        print("风险距离无效，跳过")
                        continue
                    step, min_qty = fetch_exchange_info(sym, key, secret, base)
                    qty_raw = (equity * args.risk_pct) / risk_price
                    qty = round_qty(qty_raw, step)
                    if qty < min_qty or qty <= 0:
                        print(f"计算数量过小：{qty}，最小 {min_qty}，跳过")
                        continue
                    side = "BUY" if sig["side"] == "long" else "SELL"
                    print(f"下单: {sym} {side} qty={qty} entry={sig['entry']} sl={sig['sl']} tp1={sig['tp1']} tp2={sig['tp2']}")
                    resp = place_orders(sym, side, qty, sig["entry"], sig["sl"], sig["tp1"], sig["tp2"], base, key, secret, args.price_buffer)
                    print("下单完成:", resp)
                except Exception as e:
                    print(f"{sym} 处理失败: {e}")
        except Exception as e:
            print(f"[{loop_start}] 扫描异常: {e}")
        time.sleep(args.interval_sec)


if __name__ == "__main__":
    main()
