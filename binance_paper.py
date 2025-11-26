#!/usr/bin/env python3
"""
简易脚本：对接币安永续合约（默认 TESTNET），下单/查价/查余额/一键撤单。

使用前：
  export BINANCE_API_KEY="..."
  export BINANCE_API_SECRET="..."
  # 默认走测试网；若要主网需显式 --live

示例：
  # 查价
  python binance_paper.py price --symbol BTCUSDT

  # 查询余额
  python binance_paper.py balance

  # 测试网市价开多 0.001 BTCUSDT
  python binance_paper.py order --symbol BTCUSDT --side BUY --qty 0.001 --type MARKET

  # 测试网限价开空
  python binance_paper.py order --symbol BTCUSDT --side SELL --qty 0.001 --type LIMIT --price 30000

  # 撤销某交易对所有挂单
  python binance_paper.py cancel --symbol BTCUSDT
"""

import argparse
import hashlib
import hmac
import os
import time
from typing import Dict, Any

import requests


TESTNET_URL = "https://testnet.binancefuture.com"
MAINNET_URL = "https://fapi.binance.com"


def sign_params(params: Dict[str, Any], secret: str) -> Dict[str, Any]:
    qs = "&".join(f"{k}={params[k]}" for k in sorted(params))
    sig = hmac.new(secret.encode(), qs.encode(), hashlib.sha256).hexdigest()
    params["signature"] = sig
    return params


def send_request(method: str, path: str, api_key: str, api_secret: str, base_url: str, params: Dict[str, Any] = None, signed: bool = False):
    params = params or {}
    if signed:
        params["timestamp"] = int(time.time() * 1000)
        params = sign_params(params, api_secret)
    headers = {"X-MBX-APIKEY": api_key}
    url = base_url + path
    resp = requests.request(method, url, headers=headers, params=params, timeout=10)
    resp.raise_for_status()
    return resp.json()


def cmd_price(args, api_key, api_secret, base_url):
    data = send_request("GET", "/fapi/v1/ticker/price", api_key, api_secret, base_url, params={"symbol": args.symbol})
    print(data)


def cmd_balance(args, api_key, api_secret, base_url):
    data = send_request("GET", "/fapi/v2/balance", api_key, api_secret, base_url, signed=True)
    print(data)


def cmd_order(args, api_key, api_secret, base_url):
    payload = {
        "symbol": args.symbol,
        "side": args.side.upper(),
        "type": args.type.upper(),
        "quantity": args.qty,
        "newClientOrderId": f"paper_{int(time.time()*1000)}",
    }
    if args.type.upper() == "LIMIT":
        if args.price is None:
            raise SystemExit("LIMIT 单必须提供 --price")
        payload.update({"price": args.price, "timeInForce": "GTC"})
    data = send_request("POST", "/fapi/v1/order", api_key, api_secret, base_url, params=payload, signed=True)
    print(data)


def cmd_cancel(args, api_key, api_secret, base_url):
    payload = {"symbol": args.symbol}
    data = send_request("DELETE", "/fapi/v1/allOpenOrders", api_key, api_secret, base_url, params=payload, signed=True)
    print(data)


def parse_args():
    p = argparse.ArgumentParser(description="Binance 永续合约（默认 TESTNET）简易下单/查询脚本")
    p.add_argument("action", choices=["price", "balance", "order", "cancel"], help="操作类型")
    p.add_argument("--symbol", default="BTCUSDT", help="交易对，如 BTCUSDT")
    p.add_argument("--side", choices=["BUY", "SELL"], help="下单方向")
    p.add_argument("--type", choices=["MARKET", "LIMIT"], default="MARKET", help="订单类型")
    p.add_argument("--qty", type=float, help="下单数量")
    p.add_argument("--price", type=float, help="限价单价格")
    p.add_argument("--live", action="store_true", help="使用主网（默认测试网）。谨慎！")
    return p.parse_args()


def main():
    args = parse_args()
    api_key = os.environ.get("BINANCE_API_KEY")
    api_secret = os.environ.get("BINANCE_API_SECRET")
    if not api_key or not api_secret:
        raise SystemExit("请先设置 BINANCE_API_KEY / BINANCE_API_SECRET 环境变量")

    base_url = MAINNET_URL if args.live else TESTNET_URL

    if args.action == "price":
        cmd_price(args, api_key, api_secret, base_url)
    elif args.action == "balance":
        cmd_balance(args, api_key, api_secret, base_url)
    elif args.action == "order":
        if args.qty is None:
            raise SystemExit("--qty 必填")
        cmd_order(args, api_key, api_secret, base_url)
    elif args.action == "cancel":
        cmd_cancel(args, api_key, api_secret, base_url)


if __name__ == "__main__":
    main()
