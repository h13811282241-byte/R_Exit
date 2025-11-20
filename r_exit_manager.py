#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Binance U 本位合约 R 多倍止盈 + 追踪止损管理脚本

逻辑概述：
- 你手动开仓（多或空），脚本只负责“平仓和止损管理”，不负责开仓；
- 你在启动脚本时提供初始止损价 stop_price；
- 脚本根据当前持仓均价 entry_price 和 stop_price 计算 1R；
- 按 1/3, 1/3, 1/3 三段处理：
    1）浮盈达到 2R：市价平掉 1/3 仓位，不移动止损；
        -> 如果之后打回初始止损，整单刚好 0R（打平不亏）；
    2）浮盈达到 4R：再市价平掉 1/3 仓位；
        -> 此时已实现约 2R，剩余 1/3 为“趋势尾仓”；
        -> 从这里开始，给尾仓启用 R 追踪止损：stopR = maxR - 2R；
    3）之后每当浮盈创出新高 maxR，就更新追踪止损；
        -> 止损始终在“最高浮盈 - 2R”的位置（价格层面）。

依赖：
    pip install python-binance

环境变量：
    BINANCE_API_KEY, BINANCE_API_SECRET

用法示例：
    python r_exit_manager.py --symbol ETHUSDT --stop 3430 --poll-interval 3
"""

import os
import time
import math
import argparse
from typing import Optional

from binance.client import Client
from binance.enums import (
    SIDE_BUY,
    SIDE_SELL,
    FUTURE_ORDER_TYPE_MARKET,
    FUTURE_ORDER_TYPE_STOP_MARKET,
    TIME_IN_FORCE_GTC,
)


class BinanceRExitManager:
    def __init__(self, symbol: str, stop_price: float, poll_interval: float = 3.0, testnet: bool = False):
        api_key = os.getenv("BINANCE_API_KEY")
        api_secret = os.getenv("BINANCE_API_SECRET")
        if not api_key or not api_secret:
            raise RuntimeError("请先在环境变量中设置 BINANCE_API_KEY 和 BINANCE_API_SECRET")

        self.client = Client(api_key, api_secret, testnet=testnet)

        self.symbol = symbol.upper()
        self.stop_price_input = float(stop_price)
        self.poll_interval = poll_interval

        self.price_tick_size: Optional[float] = None
        self.qty_step_size: Optional[float] = None

        self.entry_price: Optional[float] = None
        self.position_side: Optional[str] = None  # LONG or SHORT
        self.position_amt: float = 0.0
        self.initial_qty: float = 0.0
        self.tail_qty: float = 0.0

        self.R_price: Optional[float] = None
        self.tp1_done = False
        self.tp2_done = False
        self.tp1_price: Optional[float] = None
        self.tp2_price: Optional[float] = None

        self.maxR: float = 0.0
        self.current_trailing_stop_R: Optional[float] = None

    # ---------- 工具函数部分 ----------

    def load_symbol_info(self):
        """获取合约 tickSize 与 stepSize"""
        info = self.client.futures_exchange_info()
        if not isinstance(info, dict) or "symbols" not in info:
            raise RuntimeError(f"获取交易所信息失败，返回内容: {info}")

        for symbol_info in info["symbols"]:
            if symbol_info["symbol"] != self.symbol:
                continue
            for f in symbol_info["filters"]:
                if f["filterType"] == "PRICE_FILTER":
                    self.price_tick_size = float(f["tickSize"])
                elif f["filterType"] == "LOT_SIZE":
                    self.qty_step_size = float(f["stepSize"])
            break
        if self.price_tick_size is None or self.qty_step_size is None:
            raise RuntimeError(f"未能解析 {self.symbol} 的价格/数量精度")

    def round_price(self, price: float) -> float:
        """按 tick size 向下取整"""
        if self.price_tick_size is None:
            return price
        return math.floor(price / self.price_tick_size) * self.price_tick_size

    def round_qty(self, qty: float) -> float:
        """按 step size 向下取整"""
        if self.qty_step_size is None:
            return qty
        return math.floor(qty / self.qty_step_size) * self.qty_step_size

    # ---------- 交易所交互 ----------

    def fetch_position(self) -> bool:
        """读取当前 symbol 的 U 本位持仓信息"""
        positions = self.client.futures_position_information(symbol=self.symbol)
        if not positions:
            print(f"[INFO] {self.symbol} 无持仓")
            return False

        pos = positions[0]
        amt = float(pos["positionAmt"])
        if amt == 0:
            print(f"[INFO] {self.symbol} 持仓数量为 0")
            return False

        self.position_amt = amt
        self.entry_price = float(pos["entryPrice"])
        self.initial_qty = abs(amt)
        self.position_side = "LONG" if amt > 0 else "SHORT"
        print(f"[INFO] 检测到持仓: side={self.position_side}, qty={self.position_amt}, entry={self.entry_price}")
        return True

    def get_mark_price(self) -> float:
        """获取标记价格"""
        ticker = self.client.futures_mark_price(symbol=self.symbol)
        return float(ticker["markPrice"])

    def cancel_all_stop_orders(self):
        """取消现有 STOP/STOP_MARKET 类订单"""
        open_orders = self.client.futures_get_open_orders(symbol=self.symbol)
        for order in open_orders:
            if "STOP" not in order["type"]:
                continue
            try:
                self.client.futures_cancel_order(symbol=self.symbol, orderId=order["orderId"])
                print(f"[INFO] 已取消原止损单: orderId={order['orderId']}, type={order['type']}")
            except Exception as exc:
                print(f"[WARN] 取消止损单失败: {exc}")

    def place_initial_stop(self):
        """根据用户输入 stop_price_input 下注初始 closePosition 止损"""
        side = SIDE_SELL if self.position_side == "LONG" else SIDE_BUY
        stop_price = self.round_price(self.stop_price_input)
        print(f"[INFO] 下初始止损单: side={side}, stopPrice={stop_price} (closePosition=True)")
        try:
            self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=stop_price,
                closePosition=True,
                timeInForce=TIME_IN_FORCE_GTC,
            )
        except Exception as exc:
            raise RuntimeError(f"下初始止损单失败: {exc}")

    def update_trailing_stop(self, stop_price: float):
        """更新尾仓追踪止损"""
        side = SIDE_SELL if self.position_side == "LONG" else SIDE_BUY
        stop_price = self.round_price(stop_price)
        print(f"[INFO] 更新追踪止损: side={side}, stopPrice={stop_price} (closePosition=True)")
        self.cancel_all_stop_orders()
        try:
            self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type=FUTURE_ORDER_TYPE_STOP_MARKET,
                stopPrice=stop_price,
                closePosition=True,
                timeInForce=TIME_IN_FORCE_GTC,
            )
        except Exception as exc:
            print(f"[ERROR] 更新追踪止损失败: {exc}")

    def close_partial_market(self, fraction: float):
        """按初始仓位比例 fraction 市价平仓"""
        qty = self.round_qty(self.initial_qty * fraction)
        if qty <= 0:
            print(f"[WARN] 计算出的平仓数量<=0, fraction={fraction}, initial_qty={self.initial_qty}")
            return
        side = SIDE_SELL if self.position_side == "LONG" else SIDE_BUY
        print(f"[INFO] 市价平仓: side={side}, qty={qty}, fraction={fraction:.4f}")
        try:
            self.client.futures_create_order(
                symbol=self.symbol,
                side=side,
                type=FUTURE_ORDER_TYPE_MARKET,
                quantity=qty,
                reduceOnly=True,
            )
        except Exception as exc:
            print(f"[ERROR] 市价平仓失败: {exc}")

    # ---------- R 逻辑 ----------

    def compute_R_price(self):
        """根据 entry_price 和 stop_price_input 计算 1R"""
        if self.entry_price is None:
            raise RuntimeError("entry_price 为空，先调用 fetch_position()")
        if self.position_side == "LONG":
            if self.stop_price_input >= self.entry_price:
                raise RuntimeError(f"多单止损价({self.stop_price_input})应低于开仓价({self.entry_price})")
            self.R_price = self.entry_price - self.stop_price_input
        else:
            if self.stop_price_input <= self.entry_price:
                raise RuntimeError(f"空单止损价({self.stop_price_input})应高于开仓价({self.entry_price})")
            self.R_price = self.stop_price_input - self.entry_price
        print(f"[INFO] 计算得到 1R 的价格距离: R_price={self.R_price:.6f}")
        self.tp1_price = self.R_to_price(2.0)
        self.tp2_price = self.R_to_price(4.0)
        print(f"[INFO] 止盈位: 2R={self.tp1_price:.6f}, 4R={self.tp2_price:.6f}")

    def price_to_R(self, current_price: float) -> float:
        """当前价格对应的浮盈 R 值"""
        if not self.R_price:
            return 0.0
        move = current_price - self.entry_price if self.position_side == "LONG" else self.entry_price - current_price
        return move / self.R_price

    def R_to_price(self, R: float) -> float:
        """将 R 值转换为价格"""
        if self.R_price is None:
            raise RuntimeError("R_price 未初始化")
        if self.position_side == "LONG":
            return self.entry_price + R * self.R_price
        return self.entry_price - R * self.R_price

    # ---------- 主循环 ----------

    def run(self):
        print(f"[INFO] === 启动 R 出场管理: symbol={self.symbol}, stop_input={self.stop_price_input} ===")
        self.load_symbol_info()

        if not self.fetch_position():
            print("[ERROR] 当前无持仓，脚本退出。请先手动开仓。")
            return

        self.compute_R_price()
        self.cancel_all_stop_orders()
        self.place_initial_stop()

        self.tail_qty = self.initial_qty / 3.0

        print("[INFO] 开始轮询价格并执行 R 管理逻辑 ...")
        print("[INFO] 退出脚本请按 Ctrl + C")

        try:
            while True:
                if not self.fetch_position():
                    print("[INFO] 持仓已关闭，脚本结束。")
                    break

                price = self.get_mark_price()
                currentR = self.price_to_R(price)
                if currentR > self.maxR:
                    self.maxR = currentR

                print(
                    f"[TICK] price={price:.4f}, currentR={currentR:.3f}, maxR={self.maxR:.3f}, "
                    f"tp1_done={self.tp1_done}, tp2_done={self.tp2_done}, "
                    f"trailR={self.current_trailing_stop_R}"
                )

                if not self.tp1_done and currentR >= 2.0:
                    print(f"[LOGIC] 触发 2R 止盈（价格≈{self.tp1_price:.4f}），平掉 1/3 仓位（不移动止损）")
                    self.close_partial_market(1.0 / 3.0)
                    self.tp1_done = True

                if not self.tp2_done and currentR >= 4.0:
                    print(f"[LOGIC] 触发 4R 止盈（价格≈{self.tp2_price:.4f}），平掉 1/3 仓位，并启动尾仓追踪止损")
                    self.close_partial_market(1.0 / 3.0)
                    self.tp2_done = True
                    self.current_trailing_stop_R = 2.0
                    stop_price = self.R_to_price(self.current_trailing_stop_R)
                    self.update_trailing_stop(stop_price)

                if self.tp2_done:
                    desired_trail_R = self.maxR - 2.0
                    if desired_trail_R > (self.current_trailing_stop_R or -1e9) and desired_trail_R >= 2.0:
                        self.current_trailing_stop_R = desired_trail_R
                        stop_price = self.R_to_price(self.current_trailing_stop_R)
                        print(
                            f"[LOGIC] 更新尾仓追踪止损: trailR={self.current_trailing_stop_R:.3f}, "
                            f"stopPrice={stop_price:.4f}"
                        )
                        self.update_trailing_stop(stop_price)

                time.sleep(self.poll_interval)

        except KeyboardInterrupt:
            print("\n[INFO] 捕获到 Ctrl+C，手动终止脚本。")
        except Exception as exc:
            print(f"[ERROR] 主循环异常: {exc}")


def parse_args():
    parser = argparse.ArgumentParser(description="Binance U 本位合约 R 出场管理脚本")
    parser.add_argument("--symbol", required=True, help="交易对，例如 ETHUSDT, BTCUSDT")
    parser.add_argument("--stop", required=True, type=float, help="初始止损价格（你事先算好的价格）")
    parser.add_argument("--poll-interval", type=float, default=3.0, help="轮询价格的秒数间隔，默认3秒")
    parser.add_argument("--testnet", action="store_true", help="是否使用合约 testnet（需要你有 testnet 账户）")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    manager = BinanceRExitManager(
        symbol=arguments.symbol,
        stop_price=arguments.stop,
        poll_interval=arguments.poll_interval,
        testnet=arguments.testnet,
    )
    manager.run()
