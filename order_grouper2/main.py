#!/usr/bin/env python3
import argparse
from pathlib import Path
from order_grouper.io_utils import read_trades, write_orders
from order_grouper.schemas import GroupConfig
from order_grouper.grouping import group_trades_into_orders

def build_parser():
    p = argparse.ArgumentParser(description="交割单撮合成交汇总为订单（按2秒窗口合并）")
    p.add_argument("input", help="输入CSV/TSV文件路径（含表头）")
    p.add_argument("-o", "--output", default=None, help="输出汇总订单CSV路径，默认 input_orders.csv")
    p.add_argument("--detail", default=None, help="输出带订单ID的明细CSV路径（可选）")
    p.add_argument("--window", type=float, default=2.0, help="订单内最大成交间隔（秒），默认2")
    p.add_argument("--dt-format", default=None, help="明确的时间格式，例如 '%%Y-%%m-%%d %%H:%%M:%%S.%%f'（可选）")
    p.add_argument("--round-price", type=int, default=None, help="价格小数位（可选）")
    p.add_argument("--round-qty", type=int, default=None, help="数量小数位（可选）")
    p.add_argument("--round-money", type=int, default=2, help="金额/手续费/盈亏小数位（默认2）")
    return p

def main():
    args = build_parser().parse_args()
    cfg = GroupConfig(
        window_seconds=args.window,
        round_price=args.round_price,
        round_qty=args.round_qty,
        round_money=args.round_money,
    )
    df = read_trades(args.input, dt_format=args.dt_format, tz="UTC")
    orders, detailed = group_trades_into_orders(df, cfg)

    out_path = args.output or str(Path(args.input).with_suffix("").as_posix() + "_orders.csv")
    write_orders(orders, out_path)
    if args.detail:
        detailed.to_csv(args.detail, index=False, encoding="utf-8-sig")

    print(f"订单汇总已写入: {out_path}")
    if args.detail:
        print(f"带订单ID明细已写入: {args.detail}")

if __name__ == "__main__":
    main()
