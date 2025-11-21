import pandas as pd
import numpy as np
from .schemas import (
    COL_TIME, COL_SYMBOL, COL_SIDE, COL_PRICE, COL_QTY, COL_NOTIONAL, COL_FEE,
    COL_FEE_CCY, COL_PNL, COL_QUOTE, GroupConfig
)

def summarize_order(group: pd.DataFrame, cfg: GroupConfig) -> pd.Series:
    qty = group[COL_QTY].sum()
    price_vwap = (group[COL_PRICE] * group[COL_QTY]).sum() / qty if qty != 0 else np.nan
    notional = group[COL_NOTIONAL].sum()
    fee = group[COL_FEE].sum()
    pnl = group[COL_PNL].sum()
    start_ts = group[COL_TIME].min()
    end_ts = group[COL_TIME].max()
    duration = (end_ts - start_ts).total_seconds()
    side = group[COL_SIDE].iloc[0]
    symbol = group[COL_SYMBOL].iloc[0]
    fee_ccy = group[COL_FEE_CCY].mode().iloc[0] if not group[COL_FEE_CCY].empty else None
    quote = group[COL_QUOTE].mode().iloc[0] if not group[COL_QUOTE].empty else None

    out = pd.Series({
        "开始时间(UTC)": start_ts,
        "结束时间(UTC)": end_ts,
        "持续秒数": duration,
        "合约": symbol,
        "方向": side,
        "成交笔数": len(group),
        "汇总数量": qty,
        "加权均价(VWAP)": price_vwap,
        "汇总成交额": notional,
        "汇总手续费": fee,
        "手续费结算币种": fee_ccy,
        "汇总已实现盈亏": pnl,
        "计价资产": quote,
    })

    if cfg.round_qty is not None:
        out["汇总数量"] = round(out["汇总数量"], cfg.round_qty)
    if cfg.round_price is not None:
        out["加权均价(VWAP)"] = round(out["加权均价(VWAP)"], cfg.round_price)
    if cfg.round_money is not None:
        for k in ["汇总成交额", "汇总手续费", "汇总已实现盈亏"]:
            out[k] = round(out[k], cfg.round_money)
    return out
