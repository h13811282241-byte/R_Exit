from dataclasses import dataclass
from typing import Optional

COL_TIME = "时间(UTC)"
COL_SYMBOL = "合约"
COL_SIDE = "方向"
COL_PRICE = "价格"
COL_QTY = "数量"
COL_NOTIONAL = "成交额"
COL_FEE = "手续费"
COL_FEE_CCY = "手续费结算币种"
COL_PNL = "已实现盈亏"
COL_QUOTE = "计价资产"

@dataclass
class GroupConfig:
    window_seconds: float = 2.0
    round_price: Optional[int] = None
    round_qty: Optional[int] = None
    round_money: Optional[int] = 2
