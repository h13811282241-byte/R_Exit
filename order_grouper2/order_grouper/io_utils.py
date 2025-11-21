
import pandas as pd
import numpy as np
import re
from typing import Optional, Union
from .schemas import (
    COL_TIME, COL_SYMBOL, COL_SIDE, COL_PRICE, COL_QTY, COL_NOTIONAL, COL_FEE,
    COL_FEE_CCY, COL_PNL, COL_QUOTE
)

REQUIRED_COLS = [COL_TIME, COL_SYMBOL, COL_SIDE, COL_PRICE, COL_QTY,
                 COL_NOTIONAL, COL_FEE, COL_FEE_CCY, COL_PNL, COL_QUOTE]

# ---------------------- 清洗工具 ----------------------
_WS_RE = re.compile(r"\s+", re.UNICODE)

CURRENCY_SIGNS = "¥$€£₿"  # 常见货币符号
BAD_CHARS = set([","])     # 千分位逗号等

def _strip_spaces(x):
    if isinstance(x, str):
        x = _WS_RE.sub(" ", x).strip()
    return x

def _to_float(x):
    if x is None or (isinstance(x, float) and np.isnan(x)):
        return np.nan
    s = str(x).strip()
    if s == "" or s in {"-", "—", "–"}:
        return np.nan
    # 去除货币符号与千分位逗号
    for ch in CURRENCY_SIGNS:
        s = s.replace(ch, "")
    s = s.replace(",", "")
    # 括号负数: (123.45) -> -123.45
    if s.startswith("(") and s.endswith(")"):
        s = "-" + s[1:-1]
    # 百分号: 12.3% -> 0.123 （当前不预期出现，若出现按百分比处理）
    if s.endswith("%"):
        try:
            return float(s[:-1]) / 100.0
        except:
            return np.nan
    # 其它杂质字符（如单位）直接失败为NaN
    try:
        return float(s)
    except:
        return np.nan

SIDE_ALIASES = {
    "buy": "买入", "long": "买入", "open long": "买入", "开多": "买入", "买": "买入",
    "sell": "卖出", "short": "卖出", "open short": "卖出", "开空": "卖出", "卖": "卖出",
    "平多": "卖出", "平空": "买入",  # 简单近似，可按你的业务再细分
}

def _normalize_side(x: str) -> str:
    if not isinstance(x, str):
        return x
    key = x.strip().lower()
    return SIDE_ALIASES.get(key, x.strip())

# ---------------------- 读入主流程 ----------------------
def read_trades(path: Union[str, bytes], dt_format: Optional[str] = None, tz: str = "UTC") -> pd.DataFrame:
    """
    读取交割单CSV/TSV（所有内容为字符串也可）。
    - 自动识别分隔符
    - 全列先strip
    - 数值列强制从字符串清洗并转float
    - 时间列解析为UTC
    """
    df = pd.read_csv(path, sep=None, engine="python", dtype=str, na_filter=False)

    # 校验列
    missing = [c for c in REQUIRED_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"缺少必要列: {missing}，请检查表头是否为：{REQUIRED_COLS}")

    # 全表strip
    df = df.applymap(_strip_spaces)

    # 方向统一
    df[COL_SIDE] = df[COL_SIDE].map(_normalize_side)

    # 时间解析（允许多种格式，如含Z）
    if dt_format:
        df[COL_TIME] = pd.to_datetime(df[COL_TIME], format=dt_format, utc=True, errors="coerce")
    else:
        # 尝试让pandas自动识别：支持 2025-01-01 12:00:00.123 / 2025-01-01T12:00:00Z 等
        df[COL_TIME] = pd.to_datetime(df[COL_TIME].str.replace("Z","", regex=False), utc=True, errors="coerce")

    if df[COL_TIME].isna().any():
        bad = df[df[COL_TIME].isna()].index.tolist()[:10]
        raise ValueError(f"存在无法解析的时间行索引: {bad}，请检查时间字符串或提供 --dt-format。示例原值: {df.loc[bad, COL_TIME].tolist()}")

    # 数值列强制转float（从字符串）
    for c in [COL_PRICE, COL_QTY, COL_NOTIONAL, COL_FEE, COL_PNL]:
        df[c] = df[c].map(_to_float).astype(float)

    # 排序
    df = df.sort_values(COL_TIME).reset_index(drop=True)
    return df

def write_orders(df: pd.DataFrame, path: str) -> None:
    df.to_csv(path, index=False, encoding="utf-8-sig")
