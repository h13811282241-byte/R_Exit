import pandas as pd
from .schemas import COL_TIME, COL_SYMBOL, COL_SIDE, GroupConfig
from .metrics import summarize_order

def _assign_order_ids(df: pd.DataFrame, cfg: GroupConfig) -> pd.Series:
    order_ids = []
    current_id = 1
    df_sorted = df.sort_values([COL_SYMBOL, COL_SIDE, COL_TIME]).copy()
    id_series = pd.Series(index=df_sorted.index, dtype="int64")
    for (sym, side), sub in df_sorted.groupby([COL_SYMBOL, COL_SIDE]):
        prev_ts = None
        for idx, row in sub.iterrows():
            ts = row[COL_TIME]
            if prev_ts is None:
                id_series.loc[idx] = current_id
            else:
                gap = (ts - prev_ts).total_seconds()
                if gap > cfg.window_seconds:
                    current_id += 1
                id_series.loc[idx] = current_id
            prev_ts = ts
        current_id += 1
    return id_series.reindex(df.index)

def group_trades_into_orders(df: pd.DataFrame, cfg: GroupConfig):
    order_ids = _assign_order_ids(df, cfg)
    df2 = df.copy()
    df2["订单ID"] = order_ids
    orders = (df2.groupby("订单ID", group_keys=False)
                 .apply(lambda g: summarize_order(g, cfg))
                 .reset_index())
    orders["订单ID"] = orders["订单ID"].astype(str)
    return orders, df2
