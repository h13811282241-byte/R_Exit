# 交割单撮合汇总工具（按时间窗口合并为“订单”）

规则：把同一合约、同一方向、相邻成交时间间隔不超过 2 秒 的成交，合并为一单。可用 --window 调整。

输入表头（UTF-8，CSV/TSV自动识别）：
时间(UTC), 合约, 方向, 价格, 数量, 成交额, 手续费, 手续费结算币种, 已实现盈亏, 计价资产

用法示例：
python main.py your_fills.csv --window 2 -o orders.csv --detail fills_with_id.csv
