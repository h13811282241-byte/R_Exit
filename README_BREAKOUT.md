# 趋势突破策略（1h ETHUSDT 永续示例）

## 核心逻辑
- 主周期 1h，EMA 趋势过滤（EMA_100，只顺势做单）。
- Donchian 通道突破（默认 N=24，不含当前），加 ATR 缓冲（buffer = 0.1*ATR20）。
- 波动过滤：ATR20 > 最近 100 根 ATR20 中位数。
- 放量过滤：当前 volume >= 过去 20 根均量 * vol_mult（默认 1.5）。
- 开仓：突破当根收盘价进场。
- 止损：ATR 止损，距离 = k_sl * ATR（默认 1.5）。
- 止盈/止损：固定 TP = entry ± R_target*risk_abs（默认 3R），同时用 trailing stop (k_trail*ATR) 推止损。
- 手续费：单边 fee_side（默认 0.000248）。净 R = raw_R - fee_R，其中 fee_R = fee_round / risk_pct，fee_round=2*fee_side。
- 冲突判序：如同一根同时触及 TP/SL，可选下级周期（如 1m）判先后，否则保守先 SL。

## 运行示例
```bash
export BINANCE_API_KEY=... BINANCE_API_SECRET=...
python breakout_main.py --symbol ETHUSDT --interval 1h \
  --start "2025-01-01 00:00:00" --end "2025-03-01 00:00:00" \
  --market_type usdt_perp \
  --vol_spike_mult 1.5 --body_mult 1.5 --quiet_max_mult 0 \
  --k_sl 1.5 --R_target 3 --k_trail 2 \
  --lower_interval 1m \
  --us_session_mode all \
  --plot
```

## 主要参数
- 趋势/突破：`ema_span`(100), `donchian_n`(24), `atr_period`(20), `k_buffer`(0.1)
- 波动/量：`atr_median_lookback`(100), `vol_lookback`(20), `vol_mult`(1.5), `quiet_max_mult`(0 表示不限制)
- 止盈止损：`k_sl`(1.5), `R_target`(3), `k_trail`(2)
- 手续费：`fee_side` 单边费率
- 下级周期：`lower_interval`（如 1m），仅在同根 TP/SL 冲突时用于判先后
- 美股时段过滤：`us_session_mode` = all / us_only / non_us

## 输出
- 终端打印：交易笔数、胜率、平均净 R、盈亏比、最近 5 次开仓时间（北京时间）。
- 图表（可选）：价格+信号、权益曲线，输出到 `plots_breakout/`。

## 注意
- 长时间窗＋下级周期会耗时，可分段跑或去掉 `--lower_interval`。
- 输入 CSV 需含 `timestamp, open, high, low, close, volume`（UTC）。***
