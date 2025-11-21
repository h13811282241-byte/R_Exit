#!/usr/bin/env python3
"""
并行跑多组参数/时间段的小脚本。
编辑 CONFIGS 列表，执行本脚本即可按 CPU 核心数并行调用 main.py。
"""

import os
import subprocess
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import List, Dict, Any

# 根据需要调整参数组合
CONFIGS: List[Dict[str, Any]] = [
    {
        "symbol": "ETHUSDT",
        "interval": "5m",
        "start": "2025-09-01 00:00:00",
        "end": "2025-10-01 00:00:00",
        "market_type": "usdt_perp",
        "vol_spike_mult": "2.2",
        "quiet_max_mult": "0",
        "body_mult": "1.9",
        "tp_ratio": "1",
        "sl_offset_ratio": "1",
        "tp_ref": "prev",
        "sl_ref": "prev",
        "cooldown_mode": "vol",
        "cooldown_vol_mult": "1.0",
        "cooldown_quiet_bars": "3",
        "invert_side": False,
        "lower_interval": "1m",  # 留空则不拉下级周期
    },
    # 再加更多组合...
]


def build_cmd(cfg: Dict[str, Any]) -> List[str]:
    cmd = [
        "python",
        "main.py",
        "--symbol",
        cfg["symbol"],
        "--interval",
        cfg.get("interval", "5m"),
        "--start",
        cfg["start"],
        "--end",
        cfg["end"],
        "--market_type",
        cfg.get("market_type", "spot"),
        "--vol_spike_mult",
        cfg.get("vol_spike_mult", "2.0"),
        "--quiet_max_mult",
        cfg.get("quiet_max_mult", "0"),
        "--body_mult",
        cfg.get("body_mult", "1.5"),
        "--tp_ratio",
        cfg.get("tp_ratio", "0.5"),
        "--sl_offset_ratio",
        cfg.get("sl_offset_ratio", "1.0"),
        "--tp_ref",
        cfg.get("tp_ref", "prev"),
        "--sl_ref",
        cfg.get("sl_ref", "prev"),
        "--cooldown_mode",
        cfg.get("cooldown_mode", "bars"),
        "--cooldown_vol_mult",
        cfg.get("cooldown_vol_mult", "1.0"),
        "--cooldown_quiet_bars",
        cfg.get("cooldown_quiet_bars", "3"),
    ]
    if cfg.get("invert_side"):
        cmd.append("--invert_side")
    if cfg.get("lower_interval"):
        cmd.extend(["--lower_interval", cfg["lower_interval"]])
    return cmd


def run_one(idx: int, cfg: Dict[str, Any]) -> Dict[str, Any]:
    cmd = build_cmd(cfg)
    proc = subprocess.run(cmd, capture_output=True, text=True)
    return {
        "idx": idx,
        "config": cfg,
        "returncode": proc.returncode,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
    }


def main():
    max_workers = min(len(CONFIGS), os.cpu_count() or 4)
    results = []
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_idx = {executor.submit(run_one, i, cfg): i for i, cfg in enumerate(CONFIGS)}
        for future in as_completed(future_to_idx):
            results.append(future.result())

    for r in sorted(results, key=lambda x: x["idx"]):
        print("=" * 60)
        print(f"[Run {r['idx']}] returncode={r['returncode']}")
        print(f"Config: start={r['config']['start']} end={r['config']['end']} vol={r['config'].get('vol_spike_mult')}")
        if r["stdout"]:
            print(r["stdout"])
        if r["stderr"]:
            print("stderr:", r["stderr"])


if __name__ == "__main__":
    main()
