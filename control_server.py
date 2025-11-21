#!/usr/bin/env python3
"""
Flask 控制面板：用于启动/停止 Binance R Exit 管理脚本，并查看实时日志。
"""

import os
import sys
import time
import signal
import subprocess
import threading
import uuid
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
from flask import Flask, jsonify, request, render_template, send_from_directory
from werkzeug.utils import secure_filename

# 让 order_grouper 包可以被导入
ORDER_GROUPER_SRC = Path(__file__).resolve().parent / "order_grouper2"
if ORDER_GROUPER_SRC.exists() and str(ORDER_GROUPER_SRC) not in sys.path:
    sys.path.insert(0, str(ORDER_GROUPER_SRC))
from order_grouper.io_utils import read_trades, write_orders
from order_grouper.schemas import GroupConfig
from order_grouper.grouping import group_trades_into_orders

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BASE_DIR / "r_exit_manager.py"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

# 订单撮合文件目录（可用环境变量 ORDER_GROUPER_BASE 调整）
ORDER_GROUPER_BASE = Path(os.environ.get("ORDER_GROUPER_BASE", BASE_DIR / "order_data"))
ORDER_UPLOAD_DIR = ORDER_GROUPER_BASE / "uploads"
ORDER_OUTPUT_DIR = ORDER_GROUPER_BASE / "outputs"
for path in (ORDER_UPLOAD_DIR, ORDER_OUTPUT_DIR):
    path.mkdir(parents=True, exist_ok=True)
ALLOWED_UPLOAD_EXTS = {"csv", "tsv", "txt", "xls", "xlsx"}

_state_lock = threading.Lock()
_worker: Dict[str, Any] = {
    "process": None,
    "thread": None,
    "start_time": None,
    "config": None,
}
_logs = deque(maxlen=500)


def _append_log(line: str) -> None:
    timestamp = time.time()
    entry = {"timestamp": timestamp, "line": line}
    with _state_lock:
        _logs.append(entry)


def _reader_thread(proc: subprocess.Popen) -> None:
    try:
        for raw_line in proc.stdout:
            _append_log(raw_line.rstrip())
    finally:
        proc.wait()
        with _state_lock:
            _worker["process"] = None
            _worker["thread"] = None
            _worker["config"] = None
            _worker["start_time"] = None


def _is_running() -> bool:
    with _state_lock:
        proc = _worker["process"]
    return proc is not None and proc.poll() is None


@app.route("/")
def index():
    return render_template("index.html")


@app.post("/api/start")
def start_session():
    payload = request.get_json(force=True, silent=True) or {}
    symbol = payload.get("symbol")
    stop_price = payload.get("stop")
    poll_interval = payload.get("pollInterval", 3)
    testnet = bool(payload.get("testnet"))

    if not symbol or stop_price is None:
        return jsonify({"error": "symbol 与 stop 为必填项"}), 400

    with _state_lock:
        if _worker["process"] and _worker["process"].poll() is None:
            return jsonify({"error": "已有任务在运行"}), 409

    cmd = [
        sys.executable,
        str(SCRIPT_PATH),
        "--symbol",
        str(symbol).upper(),
        "--stop",
        str(stop_price),
        "--poll-interval",
        str(poll_interval),
    ]
    if testnet:
        cmd.append("--testnet")

    env = os.environ.copy()
    proc = subprocess.Popen(
        cmd,
        cwd=str(BASE_DIR),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    reader = threading.Thread(target=_reader_thread, args=(proc,), daemon=True)

    with _state_lock:
        _worker["process"] = proc
        _worker["thread"] = reader
        _worker["start_time"] = time.time()
        _worker["config"] = {
            "symbol": str(symbol).upper(),
            "stop": float(stop_price),
            "pollInterval": float(poll_interval),
            "testnet": testnet,
        }
        _logs.clear()

    _append_log(f"[SERVER] 启动命令: {' '.join(cmd)}")
    reader.start()
    return jsonify({"message": "任务已启动"}), 201


@app.post("/api/stop")
def stop_session():
    with _state_lock:
        proc: Optional[subprocess.Popen] = _worker.get("process")
    if not proc or proc.poll() is not None:
        return jsonify({"message": "当前无运行中的任务"}), 200

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
    except subprocess.TimeoutExpired:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()
    _append_log("[SERVER] 已请求停止任务")
    return jsonify({"message": "停止指令已发送"}), 200


@app.get("/api/status")
def status():
    with _state_lock:
        running = _worker["process"] is not None and _worker["process"].poll() is None
        config = _worker["config"]
        start_time = _worker["start_time"]
        logs = list(_logs)
    return jsonify(
        {
            "running": running,
            "config": config,
            "startTime": start_time,
            "logs": logs,
        }
    )


def _parse_optional_int(value: Optional[str]) -> Optional[int]:
    if value is None or str(value).strip() == "":
        return None
    return int(value)


@app.post("/api/order-group")
def order_group():
    """
    上传交割成交表（CSV/TSV/Excel），按时间窗口合并为订单，返回汇总/明细下载链接。
    可通过环境变量 ORDER_GROUPER_BASE 调整存储目录。
    """
    file = request.files.get("file")
    if not file or not file.filename:
        return jsonify({"error": "请上传文件"}), 400

    safe_name = secure_filename(file.filename) or "upload.csv"
    ext = safe_name.rsplit(".", 1)[-1].lower() if "." in safe_name else ""
    if ext not in ALLOWED_UPLOAD_EXTS:
        return jsonify({"error": f"不支持的文件类型: {ext}"}), 400

    window_seconds = float(request.form.get("windowSeconds", 2))
    dt_format = request.form.get("dtFormat") or None
    round_price = _parse_optional_int(request.form.get("roundPrice"))
    round_qty = _parse_optional_int(request.form.get("roundQty"))
    round_money = _parse_optional_int(request.form.get("roundMoney"))

    unique_prefix = f"{int(time.time())}_{uuid.uuid4().hex[:8]}"
    upload_path = ORDER_UPLOAD_DIR / f"{unique_prefix}_{safe_name}"
    file.save(upload_path)

    temp_csv = None
    input_path = upload_path
    try:
        if ext in {"xls", "xlsx"}:
            temp_csv = ORDER_UPLOAD_DIR / f"{unique_prefix}_{Path(safe_name).stem}.csv"
            df_excel = pd.read_excel(upload_path)
            df_excel.to_csv(temp_csv, index=False)
            input_path = temp_csv

        cfg = GroupConfig(
            window_seconds=window_seconds,
            round_price=round_price,
            round_qty=round_qty,
            round_money=round_money,
        )
        trades = read_trades(str(input_path), dt_format=dt_format, tz="UTC")
        orders, detailed = group_trades_into_orders(trades, cfg)

        base_stem = Path(safe_name).stem
        orders_path = ORDER_OUTPUT_DIR / f"{unique_prefix}_{base_stem}_orders.csv"
        detail_path = ORDER_OUTPUT_DIR / f"{unique_prefix}_{base_stem}_detailed.csv"
        write_orders(orders, str(orders_path))
        detailed.to_csv(detail_path, index=False, encoding="utf-8-sig")

        return (
            jsonify(
                {
                    "message": "处理成功",
                    "ordersUrl": f"/downloads/{orders_path.name}",
                    "detailUrl": f"/downloads/{detail_path.name}",
                    "storedAt": str(ORDER_OUTPUT_DIR),
                }
            ),
            201,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:
        return jsonify({"error": "处理失败", "detail": str(exc)}), 500
    finally:
        if temp_csv and temp_csv.exists():
            temp_csv.unlink(missing_ok=True)


@app.get("/downloads/<path:filename>")
def download_file(filename: str):
    safe_name = secure_filename(filename)
    target = ORDER_OUTPUT_DIR / safe_name
    if not target.exists():
        return jsonify({"error": "文件不存在"}), 404
    return send_from_directory(ORDER_OUTPUT_DIR, safe_name, as_attachment=True)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
