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
from collections import deque
from pathlib import Path
from typing import Optional, Dict, Any

from flask import Flask, jsonify, request, render_template

BASE_DIR = Path(__file__).resolve().parent
SCRIPT_PATH = BASE_DIR / "r_exit_manager.py"

app = Flask(__name__, template_folder=str(BASE_DIR / "templates"))

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


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8000"))
    app.run(host="0.0.0.0", port=port)
