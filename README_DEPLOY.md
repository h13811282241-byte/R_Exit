# 部署指南

## 1. Docker 部署（推荐）

### 1.1 复制环境变量

```bash
cp .env.docker.example .env.docker
# 编辑 .env.docker，填入真实的 BINANCE_API_KEY/SECRET
```

也可以在 `.env.docker` 中添加其他自定义变量（例如代理配置、日志级别等），容器会在启动时自动读取。

### 1.2 构建并运行

```bash
docker compose up -d --build
```

- 容器会暴露 `8000` 端口，对应 Flask 控制台（由 gunicorn 提供服务）。
- 退出或更新时，使用 `docker compose down` 或 `docker compose up -d --build`。

### 1.3 访问

浏览器打开 `http://服务器IP:8000/`：

- 配置 symbol / stop / poll interval / testnet；
- 点击“启动策略”后，容器内会执行 `r_exit_manager.py` 并根据窗口日志实时反馈；
- API（`/api/start`, `/api/stop`, `/api/status`）同样在容器内暴露，可自定义集成。

### 1.4 Docker 常用命令

```bash
# 查看日志
docker compose logs -f

# 查看容器状态
docker compose ps

# 更新依赖后重新打包
docker compose up -d --build
```

## 2. 手动部署（保留方案）

如果不使用 Docker，可以继续采用虚拟环境手动部署：

### 2.1 打包并上传

```bash
bash package.sh
scp dist/r_exit_manager_bundle_*.tar.gz user@server:/opt/r-exit/
```

### 2.2 服务器解包 & 安装

```bash
cd /opt/r-exit
tar xzf r_exit_manager_bundle_*.tar.gz
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

### 2.3 配置环境变量

```bash
export BINANCE_API_KEY="你的 key"
export BINANCE_API_SECRET="你的 secret"
```

可写到 `~/.bashrc` 或单独的 `env.sh` 中，方便 systemd/cron 引用。

### 2.4 运行

```bash
source /opt/r-exit/.venv/bin/activate
python r_exit_manager.py --symbol ETHUSDT --stop 3070 --poll-interval 3
```

- `--testnet` 用于 U 本位合约测试网。
- 运行前请先手动建立持仓，并确认止损价。

### 2.5 后台运行

#### nohup / tmux

```bash
nohup /opt/r-exit/.venv/bin/python r_exit_manager.py --symbol ... > logs/run.log 2>&1 &
```

#### systemd

`/etc/systemd/system/r-exit-manager.service`

```ini
[Unit]
Description=Binance R exit manager
After=network.target

[Service]
WorkingDirectory=/opt/r-exit
Environment="BINANCE_API_KEY=xxx"
Environment="BINANCE_API_SECRET=yyy"
ExecStart=/opt/r-exit/.venv/bin/python r_exit_manager.py --symbol ETHUSDT --stop 3070 --poll-interval 3
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now r-exit-manager.service
```

### 2.6 Flask 控制台（手动模式）

```bash
cd /opt/r-exit
source .venv/bin/activate
python control_server.py
```

访问 `http://服务器IP:8000/`，可以在页面上启动/停止策略并查看日志。同样可以用 systemd 守护：

```ini
/etc/systemd/system/r-exit-web.service

[Unit]
Description=R Exit Flask control panel
After=network.target

[Service]
WorkingDirectory=/opt/r-exit
Environment="BINANCE_API_KEY=xxx"
Environment="BINANCE_API_SECRET=yyy"
ExecStart=/opt/r-exit/.venv/bin/python control_server.py
Restart=on-failure

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now r-exit-web.service
```

## 3. 日志与升级

- Docker 模式：`docker compose logs -f`，升级时 `docker compose up -d --build`。
- 手动模式：stdout、`logs/*.log` 或 systemd 的 journald。
- 重新打包：`bash package.sh`，上传解压后覆盖旧文件并重启服务。*** End Patch*** End Patch*** End Patch*** End Patch*** End Patch*** End Patch*** End Patch*** End Patch*** End Patch*** End Patch***
