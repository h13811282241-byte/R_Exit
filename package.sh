#!/usr/bin/env bash
set -euo pipefail

DIST_DIR="dist"
mkdir -p "$DIST_DIR"

timestamp=$(date +"%Y%m%d%H%M%S")
archive="${DIST_DIR}/r_exit_manager_bundle_${timestamp}.tar.gz"

tar -czf "$archive" \
    r_exit_manager.py \
    control_server.py \
    requirements.txt \
    README_DEPLOY.md \
    templates/index.html \
    Dockerfile \
    docker-compose.yml \
    .env.docker.example \
    .dockerignore

echo "已生成打包文件: ${archive}"
