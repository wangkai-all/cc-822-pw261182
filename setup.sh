#!/usr/bin/env bash
# Create the virtual environment and install pinned dependencies.
set -euo pipefail
cd "$(dirname "$0")"
python3 -m venv .venv
.venv/bin/pip install --upgrade pip
.venv/bin/pip install -r requirements.txt
echo "依赖安装完成。运行 ./run.sh 启动应用，或 .venv/bin/pytest 运行测试。"
