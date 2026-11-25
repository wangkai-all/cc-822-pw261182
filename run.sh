#!/usr/bin/env bash
# Start the bias-correction web app on http://127.0.0.1:5050
set -euo pipefail
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
exec .venv/bin/python -m flask --app "app.web.app:create_app" run --host 127.0.0.1 --port "${PORT:-5050}"
