#!/usr/bin/env bash
set -euo pipefail

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

if command -v uv >/dev/null 2>&1; then
  exec uv run -p 3.12 --with anthropic,requests,beautifulsoup4,numpy,pandas,lxml,httpx python main.py
fi

exec python main.py
