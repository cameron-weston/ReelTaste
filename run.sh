#!/bin/bash
# Start Reel Taste at http://127.0.0.1:8765  (rebuilds the front end if sources changed)
cd "$(dirname "$0")"
if [ ! -d web/node_modules ]; then (cd web && npm install --silent); fi
if [ ! -f web/dist/index.html ] || [ -n "$(find web/src -newer web/dist/index.html 2>/dev/null)" ]; then
  (cd web && npx tsc -b && npx vite build >/dev/null)
fi
exec .venv/bin/python -m reel_taste serve "${1:-8765}"
