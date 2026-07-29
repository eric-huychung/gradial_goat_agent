#!/usr/bin/env bash
# Build agent.zip (main.py at the root + every module it imports) and,
# optionally, submit it to the event server.
#
# Usage:
#   ./ship.sh            # build agent.zip only
#   ./ship.sh --submit   # build and POST it to /api/agent/submit
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")"

if [[ -f .env ]]; then
  set -a
  # shellcheck disable=SC1091
  source .env
  set +a
fi

ZIP_NAME="agent.zip"
FILES=(
  main.py
  jeopardy.py
  config.py
  monitor.py
  runner.py
  solver.py
  strategy.py
  submission.py
  tile_selector.py
  tools.py
  requirements.txt
)

for f in "${FILES[@]}"; do
  if [[ ! -f "$f" ]]; then
    echo "error: expected file missing: $f" >&2
    exit 1
  fi
done

STAGE="$(mktemp -d)"
trap 'rm -rf "$STAGE"' EXIT

for f in "${FILES[@]}"; do
  cp "$f" "$STAGE/"
done

# Ship only non-secret dev knobs (e.g. VERBOSE=1); the four connection vars
# are injected by the runner and must never travel in the zip.
if [[ -f .env ]]; then
  grep -Ev '^\s*export\s+(JEOPARDY_BASE_URL|TEAM_API_KEY|ANTHROPIC_BASE_URL|ANTHROPIC_API_KEY)=' .env > "$STAGE/.env" || true
  [[ -s "$STAGE/.env" ]] || rm -f "$STAGE/.env"
fi

rm -f "$ZIP_NAME"
(cd "$STAGE" && zip -X -r "$OLDPWD/$ZIP_NAME" .) >/dev/null

echo "built $ZIP_NAME:"
unzip -l "$ZIP_NAME"

if [[ "${1:-}" == "--submit" ]]; then
  : "${JEOPARDY_BASE_URL:?JEOPARDY_BASE_URL not set — source .env first}"
  : "${TEAM_API_KEY:?TEAM_API_KEY not set — source .env first}"
  echo "submitting to $JEOPARDY_BASE_URL/api/agent/submit ..."
  curl -sS -X POST "$JEOPARDY_BASE_URL/api/agent/submit" \
    -H "X-Api-Key: $TEAM_API_KEY" \
    -F "file=@$ZIP_NAME"
  echo
fi
