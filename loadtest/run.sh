#!/usr/bin/env bash
# End-to-end load-test runner: build the stack, seed the conversion-tier
# topology, restart the proxy workers so they all pick up the seeded config,
# then run locust. Extra arguments are passed straight to locust.
#
#   uv run loadtest/run.sh --headless -u 100 -r 20 -t 60s
#   LOADTEST_SCENARIOS=messages uv run loadtest/run.sh --headless -u 50 -t 30s
#
# Env: LOADTEST_WORKERS, LOADTEST_REDIS (see docker-compose.loadtest.yaml),
#      LOADTEST_SCENARIOS, SKIP_BUILD=1, SKIP_SEED=1, SKIP_RESTART=1,
#      PROXY_BASE_URL (default http://localhost:8180).
set -euo pipefail

cd "$(dirname "$0")/.."

COMPOSE=(docker compose -f docker-compose.yaml -f docker-compose.loadtest.yaml)
BASE="${PROXY_BASE_URL:-http://localhost:8180}"

if [[ "${SKIP_BUILD:-0}" != "1" ]]; then
  "${COMPOSE[@]}" up -d --build
else
  "${COMPOSE[@]}" up -d
fi

if [[ "${SKIP_SEED:-0}" != "1" ]]; then
  PROXY_BASE_URL="$BASE" uv run loadtest/seed.py
fi

# Config is per-worker in memory; the seed's admin calls reload only the worker
# that served them. Restarting makes every worker load the seeded topology.
if [[ "${SKIP_RESTART:-0}" != "1" ]]; then
  "${COMPOSE[@]}" restart llm-proxy
fi

echo "waiting for $BASE ..."
for _ in $(seq 1 60); do
  if python3 - "$BASE" <<'PY'
import sys, urllib.request
try:
    urllib.request.urlopen(sys.argv[1] + "/api/health", timeout=2)
except Exception:
    sys.exit(1)
PY
  then
    break
  fi
  sleep 2
done

exec uv run locust -f loadtest/locustfile.py --host "$BASE" "$@"
