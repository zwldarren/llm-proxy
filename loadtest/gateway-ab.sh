#!/usr/bin/env bash
# Head-to-head gateway benchmark driver: llm-proxy vs LiteLLM (v1.101.0).
#
#   uv run loadtest/gateway-ab.sh
#
# Fairness contract enforced here (see loadtest/README.md):
#   * identical upstream — the deterministic zero-latency fake-upstream
#   * identical resources — cpus: 8 and 8 uvicorn workers on both gateways
#   * identical stores    — same Postgres, same Redis, same 10-conn/worker pool
#   * identical client    — same locust file, mix, ramp and duration
#   * sequential          — the idle gateway is STOPPED, never co-resident
#   * interleaved rounds  — order is reversed on round 2 to cancel drift
#
# Env: LEVELS, DUR, RAMP, REPS, OUT, MIX (core|mixed).
set -uo pipefail

cd "$(dirname "$0")/.."

export LOADTEST_WORKERS=8
export LOADTEST_LOG_LEVEL=WARNING
export LOADTEST_REDIS=true

COMPOSE=(docker compose -f docker-compose.yaml -f docker-compose.loadtest.yaml
         -f loadtest/docker-compose.bench.yaml)
OUT="${OUT:-loadtest/results/ab-$(date -u +%F)}"
LEVELS="${LEVELS:-200 400 800}"
DUR="${DUR:-60s}"
RAMP="${RAMP:-400}"
REPS="${REPS:-2}"
MIX="${MIX:-mixed}"

case "$MIX" in
  core)  SCEN="chat_openai_compat_nonstream,chat_openai_compat_stream" ;;
  mixed) SCEN="chat_openai_compat_nonstream,chat_openai_compat_stream,messages_anthropic_native_stream,embeddings_openai_compat_nonstream" ;;
  *) echo "unknown MIX=$MIX" >&2; exit 2 ;;
esac

PROXY_URL="http://localhost:8180"
LITELLM_URL="http://localhost:4000"
DIRECT_URL="http://localhost:8900"
PROXY_KEY_FILE="loadtest/.api_key-19159d47"
LITELLM_KEY="sk-litellm-loadtest"

mkdir -p "$OUT"
: > "$OUT/SWEEP.log"

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$OUT/SWEEP.log"; }

wait_http() {  # url timeout_s
  local url=$1 timeout=${2:-180} i=0
  until curl -fsS -o /dev/null "$url" 2>/dev/null; do
    i=$((i + 1))
    if (( i > timeout )); then log "TIMEOUT waiting for $url"; return 1; fi
    sleep 1
  done
}

compose() { "${COMPOSE[@]}" "$@"; }

stop_all() {
  compose stop llm-proxy litellm >/dev/null 2>&1
}

start_proxy() {
  compose stop litellm >/dev/null 2>&1
  compose up -d llm-proxy >/dev/null 2>&1
  wait_http "$PROXY_URL/api/health" 180 || return 1
}

start_litellm() {
  compose stop llm-proxy >/dev/null 2>&1
  compose up -d litellm >/dev/null 2>&1
  wait_http "$LITELLM_URL/health/liveliness" 300 || return 1
}

# Sample container CPU/mem and host load for the duration of one run.
# The container is resolved from the compose service at call time — compose
# prefixes container names with the project name (the checkout directory), so
# hardcoding "llm-proxy-*" silently yields nothing elsewhere.
sample_resources() {  # service outfile
  local service=$1 out=$2
  local cid cname
  cid=$(compose ps -q "$service" 2>/dev/null)
  if [[ -z "$cid" ]] || ! cname=$(docker inspect --format '{{.Name}}' "$cid" 2>/dev/null); then
    log "sample_resources: no running container for compose service '$service'; skipping resource samples"
    return 1
  fi
  cname=${cname#/}
  : > "$out"
  while true; do
    local line
    line=$(docker stats --no-stream --format '{{.CPUPerc}},{{.MemUsage}}' "$cname" 2>/dev/null)
    printf '%s,%s,%s\n' "$(date -u +%H:%M:%S)" "$(cut -d' ' -f1 /proc/loadavg)" "$line" >> "$out"
    sleep 3
  done
}

run_locust() {  # tag url scen key users [duration]
  local tag=$1 url=$2 scen=$3 key=$4 users=$5 dur=${6:-$DUR}
  local name="${tag}-u${users}"
  LOADTEST_SCENARIOS="$scen" API_KEY="$key" LOADTEST_LATENCY_DUMP="$OUT/$name.latency.csv" \
    uv run locust -f loadtest/locustfile.py \
    --host "$url" --headless -u "$users" -r "$RAMP" -t "$dur" \
    --csv "$OUT/$name" --only-summary --html "$OUT/$name.html" \
    > "$OUT/$name.log" 2>&1
  return 0
}

log "=== gateway A/B sweep: mix=$MIX levels='$LEVELS' dur=$DUR ramp=$RAMP reps=$REPS ==="
log "scenarios: $SCEN"

# --- one-time init: (re)create both gateways so every worker loads the seeded
# topology, and so the image is the freshly built working tree.
log "init: recreating llm-proxy (fresh image, 8 workers) ..."
compose up -d --force-recreate llm-proxy >/dev/null 2>&1
wait_http "$PROXY_URL/api/health" 240 || exit 1
log "init: llm-proxy healthy"

log "init: recreating litellm (v1.101.0, 8 workers) ..."
compose up -d --force-recreate litellm >/dev/null 2>&1
t0=$(date +%s)
wait_http "$LITELLM_URL/health/liveliness" 300 || exit 1
log "init: litellm healthy after $(( $(date +%s) - t0 ))s"

# --- warmup: prime connection pools / JIT without recording results.
log "warmup: llm-proxy 60 users 20s"
start_proxy && run_locust "warmup-proxy" "$PROXY_URL" "$SCEN" "$(cat "$PROXY_KEY_FILE")" 60 20s
log "warmup: litellm 60 users 20s"
start_litellm && run_locust "warmup-litellm" "$LITELLM_URL" "$SCEN" "$LITELLM_KEY" 60 20s

proxy_key=$(cat "$PROXY_KEY_FILE")

# --- measured rounds. Round 2 walks the levels in reverse to cancel any
# monotonic drift in host conditions.
for rep in $(seq 1 "$REPS"); do
  if (( rep % 2 == 1 )); then order=$LEVELS; else order=$(echo "$LEVELS" | tr ' ' '\n' | tac | tr '\n' ' '); fi
  log "--- round $rep (order: $order) ---"

  for users in $order; do
    # Baseline: the fake upstream directly, no gateway co-resident.
    stop_all
    log "run r$rep-direct-u$users"
    sample_resources fake-upstream "$OUT/r$rep-direct-u$users.resources.csv" &
    SAMPLE_PID=$!
    run_locust "r$rep-direct" "$DIRECT_URL" "$SCEN" "sk-none" "$users"
    kill "$SAMPLE_PID" 2>/dev/null

    log "run r$rep-proxy-u$users"
    start_proxy || { log "proxy failed to start"; continue; }
    sample_resources llm-proxy "$OUT/r$rep-proxy-u$users.resources.csv" &
    SAMPLE_PID=$!
    run_locust "r$rep-proxy" "$PROXY_URL" "$SCEN" "$proxy_key" "$users"
    kill "$SAMPLE_PID" 2>/dev/null

    log "run r$rep-litellm-u$users"
    start_litellm || { log "litellm failed to start"; continue; }
    sample_resources litellm "$OUT/r$rep-litellm-u$users.resources.csv" &
    SAMPLE_PID=$!
    run_locust "r$rep-litellm" "$LITELLM_URL" "$SCEN" "$LITELLM_KEY" "$users"
    kill "$SAMPLE_PID" 2>/dev/null
  done
done

stop_all
log "=== sweep complete -> $OUT ==="
