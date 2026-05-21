#!/usr/bin/env bash
# start-amd.sh
# AMD Windows AI Stack — Start Script
# Run from WSL2 Ubuntu: ./start-amd.sh
# Options:
#   --no-frontend     skip frontend dev server
#   --backend-only    same as --no-frontend
#   --skip-ollama     skip Ollama check (if already running)

set -euo pipefail

# ─── Configuration ────────────────────────────────────────────────────────────
LLAMA_EXE="E:\\AI\\llama.cpp\\llama-server.exe"
LLAMA_EXE_WIN="/mnt/e/AI/llama.cpp/llama-server.exe"
MODEL_PATH="E:\\AI\\models\\Qwen\\Qwen2.5-Coder-14B-Instruct-Q5_K_M.gguf"
LLAMA_PORT=8001
# Context: 6144 leaves ~3.5GB VRAM free for display, browser, VS Code on a shared desktop
LLAMA_CTX=6144
# Batch: 1024 avoids GPU stall long enough to freeze display during prompt processing
LLAMA_BATCH=1024
LLAMA_UBATCH=512
# Threads: 4 physical cores, 8 logical — leaves 2 cores free for desktop apps
LLAMA_THREADS=4
LLAMA_THREADS_BATCH=8
OLLAMA_PORT=11434
ORCHESTRATOR_DIR="$HOME/orchestrator"
COMPOSE_FILE="docker-compose.windows.yml"
BACKEND_PORT=8080
FRONTEND_PORT=3000
# Resolve Windows host IP from WSL2 default gateway.
# Falls back to /etc/resolv.conf nameserver (also reliable in WSL2).
# If both fail, exits early — all Windows-side services depend on this address.
WINDOWS_HOST=$(ip route | grep default | awk '{print $3}' 2>/dev/null | head -1)
if [ -z "$WINDOWS_HOST" ]; then
    WINDOWS_HOST=$(grep -m1 nameserver /etc/resolv.conf 2>/dev/null | awk '{print $2}')
fi
if [ -z "$WINDOWS_HOST" ]; then
    echo -e "    ${RED}[FAIL]${NC} Cannot determine Windows host IP (tried ip route and /etc/resolv.conf)."
    echo    "    Set WINDOWS_HOST manually: export WINDOWS_HOST=<your-windows-ip>"
    exit 1
fi
# ──────────────────────────────────────────────────────────────────────────────

# ─── Flags ────────────────────────────────────────────────────────────────────
NO_FRONTEND=false
SKIP_OLLAMA=false
for arg in "$@"; do
    case $arg in
        --no-frontend|--backend-only) NO_FRONTEND=true ;;
        --skip-ollama) SKIP_OLLAMA=true ;;
    esac
done
# ──────────────────────────────────────────────────────────────────────────────

CYAN='\033[0;36m'
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[1;33m'
GRAY='\033[0;37m'
NC='\033[0m'

step()  { echo -e "\n${CYAN}==> $1${NC}"; }
ok()    { echo -e "    ${GREEN}[OK]${NC} $1"; }
fail()  { echo -e "    ${RED}[FAIL]${NC} $1"; exit 1; }
warn()  { echo -e "    ${YELLOW}[WARN]${NC} $1"; }
info()  { echo -e "    ${GRAY}$1${NC}"; }

wait_port() {
    local port=$1
    local label=$2
    local timeout=${3:-30}
    local host=${4:-localhost}
    local elapsed=0
    while [ $elapsed -lt $timeout ]; do
        if curl -s -o /dev/null --connect-timeout 2 "http://${host}:$port" 2>/dev/null; then
            return 0
        fi
        sleep 2
        elapsed=$((elapsed + 2))
        info "Waiting for $label on port $port... (${elapsed}s)"
    done
    return 1
}

# ─── 1. Ollama ────────────────────────────────────────────────────────────────
# Ollama provides embeddings for Qdrant / the knowledge layer.
# If it is not available the backend and dashboard still launch — semantic
# search and knowledge retrieval will be degraded until Ollama is running.
OLLAMA_OK=false
if [ "$SKIP_OLLAMA" = true ]; then
    OLLAMA_OK=true
else
    step "Checking Ollama (embeddings for Qdrant / knowledge layer)"
    info "Windows host resolved to: ${WINDOWS_HOST}"
    # -f makes curl exit non-zero on HTTP 4xx/5xx so a bad response isn't treated as success
    if curl -sf -o /dev/null --connect-timeout 3 "http://${WINDOWS_HOST}:${OLLAMA_PORT}/api/tags" 2>/dev/null; then
        ok "Ollama already running"
        OLLAMA_OK=true
    else
        info "Starting Ollama via PowerShell..."
        powershell.exe -Command "Start-Process ollama -ArgumentList 'serve' -WindowStyle Hidden" 2>/dev/null || true
        sleep 4
        if curl -sf -o /dev/null --connect-timeout 5 "http://${WINDOWS_HOST}:${OLLAMA_PORT}/api/tags" 2>/dev/null; then
            ok "Ollama started"
            OLLAMA_OK=true
        else
            # NOTE: Ollama not detected — embeddings / knowledge layer will be unavailable.
            # The backend and frontend dashboard will still start normally.
            # To enable semantic search, run 'ollama serve' in PowerShell and restart.
            warn "Ollama not detected on ${WINDOWS_HOST}:${OLLAMA_PORT} — continuing without it."
            warn "Qdrant embeddings and knowledge retrieval will be degraded."
            warn "To fix: run 'ollama serve' in PowerShell, then restart this script."
        fi
    fi
fi

# ─── 2. llama-server ──────────────────────────────────────────────────────────
step "Checking llama-server (Vulkan)"
if curl -sf -o /dev/null --connect-timeout 3 "http://${WINDOWS_HOST}:${LLAMA_PORT}/v1/models" 2>/dev/null; then
    ok "llama-server already running on port $LLAMA_PORT"
else
    if [ ! -f "$LLAMA_EXE_WIN" ]; then
        fail "llama-server.exe not found at $LLAMA_EXE_WIN"
    fi

    info "Starting llama-server (model load takes 20-60 seconds)..."
    powershell.exe -Command "
        Start-Process -FilePath '$LLAMA_EXE' \`
            -ArgumentList '-m \"$MODEL_PATH\" --host 0.0.0.0 --port $LLAMA_PORT -ngl 99 -c $LLAMA_CTX -b $LLAMA_BATCH -ub $LLAMA_UBATCH --threads $LLAMA_THREADS --threads-batch $LLAMA_THREADS_BATCH -ctk q8_0 -ctv q8_0 --flash-attn --jinja' \`
            -WindowStyle Normal
    " 2>/dev/null &

    if wait_port $LLAMA_PORT "llama-server" 90 "$WINDOWS_HOST"; then
        ok "llama-server ready on port $LLAMA_PORT"
    else
        fail "llama-server did not start within 90s. Check the llama-server window for errors."
    fi
fi

# ─── 3. Docker backend ────────────────────────────────────────────────────────
step "Starting Docker backend"

[ -f "$ORCHESTRATOR_DIR/.env" ] || fail ".env not found at $ORCHESTRATOR_DIR/.env"

PROJECTS_DIR=$(grep "^WINDOWS_PROJECTS_DIR=" "$ORCHESTRATOR_DIR/.env" | cut -d= -f2)
PROJECTS_DIR=${PROJECTS_DIR:-"$HOME/projects"}
mkdir -p "$PROJECTS_DIR"

cd "$ORCHESTRATOR_DIR"
export WINDOWS_PROJECTS_DIR="$PROJECTS_DIR"
docker compose -f "$COMPOSE_FILE" up -d 2>&1 | tail -5

if wait_port $BACKEND_PORT "orchestrator backend" 60; then
    ok "Backend ready on port $BACKEND_PORT"
else
    fail "Backend did not become healthy within 60s. Run: docker compose -f $COMPOSE_FILE logs"
fi

# ─── 4. Frontend ──────────────────────────────────────────────────────────────
if [ "$NO_FRONTEND" = false ]; then
    step "Starting frontend"
    cd "$ORCHESTRATOR_DIR/frontend"
    nohup env VITE_API_URL="http://localhost:${BACKEND_PORT}/api/v1" pnpm dev \
        > /tmp/frontend.log 2>&1 &
    echo $! > /tmp/frontend.pid
    sleep 5
    if wait_port $FRONTEND_PORT "frontend" 30; then
        ok "Frontend ready on port $FRONTEND_PORT"
    else
        warn "Frontend may still be starting. Check http://localhost:$FRONTEND_PORT in a moment."
        info "Logs: tail -f /tmp/frontend.log"
    fi
fi

# ─── Summary ──────────────────────────────────────────────────────────────────
echo -e "\n${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo -e "${CYAN} Stack is up${NC}"
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}"
echo    "  llama-server : http://${WINDOWS_HOST}:${LLAMA_PORT}/v1/models"
if [ "$OLLAMA_OK" = true ]; then
    echo "  Ollama       : http://${WINDOWS_HOST}:${OLLAMA_PORT}/api/tags"
else
    echo -e "  Ollama       : ${YELLOW}NOT RUNNING${NC} — embeddings/knowledge layer degraded"
fi
echo    "  Backend API  : http://localhost:${BACKEND_PORT}/health"
echo    "  API docs     : http://localhost:${BACKEND_PORT}/docs"
if [ "$NO_FRONTEND" = false ]; then
    echo "  Dashboard    : http://localhost:${FRONTEND_PORT}"
fi
echo ""
echo    "  Stop backend : cd ~/orchestrator && docker compose -f $COMPOSE_FILE down"
echo    "  Stop llama   : Close the llama-server window in Windows"
if [ "$NO_FRONTEND" = false ]; then
    echo "  Stop frontend: kill \$(cat /tmp/frontend.pid)"
fi
echo -e "${CYAN}━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━${NC}\n"