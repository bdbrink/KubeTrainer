#!/usr/bin/env bash
set -euo pipefail

# agent-entrypoint.sh
# Usage:
#   ./agent-entrypoint.sh gpu-detect [args...]
#   ./agent-entrypoint.sh data-collector [args...]
#   ./agent-entrypoint.sh both

CMD=${1:-"--help"}
shift || true

case "$CMD" in
  gpu-detect)
    echo "[agent] running gpu-detect $*"
    ./gpu-detect "$@"
    ;;
  data-collector)
    echo "[agent] running k8s-data-collector $*"
    ./k8s-data-collector "$@"
    ;;
  both)
    echo "[agent] starting both agents (foreground)"
    # run data collector in background and keep gpu-detect in foreground (or supervisord)
    ./k8s-data-collector "$&"
    exec ./gpu-detect
    ;;
  --help|-h|help)
    cat <<EOF
Agent entrypoint
Commands:
  gpu-detect [args]       Run GPU detection binary
  data-collector [args]   Run k8s data collector binary
  both                    Run both (data-collector background + gpu-detect foreground)
EOF
    ;;
  *)
    exec "$CMD" "$@"
    ;;
esac
