#!/usr/bin/env bash
set -euo pipefail

# entrypoint for kubetrainer infra training image
# Usage:
#   ./entrypoint.sh train --config /config/train.yaml
#   ./entrypoint.sh serve --host 0.0.0.0 --port 8000
#   ./entrypoint.sh shell

SUBCOMMAND=${1:-"--help"}
shift || true

# create the ready file to signal container is up (can be overwritten by app)
touch /tmp/kubetrainer_ready

function run_train {
  echo "[kubetrainer] invoking training: python infra_training/infra_learning.py $*"
  python infra_training/infra_learning.py "$@"
}

function run_serve {
  # if you have a uvicorn app in infra_training/serve.py or similar:
  echo "[kubetrainer] starting inference server"
  # Example: uvicorn infra_training.server:app --host 0.0.0.0 --port 8000 --workers 1
  uvicorn infra_training.server:app --host 0.0.0.0 --port ${PORT:-8000} --proxy-headers "$@"
}

case "${SUBCOMMAND}" in
  train)
    run_train "$@"
    ;;
  serve)
    run_serve "$@"
    ;;
  eval)
    echo "[kubetrainer] running eval"
    python infra_training/eval.py "$@"
    ;;
  shell)
    /bin/bash
    ;;
  --help|-h|help)
    cat <<EOF
KubeTrainer infra_training container entrypoint
Usage:
  entrypoint.sh train [args...]    # run training script
  entrypoint.sh serve [args...]    # run inference server
  entrypoint.sh eval [args...]     # run evaluation
  entrypoint.sh shell              # open a shell
EOF
    ;;
  *)
    # default: run as python module
    python "$SUBCOMMAND" "$@"
    ;;
esac
