#!/bin/sh
# grader.sh {start|stop|status}: manage the local Qwen3Guard-Gen-8B ROCm llama-server on 127.0.0.1:8088.
# Stops ik2-ablate-lab on start (unless KEEP_LAB=1) so the 24 GiB RX 7900 XTX VRAM is clean.
set -eu

PORT="${GRADER_PORT:-8088}"
IMAGE="${GRADER_IMAGE:-hyperion/llama.cpp:server-rocm-b12600}"
MODELS_DIR="${GRADER_MODELS_DIR:-/home/sergio/ai/models}"
MODEL_REL="${GRADER_MODEL_REL:-qwen3guard-gen-8b/Qwen3Guard-Gen-8B.Q6_K.gguf}"
ALIAS="${GRADER_ALIAS:-qwen3guard-gen-8b}"
SLOTS="${GRADER_SLOTS:-16}"
CTX="${GRADER_CTX:-65536}"
NAME="${GRADER_CONTAINER:-ik2-grader}"

case "${1:-start}" in
  start)
    if [ "${KEEP_LAB:-0}" != "1" ]; then
      docker rm -f ik2-ablate-lab >/dev/null 2>&1 || true
    fi
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    if [ ! -f "$MODELS_DIR/$MODEL_REL" ]; then
      echo "error: grader GGUF not found at $MODELS_DIR/$MODEL_REL" >&2
      exit 1
    fi
    docker run -d --name "$NAME" \
      --device /dev/dri --device /dev/kfd \
      -v "$MODELS_DIR:/models:ro" \
      -p "127.0.0.1:${PORT}:8080" \
      "$IMAGE" \
      -m "/models/${MODEL_REL}" \
      --alias "$ALIAS" \
      --host 0.0.0.0 \
      --port 8080 \
      --jinja \
      -ngl 99 \
      -np "$SLOTS" \
      --ctx-size "$CTX" \
      -fa on \
      --cache-type-k q8_0 \
      --cache-type-v q8_0 >/dev/null
    for i in $(seq 180); do
      if curl -sf "http://127.0.0.1:${PORT}/health" >/dev/null 2>&1; then
        echo "grader up on 127.0.0.1:${PORT} after ${i}s"
        exit 0
      fi
      if [ "$(docker inspect -f '{{.State.Running}}' "$NAME" 2>/dev/null || echo false)" != "true" ]; then
        docker logs --tail 40 "$NAME" >&2 || true
        exit 1
      fi
      sleep 1
    done
    docker logs --tail 40 "$NAME" >&2 || true
    exit 1
    ;;
  stop)
    docker rm -f "$NAME" >/dev/null 2>&1 || true
    echo "grader stopped"
    ;;
  status)
    docker ps -a --filter "name=^/${NAME}$" --format "table {{.Names}}\t{{.Status}}\t{{.Ports}}"
    curl -sf "http://127.0.0.1:${PORT}/health" && echo "" || true
    ;;
  *)
    echo "usage: $0 {start|stop|status}" >&2
    exit 2
    ;;
esac
