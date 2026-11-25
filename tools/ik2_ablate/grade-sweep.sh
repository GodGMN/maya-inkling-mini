#!/bin/sh
# grade-sweep.sh [JSONL...]: one-swap batch grader for saved candidate answers (runtime presets OR baked GGUFs).
# Stops ik2-ablate-lab once, starts Qwen3Guard-Gen-8B via grader.sh, grades all given JSONLs (default:
# eval/*-answers.jsonl), writes *.graded.jsonl + review markdowns under graded/, and stops the grader.
set -eu

cd "$(dirname "$0")"
PORT="${GRADER_PORT:-8088}"
OUT_DIR="${GRADE_OUT_DIR:-graded}"
REVIEW_DIR="${GRADE_REVIEW_DIR:-graded/review}"

if [ "$#" -eq 0 ]; then
  set -- eval/*-answers.jsonl
  if [ ! -e "$1" ]; then
    echo "no eval/*-answers.jsonl files found; pass one or more response JSONLs explicitly" >&2
    exit 1
  fi
fi

cleanup() {
  if [ "${KEEP_GRADER:-0}" != "1" ]; then
    ./grader.sh stop >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

./grader.sh start
mkdir -p "$OUT_DIR" "$REVIEW_DIR"
python3 grade.py \
  --url "http://127.0.0.1:${PORT}" \
  --model "${GRADER_ALIAS:-qwen3guard-gen-8b}" \
  --concurrency "${GRADER_SLOTS:-16}" \
  --out-dir "$OUT_DIR" \
  --review-dir "$REVIEW_DIR" \
  "$@"
