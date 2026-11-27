#!/bin/sh
set -eu

config=${TINTERO_CONFIG:?set TINTERO_CONFIG to a preset JSON path}
# The server applies config.env after inheriting the environment.
# Give explicit container settings precedence without changing the mounted preset.
config=$(python3 - "$config" <<'PY'
import json
import os
import sys
import tempfile

with open(sys.argv[1], encoding="utf-8-sig") as f:
    cfg = json.load(f)
env = cfg.setdefault("env", {})
overrides = {k: os.environ[k] for k in env if k.startswith("IK2_") and k in os.environ}
model = os.getenv("TINTERO_MODEL")
if model or overrides:
    if model:
        args = cfg["args"]
        args[args.index("--model") + 1] = model
        if cfg.get("vision"):
            cfg["vision"]["model"] = model
    env.update(overrides)
    with tempfile.NamedTemporaryFile(mode="w", dir=tempfile.mkdtemp(), suffix=".json", delete=False) as tmp:
        json.dump(cfg, tmp)
    print(tmp.name)
else:
    print(sys.argv[1])
PY
)
exec python3 serve/server.py --engine maya --config "$config" \
    --host "${TINTERO_HOST:-0.0.0.0}" --port "${TINTERO_PORT:-8080}" "$@"
