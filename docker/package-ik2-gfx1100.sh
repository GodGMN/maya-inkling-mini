#!/bin/sh
# Package a built gfx1100 engine and a tokenizer exported from the target GGUF.
set -eu
if [ "$#" -lt 3 ] || [ "$#" -gt 5 ]; then
    echo "usage: $0 MAYA_IK2 TOKENIZER_DIRECTORY IMAGE_TAG [CPU_VISION [HIP_VISION]]" >&2
    exit 2
fi
root=$(CDPATH= cd -- "$(dirname -- "$0")/.." && pwd)
test -x "$1"
test -r "$2/chat_template.jinja"
context=$(mktemp -d)
trap 'rm -rf "$context"' EXIT
cp "$root/LICENSE" "$root/VERSION" "$context/"
cp "$1" "$context/maya-ik2"
cp -R "$2" "$context/tokenizer"
cp "$root/docs/inkling2-mini/ik2-mini-gfx1100.json" "$root/docker/ik2-gfx1100.Dockerfile" "$context/"
mkdir "$context/vision"
if [ "$#" -ge 4 ]; then
    test -x "$4"
    cp "$4" "$context/vision/maya-vision-cpu"
    if [ "$#" -eq 5 ]; then
        test -x "$5"
        cp "$5" "$context/vision/maya-vision-hip"
    fi
    python3 - "$context" <<'PYCONFIG'
import json
import sys
from pathlib import Path

directory = Path(sys.argv[1])
default = directory / "ik2-mini-gfx1100.json"
cfg = json.loads(default.read_text())
(directory / "ik2-mini-gfx1100-text.json").write_text(default.read_text())
cfg["vision"] = dict(exe="vision/maya-vision-cpu", model=cfg["args"][1], mmproj="models/mmproj-F16.gguf",
                     threads=4, max_tokens=1024, gpu=False, resident=True)
default.write_text(json.dumps(cfg, indent=2) + "\n")
if (directory / "vision/maya-vision-hip").exists():
    cfg["vision"].update(exe="vision/maya-vision-hip", gpu=True)
    cfg["args"][cfg["args"].index("--chunk") + 1] = "256"
    (directory / "ik2-mini-gfx1100-hip-vision.json").write_text(json.dumps(cfg, indent=2) + "\n")
PYCONFIG
fi
cp -R "$root/serve" "$root/tools" "$context/"
find "$context" -name __pycache__ -type d -prune -exec rm -rf '{}' +
docker buildx build --load --resource memory=6g --resource memory-swap=6g \
    --resource cpu-period=100000 --resource cpu-quota=400000 \
    -t "$3" -f "$context/ik2-gfx1100.Dockerfile" "$context"
