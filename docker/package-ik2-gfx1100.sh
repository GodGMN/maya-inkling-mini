#!/bin/sh
# Package a built gfx1100 engine and a tokenizer exported from the target GGUF.
set -eu
if [ "$#" -ne 3 ]; then
    echo "usage: $0 MAYA_IK2 TOKENIZER_DIRECTORY IMAGE_TAG" >&2
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
cp -R "$root/serve" "$root/tools" "$context/"
find "$context" -name __pycache__ -type d -prune -exec rm -rf '{}' +
docker buildx build --load --resource memory=6g --resource memory-swap=6g \
    --resource cpu-period=100000 --resource cpu-quota=400000 \
    -t "$3" -f "$context/ik2-gfx1100.Dockerfile" "$context"
