#!/usr/bin/env bash
# Run from the isolated Prometheus build, after the GPU owner releases the card.
set -euo pipefail
stage=$(cd "$(dirname "$0")/../.." && pwd)
mode=${1:-}
case "$mode" in
    --check|--run) ;;
    *) echo "usage: $0 --check | --run" >&2; exit 2 ;;
esac
image=hyperion/maya-ik2:gfx1100-service-r4
model_dir=$HOME/ai/models/inkling-2-mini
mounts=(-v "$stage:/trial:ro" -v "$model_dir:/weights:ro")
model=/weights/Inkling-2-Mini-IQ4_NL.gguf
docker run --rm --network none "${mounts[@]}" --entrypoint sh "$image" -ec '
    test ! -e /dev/kfd
    test -r "$1"
    for binary in /trial/build/maya-ik2 /trial/build/ik2_kernels_test; do
        deps=$(ldd "$binary"); if echo "$deps" | grep "not found"; then exit 1; fi
    done
    python3 /trial/src/tests/ik2_sparse_decode.py run --help >/dev/null
    cd /trial
    sha256sum --quiet -c SHA256SUMS
' sh "$model"
if [[ $mode == --check ]]; then echo "CPU preflight passed"; exit 0; fi

# This host has one discrete AMD GPU. Refuse to displace a resident model.
vram=(/sys/class/drm/card*/device/mem_info_vram_used)
if [[ ${#vram[@]} != 1 || ! -r ${vram[0]} ]]; then echo "Cannot identify the Prometheus GPU" >&2; exit 1; fi
if (( $(cat "${vram[0]}") > 1024 * 1024 * 1024 )); then
    echo "GPU still occupied (more than 1 GiB allocated). Retry after it is released." >&2; exit 1
fi
mkdir -p "$stage/results"
out=$(mktemp -d "$stage/results/run.XXXXXX")
echo "Saving trial to $out"
docker run --rm --network none --name maya-ik2-sparse-trial --user "$(id -u):$(id -g)" \
    --group-add "$(stat -c %g /dev/kfd)" --group-add "$(stat -c %g /dev/dri/renderD128)" \
    --device /dev/kfd --device /dev/dri "${mounts[@]}" -v "$out:/out" \
    -e HOME=/tmp -e PYTHONDONTWRITEBYTECODE=1 -e PYTHONUNBUFFERED=1 --entrypoint sh "$image" -ec '
        /trial/build/ik2_kernels_test > /out/kernels.log 2>&1 || { tail -50 /out/kernels.log; exit 1; }
        exec python3 /trial/src/tests/ik2_sparse_decode.py run --engine /trial/build/maya-ik2 \
            --model "$1" --tokenizer ik2/tokenizer --fixtures /trial/fixtures --output /out/quality
    ' sh "$model" | tee "$out/progress.log"
