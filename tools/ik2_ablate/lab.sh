#!/bin/sh
# lab.sh PRESET: (re)start the lab engine on 127.0.0.1:8087 with presets/PRESET.json and wait until healthy
set -eu
L=$(cd "$(dirname "$0")" && pwd)
docker rm -f ik2-ablate-lab >/dev/null 2>&1 || true
docker run -d --name ik2-ablate-lab --device /dev/dri --device /dev/kfd --memory 10g --memory-swap 10g \
  -v /home/sergio/ai/models/inkling-2-mini:/opt/maya/models:ro \
  -v /home/sergio/ai/build/maya-ik2-hip/build-ik2/maya-ik2:/opt/maya/build-ik2/maya-ik2:ro \
  -v "$L":/lab -e MAYA_IK2_PRESET=/lab/presets/$1.json \
  -p 127.0.0.1:8087:8080 hyperion/maya-ik2:gfx1100-service-r4 >/dev/null
for i in $(seq 300); do
  if curl -sf http://127.0.0.1:8087/health >/dev/null 2>&1; then echo "up after ${i}x2s"; exit 0; fi
  if [ "$(docker inspect -f {{.State.Running}} ik2-ablate-lab)" != true ]; then docker logs --tail 40 ik2-ablate-lab; exit 1; fi
  sleep 2
done
docker logs --tail 40 ik2-ablate-lab; exit 1
