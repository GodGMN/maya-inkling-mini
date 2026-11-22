#!/bin/sh
set -eu

preset=ik2-mini-gfx1100.json
if [ "${IK2_CACHE_SLOTS:-1}" = 1 ] && [ -f ik2-mini-gfx1100-hip-vision.json ]; then
    preset=ik2-mini-gfx1100-hip-vision.json
fi
exec python3 serve/server.py --engine maya --config "${MAYA_IK2_PRESET:-$preset}" "$@"
