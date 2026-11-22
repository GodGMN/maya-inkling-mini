#!/bin/sh
set -eu

preset=ik2-mini-gfx1100.json
if [ -f ik2-mini-gfx1100-hip-host-vision.json ]; then
    case "${IK2_CACHE_SLOTS:-1}" in
        1|2) preset=ik2-mini-gfx1100-hip-host-vision.json ;;
        3|4) preset=ik2-mini-gfx1100-hip-host-vision-512.json ;;
    esac
elif [ "${IK2_CACHE_SLOTS:-1}" = 1 ] && [ -f ik2-mini-gfx1100-hip-vision.json ]; then
    preset=ik2-mini-gfx1100-hip-vision.json
fi
exec python3 serve/server.py --engine maya --config "${MAYA_IK2_PRESET:-$preset}" "$@"
