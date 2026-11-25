# Inkling-2-Mini RX 7900 XTX (`gfx1100`) — Runtime Abliteration (`r5`)

`hyperion/maya-ik2:gfx1100-service-r5` packages the compiled `maya-ik2` HIP engine (with `IK2_ABLATE_KEEP_NORM`, `IK2_ABLATE_VISITS`, and native `attn_output` LoRA support) plus the 9 KB norm-preserving orthogonalized refusal direction (`v21-pc4.f32`) right inside the image.

- **Zero extra model download:** uses the exact same stock `Inkling-2-Mini-IQ4_NL.gguf` (17.6 GB).
- **Single feature flag:** off by default (100% stock Inkling-2-Mini); flip `IK2_ABLATE_MODE=on` (aliases: `1`, `true`, `v21-pc4-kn`, `cyber`) to enable our verified `v21-pc4 @ s=1.6, VISITS=22:48, KEEP_NORM=1` configuration (~114.6 tok/s vs. 118.6 tok/s stock, ~3% overhead).

## Quick Start (RX 7900 XTX, ROCm)

### Abliterated Mode (`3 × 131K` FP8 KV, text-only — no `mmproj` required)
```bash
docker run -d --name inkling-2-mini \
  --device /dev/dri --device /dev/kfd \
  --memory 10g --memory-swap 10g \
  -v /path/to/inkling-2-mini:/opt/maya/models:ro \
  -e IK2_CACHE_SLOTS=3 \
  -e IK2_TEXT_ONLY=1 \
  -e IK2_ABLATE_MODE=on \
  -p 127.0.0.1:8085:8080 \
  hyperion/maya-ik2:gfx1100-service-r5
```

### Stock Mode (A/B comparison on the same GGUF)
Omit `IK2_ABLATE_MODE` (or set `IK2_ABLATE_MODE=off`).

### Multimodal Mode (with `mmproj-F16.gguf` in `/path/to/inkling-2-mini`)
Omit `IK2_TEXT_ONLY=1` (uses `hip-host-vision` automatically; recommended `IK2_CACHE_SLOTS=1` or `2` on 24 GiB VRAM).

## Environment Knobs

| Variable | Default | Description |
|---|---|---|
| `IK2_ABLATE_MODE` | `off` | `off` (stock) or `cyber` / `v21-pc4-kn` (loads `v21-pc4.f32` at `s=1.6, 22:48, keep_norm=1`). |
| `IK2_ABLATE` | auto | Prefers `/opt/maya/models/v21-pc4.f32` if present, else built-in `/opt/maya/ablate/v21-pc4.f32`. |
| `IK2_ABLATE_SCALE` | `1.6` | Steering strength override when `IK2_ABLATE_MODE=cyber`. |
| `IK2_ABLATE_VISITS` | `22:48` | Residual visit range override. |
| `IK2_ABLATE_KEEP_NORM` | `1` | Preserves L2 residual norm after projection. |
| `IK2_CACHE_SLOTS` | `1` | Parallel KV cache slots (`3` gives `3 × 131,072` FP8 KV on 24 GiB). |
| `IK2_TEXT_ONLY` | `0` | Set `1` to force `ik2-mini-gfx1100-text.json` without loading `mmproj-F16.gguf`. |

## Exporting the Image for a Betatester (7900 XTX)
```bash
docker save hyperion/maya-ik2:gfx1100-service-r5 | zstd -T0 -3 -o maya-ik2-gfx1100-r5.tar.zst
# On the betatester machine:
zstd -dc maya-ik2-gfx1100-r5.tar.zst | docker load
```
