# Inkling-2-Mini presets

Choose a JSON file from `docs/inkling2-mini/`.

| GPU | Workload | Preset | GGUF | KV cache | Context budget |
| --- | --- | --- | --- | --- | ---: |
| RX 7900 XTX (`gfx1100`) | Single conversation | `ik2-mini-gfx1100.json` | IQ4_NL | FP8 | 393,216 |
| Strix Halo (`gfx1151`, 64–128 GB) | Long context on unified RAM | `strix-halo-gfx1151.json` | IQ4_NL | Paged FP4 | 524,288 |
| RTX 3090, 4090 or 5090 with 24 GB+ | Long chat on one card | `cuda-24gb-fp4.json` | IQ4_NL | FP4 global | 262,144 |
| A100 or H100 with 80 GB | Long chat with FP8 KV | `cuda-80gb.json` | IQ4_NL | FP8 | 393,216 |
| DGX Spark GB10, 121 GiB | Single user | `dgx-spark-quality.json` | IQ4_NL | FP8 | 262,144 |
| DGX Spark GB10, 121 GiB | Coding with DFlash2 (1–8 streams) | `dgx-spark-dflash2.json` | UD-Q4_K_XL | FP8 | 262,144 |
| DGX Spark GB10, 121 GiB | 32 concurrent streams | `dgx-spark-batched-nvfp4.json` | UD-Q4_K_XL | FP8 | 131,072 total (4,096 per stream) |
| DGX Spark GB10, 121 GiB | One very long conversation | `dgx-spark-longctx-fp4.json` | IQ4_NL | FP4 global | 1,048,576 |

On the DGX Spark, `dgx-spark-dflash2.json` runs the Q8-quantized DFlash2 drafter (`341 MiB`) with lossless probabilistic drafting (`55.6 tok/s` at `B=1` / `119.7 tok/s` at `B=8` on `T=0.7` coding; `162.7 tok/s` steady at `T=0`, `c=8`) and 32-row Blackwell `OMMA.SF` prefill (`~2,050–2,089 tok/s`). Plain 32-stream batching reaches `130.3 tok/s` with `UD-Q4_K_XL` (native NVFP4) and `119.0 tok/s` with `IQ4_NL`.

The project reports hardware measurements for the RX 7900 XTX, A100-SXM4-80GB and GB10. The `strix-halo-gfx1151.json` preset is compiled for `gfx1151` and verified on `gfx1100` hardware. The context budgets above are configuration values; available memory and other GPU workloads affect how much fits.

## Run a preset

Put the matching GGUF in `models/` and export its tokenizer from the package root.

```sh
python3 tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-IQ4_NL.gguf \
  --out ik2 --chat-template tools/ik2/chat_template.jinja
```

For `dgx-spark-dflash2.json`, also download the DFlash2 drafter weights:

```sh
hf download incoai/Inkling-2-Mini-DFlash2 model.safetensors --local-dir models/dflash2
```

For CUDA containers, mount both directories under `/opt/tintero`.

```sh
docker run --rm --gpus all -p 127.0.0.1:8080:8080 \
  -v "$PWD/models:/opt/tintero/models:ro" -v "$PWD/ik2:/opt/tintero/ik2:ro" \
  -e TINTERO_CONFIG=docs/inkling2-mini/dgx-spark-dflash2.json \
  ghcr.io/godgmn/tintero:v0.2.0-cuda
```

The HIP image uses `v0.2.0-hip-gfx1100` (compiled for both `gfx1100` and `gfx1151`) and defaults to the RX 7900 XTX preset. Pass `--device=/dev/kfd --device=/dev/dri` for ROCm. Set `TINTERO_MODEL` to override the GGUF path inside the container. Explicit `IK2_*` environment settings override preset values (for example `-e IK2_KV=fp4` cuts RX 7900 XTX 393K KV cache VRAM from `6.16` to `3.46 GiB`). The server listens on `TINTERO_HOST` (default `0.0.0.0`) and `TINTERO_PORT` (default `8080`).

For a release tarball, extract it into an empty directory on Ubuntu 24.04 with the matching CUDA 13.1 or ROCm 7.2 runtime libraries. From that directory, activate a Python virtual environment, run `pip install -r requirements.txt`, export the tokenizer, and start the server.

```sh
TINTERO_CONFIG=docs/inkling2-mini/cuda-24gb-fp4.json ./entrypoint.sh --host 127.0.0.1
```
