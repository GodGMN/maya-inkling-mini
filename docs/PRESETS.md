# Inkling-2-Mini presets

Choose a JSON file from `docs/inkling2-mini/`. Each preset uses a 1,024-token prefill chunk.

| GPU | Workload | Preset | GGUF | KV cache | Context budget |
| --- | --- | --- | --- | --- | ---: |
| RX 7900 XTX | Single conversation | `ik2-mini-gfx1100.json` | IQ4_NL | FP8 | 393,216 |
| RTX 3090, 4090 or 5090 with 24 GB+ | Long chat on one card | `cuda-24gb-fp4.json` | IQ4_NL | FP4 global | 262,144 |
| A100 or H100 with 80 GB | Long chat with FP8 KV | `cuda-80gb.json` | IQ4_NL | FP8 | 393,216 |
| DGX Spark GB10, 121 GiB | Single user | `dgx-spark-quality.json` | IQ4_NL | FP8 | 262,144 |
| DGX Spark GB10, 121 GiB | 32 concurrent streams | `dgx-spark-batched-nvfp4.json` | UD-Q4_K_XL | FP8 | 131,072 total (4,096 per stream) |
| DGX Spark GB10, 121 GiB | One very long conversation | `dgx-spark-longctx-fp4.json` | IQ4_NL | FP4 global | 1,048,576 |

The project reports hardware measurements for the RX 7900 XTX, A100-SXM4-80GB and GB10. The context budgets above are configuration values. Available memory and other GPU workloads affect how much fits.

## Run a preset

Put the matching GGUF in `models/` and export its tokenizer from the package root.

```sh
python3 tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-IQ4_NL.gguf \
  --out ik2 --chat-template tools/ik2/chat_template.jinja
```

For CUDA containers, mount both directories under `/opt/tintero`.

```sh
docker run --rm --gpus all -p 127.0.0.1:8080:8080 \
  -v "$PWD/models:/opt/tintero/models:ro" -v "$PWD/ik2:/opt/tintero/ik2:ro" \
  -e TINTERO_CONFIG=docs/inkling2-mini/dgx-spark-quality.json \
  ghcr.io/godgmn/tintero:v0.1.0-cuda
```

The HIP image uses `v0.1.0-hip-gfx1100` and defaults to the RX 7900 XTX preset. Pass `--device=/dev/kfd --device=/dev/dri` for ROCm. Set `TINTERO_MODEL` to override the GGUF path inside the container. Explicit `IK2_*` environment settings override preset values. The server listens on `TINTERO_HOST` (default `0.0.0.0`) and `TINTERO_PORT` (default `8080`).

For a release tarball, extract it into an empty directory on Ubuntu 24.04 with the matching CUDA 13.1 or ROCm 7.2 runtime libraries. From that directory, activate a Python virtual environment, run `pip install -r requirements.txt`, export the tokenizer, and start the server.

```sh
TINTERO_CONFIG=docs/inkling2-mini/cuda-24gb-fp4.json ./entrypoint.sh --host 127.0.0.1
```
