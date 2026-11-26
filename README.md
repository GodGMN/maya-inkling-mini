<h1><picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/tintero-wordmark-dark.png">
  <img src="assets/tintero-wordmark.png" alt="Tintero" width="300">
</picture></h1>

Tintero is a local Linux inference server for [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) supporting **AMD (HIP)** and **NVIDIA (CUDA)** GPUs. It is tailored to keep the full quantized model resident on a single **24 GB+ VRAM** card, with FP8/FP4 KV context scaling to whatever memory you have available and optional zero-retraining runtime abliteration (`IK2_ABLATE_MODE=on`).

### Tested hardware

| Backend | Target | Tested GPU | Single-stream decode (512p) | Abliterated (`v21-pc4`) | Batched / Prefill notes |
|---|---|---|---:|---:|---|
| **AMD HIP** | `gfx1100` (RDNA3) | RX 7900 XTX (24 GB) | **117.1 tok/s** | 114.6 tok/s (`-3.4%`) | Up to **327.8 tok/s** across 32 streams (`IK2_BATCH`); 393K FP8 KV |
| **NVIDIA CUDA** | `sm_80+` (Ampere–Blackwell) | A100-SXM4-80GB (`sm_80`) | **98.8 tok/s** | 95.6 tok/s (`-3.3%`) | **2,155–2,184 tok/s** prefill (4K–32K); 393K FP8 KV; single-stream (default CUDA flags) |

[![RX 7900 XTX: one chat generates 117.1 tokens/s after a 512-token prompt. Total batched throughput reaches 327.8 tokens/s across 32 streams.](docs/assets/rx7900xtx-speed.svg)](docs/HIP-PORT.md#full-model-measurements-at-393216-reserved-tokens)

> [!NOTE]
> The RX 7900 XTX (`gfx1100`) is currently the most heavily tuned backend (RDNA3 decode kernels, `IK2_BATCH` continuous batching, sparse long-context decode, and the r6 Docker image). NVIDIA CUDA (`sm_80+`, Linux x86_64 & ARM64) passes the full `ik2_kernels_test` suite and runs full-resident inference + runtime abliteration single-stream today, with CUDA decode tuning and `IK2_BATCH` parity planned next.

## Quick start

> [!TIP]
> Using a coding agent? Send it this README and ask: “Set up Tintero on this machine and verify it with the test request below.”

This starts a text-only OpenAI-compatible server on Linux. You’ll need a GPU with **24 GB+ VRAM** (AMD ROCm under `/opt/rocm` for `gfx1100`, or NVIDIA CUDA Toolkit for `sm_80+`), a C++20 compiler, CMake 3.24+, Ninja, Git, and Python 3.10+ with `venv`. The `IQ4_NL` model download is about 18 GB.

**1. Clone and build Tintero.** Pick the command for your GPU:

```sh
git clone https://github.com/GodGMN/tintero
cd tintero
```

**AMD HIP (`gfx1100`, tested with ROCm 7.2.1):**

```sh
cmake -S . -B build-ik2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGETS=gfx1100
cmake --build build-ik2 --target tintero -j4
```

**NVIDIA CUDA (`sm_80+` — RTX 30/40/50 series, A100, H100, DGX Spark):**

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_CUDA=ON -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_CUDA_ARCHITECTURES=native
cmake --build build --target tintero ik2_kernels_test -j4
./build/ik2_kernels_test
```

**2. Download the model and start the server.** Run these commands from the same directory.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install huggingface_hub numpy jinja2 regex pyyaml requests pillow psutil 'jsonschema>=4.18,<5'
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir models
python tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-IQ4_NL.gguf \
  --out ik2 --chat-template tools/ik2/chat_template.jinja
```

Start with the preset for your backend:

```sh
# AMD RX 7900 XTX (HIP)
python serve/server.py --engine maya --config docs/inkling2-mini/ik2-mini-gfx1100.json \
  --host 127.0.0.1 --port 8085

# NVIDIA GPU (CUDA) — update the model filename in docs/inkling2-mini/ik2-mini-text.json if using IQ4_NL
python serve/server.py --engine maya --config docs/inkling2-mini/ik2-mini-text.json \
  --host 127.0.0.1 --port 8085
```

<details>
<summary>Already have the AMD r6 Docker image? Start it here instead.</summary>

With the Hugging Face CLI installed and the image loaded locally (runs stock Inkling by default; pass `-e IK2_ABLATE_MODE=on` for runtime abliteration):

```sh
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir models
docker run -d --name inkling-2-mini --device /dev/kfd --device /dev/dri \
  -v "$PWD/models:/opt/maya/models:ro" -e IK2_TEXT_ONLY=1 \
  -p 127.0.0.1:8085:8080 hyperion/tintero:gfx1100-service-r6
```

</details>

**3. Send a message.** Once the model has loaded, run this in another terminal:

```sh
curl -fsS http://127.0.0.1:8085/v1/chat/completions \
  -H 'Content-Type: application/json' \
  -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 17 + 25? Answer with the number only."}],"reasoning_effort":"none","temperature":0,"max_tokens":48}'
```

The answer is in `choices[0].message.content`. To connect a client, set its base URL to `http://127.0.0.1:8085/v1` and its model to `inkling-2-mini`.

## Configuration and docs

- [Runtime settings](docs/README.md#settings) cover context size, concurrent requests, sparse decode, and runtime abliteration (`IK2_ABLATE*`).
- [Vision](docs/HIP-PORT.md#image-encoder-on-the-ryzen-5-3600) requires the image encoder and `mmproj-F16.gguf`. The AMD r6 Docker image includes the encoder.
- [HIP port & benchmarks](docs/HIP-PORT.md) document the full 393K FP8/FP4 context measurements and kernel parity tests.

## Credits

Tintero builds on [Maya’s Inkling engine](https://github.com/alesha-pro/maya-inkling-mini). [Project Maya](https://github.com/mw00/project-maya) grew out of [Strata](https://github.com/Niko1221/Strata). Thinking Machines released the model, and [ggml-org](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) provides the IQ4_NL weights. [MIT license](LICENSE)
