<h1><picture>
  <source media="(prefers-color-scheme: dark)" srcset="assets/tintero-wordmark-dark.png">
  <img src="assets/tintero-wordmark.png" alt="Tintero" width="300">
</picture></h1>

Tintero is a local inference server for [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini). It runs the full quantized model on one 24 GB Radeon RX 7900 XTX.

The Docker image runs stock Inkling by default. Set `IK2_ABLATE_MODE=on` to use the abliterated version with the same weights. On the RX 7900 XTX, it generated 114.6 tokens/s with the flag on and 118.6 tokens/s with it off.

[![RX 7900 XTX: one chat generates 117.1 tokens/s after a 512-token prompt. Total batched throughput reaches 327.8 tokens/s across 32 streams.](docs/assets/rx7900xtx-speed.svg)](docs/HIP-PORT.md#full-model-measurements-at-393216-reserved-tokens)

## Quick start on RX 7900 XTX

> [!TIP]
> Using a coding agent? Send it this README and ask: “Set up Tintero on this machine and verify it with the test request below.”

This starts a text-only server on Linux. You’ll need the ROCm development tools under `/opt/rocm` (tested with 7.2.1), a C++ compiler, CMake 3.24+, Ninja, Git, and Python 3.10+ with `venv`. The model download is about 18 GB.

**1. Build Tintero.**

```sh
git clone https://github.com/GodGMN/tintero
cd tintero
cmake -S . -B build-ik2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_PREFIX_PATH=/opt/rocm -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGETS=gfx1100
cmake --build build-ik2 --target tintero -j4
```

**2. Download the model and start the server.** Run these commands from the same directory.

```sh
python3 -m venv .venv
. .venv/bin/activate
python -m pip install huggingface_hub numpy jinja2 regex pyyaml requests pillow psutil 'jsonschema>=4.18,<5'
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir models
python tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-IQ4_NL.gguf \
  --out ik2 --chat-template tools/ik2/chat_template.jinja
python serve/server.py --engine maya --config docs/inkling2-mini/ik2-mini-gfx1100.json \
  --host 127.0.0.1 --port 8085
```

<details>
<summary>Already have the r6 Docker image? Start it here instead.</summary>

With the Hugging Face CLI installed and the image loaded locally:

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

## Configuration and hardware

- [Vision](docs/HIP-PORT.md#image-encoder-on-the-ryzen-5-3600) requires the image encoder and `mmproj-F16.gguf`. The r6 image includes the encoder.
- [Runtime settings](docs/README.md#settings) cover context size, concurrent requests, sparse decode, and runtime ablation.
- [CUDA on the RTX 3090](docs/HIP-PORT.md#cuda-build-check) compiled on an earlier revision. The current release has been tested on the RX 7900 XTX only.

## Credits

Tintero builds on [Maya’s Inkling engine](https://github.com/alesha-pro/maya-inkling-mini). [Project Maya](https://github.com/mw00/project-maya) grew out of [Strata](https://github.com/Niko1221/Strata). Thinking Machines released the model, and [ggml-org](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) provides the IQ4_NL weights. [MIT license](LICENSE)
