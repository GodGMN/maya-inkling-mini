# Inkling 2 Mini on AMD Radeon RX 7900 XTX

An AMD (HIP/ROCm) port of [alesha-pro/maya-inkling-mini](https://github.com/alesha-pro/maya-inkling-mini), the CUDA engine for Thinking Machines' 26B looped MoE. The engine, the RTX 3090 results and everything under [About this repo](#about-this-repo) are their work and that of the Project Maya contributors. This repo adds the ROCm build, RDNA3 (gfx1100) kernels and RX 7900 XTX measurements.

On one RX 7900 XTX with the [ggml-org IQ4_NL weights](https://huggingface.co/ggml-org/Inkling-2-Mini-GGUF) and an FP8 KV cache, with 393,216 context positions reserved:

| | tok/s |
|---|---:|
| Decode after 512 input tokens | **117.1** |
| Decode after 8,192 | 107.5 |
| Decode after 32,768 | 93.7 |
| Prefill, uncached 8,192-token prompt | 3,365.7 |
| Editing code (Python rename returning 886 tokens) | 693.6 |

Each figure is the median of three greedy requests. [Measurements and commands](docs/HIP-PORT.md#full-model-measurements-at-393216-reserved-tokens)

* **Quality.** The gfx1100 kernel suite passes, including FP64 checks at 392,957 keys. 184/192 on GSM8K and 207/256 on eight MMLU subjects; with sliding-window KV in FP16, 188/192 and 206/256. [Quality checks](docs/HIP-PORT.md#full-model-measurements-at-393216-reserved-tokens)
* **Tools.** OpenAI and Anthropic tool calls, including streamed parallel calls and multi-turn results. A conversation that grew from 20K to 300K tokens reused every previously processed token on each turn. [Tool conversations](docs/HIP-PORT.md#growing-conversations)
* **Several chats.** `IK2_CACHE_SLOTS=2`, `3` or `4` splits the context into independent caches, so each follow-up processes only its new tokens. Requests run one at a time. [Slot sizes and VRAM](docs/HIP-PORT.md#separate-conversation-caches)
* **Images.** The GPU image encoder reads a new screenshot in 0.7 to 1.6 s at full context and adds 404 MiB of VRAM. [Image timings](docs/HIP-PORT.md#image-latency-on-port-8085)
* **Optional runtime ablation.** Toggle `IK2_ABLATE_MODE=cyber` on the packaged `r5` image (or pass `IK2_ABLATE=tools/ik2_ablate/directions/v21-pc4.f32`) against the same stock GGUF. [Runtime ablation guide](docs/inkling2-mini/runtime-ablation-gfx1100.md)

## ROCm quick start

Linux, an RX 7900 XTX (`gfx1100`, 24 GB) and ROCm. The 20 GB 7900 XT and 16 GB 7900 GRE share the target but don't fit the full-context configuration; other RDNA3 cards need their own build. [Which GPUs can run it](docs/HIP-PORT.md#which-gpus-and-models-can-run-it)

```sh
git clone https://github.com/GodGMN/maya-inkling-mini && cd maya-inkling-mini

# llama.cpp at the revision Maya's CMake pins
curl -fLsS https://codeload.github.com/ggml-org/llama.cpp/tar.gz/3cf03257f219afbe7334045ff7c6a06ac68c627d | tar xz -C ..

cmake -S . -B build-ik2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=OFF \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGETS=gfx1100 \
  -DSTRATA_GGML_DIR="$PWD/../llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d"
cmake --build build-ik2 --target maya-ik2

hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir models
python3 tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-IQ4_NL.gguf --out ik2 \
  --chat-template tools/ik2/chat_template.jinja
python3 serve/server.py --engine maya --config docs/inkling2-mini/ik2-mini-gfx1100.json \
  --host 127.0.0.1 --port 18085
```

Then point any OpenAI or Anthropic client at `http://127.0.0.1:18085/v1`, model `inkling-2-mini`. If your machine has no ROCm toolchain, the [build guide](docs/HIP-PORT.md#build-on-prometheus-without-a-gpu) runs the same build inside the `ghcr.io/ggml-org/llama.cpp:server-rocm-b11176` container. A [Docker image with the image encoder](docs/HIP-PORT.md#run-the-packaged-image-on-prometheus) is also described there.

## Upstream CUDA results on RTX 3090

<p align="center"><b>Thinking Machines' 26B looped MoE, all of it in 24 GB of VRAM. 128 tok/s plain decode, 150 to 210 tok/s on code and edits with the official DSpark drafter, 2,700 tok/s prefill, up to 344K context, images and tools. llama.cpp gets 57 tok/s on the same card.</b></p>

<p align="center">
<a href="docs/INKLING2-MINI.md"><b>Read the guide</b></a> ·
<a href="#cuda-quick-start"><b>CUDA quick start</b></a> ·
<a href="https://huggingface.co/unsloth/Inkling-2-Mini-GGUF"><b>Weights on Hugging Face</b></a>
</p>

| Text, greedy | Plain decode, tok/s | With the drafter, tok/s | llama.cpp b12590 |
|---|---:|---:|---:|
| Prose | 126 | 131 | 57.0 |
| New code | 125 | 149 (through the server 182 to 207) | |
| Editing a document | 124 | 212 | |
| Copying code | 124 | 188 | |
| A 64K token document | 104 | 125 | 42.7 (65,536 deep) |

One RTX 3090 at a 300 W limit, Unsloth `UD-Q4_K_XL`, FP8 KV cache, a cold prompt every time. Prefill runs at 2,742 tok/s on 4K tokens, 2,361 on 64K and 2,079 on 128K. Every launch and log is in the [guide](docs/INKLING2-MINI.md#numbers).

## CUDA engine features

* **`maya-ik2`**, an engine for [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini): the 24 layers, the loop (layers 6 to 17 run 3 times, 48 layer visits per token), the relative bias, the short convs and the attention residuals.
* **The model stays in VRAM.** All 3,072 experts (14.7 GiB), the dense weights and an FP8 KV cache fit in 24 GB: 344,064 tokens without the drafter, 262,144 with it. Nothing is read from RAM or disk while it answers.
* **Speculative decoding with the official drafter.** Thinking Machines ships a 3-layer DSpark drafter with the model. The engine checks its guesses 2 to 8 at a time in one pass, and drafts only when that pays: code and edits get a lot faster, prose stays as fast as plain decode. Greedy and sampled output are the model's own.
* **The original weights.** The experts of layers 0 to 19 are Thinking Machines' own NVFP4 bits, the way Unsloth's GGUF carries them. A 3090 has no FP4 hardware, so the engine decodes the 4-bit codes in registers and feeds the tensor cores.
* **Images, tools, reasoning control, streaming** through OpenAI and Anthropic compatible APIs on `/v1`.

## CUDA quick start

Linux, an RTX 3090 (or another 24 GB `sm_86` card), CUDA 12.8, Python 3.10 or newer.

```bash
git clone https://github.com/alesha-pro/maya-inkling-mini && cd maya-inkling-mini
pip install numpy jinja2 regex pyyaml tqdm requests pillow psutil "jsonschema>=4.18,<5" huggingface_hub

# the engine
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release -DSTRATA_ENABLE_CUDA=ON \
  -DCMAKE_CUDA_ARCHITECTURES=86 -DSTRATA_BUILD_TESTS=OFF
cmake --build build -j 24 --target maya-ik2

# the image encoder (skip it for text only)
git clone --depth 1 -b b12590 https://github.com/ggml-org/llama.cpp third_party/llama.cpp-b12590
cmake -S tools/vision -B build-vision -G Ninja -DCMAKE_BUILD_TYPE=Release -DLLAMA_DIR=third_party/llama.cpp-b12590
cmake --build build-vision -j 24 --target maya-vision

# the weights (17.6 GB), the drafter (0.6 GB) and the tokenizer files
hf download unsloth/Inkling-2-Mini-GGUF Inkling-2-Mini-UD-Q4_K_XL.gguf mmproj-BF16.gguf --local-dir models
hf download thinkingmachines/Inkling-2-Mini mtp.safetensors --local-dir models
python tools/strata_tokenizer.py --gguf models/Inkling-2-Mini-UD-Q4_K_XL.gguf --out ik2 \
  --chat-template tools/ik2/chat_template.jinja --check

python serve/server.py --engine maya --config docs/inkling2-mini/ik2-mini.json --host 127.0.0.1 --port 18000
```

Then talk to `http://127.0.0.1:18000/v1` with any OpenAI client, model `inkling-2-mini`. There are four CUDA configs in [`docs/inkling2-mini/`](docs/inkling2-mini/): the default (the drafter, 262K context, image encoder on the CPU), the longest context (344K, no drafter), text only, and the image encoder on the GPU.

## Why the CUDA engine is faster than llama.cpp

llama.cpp runs about 7,350 kernel launches per token for this model. A token here is 757 launches replayed as one CUDA graph, and the GPU never waits for the host between the embedding and the logits.

After that a token costs what its weights weigh: 3.7 GB read from VRAM, half of it experts. So the kernels are built around memory speed. Attention reads the KV cache at up to 846 GB/s, the NVFP4 experts go through the tensor cores at 745 GB/s, and an FP8 cache beats FP16 deep in a context because it is half the bytes.

The drafter's window has its own path: the tokens' expert picks are grouped on the GPU so an expert is read once for all the tokens that picked it, and the window runs as one graph too. 5 tokens cost 15.6 ms against 7.8 ms for one.

The [guide](docs/INKLING2-MINI.md#how-it-works) has the details, and what I tried that didn't work.

## Upstream CUDA quality checks

The engine computes the model the GGUF holds, no pruning and no requantizing. On 96 GSM8K questions, greedy: 74 solved with plain decode and 77 with the drafter (the drafter only changes which pass computes a token, so the difference is noise). Against the float64 reference in `ref/inkling2.py` the engine picks the same top token on 45 to 46 of 47 positions, and the reference agrees with llama.cpp on 47 of 47.

## Upstream CUDA limits

* One request at a time. Two GPUs run as expert parallel (`IK2_EP=1`, tested on Windows with an RTX 5080 and an RTX 4060 Ti by @cobra91), the drafter is one GPU's.
* It reads `UD-Q4_K_XL` only. `UD-Q8_K_XL` keeps the last 4 layers' experts in BF16 and the engine doesn't load them.
* Decode slows down with depth, because 8 of the 48 layer visits attend over the full prefix: 16 KB of cache per token of context is read for every new token.
* The drafter helps code, edits and structured text. On free prose it accepts about 2 tokens per window and the engine mostly decodes plain.
* The image encoder runs on the CPU in the default configs. A new image takes about 4.5 s on a 48-core EPYC, the same image again 0.5 s.
* Tested on RTX 3090s with the `gpu-unlock` kernel module, Ubuntu 22.04, driver 610.43. Windows builds since @cobra91's port; other cards are untested.

## About this repo

This is [Project Maya](https://github.com/mw00/project-maya) v1.1.70, the engine family by Peasant Smith ([@mw00](https://github.com/mw00)) and the Maya contributors, which grew out of [Strata](https://github.com/Niko1221/Strata) by Niko1221. `maya-ik2` started as the Inkling 2 Small engine from Maya's pull requests [#981](https://github.com/mw00/project-maya/pull/981) and [#1079](https://github.com/mw00/project-maya/pull/1079). alesha-pro's repo adds the loop, the path that keeps every expert in VRAM, the tensor core kernels for attention and experts, FP8 KV for prefill, GPU sampling candidates, speculative decoding with the official drafter, and the Mini profile of the server. The Windows / CUDA 13 build and two-GPU expert parallel are by [@cobra91](https://github.com/cobra91). Most of it is offered upstream in [mw00/project-maya#1095](https://github.com/mw00/project-maya/pull/1095). The rest of Maya (GLM-5.3, DeepSeek-V4.1-Flash, MiniMax-M3.1-Flash, Hy4-flash and the app around them) is still in the tree but isn't maintained here: use [upstream Maya](https://github.com/mw00/project-maya) for those.

Weights: [Inkling 2 Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini) and its drafter by Thinking Machines, GGUF by [Unsloth](https://huggingface.co/unsloth/Inkling-2-Mini-GGUF).

## License

The code is MIT, as upstream ([LICENSE](LICENSE)). The model weights are Apache 2.0.
