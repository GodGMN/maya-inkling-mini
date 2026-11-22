# Inkling 2 Mini on gfx1100

On Prometheus’s RX 7900 XTX, the Radeon server preset runs the installed ggml-org IQ4_NL model with all routed experts in VRAM. See the [decode measurements](#fused-decode-projections), [uncached prefill measurements](#cooperative-prefill-and-depth-dependent-decode-attention) and [long-context retrieval results](#exact-phrase-retrieval-at-long-context).

## Build on Prometheus without a GPU

The source tree is under `~/ai/build/maya-ik2-hip/src`. The build container mounts its parent at `/work`. The following commands reproduce the environment from a synced source tree. The llama.cpp archive is the source revision pinned by Maya's top-level CMake file.

```sh
cd ~/ai/build/maya-ik2-hip
curl -fLsS https://codeload.github.com/ggml-org/llama.cpp/tar.gz/3cf03257f219afbe7334045ff7c6a06ac68c627d | tar xz
docker run -d --name maya-ik2-hip-build --memory 6g --memory-swap 6g --cpus 4 \
  -v "$PWD:/work" -w /work --entrypoint sleep ghcr.io/ggml-org/llama.cpp:server-rocm-b11176 infinity
docker exec -it maya-ik2-hip-build bash
```

Run the following inside that container.

```sh
apt-get update -qq
apt-get install -y --no-install-recommends build-essential cmake ninja-build git python3 python3-pip libomp-dev
cmake -S src -B build-ik2 -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DSTRATA_ENABLE_HIP=ON -DSTRATA_PREFILL_MMQ=ON -DSTRATA_BUILD_TESTS=ON \
  -DCMAKE_HIP_COMPILER=/opt/rocm/llvm/bin/clang++ -DCMAKE_PREFIX_PATH=/opt/rocm \
  -DCMAKE_HIP_ARCHITECTURES=gfx1100 -DGPU_TARGETS=gfx1100 \
  -DSTRATA_GGML_DIR=/work/llama.cpp-3cf03257f219afbe7334045ff7c6a06ac68c627d
cmake --build build-ik2 -j4 --target maya-ik2 ik2_kernels_test ik2_host_ref_test
```

The base library uses the same source and target architecture. For the separate base build directory on Prometheus, the build command is `cmake --build build -j4 --target strata_engine`. The engine links its own kernels and hipBLAS directly.

## Serve the IQ4_NL model on gfx1100

The Radeon preset runs a single text session with FP8 KV. From the repository root inside the ROCm environment, set the paths in the preset to the built executable and the installed GGUF, then export its tokenizer and start the HTTP server.

```sh
MODEL=models/Inkling-2-Mini-IQ4_NL.gguf
python3 tools/strata_tokenizer.py --gguf "$MODEL" --out ik2 --chat-template tools/ik2/chat_template.jinja
python3 serve/server.py --engine maya --config docs/inkling2-mini/ik2-mini-gfx1100.json \
    --host 127.0.0.1 --port 18085
```

On Prometheus the executable and installed weights were mounted at `/work/specialized` and `/weight.gguf`. The loopback HTTP check used a 32,768-token context and 1,024-token chunks. `GET /v1/models` returned `inkling-2-mini`. The ordinary chat request returned “Paris” and the streaming request returned “42”, a stop reason, and the final SSE `[DONE]`.

```sh
curl -fsS http://127.0.0.1:18085/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is the capital of France? Reply in one word."}],"temperature":0,"reasoning_effort":"none","max_tokens":512}'
curl -NsS http://127.0.0.1:18085/v1/chat/completions -H 'Content-Type: application/json' \
    -d '{"model":"inkling-2-mini","messages":[{"role":"user","content":"What is 17 plus 25? Reply with digits only."}],"temperature":0,"reasoning_effort":"none","max_tokens":512,"stream":true}'
```

## CUDA and HIP differences in the initial scalar port

| Area | Phase 1 implementation |
| --- | --- |
| Wave size | The lane index, reductions, and broadcasts assume 32 lanes. The target is restricted to gfx1100. Strata's HIP intrinsics header supplies CUDA-compatible byte selection and warp operations. |
| Decode and expert MMA | `mma.sync.m16n8k16`, packed-half PTX and Hopper `wgmma` use NVIDIA-specific register layouts. HIP omits those kernels and dispatches the existing scalar products. The CUDA switches keep their original behavior on NVIDIA. |
| Prefill | The CUDA flash kernel uses `ldmatrix` and pipelined `cp.async`. Its dynamic shared-memory request is 88,064 bytes, above the 65,536-byte gfx1100 workgroup limit. HIP uses the existing tiled logits, softmax and strided batched GEMM path. With FP8 KV it unpacks cache chunks to FP16. This requires more scratch than CUDA flash. |
| Cache and weights | gfx1100 has no native FP8 or FP4 math. KV uses the software E4M3FN conversion, including subnormals. NVFP4 uses E2M1 nibbles and unsigned block scales. Device compilation selects device-resident IQ lookup tables. |
| Graphs | Global stream capture maps to HIP global capture. Instantiate and replay use the existing hipGraph wrappers. `IK2_GRAPH=0` runs individual launches for comparison. Replay and captured host/device copies need a card test. |
| Memory | The compatibility header maps managed allocation, advice, prefetch and the CPU device ID to HIP. Managed KV remains opt-in and checks the device's concurrent managed access attribute. Ordinary resident decode uses device allocations. |
| Speculative block verification | Block attention and its dense Q8 tile require CUDA MMA. HIP disables the block path. Serving rejects a drafter because the server verifies its proposals with that path. Explicit nonzero requests for `IK2_BLOCK`, `IK2_Q8_TILE`, `IK2_GROUP_MMA`, `IK2_EXPERT_MMA`, `IK2_ATTN_MMA`, `IK2_FLASH` or `IK2_WGMMA` fail at startup. |

## Verification

`/opt/rocm/llvm/bin/llvm-objdump --offloading build-ik2/maya-ik2` identified gfx1100 code objects in the linked executable. After extraction, `/opt/rocm/llvm/bin/llvm-readelf --notes build-ik2/maya-ik2.*gfx1100` reported a wavefront size of 32. `ldd build-ik2/maya-ik2` resolved the ROCm 7.2.1 HIP runtime and hipBLAS.

Generate the synthetic input on a CPU with NumPy and PyTorch. The fixture uses PyTorch to decode the FP8 scales and calls `ref/inkling2.py`'s `Ref.moe` for the expected output. It requires no model files.

```sh
python3 tests/ik2_host_ref_fixture.py > /tmp/ik2-host-ref.bin
scp /tmp/ik2-host-ref.bin sergio@192.168.100.111:ai/build/maya-ik2-hip/
```

Inside the build container, run `build-ik2/ik2_host_ref_test < /work/ik2-host-ref.bin`. On the Ryzen 5 3600 the host decoder matched every packed weight exactly. The C++ CPU expert's maximum absolute difference from the one-expert `Ref.moe` output was `1.90734863e-06`. This fixture covers one input through gate, up, SwiGLU and down. It does not exercise device execution or whole-model routing.

`python3 -m unittest tools.test_hip_compat` passed the repository's CUDA name mapping checks. The CUDA 13 managed-memory shims are excluded from HIP. Both builds completed with the build container limited to four CPUs and 6 GiB of RAM with swap disabled. At completion, `df -h /work` reported 21G available on the root filesystem.

To rerun the device suite, use `ctest --test-dir build-ik2 -R '^ik2_kernels_test$' --output-on-failure` with the GPU available. The suite contains scalar expert, routing, FP8 and FP16 attention checks against numerical references. The GPU validation below uses the installed ggml-org weights. Compare logits on identical forced tokens, generated text, and graph-on/off output before timing prefill and decode at different prompt depths.

### Forced decode comparison

Use identical prompt and forced token IDs in each run. The CLI writes its first raw logit row **after consuming the first forced token**. For `--gen N` it writes N−1 rows and omits the prompt prediction. The comparator checks the expected byte counts and reports top-1 agreement, mean/median/max KL, and the input position with the largest KL. It reads one vocabulary row at a time.

Example for the GPU validation session, with `MODEL` pointing to the approved Mini GGUF, in a scratch directory. The token IDs below exercise the alignment only. For quality measurements use tokenized real text and the same cache format, prompt, and forced file in both runs.

```sh
printf '%s\n' '101,102,103' > force.csv
for graph in 0 1; do
  IK2_KV=fp8 IK2_GRAPH="$graph" IK2_FORCE=force.csv IK2_LOGITS_OUT="graph-$graph.f32" \
    /work/build-ik2/maya-ik2 --model "$MODEL" --tokens 1,2 --gen 4 --ctx 256
done
python3 /work/src/tools/ik2_compare_decode.py graph-1.f32 --baseline graph-0.f32 \
  --prompt-length 2 --gen 4 --vocab 200058
```

For comparison against the independent model reference, run the reference on a CPU machine with sufficient RAM for the GGUF forward pass. Pass the prompt followed by the first N−1 forced tokens. The example compares the same three decode rows as above. `--logprobs-out` saves the final array without retaining intermediate layer dumps.

```sh
python3 ref/inkling2.py "$MODEL" --tokens 1,2,101,102,103 --logprobs-out ref.npy
python3 tools/ik2_compare_decode.py graph-1.f32 --ref ref.npy --prompt-length 2 --gen 4 --vocab 200058
```

Run throughput measurements separately with logit export disabled, since exporting the full vocabulary adds a device-to-host copy per step.

## Prometheus GPU validation

The installed ggml-org IQ4_NL GGUF can be mounted read-only at `/weight.gguf`. It stores 24 physical layers under `inkling2_loop`. The three passes over layers 6–17 produce 48 visits. Each core layer stores its four message queries in q, k, v, FFN order. Twenty layers use IQ4_NL experts and four use Q8_0. Both the engine and CPU oracle read this layout directly.

The gfx1100 run of `build-ik2/ik2_kernels_test` passed, including the new IQ4_NL comparisons. HIP FP contraction across statements changed the MoE convolution sum. The kernel target now contracts only within a source expression. KV unpacking preserves the E4M3 sign bit through rescaling, including negative zero. The tests exercise scalar and packed reads at inverse scales 0.5, 1 and 2. The CUDA-only MMA and flash tests are excluded from this build.

The five-token prompt below is “The capital of France is”. On the three forced positions the scalar graph and eager logits agreed exactly by the comparator. The float64 reference and FP8-KV engine picked the same top token at all three positions, with mean KL 0.00443123 and max KL 0.00697867. This short check exercises the loop wiring and token alignment.

```sh
cd /work
export IK2_HOST=mmap IK2_KV=fp8 IK2_GATE_BF16=1 IK2_GATE_TILE=64 IK2_Q8_UNROLL=1 IK2_BATCH_PREFILL=1
printf '%s\n' '152717,11,149880' > paris-force.csv
for graph in 0 1; do
  IK2_GRAPH=$graph IK2_FORCE=paris-force.csv IK2_LOGITS_OUT=paris-$graph.f32 \
    build-ik2/maya-ik2 --model /weight.gguf --ctx 1024 --chunk 128 --tokens 371,151451,149879,151194,149884 --gen 4
done
oracle-env/bin/python src/ref/inkling2.py /weight.gguf --tokens 371,151451,149879,151194,149884,152717,11,149880 --logprobs-out paris-ref.npy
oracle-env/bin/python src/tools/ik2_compare_decode.py paris-1.f32 --baseline paris-0.f32 --prompt-length 5 --gen 4 --vocab 200058
oracle-env/bin/python src/tools/ik2_compare_decode.py paris-0.f32 --ref paris-ref.npy --prompt-length 5 --gen 4 --vocab 200058
```

### Expert decode baseline

With the environment above and graph replay enabled, `build-ik2/maya-ik2 --model /weight.gguf --ctx 1024 --chunk 128 --tokens 371,151451,149879,151194,149884 --gen 65` measured 32.65 tok/s for 64 scalar decode steps. The prompt predicted the first generated token. On the same prompt, `IK2_GRAPH=0 rocprofv3 --kernel-trace --stats -f csv -d /work/profile-scalar -- /work/scalar-baseline --model /weight.gguf --ctx 1024 --chunk 128 --tokens 371,151451,149879,151194,149884 --gen 33` attributed 43.83% of the traced kernel time to single-token gate/up and 21.88% to down projections.

The vectorized IQ4_NL path reads four packed bytes per lane and uses gfx1100 byte-permute instructions to decode eight weights. Input activations and dot accumulators remain FP32. The GPU kernel suite passed. For a text check, tokenize “Water freezes at” and the continuation “ zero degrees Celsius and boils at one hundred degrees Celsius at sea level. At higher altitudes the boiling point falls because atmospheric pressure is lower.” with `tools/strata_tokenizer.py`. On this GGUF they yield 4 prompt and 30 continuation tokens. Force the continuation with `--gen 31`, export logits and run the oracle on all 34 tokens. `oracle-env/bin/python src/tools/ik2_compare_decode.py v1-water.f32 --ref water-ref.npy --prompt-length 4 --gen 31 --vocab 200058` measured 30/30 top-1, mean KL 0.0018280142 and max KL 0.0081613966.

### RDNA3 expert and attention kernels

`IK2_EXPERT_LIMB=60` uses signed integer dot products for IQ4_NL and Q8_0 experts during decode. Each group of sixteen activations shares a scale and uses two signed bytes per value. `IK2_BATCH_PREFILL=1 IK2_PREFILL_WMMA=1` groups routed tokens for native 16×16×16 FP16 matrix instructions with FP32 accumulation. `IK2_ATTN_MMA=1` enables WMMA decode attention on every visit, and `2` selects full-attention visits only. `IK2_FLASH=1` enables WMMA prefill attention. Both attention kernels accept FP16 and software-decoded E4M3 KV, including ring buffers, relative bias and the sink. At this stage HIP still rejected the CUDA block verifier. A later revision added native RDNA3 block attention.

These are plain greedy rates, using three uncached requests per depth in one resident process. The server predicts the first output token during prefill. Each decode timing covers the following 128 tokens. For this measurement, attention used the HIP scalar decode and hipBLAS prefill paths. The reported 64 tok/s for the existing llama.cpp service came from a separate run.

| Prompt tokens | Decode tok/s, three runs | Uncached prefill tok/s, three runs |
| ---: | --- | --- |
| 32 | 95.10, 95.10, 96.24 | 12.90, 48.71, 49.00 |
| 512 | 91.69, 91.69, 91.56 | 890.43, 888.89, 890.43 |
| 2,048 | 87.55, 87.55, 87.55 | 818.87, 825.14, 819.86 |

Run inside the ROCm container, with the installed IQ4_NL GGUF mounted at `/weight.gguf`. The benchmark alternates the first prompt token, checks the server's cached-token count, and requests greedy decoding without a stop token. The water continuation described above supplies these input IDs.

```sh
cd /work
echo '151712,156358,179692,149880,167442,82,149922,150011,156258,156358,179692,149922,151333,150628,13,150847,151197,178091,82,149876,167442,279,150279,159383,150301,186867,152023,149884,151214,13' > water.force
IK2_HOST=mmap IK2_KV=fp8 IK2_GATE_BF16=1 IK2_GATE_TILE=64 IK2_Q8_UNROLL=1 \
IK2_BATCH_PREFILL=1 IK2_PREFILL_WMMA=1 IK2_EXPERT_LIMB=60 IK2_ATTN_MMA=0 IK2_FLASH=0 \
python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
    --tokens water.force --depths 32,512,2048 --gen 129 --repeats 3 --chunk 128
```

The four full-vocabulary checks below used the same WMMA attention and integer-dot decode configuration. Each row compares every forced step with the independent float64 forward, including the prediction after the final forced token. The three chat checks used terminal token `<|content_model_end_sampling|>` and the normal chat template at effort `none`. They returned Paris, `12 - 3 - 4 = 5 apples`, and an explanation of ice's lower density and open hydrogen-bonded lattice.

| Passage | Prompt | Forced positions | Top-1 matches | Mean KL | Max KL |
| --- | ---: | ---: | ---: | ---: | ---: |
| Water | 4 | 30 | 30 | 0.00147747 | 0.00451225 |
| Fibonacci | 8 | 33 | 33 | 0.00048531 | 0.00675873 |
| Apples | 7 | 33 | 33 | 0.00152825 | 0.00606835 |
| Library | 114 | 13 | 11 | 0.01083507 | 0.05335118 |

For the library check, repeat “The town library keeps its reading room open on Saturdays. Visitors can borrow a book for three weeks and renew it online. ” four times and append “A visitor borrows a book on Saturday.” Force the continuation “ When is the book due? It is due three weeks later.” Starting from the benchmark environment above, enable `IK2_FLASH=1 IK2_ATTN_MMA=1`. Run the CLI with `--tokens @long.prompt --gen 14 --ctx 1024 --chunk 128`, `IK2_FORCE=long.force`, and `IK2_LOGITS_OUT=all-flash-long.f32`. Compare using `python3 tools/ik2_compare_decode.py all-flash-long.f32 --ref long-ref.npy --prompt-length 114 --gen 14 --vocab 200058`. The two top-1 changes were at positions 114 and 119. These are distribution checks on four passages.

The Loom bundle uses the original Unsloth quant and an external drafter on Strix Halo. Its report identifies speculative acceptance separately for each context. Here the weights are ggml-org IQ4_NL and the decoder predicts one token per step. The Loom code identifies two relevant operations for the port, grouping routed experts for WMMA and accumulating relative-position attention with an explicit sink. [Loom measurement report](https://huggingface.co/loom-fun/unsloth-Inkling-2-Mini-loom/blob/a78779814f545e254cffa9b346e85f236d54ea7c/REPORT.md)

The upstream Mini pull request describes the CUDA engine this fork ports, including graph replay and the DSpark verifier. The other SM80 pull request targets NVIDIA Ampere with E5M2 KV and tensor parallelism. Prometheus uses one gfx1100 device and the port's tested E4M3 cache. [Mini engine source](https://github.com/alesha-pro/maya-inkling-mini) · [Project Maya #1095](https://github.com/mw00/project-maya/pull/1095) · [SM80 #1](https://github.com/snailbrainx/inkling-2-mini-sm80/pull/1)

The inference optimization playbook studies larger MoE models whose experts spill into system RAM. Its advice on measuring prefix reuse applies to the two servers here. All experts of the installed Mini checkpoint fit in VRAM, which changes the value of its CPU and PCIe strategies. [Inference optimization playbook](https://github.com/alesha-pro/inference-optimization-playbook)

### FP32 expert decode and packed E4M3 reads

The IQ4_NL decode kernel reads eight GGUF blocks per wave and reuses each FP32 activation across gate and up rows. It also reads the FP16 down input directly from registers after conversion. The 32- and 128-row FP16 projections use four waves per output row. For attention, a pair of E4M3 bytes is expanded to FP16 in two lanes. Scaling the packed FP16 result by 256 preserves E4M3 subnormals. The WMMA prefill kernel stages Q and transposes V in LDS. The device suite checks the packed conversions, signed zero, small dense products, fused SwiGLU, down projections, and both attention formats against numerical references.

The following rates are medians over three uncached prompts, 128 timed greedy decode steps each. A single resident process served all four depths. The scalar attention columns use the same packed KV conversion and FP32 expert inputs. The experiment changed one expert or attention switch per comparison.

| Prompt tokens | Direct IQ4 + scalar attention | Previous IQ4 + scalar attention | Direct IQ4 + WMMA attention | Direct IQ4 uncached prefill |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 105.52 | 101.27 | 99.07 | 323.23 |
| 512 | 101.27 | 97.19 | 88.52 | 1,003.92 |
| 2,048 | 97.12 | 93.23 | 84.04 | 973.85 |
| 8,192 | 92.69 | 89.20 | 80.30 | 949.03 |

All throughput columns are in tokens per second. Run the following from `/work` with `water.force` as above. Set `IK2_IQ4_MV=0` for the previous expert kernel or `IK2_ATTN_MMA=1` for WMMA decode attention.

```sh
export IK2_HOST=mmap IK2_KV=fp8 IK2_GATE_BF16=1 IK2_GATE_TILE=64 IK2_Q8_UNROLL=1
export IK2_BATCH_PREFILL=1 IK2_PREFILL_WMMA=1 IK2_FLASH=1 IK2_Q8_TILE=1
export IK2_EXPERT_LIMB=0 IK2_IQ4_MV=1 IK2_F16_MV=1 IK2_ATTN_MMA=0
python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
    --tokens water.force --depths 32,512,2048,8192 --gen 129 --repeats 3 --chunk 128
```

### Comparison with the production llama.cpp service

The same `ik2_bench.py` replayed the same input IDs, request order and 129-token output length against the patched llama.cpp HTTP service. Both decoders ran alone on the 7900 XTX. The service retained its deployed 393,216-token context, Q4_0 K/V, batch 2,048 and microbatch 256. Maya reserved 8,321 positions for this run, used E4M3 K/V and a 128-token chunk. The benchmark rejects a cached prompt or incomplete output and divides the generation time by the 128 steps after prefill.

```sh
python3 src/tools/ik2_bench.py --llama-url http://127.0.0.1:8085 --tokens water.force \
    --depths 32,512,2048,8192 --gen 129 --repeats 3
```

| Prompt | Maya decode | llama.cpp decode | Maya prefill | llama.cpp prefill |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 105.52 | 64.14 | 323.23 | 247.14 |
| 512 | 101.27 | 63.96 | 1,003.92 | 1,304.86 |
| 2,048 | 97.12 | 62.89 | 973.85 | 1,511.03 |
| 8,192 | 92.69 | 61.97 | 949.03 | 1,488.44 |

Medians in tok/s. The wider comparison is one GPU, one stream and one installed GGUF. Each runtime uses its own KV representation and greedy trajectories can diverge.

### GSM8K

The first 96 GSM8K test questions use the official Inkling chat template with reasoning effort `none`, greedy sampling, and up to 1,024 generated tokens. The scorer reads the final content channel and compares its last number to the labeled answer. Missing final answers count as incorrect. With the FP32 expert configuration above, Maya scored **90/96** and the production llama.cpp service scored **93/96**. Both missed questions 12, 40 and 93 (zero-based). Maya reached the output limit before the final answer on questions 41, 86 and 87. The service reached the limit on 12 and 40. Both engines returned a final answer on the other 91 questions. This sample measures arithmetic and instruction following. [OpenAI GSM8K dataset](https://github.com/openai/grade-school-math/tree/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data)

```sh
cd /work
curl -fLsS https://raw.githubusercontent.com/openai/grade-school-math/3101c7d5072418e28b9008a6636bde82a006892c/grade_school_math/data/test.jsonl -o gsm8k-test.jsonl
oracle-env/bin/python src/tests/ik2_gsm8k.py --data gsm8k-test.jsonl --model /weight.gguf --count 96 >gsm96-requests.txt
# Use the environment from the FP32 expert benchmark.
./build-ik2/maya-ik2 --serve-ik2 --model /weight.gguf --ctx 2048 --chunk 128 <gsm96-requests.txt >gsm96.log
oracle-env/bin/python src/tests/ik2_gsm8k.py --data gsm8k-test.jsonl --model /weight.gguf --count 96 --score gsm96.log
oracle-env/bin/python src/tests/ik2_gsm8k.py --data gsm8k-test.jsonl --model /weight.gguf --count 96 \
    --llama-url http://192.168.100.111:8085
```

At a 4,096-token cap, the cooperative RDNA3 configuration answered **93/96** on questions 0–95 and **94/96** on 96–191. All 192 requests finished. The three misses on the first set were questions 12, 40 and 93. On the second set they were 100 and 119. Across the two runs, generation averaged 108.41 and 108.15 tok/s respectively, computed as the sum of generated tokens after the first token divided by the sum of decode seconds. Neither run reused a prompt prefix. These raw token requests let the model emit a reasoning channel.

```sh
# Use the cooperative prefill environment below with IK2_EXPERT_TILE_LDS=1.
for offset in 0 96; do
    oracle-env/bin/python src/tests/ik2_gsm8k.py --data gsm8k-test.jsonl --model /weight.gguf \
        --count 96 --offset "$offset" --max-tokens 4096 >"gsm-$offset.requests"
    ./build-ik2/maya-ik2 --serve-ik2 --model /weight.gguf --ctx 8192 --chunk 1024 \
        <"gsm-$offset.requests" >"gsm-$offset.log"
    oracle-env/bin/python src/tests/ik2_gsm8k.py --data gsm8k-test.jsonl --model /weight.gguf \
        --count 96 --offset "$offset" --score "gsm-$offset.log"
done
```

### Larger prefill chunks and exact router accumulation

On the installed IQ4_NL model, the 56 attention projection matrices occupy 1,228 MiB when dequantized to FP16. `IK2_PREFILL_DENSE=1` retains those matrices when VRAM has room after the expert cache and headroom. Decode continues to read the original planar Q8. The grouped expert kernel expands signed byte pairs and applies the original FP16 block scale. It can reuse one loaded weight tile across 16, 32 or 64 routed tokens. The table uses the 16-token tile.

| Prompt | Chunk | Uncached prefill, both runs (tok/s) | Decode, both runs (tok/s) |
| ---: | ---: | ---: | ---: |
| 512 | 512 | 1,625.40, 1,635.78 | 101.91, 101.99 |
| 8,192 | 512 | 1,485.94, 1,485.94 | 93.36, 93.43 |
| 8,192 | 1,024 | 1,589.14, 1,586.67 | 93.50, 93.50 |
| 8,192 | 2,048 | 1,666.40, 1,663.35 | 93.64, 93.57 |
| 32,768 | 2,048 | 1,339.38, 1,339.82 | 79.06, 79.11 |

The production comparison above used identical tokens at 8,192 and reached 1,488.44 tok/s prefill and 61.97 tok/s decode. The 2,048-token chunk improves prefill, costs workspace and adds startup latency on the first matrix product.

```sh
# Start with the FP32 expert environment above.
export IK2_PREFILL_DENSE=1 IK2_PREFILL_TILE=16 IK2_Q8_MV=1 IK2_F16_MV=2
for chunk in 512 1024; do
    python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
        --tokens water.force --depths 32,512,2048,8192,32768 --gen 129 --repeats 2 --chunk "$chunk"
done
python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
    --tokens water.force --depths 2048,8192,32768 --gen 129 --repeats 2 --chunk 2048
```

`IK2_F16_MV=2` preserves the scalar router's FP32 FMA sequence within each lane and its wave reduction order. The gfx1100 device test compares its output bit for bit to this CPU computation. The library passage with E4M3 KV matched 12/13 CPU top-1 predictions (mean KL 0.02426905, maximum 0.22447712). With FP16 KV and uncompressed loop-gate weights it matched 9/13, with a lower mean KL of 0.00308744 and maximum 0.01372127. Both are full distributions over 200,058 tokens. Reproduce both with the library command above, `--chunk 128`, `IK2_ATTN_MMA=0 IK2_EXPERT_LIMB=0 IK2_IQ4_MV=1 IK2_Q8_MV=1 IK2_F16_MV=2 IK2_PREFILL_DENSE=1`. For the second run additionally set `IK2_KV=f16 IK2_GATE_BF16=0`.

### Cooperative prefill and depth-dependent decode attention

A 64×64 WMMA projection tile replaced the hipBLAS FP16 products in large prefill batches. Four wave32 groups now share the K and transposed V for 64 query tokens. In decode, eight lanes cooperate on one key for sliding attention and short global attention. Global visits switch to WMMA at 8,192. The engine recaptures the graph when the split count or attention kernel changes. Expert matvecs specialize the 2,304- and 1,152-wide matrices of the installed model.

| Prompt | Maya prefill | Maya decode | Deployed llama.cpp prefill | Deployed llama.cpp decode |
| ---: | ---: | ---: | ---: | ---: |
| 32 | 376.47 | 111.21 | 247.14 | 64.14 |
| 512 | 2,370.37 | 109.50 | 1,304.86 | 63.96 |
| 2,048 | 3,362.89 | 107.29 | 1,511.03 | 62.89 |
| 8,192 | 3,558.64 | 100.63 | 1,488.44 | 61.97 |
| 32,768 | 3,010.93 | 87.67 | | |

All five rows use the fixed asynchronous input transfer. A fast resident prefill could previously overwrite pinned embeddings or positions while their copy was pending. Reproduce the warm vs cold continuation check with `python3 src/tests/ik2_prefix_cache.py --engine ./build-ik2/maya-ik2 --model /weight.gguf --tokens water.force`.

Medians in tok/s for three uncached requests per depth. Each request generates 129 tokens with 128 timed decode steps. Maya reserved 32,897 positions with FP8 KV. The service retained its deployed 393,216 positions with Q4_0 KV. On the first 32-token request the initial matrix-library setup took the prefill rate down to 18.52 tok/s. The other two measured 376.47 tok/s.

```sh
export IK2_HOST=mmap IK2_KV=fp8 IK2_GATE_BF16=1 IK2_GATE_TILE=64 IK2_Q8_UNROLL=1
export IK2_BATCH_PREFILL=1 IK2_PREFILL_WMMA=1 IK2_PREFILL_TILE=64 IK2_EXPERT_TILE_LDS=1 IK2_SHARED_COMBINE=1
export IK2_F16_MV=2 IK2_IQ4_MV=1 IK2_Q8_MV=1 IK2_Q8_TILE=1 IK2_EXPERT_LIMB=0
export IK2_PREFILL_DENSE=1 IK2_DENSE_WMMA=1 IK2_FLASH=1 IK2_FLASH_WAVES=4
export IK2_ATTN_MMA=3 IK2_DECODE_SUBWAVE=1 IK2_ATTN_STAGED=1 IK2_RESIDENT_ONLY=1
python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 --model /weight.gguf --tokens water.force \
    --depths 32,512,2048,8192,32768 --gen 129 --repeats 3 --chunk 2048
```

The gfx1100 device suite checks WMMA against FP64 on partial tiles and padded outputs. The attention tests cover E4M3 and FP16, sink and relative bias, ring wraparound, causal masking and multi-token block verification. Run `IK2_EXPERT_TILE_LDS=1 ./build-ik2/ik2_kernels_test` with the environment above. Two output rows sharing each Q8 input load slowed decode at every measured depth, so the engine keeps one row per wave.

For long-context retrieval, `tests/ik2_retrieval.py` places a seeded four-word phrase among GSM8K distractor questions at an exact prompt depth. The score records the final content channel, the number of cached tokens and the stop reason. To reproduce the midpoint test, pass `--tokenizer ik2-tokenizer/tokenizer --data gsm8k-test.jsonl --depths 32768,131072,392960`. Feed stdout to a resident server with `--ctx 393216 --chunk 1024`, then run the same Python command with `--score SERVER.log`.

### Copy and edit requests without a drafter

Maya can propose a continuation from the request's own tokens and check it against the target model. On Prometheus, a 64-token verification window copied an 81-box inventory at **271.15 tok/s** and made a single requested edit at **196.72 tok/s**. Plain decoding produced the same two final answers at 105.23 and 106.21 tok/s. Each figure is one uncached greedy request with a 3,072-token output limit. Timing excludes prefill and the first token. The edit took 1,586 generated tokens with verification and 1,601 without it, including the model's reasoning. These examples measure repeated text.

| Verification window | Verbatim copy | Single edit | Exact final answers |
| ---: | ---: | ---: | ---: |
| Off | 105.23 | 106.21 | 2/2 |
| 8 | 183.02 | 177.24 | 2/2 |
| 64 | 271.15 | 196.72 | 2/2 |

Use the preceding cooperative prefill environment with `IK2_EXPERT_TILE_LDS=1`. The GGUF and tokenizer are the same ones used for the throughput runs.

```sh
oracle-env/bin/python src/tests/ik2_copy_edit.py --tokenizer ik2-tokenizer/tokenizer >copy.requests
for window in 0 8 64; do
    IK2_SUFFIX=$window ./build-ik2/maya-ik2 --serve-ik2 --model /weight.gguf \
        --ctx 4096 --chunk 512 <copy.requests >"copy-$window.log"
    oracle-env/bin/python src/tests/ik2_copy_edit.py --tokenizer ik2-tokenizer/tokenizer --score "copy-$window.log"
done
```

The direct-answer HTTP template prefills the final content channel. With that same prompt ending, the Python task renames one function in `tests/ik2_compare_decode_test.py` and asks for the complete file. The inventory tasks use the same text as above. The table gives median decode tok/s over three uncached runs. Every final response matched the expected string, and every setting generated the same number of tokens for each task.

| Verification window | Inventory copy, 582 tokens | Inventory edit, 583 tokens | Python rename, 886 tokens |
| ---: | ---: | ---: | ---: |
| Off | 108.40 | 108.36 | 107.83 |
| 64 | 523.42 | 412.18 | 409.91 |
| 128 | 750.65 | 482.59 | 445.84 |

```sh
# Use the preceding cooperative prefill environment with IK2_EXPERT_TILE_LDS=1.
for task in inventory python; do
    for window in 0 64 128; do
        oracle-env/bin/python src/tests/ik2_copy_edit.py --tokenizer ik2-tokenizer/tokenizer \
            --direct-answer --workload "$task" --repeats 3 >"copy-$task.requests"
        IK2_SUFFIX=$window ./build-ik2/maya-ik2 --serve-ik2 --model /weight.gguf \
            --ctx 8192 --chunk 512 <"copy-$task.requests" >"copy-$task-$window.log"
        oracle-env/bin/python src/tests/ik2_copy_edit.py --tokenizer ik2-tokenizer/tokenizer \
            --direct-answer --workload "$task" --repeats 3 --score "copy-$task-$window.log"
    done
done
```

### Exact phrase retrieval at long context

In the run before the host input copy fix, with `--ctx 393216 --chunk 1024`, the resident IQ4_NL engine returned the exact four-word phrase at 32,768 and 131,072 tokens, placed halfway through each uncached prompt. At 392,960 input tokens the engine finished prefill in 470.020 seconds and used all 200 generation tokens without reaching its final answer. That request scored incorrect. All 3,072 experts remained on the card.

```sh
oracle-env/bin/python src/tests/ik2_retrieval.py --tokenizer ik2-tokenizer/tokenizer --data gsm8k-test.jsonl \
    --depths 32768,131072,392960 >retrieval.requests
./build-ik2/maya-ik2 --serve-ik2 --model /weight.gguf --ctx 393216 --chunk 1024 <retrieval.requests >retrieval.log
oracle-env/bin/python src/tests/ik2_retrieval.py --tokenizer ik2-tokenizer/tokenizer --data gsm8k-test.jsonl \
    --depths 32768,131072,392960 --score retrieval.log
```

With the pinned-input fix, Maya again returned the exact midpoint phrase on the two shorter prompts. The same process reserved 393,216 positions and kept all experts on the GPU. Each retrieval started with an empty prefix. A second request extended the prompt by one token and generated 129 tokens with stopping disabled. Its decode time covers the last 128 tokens.

| Input tokens | Expected phrase and actual final answer | Uncached prefill | Tokens before stop | Fixed-length decode |
| ---: | --- | ---: | ---: | ---: |
| 32,768 | falcon saffron cobalt meadow | 14.056 s | 119 | 87.13 tok/s |
| 131,072 | falcon meadow lantern glacier | 78.134 s | 80 | 59.76 tok/s |

These are single runs with token-major E4M3. The longer prompt fully recovered its 131,072-token prefix for the fixed-length request. To reproduce, add `--decode-steps 128` to both Python commands above and use `--depths 32768,131072`. Set `IK2_EXPERT_TILE_LDS=1` and `IK2_SHARED_COMBINE=1` with the cooperative prefill environment.


### CUDA build check

CUDA 12.8 compiled and linked the engine and its device tests for sm_86 with the following commands. The host used for compilation has no NVIDIA GPU.

```sh
cmake -S . -B build-cuda -G Ninja -DSTRATA_ENABLE_CUDA=ON -DCMAKE_CUDA_ARCHITECTURES=86 -DBUILD_TESTING=ON
cmake --build build-cuda --target maya-ik2 ik2_kernels_test -j4
```

### Held-out distribution fixture

Use the first 256 tokenizer IDs from GSM8K test questions and solutions 192 through 199, numbered from zero. The first 224 IDs form the decode prompt and the remaining 32 are forced in order. The arithmetic evaluation above used questions 0 through 191.

```sh
PYTHONPATH=src/tools oracle-env/bin/python - <<'PYDATA'
import json
from pathlib import Path
from strata_tokenizer import Tokenizer
rows = [json.loads(line) for line in Path("gsm8k-test.jsonl").read_text().splitlines()]
text = "\n\n".join(r["question"] + "\n" + r["answer"] for r in rows[192:200])
ids = Tokenizer.from_dir("ik2-tokenizer/tokenizer").encode(text)[:256]
for name, seq in [("all", ids), ("prompt", ids[:224]), ("force", ids[224:])]:
    Path("heldout-teacher." + name).write_text(",".join(map(str, seq)))
PYDATA
OPENBLAS_NUM_THREADS=1 oracle-env/bin/python src/ref/inkling2.py /weight.gguf \
    --tokens "$(cat heldout-teacher.all)" --logprobs-out heldout-teacher-ref.npy
```

### Global E4M3 pages

`IK2_PAGED_KV=1` groups 16 positions per KV head and transposes V inside each page. Sliding attention retains its ring. A 131,072-token midpoint retrieval with the direct page reader returned the exact same four-word phrase as the token-major layout. The 128-step decode measured 65.07 tok/s with pages and 59.76 with token-major KV. Cold prefill took 101.342 and 78.134 seconds respectively. These are single runs with `--ctx 393216 --chunk 1024` and `--decode-steps 128` in the retrieval command above.

The paged prefill agreed with the FP64 reference on 251 of 256 held-out next-token predictions. Mean KL was 0.0022271 across the full vocabulary. Next-token perplexity on the 255 labels was 3.9611 for Maya and 3.9677 for the reference. The device tests cover page boundaries, partial pages, transposed V, causal masking and the stored K/V bytes.

```sh
export IK2_PAGED_KV=1
./build-ik2/ik2_kernels_test
IK2_PREFILL_LOGITS_OUT=teacher-pages.f32 ./build-ik2/maya-ik2 --model /weight.gguf \
    --ctx 1024 --chunk 64 --tokens @heldout-teacher.all --gen 0
oracle-env/bin/python src/tools/ik2_compare_decode.py teacher-pages.f32 --ref heldout-teacher-ref.npy \
    --tokens heldout-teacher.all --vocab 200058 --prefill --prompt-length 256
```

### Fused decode projections

The Radeon preset computes the FP16 attention sink and dynamic convolution taps alongside their Q8 projection. For short prompts this removes 16 launches per generated token. Across 32 forced steps the two paths produced byte-identical full-vocabulary logits. With the installed IQ4_NL GGUF, three 256-step greedy measurements gave the following medians in tok/s.

| Input tokens | Separate projections | Fused projections |
| ---: | ---: | ---: |
| 32 | 111.06 | 117.06 |
| 512 | 109.35 | 115.16 |
| 8,192 | 100.95 | 105.74 |

```sh
# Export the cooperative prefill environment above.
for flag in 0 1; do
    IK2_DECODE_PROJ=$flag python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
        --tokens water.force --gen 257 --depths 32,512,8192 --repeats 3 --chunk 2048
    IK2_DECODE_PROJ=$flag IK2_FORCE=heldout-teacher.force IK2_LOGITS_OUT="projection-$flag.f32" ./build-ik2/maya-ik2 --model /weight.gguf \
        --ctx 1024 --chunk 128 --tokens @heldout-teacher.prompt --gen 33
done
cmp projection-0.f32 projection-1.f32
```


### Direct multiple-choice answers

On 256 questions from eight MMLU subjects, the paged E4M3 engine with the cancellation fix answered **207 correctly**. Every response supplied a valid letter. The first 32 test examples from each subject were selected before running either engine. Each chat requested one letter in the final content channel, with greedy decoding and an eight-token limit. The scorer accepts the letter with optional parentheses or a final period. [Pinned CAIS dataset](https://huggingface.co/datasets/cais/mmlu/tree/c30699e8356da336a370243923dbaf21066bb9fe)

Prepare the small JSONL with PyArrow. This command downloads the eight evaluation splits.

```sh
python3 - <<'PYDATA'
import json
from urllib.request import urlopen
import pyarrow as pa
import pyarrow.parquet as pq
subjects = ("astronomy", "college_mathematics", "conceptual_physics", "high_school_european_history",
            "machine_learning", "philosophy", "professional_psychology", "world_religions")
revision = "c30699e8356da336a370243923dbaf21066bb9fe"
with open("mmlu256.jsonl", "w") as out:
    for subject in subjects:
        url = f"https://huggingface.co/datasets/cais/mmlu/resolve/{revision}/{subject}/test-00000-of-00001.parquet"
        with urlopen(url) as response:
            rows = pq.read_table(pa.BufferReader(response.read())).slice(0, 32).to_pylist()
        for i, r in enumerate(rows):
            print(json.dumps(dict(subject=subject, index=i, question=r["question"], choices=r["choices"], answer=r["answer"])), file=out)
PYDATA
```

Use the cooperative prefill environment with fused projections and paged KV. Both engines receive the same token IDs.

```sh
oracle-env/bin/python src/tests/ik2_multiple_choice.py --data mmlu256.jsonl --tokenizer ik2-tokenizer/tokenizer >mmlu.requests
IK2_DECODE_PROJ=1 IK2_PAGED_KV=1 ./build-ik2/maya-ik2 --model /weight.gguf --serve-ik2 --ctx 4096 --chunk 512 \
    <mmlu.requests >mmlu-maya.log
oracle-env/bin/python src/tests/ik2_multiple_choice.py --data mmlu256.jsonl --tokenizer ik2-tokenizer/tokenizer --score mmlu-maya.log
oracle-env/bin/python src/tests/ik2_multiple_choice.py --data mmlu256.jsonl --tokenizer ik2-tokenizer/tokenizer --llama-url http://127.0.0.1:8085
```

### Attention workgroups and KV layout

The Radeon kernel splits the keys according to the number of query workgroups. Each wave covers 16 queries. Larger decode graphs use additional key splits at 32K, 128K and 256K positions. The paged decode kernel reads global KV directly, while prefill shares each page between waves. Four-wave measurements on the installed IQ4_NL model gave the following medians in tok/s. Each depth had three uncached requests and 128 timed decode steps.

| Input tokens | Token-major prefill | Paged prefill | Token-major decode | Paged decode |
| ---: | ---: | ---: | ---: | ---: |
| 512 | 2,393 | 2,327 | 113.98 | 112.87 |
| 8,192 | 3,660 | 3,674 | 104.58 | 102.48 |
| 32,768 | 3,159 | 3,233 | 91.76 | 93.77 |

The 32K server preset uses token-major KV. The page layout takes the same bytes per token. For the two- and eight-wave boundary cases, the device tests checked results against the CPU attention reference.

```sh
# Use the cooperative prefill environment above.
export IK2_DECODE_PROJ=1 IK2_FLASH_WAVES=4
for page in 0 1; do
    IK2_PAGED_KV=$page python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
        --tokens water.force --depths 512,8192,32768 --gen 129 --repeats 3 --chunk 2048
done
for waves in 2 8; do
    IK2_FLASH_WAVES=$waves IK2_PAGED_KV=1 ./build-ik2/ik2_kernels_test
done
```

### Looped-layer message gates

On a single decode token, 128 threads now read four input channels each for the two BF16 projections. One rank computes the input norms. The next kernel applies the two output projections to 64 channels per workgroup. The device test checks three simultaneous in-place gates against FP64 across all 6,912 outputs.

With `IK2_GATE_TILE=256`, median decode at 512 tokens rose from 113.93 to **117.38 tok/s**. At 8,192 it rose from 105.05 to **107.61 tok/s**. Across 32 forced held-out steps, both variants agreed with the CPU reference on 30 top tokens. Mean full-vocabulary KL changed from 0.002206 to 0.002220. The Radeon preset enables the new gate.

```sh
# Use the cooperative prefill environment above with IK2_DECODE_PROJ=1.
for tile in 64 256; do
    IK2_GATE_TILE=$tile python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
        --tokens water.force --depths 32,512,8192 --gen 257 --repeats 3 --chunk 2048
    IK2_GATE_TILE=$tile IK2_FORCE=heldout-teacher.force IK2_LOGITS_OUT="gate-$tile.f32" \
        ./build-ik2/maya-ik2 --model /weight.gguf --ctx 1024 --chunk 128 --tokens @heldout-teacher.prompt --gen 33
    oracle-env/bin/python src/tools/ik2_compare_decode.py "gate-$tile.f32" --ref heldout-teacher-ref.npy \
        --tokens heldout-teacher.all --vocab 200058 --gen 33 --prompt-length 224
done
IK2_GATE_TILE=256 ./build-ik2/ik2_kernels_test
```

### Repeated requests against a shared prefix

The server has eight snapshots of the sliding KV and convolution state. Previously, changing the end of a request invalidated newer snapshots, but the next save could overwrite the shared prefix while empty slots were available. In a replay with a 256-token prefix the ninth branch recomputed all 256 tokens. The server now chooses an empty slot first. All 12 branches reused the full prefix and the last one produced the same four predictions as a cold replay.

```sh
# Use the Radeon preset environment.
python3 src/tests/ik2_prefix_cache.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
    --tokens water.force --branches 12
```

### Host RAM on the production comparison server

Prometheus gives the `inkling-2-mini` container 10 GiB of host RAM and the same combined RAM-plus-swap limit. The RX 7900 XTX has a separate 24 GB of VRAM. The service image defaults to a prompt cache of up to 8,192 MiB and 32 context checkpoints per slot. A checkpoint at 229,211 tokens occupied 48.670 MiB in the server log. Thirty-two checkpoints of that size would occupy 1.52 GiB. The crash did not retain an allocator-level breakdown.

The deployed Compose command now sets `--cache-ram 0 --ctx-checkpoints 2`. It keeps one slot, 393,216 context positions, Q4_0 K/V and the 10 GiB cgroup limit. This bounds the two host-side caches while preserving prefix reuse in the active slot. The host kernel also charges file-backed pages to the cgroup, so the resident-memory total includes reclaimable file data.

```sh
docker run --rm --memory 1g --memory-swap 1g --cpus 1 \
    hyperion/llama.cpp:server-rocm-inkling-1427-r2 --help 2>&1 | grep -A3 -E -- '--cache-ram|--ctx-checkpoints'
docker inspect -f '{{.HostConfig.Memory}} {{.HostConfig.MemorySwap}} {{json .Config.Cmd}}' inkling-2-mini
docker logs inkling-2-mini 2>&1 | grep 'erasing old context checkpoint' | tail -1
```

### Fused expert activation in prefill

An 8,192-token natural prompt spent 809 ms of GPU work in IQ4_NL expert products and 44 ms in the separate SwiGLU kernel. With `IK2_PREFILL_SWIGLU=1`, a wave computes adjacent gate and up tiles and writes the scaled, capped FP16 activation. The device tests compare each stored half to the separate kernels and to an FP64 calculation for IQ4_NL and Q8_0. Across all 256 positions and 200,058 vocabulary entries of the held-out fixture, the fused and separate paths produced identical FP32 logits.

| Input tokens | Separate prefill, tok/s | Fused prefill, tok/s |
| ---: | ---: | ---: |
| 2,048 | 3,419 | 3,501 |
| 8,192 | 3,674 | 3,732 |
| 32,768 | 3,167 | 3,214 |

The table uses three uncached requests per depth, `--chunk 2048` and the installed IQ4_NL GGUF. At 512 tokens the fused run took 1.660, 0.294 and 0.210 seconds. The separate run took 1.664, 0.213 and 0.213 seconds.

```sh
# Export the cooperative prefill environment, IK2_DECODE_PROJ=1 and IK2_GATE_TILE=256.
export IK2_WMMA_K=32
for flag in 0 1; do
    IK2_PREFILL_SWIGLU=$flag IK2_PREFILL_LOGITS_OUT="swiglu-$flag.f32" ./build-ik2/maya-ik2 \
        --model /weight.gguf --ctx 1024 --chunk 64 --tokens @heldout-teacher.all --gen 0
    IK2_PREFILL_SWIGLU=$flag python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
        --tokens water.force --depths 512,2048,8192,32768 --gen 129 --repeats 3 --chunk 2048
done
cmp swiglu-0.f32 swiglu-1.f32
for k in 32 64; do IK2_WMMA_K=$k IK2_PREFILL_SWIGLU=1 ./build-ik2/ik2_kernels_test; done
```

### Q8 attention output projection

The 3,072-input Q8 projection handles eight output rows per workgroup. It reads the original quantized weights and accumulates each row in the same order. A 37-row device test covers the partially filled last workgroup against FP64. For 32 held-out decode steps, the full-vocabulary FP32 logits matched the prior build byte for byte.

With the loop-gate optimization enabled, the median of three 256-step requests changed from 117.38 to 118.14 tok/s at 512 tokens and from 107.61 to 108.61 tok/s at 8,192 tokens. The command uses the same input IDs on each build.

```sh
# Run on the parent of the projection commit, then on the commit itself.
export IK2_GATE_TILE=256 IK2_Q8_UNROLL=1 IK2_DECODE_PROJ=1
python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
    --tokens water.force --depths 32,512,8192 --gen 257 --repeats 3 --chunk 2048
IK2_Q8_UNROLL=1 ./build-ik2/ik2_kernels_test
IK2_FORCE=heldout-teacher.force IK2_LOGITS_OUT=projection.f32 ./build-ik2/maya-ik2 \
    --model /weight.gguf --ctx 1024 --chunk 128 --tokens @heldout-teacher.prompt --gen 33
sha256sum projection.f32
```

### Quality checks with background text

The multiple-choice runner can place the same archive before each question. It sends one warm-up request and checks how much of the prefix each later request actually reused. With 256 archive tokens, the Radeon engine answered 24 of 32 questions correctly and returned a valid letter every time. Each question reused all 256 tokens. The sample contains indices 0 through 3 from each of the eight subjects in the 256-question fixture above.

A separate inventory task places three shipment records at 5%, 50% and 95% of the input. The prompt asks for the two deliveries minus the damaged boxes. At 32,768 tokens the direct-answer run returned 1,008. The expected answer was 986. This task checks whether the model can retrieve and combine the records.

```sh
# Use the Radeon preset environment and the MMLU and GSM8K data above.
python3 -c 'import json; print("".join(s for s in open("mmlu256.jsonl") if json.loads(s)["index"] < 4), end="")' >mmlu32.jsonl
quiz() {
    oracle-env/bin/python src/tests/ik2_multiple_choice.py --data mmlu32.jsonl \
        --tokenizer ik2-tokenizer/tokenizer --prefix-tokens 256 --background gsm8k-test.jsonl "$@"
}
quiz | ./build-ik2/maya-ik2 --serve-ik2 --model /weight.gguf --ctx 4096 --chunk 512 >quiz.log
quiz --min-cached 256 --score quiz.log
ledger() {
    oracle-env/bin/python src/tests/ik2_retrieval.py --data gsm8k-test.jsonl \
        --tokenizer ik2-tokenizer/tokenizer --depths 32768 --inventory --direct-answer --max-tokens 32 "$@"
}
ledger | IK2_PAGED_KV=1 ./build-ik2/maya-ik2 --serve-ik2 --model /weight.gguf --ctx 393216 --chunk 1024 >ledger.log
ledger --score ledger.log
```

### Larger copy and edit windows

The HIP verifier can check up to 512 proposed inputs per target pass. With a window of at least 32 it computes up to 64 vocabulary rows together for greedy verification. The window is bounded by the prefill chunk size. Plain generation is the default.

| Verification window | Inventory copy, tok/s | Inventory edit, tok/s | Python rename, tok/s |
| ---: | ---: | ---: | ---: |
| 128 | 833.57 | 527.65 | 479.67 |
| 256 | 881.64 | 553.23 | 693.57 |
| 512 | 996.57 | 488.66 | 637.15 |

These are medians of three direct-answer requests per cell. All 27 final answers exactly matched the requested text. On the longer, 27-line inventory, the client sent STOP after reading 750 tokens. The server ended the verified batch after emitting 1,164 and reused 2,939 tokens on the next request. Its eight predictions after resumption matched a cold replay.

```sh
# Use the preceding copy/edit commands with a 512-token chunk and windows 128, 256 and 512.
for task in inventory python; do
    oracle-env/bin/python src/tests/ik2_copy_edit.py --tokenizer ik2-tokenizer/tokenizer \
        --direct-answer --workload "$task" --repeats 3 >"$task.requests"
    for w in 128 256 512; do
        IK2_SUFFIX=$w ./build-ik2/maya-ik2 --serve-ik2 --model /weight.gguf --ctx 8192 \
            --chunk 512 <"$task.requests" >"$task-$w.log"
        oracle-env/bin/python src/tests/ik2_copy_edit.py --tokenizer ik2-tokenizer/tokenizer \
            --direct-answer --workload "$task" --repeats 3 --score "$task-$w.log"
    done
done
oracle-env/bin/python src/tests/ik2_copy_edit.py --tokenizer ik2-tokenizer/tokenizer \
    --direct-answer --inventory-lines 27 --max-tokens 2048 >long-copy.requests
IK2_SUFFIX=512 python3 src/tests/ik2_prefix_cache.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
    --tokens water.force --chunk 512 --cancel-after 750 --cancel-request long-copy.requests
```

### Precision of split E4M3 attention

Two keys can have exactly equal dot products and opposite values. In a gfx1100 regression test, permuting the entries of one key changed its FP32 reduction order. The scalar and resident kernels returned −0.0000153 and −0.0000305 on a channel whose exact result is zero. The scalar E4M3 path now forms the score in FP64. The eight-lane resident kernel refines a score when its magnitude reaches 16, where FP32 spacing reaches 2⁻¹⁹. Both kernels returned zero on all 128 channels of the device regression. The separate 16,640-key stress case also passed at its original tolerance.

On 32 forced held-out steps, mean full-vocabulary KL was 0.002114 and top-1 agreement was 30/32. Before the score change the figures were 0.002220 and 30/32. With Q8 output projection enabled, median decode after 8,192 tokens was 108.11 tok/s on the corrected build and 108.61 on the prior build. These runs used the short-context reservation and three 256-step repetitions. The corrected speed run used a prefill K tile of 64. Repeating the forced-logit check with a tile of 32 produced the same FP32 file.

```sh
# Export the Radeon preset environment, including IK2_DECODE_SUBWAVE=1 and IK2_GATE_TILE=256.
./build-ik2/ik2_kernels_test
IK2_FORCE=heldout-teacher.force IK2_LOGITS_OUT=attention.f32 ./build-ik2/maya-ik2 \
    --model /weight.gguf --ctx 1024 --chunk 128 --tokens @heldout-teacher.prompt --gen 33
oracle-env/bin/python src/tools/ik2_compare_decode.py attention.f32 --ref heldout-teacher-ref.npy \
    --tokens heldout-teacher.all --vocab 200058 --gen 33 --prompt-length 224
IK2_WMMA_K=64 python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
    --tokens water.force --depths 32,512,8192 --gen 257 --repeats 3 --chunk 2048
```

### Dense WMMA tile depth

The gfx1100 test suite exercises the 3,072-input by 2,304-output attention projection with 513 tokens. It checks the FP32 output against FP64 at the first and last channels and around tile boundaries. A 64-channel K tile accelerated this projection in isolation. Selecting it only for this shape gave mixed results on uncached natural prompts. The preset keeps the 32-channel depth.

```sh
for k in 32 64; do IK2_WMMA_K=$k ./build-ik2/ik2_kernels_test; done
```

### Short prompt tiles and inventory control

The Radeon preset chooses the MoE tile height from the number of routed rows per expert. On this model it uses 16 rows through 128 input tokens, 32 rows through 256, and 64 rows above that. The 64-row case uses fused SwiGLU. Median prefill time for 16, 32, 64 and 128 new tokens fell from 63, 88, 101 and 116 ms to 57, 80, 91 and 108 ms. At 256, 512, 1,024 and 2,048 tokens both settings measured 141, 180, 290 and 513 ms. At chunk sizes 64 and 256, all 256 held-out prefill rows matched the fixed-height run byte for byte across the full vocabulary.

```sh
# Use the Radeon preset and the GSM8K test file and tokenizer from the retrieval check.
oracle-env/bin/python src/tests/ik2_retrieval.py --tokenizer ik2-tokenizer/tokenizer --data gsm8k-test.jsonl \
    --depths 32768 >needle-32768.request
python3 -c 'from pathlib import Path; ids=Path("needle-32768.request").read_text().splitlines()[0].split()[-1].split(","); Path("profile-pp-ids").write_text(",".join(ids[:8192]))'
for rows in auto 64; do
    IK2_WMMA_K=32 IK2_PREFILL_TILE=$rows python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 \
        --model /weight.gguf --tokens profile-pp-ids --depths 16,32,64,128,256,512,1024,2048 --chunk 2048 --gen 3 --repeats 5
    for chunk in 64 256; do
        IK2_WMMA_K=32 IK2_PREFILL_TILE=$rows IK2_PREFILL_LOGITS_OUT="tile-$chunk-$rows.f32" ./build-ik2/maya-ik2 \
            --model /weight.gguf --ctx 1024 --chunk "$chunk" --tokens @heldout-teacher.all --gen 0
    done
done
for chunk in 64 256; do cmp "tile-$chunk-auto.f32" "tile-$chunk-64.f32"; done
```

On the three-record inventory, the direct-answer prompt returned the correct 986 at 256 and 2,048 input tokens and 1,008 at 32,768. Allowing the model to reason before answering produced 986 at all three depths. The model used 355, 169 and 488 generated tokens respectively. Switching all KV entries to FP16 did not recover the 32,768-token direct answer. Run the ledger command above with `--depths 256,2048,32768`, then remove `--direct-answer` and use `--max-tokens 512` for the reasoning control.

### NVFP4 expert WMMA on RDNA3

The Radeon kernel unpacks GGUF NVFP4 in registers. Each 36-byte block holds 64 E2M1 weights and four unsigned E4M3 scales. Batched prefill can group resident experts and apply the same capped SwiGLU as the IQ4_NL path. For single-token NVFP4 decode, `IK2_EXPERT_MMA=1` enables a WMMA gate/up product with a two-half representation of the FP32 input and a down projection over the FP16 hidden state. The default for single-token NVFP4 on HIP remains scalar.

In the first grouped implementation, adding each matrix tile directly to its WMMA accumulator gave −2.19387 for an FP64 reference of −2.19205. One output of 4,512 exceeded its test tolerance. Adding independently rounded 16-channel dot products in FP32 passed the original test. The complete Radeon suite passed with register and LDS tiles, including tile heights 16, 32 and 64 and both 32- and 64-channel K depths. The single-token test compares all 640 gate/up outputs and 1,280 down outputs to independent FP64 evaluations of the packed bytes. These NVFP4 results use generated expert weights. All full-model performance and quality results for this port use the installed IQ4_NL checkpoint.

```sh
for k in 32 64; do
    IK2_EXPERT_MMA=1 IK2_WMMA_K=$k IK2_EXPERT_TILE_LDS=1 ./build-ik2/ik2_kernels_test
done
IK2_EXPERT_MMA=1 IK2_WMMA_K=32 IK2_EXPERT_TILE_LDS=0 ./build-ik2/ik2_kernels_test
```

### Resident prefill workspace and projection cache

The grouped expert gate/up and down projections use one FP32 workspace in sequence. SwiGLU places its FP16 output in a separate buffer. With an 8,192-token chunk this saves 576 MiB of expert scratch, enough for the engine to keep all 56 dense attention projections in FP16 under the existing VRAM headroom setting. On 8,192 and 32,768 input tokens, median prefill fell from 2.175 and 9.756 seconds to 2.078 and 9.612 seconds respectively. At 512 tokens it went from 0.215 to 0.221 seconds. All 256 rows of held-out prefill logits matched the previous allocation byte for byte, both with a 64-token chunk and with a 256-token chunk.

The Radeon preset uses `IK2_PREFILL_DENSE=2` to keep just the 24 fused QKV projections in FP16 (868 MiB). Q8 projections go directly through wave32 WMMA on batches of 32 through 511 tokens. At a chunk size of 2,048, the selective cache took 0.151, 2.214 and 10.333 seconds for 256, 8,192 and 32,768 input tokens. The 56-matrix cache took 0.160, 2.219 and 10.368 seconds. The uncached, selective and full caches produced identical full-vocabulary logits on the 256-token passage with chunk size 128. All times are medians of three requests.

```sh
# Use the preset environment and profile-pp-ids from the short-prompt test above.
# Compare the workspace on commit 3b2e4ac and the compact build using the same 56-matrix cache.
IK2_PREFILL_DENSE=1 python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
    --tokens profile-pp-ids --depths 512,8192,32768 --chunk 8192 --gen 3 --repeats 3

for cache in 1 0 2; do
    IK2_PREFILL_DENSE=$cache python3 src/tools/ik2_bench.py --engine ./build-ik2/maya-ik2 --model /weight.gguf \
        --tokens profile-pp-ids --depths 64,256,512,2048,8192,32768 --chunk 2048 --gen 3 --repeats 3
    IK2_PREFILL_DENSE=$cache IK2_PREFILL_LOGITS_OUT="cache-$cache.f32" ./build-ik2/maya-ik2 \
        --model /weight.gguf --ctx 1024 --chunk 128 --tokens @heldout-teacher.all --gen 0
done
cmp cache-0.f32 cache-1.f32 && cmp cache-1.f32 cache-2.f32
```

### Alternating prompt lengths

The preset keeps the two most recently used decode graphs for each output mode. When successive requests cross an attention depth boundary, the engine can replay the previous graph. On 24 alternating uncached prompts of 128 and 2,048 tokens, both configurations returned the same 120 greedy token IDs. After the first two requests of each length, the median time for four decode steps fell from 38 to 34 ms at 128 tokens and from 40 to 36 ms at 2,048. The second graph used another 2 MiB of reported device memory at a context reservation of 393,216. A separate run visited depths 128, 2,048 and 8,192 in the order A, B, C, A, C, B, first greedy and then with temperature 0.7, top-p 0.95 and seed 1. All 60 generated token IDs matched with the cache enabled and disabled. Expert placement in managed or host memory retains its existing graph lifecycle.

```sh
python3 - <<'PY'
from pathlib import Path
ids = Path("profile-pp-ids").read_text().strip().split(",")
with Path("mixed.requests").open("w") as out:
    for r in range(12):
        for j, n in enumerate((128, 2048)):
            prompt = [str(1000 + 2 * r + j)] + ids[r:r+n-1]
            print("GEN 5 0 1 1 -", ",".join(prompt), file=out)
    print("QUIT", file=out)
PY
for pair in 0 1; do
    IK2_GRAPH_PAIR=$pair ./build-ik2/maya-ik2 --model /weight.gguf --serve-ik2 --ctx 393216 --chunk 1024 \
        <mixed.requests >"mixed-$pair.log"
    grep '^@TOK ' "mixed-$pair.log" >"mixed-$pair.ids"
done
cmp mixed-0.ids mixed-1.ids
python3 - <<'PY'
from statistics import median
for pair in (0, 1):
    rows = [l.split() for l in open(f"mixed-{pair}.log") if l.startswith("@END ")][4:]
    for n in (128, 2048):
        times = [float(r[5]) for r in rows if int(r[2]) == n]
        assert len(times) == 10
        print(pair, n, 1000 * median(times), "ms")
PY
```

For single-token WMMA, the E2M1 lookup now constructs the four FP16 values directly from their high bytes. Sixteen waves split the gate/up dot products, and four split each down projection. At model width, all 1,152 activated gate/up outputs and 2,304 down outputs passed the FP64 check with the same tolerances as the smaller case. The device suite passed at both K depths and with register tiles.

With eight of 128 generated experts active and width 2,304, the median gate/up kernel time fell from 60.455 to 42.411 microseconds. The down projection went from 25.720 to 27.730 microseconds. Scalar gate/up and down measured 37.715 and 23.225 microseconds on the new binary. These are device-event times across 96 expert selections per repetition. They exclude routing and input conversion. The scalar default still wins for single-token NVFP4 on this fixture.

```sh
# From /work in the CPU build container. For the earlier comparison, link the same tool against the previous library.
hipcc --offload-arch=gfx1100 -O3 -DSTRATA_USE_HIP=1 -Isrc/include/strata/hip_compat -Isrc/src/ik2 \
    src/tools/ik2_nv_bench.hip -x none build-ik2/libik2_kernels.a -o ik2-nv-bench
# In the GPU container.
IK2_EXPERT_MMA=1 ./ik2-nv-bench
```
