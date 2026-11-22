# Inkling 2 Mini on gfx1100

The port runs on Prometheus's RX 7900 XTX. Build instructions and the original scalar port are below. The [RDNA3 results](#rdna3-expert-and-attention-kernels) give the measured configuration and uncached benchmark command.

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
| 32 | 415.58 | 110.54 | 247.14 | 64.14 |
| 512 | 2,612.24 | 109.12 | 1,304.86 | 63.96 |
| 2,048 | 3,205.01 | 107.02 | 1,511.03 | 62.89 |
| 8,192 | 3,126.72 | 100.31 | 1,488.44 | 61.97 |
| 32,768 | 2,652.21 | 87.07 | | |

The 8,192- and 32,768-token rows below the single-chunk range were collected before the host input copy fix. A fast resident prefill could overwrite a pinned embedding or position buffer while the GPU had queued its copy. The 1,792-token prefix regression exposed the corruption. The fixed build waits for a transfer event before releasing those buffers and predicts identical continuations after a warm or cold replay. Reproduce with `python3 src/tests/ik2_prefix_cache.py --engine ./build-ik2/maya-ik2 --model /weight.gguf --tokens water.force`.

Medians in tok/s for three uncached requests per depth. Each request generates 129 tokens with 128 timed decode steps. Maya reserved 32,897 positions with FP8 KV. The service retained its deployed 393,216 positions with Q4_0 KV. On the first 32-token request the initial matrix-library setup took the prefill rate down to 17.5 tok/s. The other two measured 415.58 tok/s.

```sh
export IK2_HOST=mmap IK2_KV=fp8 IK2_GATE_BF16=1 IK2_GATE_TILE=64 IK2_Q8_UNROLL=1
export IK2_BATCH_PREFILL=1 IK2_PREFILL_WMMA=1 IK2_PREFILL_TILE=64 IK2_EXPERT_TILE_LDS=0
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
