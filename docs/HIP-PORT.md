# Inkling 2 Mini on gfx1100

Phase 1 targets the RX 7900 XTX on Prometheus. The HIP executable uses the resident expert path, scalar NVFP4 and Q8_0 expert products, scalar decode attention, and hipBLAS for prompt GEMMs. Use the text-only configuration as the starting point for validation, with a short context first. The CUDA configuration's maximum context and speed have not been measured on AMD.

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

## CUDA and HIP differences

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

After GPU release, run `ctest --test-dir build-ik2 -R '^ik2_kernels_test$' --output-on-failure` with the GPU available. The suite contains scalar expert, routing, FP8 and FP16 attention checks against numerical references. Full-model validation also requires an approved storage location for the unsloth GGUF. Compare logits on identical forced tokens, generated text, and graph-on/off output before timing prefill and decode at different prompt depths.

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

`IK2_EXPERT_LIMB=60` uses signed integer dot products for IQ4_NL and Q8_0 experts during decode. Each group of sixteen activations shares a scale and uses two signed bytes per value. `IK2_BATCH_PREFILL=1 IK2_PREFILL_WMMA=1` groups routed tokens for native 16×16×16 FP16 matrix instructions with FP32 accumulation. `IK2_ATTN_MMA=1` enables WMMA decode attention on every visit, and `2` selects full-attention visits only. `IK2_FLASH=1` enables WMMA prefill attention. Both attention kernels accept FP16 and software-decoded E4M3 KV, including ring buffers, relative bias and the sink. The CUDA block verifier and drafter server path still require NVIDIA kernels.

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
