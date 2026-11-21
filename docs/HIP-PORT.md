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
