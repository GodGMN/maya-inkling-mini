# Contributing to Tintero

Base pull requests on `main` and keep each change to one topic. For a bug report, include the commit, GPU, driver and toolchain versions, model quant, preset, and the engine log around the failure.

For performance changes, compare the same prompt, cache profile and output length before and after the change. Report prefill and decode separately. The [HIP guide](docs/HIP-PORT.md) describes the test conditions for the RX 7900 XTX.

Run the local checks from the repository root.

```sh
cmake -S . -B build-check -DSTRATA_ENABLE_HIP=OFF -DSTRATA_ENABLE_CUDA=OFF
python tools/check_hip_compat.py
python -m unittest discover -s serve -t .
python -m unittest discover -s tools
python tests/ik2_compare_decode_test.py
node --test tools/test_*.mjs
```

For kernel changes, build the engine and numerical tests for the affected backends. Include the GPU architecture, kernel test result and host-reference result in the pull request. [HIP verification](docs/HIP-PORT.md#verification) · [CUDA build](docs/INKLING2-MINI.md)
