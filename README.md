<img src="assets/tintero.png" alt="" width="96">

# Tintero

Tintero runs Thinking Machines' Inkling-2-Mini on a single RX 7900 XTX. The 24 GB card holds the full IQ4_NL model and serves text and images. With an FP8 cache, three chats can each use 131,072 tokens of context.

[Runtime settings](docs/README.md#settings) · [Benchmarks and tests](docs/README.md#tests-and-measurements)

| GPU | Backend | Tintero support |
| --- | --- | --- |
| AMD RX 7900 XTX (`gfx1100`) | HIP | [Tested on hardware](docs/HIP-PORT.md#full-model-measurements-at-393216-reserved-tokens) |
| NVIDIA RTX 3090 (`sm_86`) | CUDA | [Earlier revision compiled](docs/HIP-PORT.md#cuda-build-check). Current runtime unverified. |

## Quick start on RX 7900 XTX

For a source install, [build the HIP executable](docs/HIP-PORT.md#build-on-prometheus-without-a-gpu) and [start the server](docs/HIP-PORT.md#serve-the-iq4_nl-model-on-gfx1100).

If you have the r6 Docker image loaded locally, start a text-only server on Linux:

```sh
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir models
docker run -d --name inkling-2-mini --device /dev/kfd --device /dev/dri \
  -v "$PWD/models:/opt/maya/models:ro" -e IK2_TEXT_ONLY=1 \
  -p 127.0.0.1:8085:8080 hyperion/tintero:gfx1100-service-r6
```

Check the model ID once the server is ready:

```sh
curl -fsS http://127.0.0.1:8085/v1/models
```

Use `inkling-2-mini` for requests to `/v1/chat/completions`.

## Credits

This fork builds on [alesha-pro/maya-inkling-mini](https://github.com/alesha-pro/maya-inkling-mini), the Inkling engine from [Project Maya](https://github.com/mw00/project-maya). Maya grew out of [Strata](https://github.com/Niko1221/Strata). Thinking Machines released [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini). The engine code is [MIT-licensed](LICENSE).
