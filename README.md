# Tintero

Inference engine for Thinking Machines' Inkling-2-Mini. The IQ4_NL model fits in 24 GB of VRAM on an RX 7900 XTX.

| GPU | Backend | Status in this fork |
| --- | --- | --- |
| AMD RX 7900 XTX (`gfx1100`) | HIP | [Tested on hardware](docs/HIP-PORT.md#full-model-measurements-at-393216-reserved-tokens) |
| NVIDIA RTX 3090 (`sm_86`) | CUDA | [Earlier revision compiled](docs/HIP-PORT.md#cuda-build-check). Current runtime unverified. |

## Quick start on RX 7900 XTX

On Linux with Docker, the Hugging Face CLI (`hf`), and the packaged r6 image loaded locally, run:

```sh
hf download ggml-org/Inkling-2-Mini-GGUF Inkling-2-Mini-IQ4_NL.gguf --local-dir models
docker run -d --name inkling-2-mini --device /dev/kfd --device /dev/dri \
  -v "$PWD/models:/opt/maya/models:ro" -e IK2_TEXT_ONLY=1 \
  -p 127.0.0.1:8085:8080 hyperion/tintero:gfx1100-service-r6
```

Once loaded, the server lists `inkling-2-mini` at `http://127.0.0.1:8085/v1/models`. The source build produces the `tintero` executable. For a source install, follow the [HIP build](docs/HIP-PORT.md#build-on-prometheus-without-a-gpu) and [server setup](docs/HIP-PORT.md#serve-the-iq4_nl-model-on-gfx1100).

[Inkling settings, benchmarks, and tests](docs/README.md)

## Credits

This fork builds on [alesha-pro/maya-inkling-mini](https://github.com/alesha-pro/maya-inkling-mini), the Inkling engine from [Project Maya](https://github.com/mw00/project-maya). Maya grew out of [Strata](https://github.com/Niko1221/Strata). Thinking Machines released [Inkling-2-Mini](https://huggingface.co/thinkingmachines/Inkling-2-Mini). The engine code is [MIT-licensed](LICENSE).
