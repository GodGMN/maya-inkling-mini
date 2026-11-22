FROM ghcr.io/ggml-org/llama.cpp:server-rocm-b11176
RUN apt-get update -qq && apt-get install -y --no-install-recommends \
    python3-jinja2 python3-jsonschema python3-numpy python3-pil python3-regex python3-requests python3-yaml \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/maya
COPY LICENSE VERSION ./
COPY maya-ik2 build-ik2/maya-ik2
COPY serve/ serve/
COPY tools/ tools/
COPY tokenizer/ ik2/tokenizer/
COPY vision/ vision/
COPY ik2-mini-gfx1100*.json ./
ENTRYPOINT ["python3", "serve/server.py", "--engine", "maya"]
CMD ["--config", "ik2-mini-gfx1100.json", "--host", "0.0.0.0", "--port", "8080"]
