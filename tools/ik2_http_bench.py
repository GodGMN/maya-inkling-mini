"""Measure client-visible latency on a local OpenAI-compatible streaming chat endpoint."""
import argparse
import json
from time import perf_counter
from urllib.request import Request, urlopen


PROMPTS = {
    "islands": "Explain in at least 250 words how a volcanic island forms, from the first eruption to an established forest.",
    "printing": "Describe in at least 250 words how a newspaper was printed and distributed in the early nineteenth century.",
    "bridges": "Explain in at least 250 words how a suspension bridge carries the weight of vehicles from its deck to the ground.",
}


def stream(url, model, prompt, tokens):
    body = dict(model=model, messages=[dict(role="user", content=prompt)], temperature=0, max_tokens=tokens,
                reasoning_effort="none", stream=True, stream_options=dict(include_usage=True))
    request = Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
    start = perf_counter()
    first_delta = usage = finish = None
    with urlopen(request, timeout=300) as response:
        for line in response:
            if not line.startswith(b"data:"):
                continue
            event = line[5:].strip()
            if event == b"[DONE]":
                break
            chunk = json.loads(event)
            usage = chunk.get("usage") or usage
            for choice in chunk.get("choices", []):
                delta = choice.get("delta", {})
                if first_delta is None and (delta.get("content") or delta.get("reasoning_content")):
                    first_delta = perf_counter()
                finish = choice.get("finish_reason") or finish
        else:
            raise ValueError("stream ended before [DONE]")
    elapsed = perf_counter() - start
    if first_delta is None or not usage or finish not in ("length", "stop"):
        raise ValueError(f"incomplete chat response: finish={finish!r}, usage={usage!r}")
    count = usage["completion_tokens"]
    if not 0 < count <= tokens:
        raise ValueError(f"invalid completion token count {count} for limit {tokens}")
    return dict(prompt_tokens=usage["prompt_tokens"], generated=count, finish=finish,
                first_text_ms=round(1000 * (first_delta - start), 2), elapsed_s=round(elapsed, 5),
                request_tok_s=round(count / elapsed, 2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="server base URL including port")
    parser.add_argument("--model", help="model ID, inferred for servers listing one model")
    parser.add_argument("--max-tokens", type=int, default=129)
    parser.add_argument("--repeats", type=int, default=3)
    args = parser.parse_args()
    if args.max_tokens < 1 or args.repeats < 1:
        parser.error("a positive token limit and repeat count are required")
    url = args.url.rstrip("/")
    model = args.model
    if model is None:
        with urlopen(url + "/v1/models", timeout=10) as response:
            models = json.load(response)["data"]
        if len(models) != 1:
            parser.error("pass --model to select from the models the server lists")
        model = models[0]["id"]
    for repeat in range(1, args.repeats + 1):
        for name, prompt in PROMPTS.items():
            print(json.dumps(dict(case=name, repeat=repeat, **stream(url, model, prompt, args.max_tokens))), flush=True)


if __name__ == "__main__":
    main()
