"""Prepare or score exact-length Inkling chat requests with a phrase at a known position."""
import argparse
import json
from pathlib import Path
import random
import sys
from urllib.request import Request, urlopen

from jinja2 import Environment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from needle_bench import WORDS
from strata_tokenizer import Tokenizer

FINAL = "<|message_model|><|content_text|>"
STOP = "<|content_model_end_sampling|>"


def requests(tok, depths, percent, corpus, direct_answer=False):
    template = Environment().from_string((ROOT / "tools/ik2/chat_template.jinja").read_text())
    content = ("Read these archive notes.\n@FILL@\n"
               "What is the archive verification phrase? Reply with the exact phrase.")
    prefix, suffix = template.render(messages=[{"role": "user", "content": content}], tools=[],
                                     add_generation_prompt=True, reasoning_effort="none").split("@FILL@")
    left = tok.encode(prefix, parse_special=True)
    right = tok.encode(suffix + (FINAL if direct_answer else ""), parse_special=True)
    filler = tok.encode(corpus)
    if not filler:
        raise ValueError("empty distractor corpus")
    for depth in depths:
        seed = depth if percent == 50 else depth * 101 + percent
        expected = " ".join(random.Random(seed).sample(WORDS, 4))
        if expected in corpus.lower():
            raise ValueError("generated phrase already appears in the distractors")
        marker = tok.encode("\nArchive register: the archive verification phrase is " + expected + ".\n")
        n = depth - len(left) - len(right) - len(marker)
        if n < 0:
            raise ValueError(f"{depth} tokens cannot hold the question and phrase")
        notes = (filler * ((n + len(filler) - 1) // len(filler)))[:n]
        at = n * percent // 100
        yield expected, left + notes[:at] + marker + notes[at:] + right


def responses(log):
    ids = []
    for line in log:
        if line.startswith("@TOK "):
            ids.append(int(line.split()[1]))
        elif line.startswith("@END "):
            _, reason, prompt, cached, pp, tg = line.split()
            yield ids, int(prompt), int(cached), reason, float(pp), float(tg)
            ids = []
    if ids:
        raise ValueError("truncated engine response")


def llama_completion(url, prompt, limit, fixed_length=False):
    body = dict(prompt=prompt, n_predict=limit, temperature=0, seed=1, return_tokens=True, stream=False,
                cache_prompt=fixed_length, ignore_eos=fixed_length)
    req = Request(url.rstrip("/") + "/completion", data=json.dumps(body).encode(),
                  headers={"Content-Type": "application/json"})
    with urlopen(req, timeout=7200) as f:
        result = json.load(f)
    timings = result["timings"]
    cached = timings.get("cache_n", 0)
    return (result["tokens"], timings["prompt_n"] + cached, cached, result["stop_type"],
            timings["prompt_ms"] / 1000, timings["predicted_ms"] / 1000)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--data", required=True, type=Path, help="GSM8K JSONL distractor questions")
    ap.add_argument("--depths", default="32768,131072,392960")
    ap.add_argument("--percent", type=int, default=50)
    ap.add_argument("--max-tokens", type=int, default=200)
    ap.add_argument("--direct-answer", action="store_true",
                    help="prefill the final text channel as serve/server.py does for explicit reasoning_effort=none")
    ap.add_argument("--decode-steps", type=int, default=0,
                    help="also time this many decode steps after extending the same prompt by one token")
    mode = ap.add_mutually_exclusive_group()
    mode.add_argument("--score", type=Path, help="stdout of maya-ik2 --serve-ik2")
    mode.add_argument("--llama-url", help="run the same token IDs through llama.cpp /completion")
    args = ap.parse_args()
    depths = [int(x) for x in args.depths.split(",")]
    if not 0 <= args.percent <= 100 or args.max_tokens < 1 or args.decode_steps < 0:
        ap.error("percent must be between 0 and 100 with positive max-tokens and nonnegative decode-steps")
    tok = Tokenizer.from_dir(args.tokenizer)
    stop, = tok.encode(STOP, parse_special=True)
    corpus = "\n".join(json.loads(line)["question"] for line in args.data.read_text().splitlines())
    recorded = []
    if args.score:
        with args.score.open() as f:
            recorded = list(responses(f))
        count = len(depths) * (2 if args.decode_steps else 1)
        if len(recorded) != count:
            raise ValueError(f"expected {count} complete responses, got {len(recorded)}")
    for index, (expected, prompt) in enumerate(requests(tok, depths, args.percent, corpus, args.direct_answer)):
        if not args.score and not args.llama_url:
            payload = ",".join(map(str, prompt))
            print(f"GEN {args.max_tokens} 0 1 1 {stop} " + payload)
            if args.decode_steps:
                print(f"GEN {args.decode_steps + 1} 0 1 1 - " + payload + ",371")
            continue
        stride = 2 if args.decode_steps else 1
        result = llama_completion(args.llama_url, prompt, args.max_tokens) if args.llama_url else recorded[index * stride]
        ids, seen, cached, reason, pp, tg = result
        extra = {}
        if args.decode_steps:
            steps = args.decode_steps
            replay = (llama_completion(args.llama_url, prompt + [371], steps + 1, True) if args.llama_url
                      else recorded[index * stride + 1])
            output, replay_seen, reused, _, _, seconds = replay
            if len(output) != steps + 1 or replay_seen != len(prompt) + 1 or seconds <= 0:
                raise ValueError(f"incomplete fixed-length replay at {len(prompt)} tokens")
            extra = dict(replay_steps=steps, replay_cached=reused, replay_decode_s=seconds,
                         replay_decode_tok_s=round(steps / seconds, 2))
        if seen != len(prompt):
            raise ValueError(f"engine processed {seen} out of {len(prompt)} prompt tokens")
        text = tok.decode(ids)
        if args.direct_answer:
            text = FINAL + text
        answer = text.rsplit(FINAL, 1)[1].split("<|end_message|>")[0].strip() if FINAL in text else ""
        print(json.dumps(dict(prompt=seen, percent=args.percent, direct_answer=args.direct_answer, expected=expected, content=answer,
                              found=expected in answer.lower(), exact=answer.lower().rstrip(".") == expected,
                              cached=cached, finish=reason, generated=len(ids), prefill_s=pp, decode_s=tg, **extra)), flush=True)
    if not args.score and not args.llama_url:
        print("QUIT")


if __name__ == "__main__":
    main()
