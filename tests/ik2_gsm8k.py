"""Prepare greedy GSM8K requests or score the final answers from maya-ik2's server protocol."""
import argparse
from decimal import Decimal
import json
from pathlib import Path
import re
import sys
from urllib.request import Request, urlopen

from jinja2 import Environment

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import Tokenizer

NUMBER = re.compile(r"-?\d[\d,]*(?:\.\d+)?")
END = "<|content_model_end_sampling|>"
TEXT = "<|message_model|><|content_text|>"


def score(reason, tokens, tokenizer, sample):
    if reason not in ("stop", "length"):
        raise ValueError(f"unexpected finish reason: {reason}")
    full = tokenizer.decode(tokens)
    final = full.rsplit(TEXT, 1)[1].split("<|end_message|>")[0] if TEXT in full else ""
    numbers = NUMBER.findall(final)
    predicted = Decimal(numbers[-1].replace(",", "")) if numbers else None
    expected = Decimal(sample["answer"].rsplit("####", 1)[1].strip().replace(",", ""))
    return dict(correct=predicted == expected, expected=str(expected), predicted=str(predicted) if numbers else None,
                finish=reason, generated=len(tokens))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data", required=True, type=Path, help="OpenAI GSM8K test.jsonl")
    parser.add_argument("--count", type=int, default=96, help="take the first N questions in file order")
    parser.add_argument("--max-tokens", type=int, default=1024)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--score", type=Path, help="server stdout from the prepared inputs")
    mode.add_argument("--llama-url", help="score the same tokenized inputs against a running llama.cpp service")
    args = parser.parse_args()
    with args.data.open() as f:
        samples = [json.loads(line) for line in f][:args.count]
    if len(samples) != args.count or args.count < 1:
        parser.error("invalid question count")
    tok = Tokenizer.from_gguf(args.model)
    if args.score:
        tokens, results = [], []
        with args.score.open() as f:
            for line in f:
                if line.startswith("@TOK "):
                    tokens.append(int(line.split()[1]))
                elif line.startswith("@END "):
                    result = score(line.split()[1], tokens, tok, samples[len(results)])
                    print(json.dumps(dict(question=len(results), **result)))
                    results.append(result)
                    tokens = []
        if len(results) != len(samples) or tokens:
            raise ValueError(f"incomplete run: {len(results)} of {len(samples)} responses")
        print(json.dumps(dict(correct=sum(r["correct"] for r in results), total=len(results),
                              truncated=sum(r["finish"] == "length" for r in results))))
        return
    template = Environment().from_string((ROOT / "tools/ik2/chat_template.jinja").read_text())
    stop, = tok.encode(END, parse_special=True)
    results = []
    for index, sample in enumerate(samples):
        question = sample["question"] + "\nEnd your answer with #### followed by the number."
        prompt = template.render(messages=[{"role": "user", "content": question}], add_generation_prompt=True,
                                 tools=[], reasoning_effort="none")
        ids = tok.encode(prompt, parse_special=True)
        if args.llama_url:
            payload = dict(prompt=ids, n_predict=args.max_tokens, temperature=0, seed=1, return_tokens=True,
                           cache_prompt=False, stream=False)
            req = Request(args.llama_url.rstrip("/") + "/completion", data=json.dumps(payload).encode(),
                          headers={"Content-Type": "application/json"})
            with urlopen(req, timeout=300) as response:
                completion = json.load(response)
            reason = "length" if completion["stop_type"] == "limit" else "stop"
            if not completion["tokens"]:
                raise ValueError("llama.cpp returned no tokens")
            result = score(reason, completion["tokens"], tok, sample)
            print(json.dumps(dict(question=index, **result)), flush=True)
            results.append(result)
        else:
            print(f"GEN {args.max_tokens} 0 1 1 {stop} " + ",".join(map(str, ids)))
    if args.llama_url:
        print(json.dumps(dict(correct=sum(r["correct"] for r in results), total=len(results),
                              truncated=sum(r["finish"] == "length" for r in results))))
    else:
        print("QUIT")


if __name__ == "__main__":
    main()
