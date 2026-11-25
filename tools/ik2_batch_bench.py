"""Measure aggregate and per-request HTTP generation rates for simultaneous short prompts."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
from statistics import median
from threading import Barrier
from time import perf_counter
from urllib.request import Request, urlopen


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", default="inkling-2-mini")
    parser.add_argument("--tokens", type=int, default=512)
    parser.add_argument("--concurrency", type=int, nargs="+", default=[1, 4, 8, 16, 32])
    args = parser.parse_args()

    def request(i, n, barrier):
        prompt = (f"Lesson {n}.{i} in a series for a school class. Explain how a city supplies "
                  "drinking water. Describe each stage and give several examples. Write at least 2000 words.")
        body = dict(model=args.model, messages=[dict(role="user", content=prompt)], temperature=0,
                    reasoning_effort="medium", max_tokens=args.tokens)
        req = Request(args.url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                      {"Content-Type": "application/json"})
        barrier.wait()
        start = perf_counter()
        with urlopen(req, timeout=900) as response:
            result = json.load(response)
        end = perf_counter()
        count = result["usage"]["completion_tokens"]
        if result["choices"][0]["finish_reason"] != "length" or count != args.tokens:
            raise RuntimeError(f"request {i} finished after {count} tokens")
        return start, end, count

    for n in args.concurrency:
        barrier = Barrier(n)
        with ThreadPoolExecutor(n) as executor:
            results = list(executor.map(lambda i: request(i, n, barrier), range(n)))
        elapsed = max(e for _, e, _ in results) - min(s for s, _, _ in results)
        count = sum(c for _, _, c in results)
        print(json.dumps(dict(concurrency=n, tokens=count, elapsed_s=round(elapsed, 3),
                              aggregate_tok_s=round(count / elapsed, 2),
                              median_request_tok_s=round(median(c / (e - s) for s, e, c in results), 2))), flush=True)


if __name__ == "__main__":
    main()
