"""Measure uncached prefill and decode on maya-ik2 or a running llama.cpp service."""
import argparse
from contextlib import contextmanager
import json
import pathlib
import subprocess
import sys
from urllib.request import Request, urlopen


@contextmanager
def maya_requests(args, capacity):
    cmd = [args.engine, "--model", args.model, "--serve-ik2", "--ctx", str(capacity), "--chunk", str(args.chunk)]
    if args.draft:
        cmd.extend(["--draft", str(args.draft)])
    with subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True, bufsize=1) as engine:
        assert engine.stdin and engine.stdout

        def receive(prefix, expected_tokens=0):
            generated = 0
            for line in engine.stdout:
                if line.startswith(prefix):
                    if generated != expected_tokens:
                        raise RuntimeError(f"expected {expected_tokens} tokens, received {generated}")
                    return line.strip().split()
                if line.startswith("@TOK "):
                    generated += 1
                else:
                    print(line.rstrip(), file=sys.stderr, flush=True)
            raise RuntimeError(f"engine exited with {engine.wait()} before {prefix}")

        def request(prompt, gen):
            ids = ",".join(map(str, prompt))
            engine.stdin.write(f"GEN {gen} {args.temperature} {args.top_p} 1 - {ids}\n")
            engine.stdin.flush()
            fields = receive("@END ", gen)
            if len(fields) != 6 or fields[1] != "length" or int(fields[2]) != len(prompt) or int(fields[3]):
                raise RuntimeError("unexpected engine result: " + " ".join(fields))
            return float(fields[4]), float(fields[5])

        receive("@READY ")
        yield request
        engine.stdin.write("QUIT\n")
        engine.stdin.flush()
        engine.stdin.close()
        if engine.wait():
            raise RuntimeError("engine exited unsuccessfully")


@contextmanager
def llama_requests(args):
    def request(prompt, gen):
        payload = dict(prompt=prompt, n_predict=gen, temperature=args.temperature, top_p=args.top_p, top_k=1000, seed=1, cache_prompt=False,
                       ignore_eos=True, stream=False)
        req = Request(args.llama_url.rstrip("/") + "/completion", data=json.dumps(payload).encode(),
                      headers={"Content-Type": "application/json"})
        with urlopen(req, timeout=3600) as response:
            result = json.load(response)
        times = result["timings"]
        if result["tokens_predicted"] != gen or times["prompt_n"] != len(prompt) or times.get("cache_n", 0):
            raise RuntimeError(f"unexpected llama.cpp result: {times}")
        return times["prompt_ms"] / 1000, times["predicted_ms"] / 1000

    yield request


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--engine", help="path to maya-ik2")
    target.add_argument("--llama-url", help="base URL of llama.cpp server")
    parser.add_argument("--model", help="GGUF for --engine")
    parser.add_argument("--draft", type=pathlib.Path, help="optional DSpark mtp.safetensors for --engine")
    parser.add_argument("--tokens", required=True, type=pathlib.Path, help="comma-separated token IDs, cycled to each depth")
    parser.add_argument("--depths", default="32,512,4096")
    parser.add_argument("--gen", type=int, default=129, help="includes the first token predicted by prefill")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--ctx", type=int, help="reserve at least this many context positions in Maya")
    parser.add_argument("--temperature", type=float, default=0, help="zero selects greedy decode")
    parser.add_argument("--top-p", type=float, default=1, help="nucleus probability for sampled decode")
    args = parser.parse_args()
    if not 0 <= args.temperature < float("inf") or not 0 < args.top_p <= 1:
        parser.error("temperature must be finite and nonnegative and top-p must be in (0, 1]")
    if args.engine and not args.model:
        parser.error("--engine requires --model")
    if args.draft and not args.engine:
        parser.error("--draft requires --engine")
    ids = [int(s) for s in args.tokens.read_text().strip().split(",")]
    depths = [int(s) for s in args.depths.split(",")]
    if len(set(ids)) < 2 or min(depths) < 1 or args.gen < 2 or args.repeats < 1:
        parser.error("provide distinct token IDs, positive depths and repeats, and at least two generated tokens")
    shift = next(i for i, tok in enumerate(ids) if tok != ids[0])
    capacity = max(max(depths) + args.gen, args.ctx or 64)
    with (maya_requests(args, capacity) if args.engine else llama_requests(args)) as request:
        request([ids[shift]], 9)
        run = 0
        for depth in depths:
            for repeat in range(args.repeats):
                offset = shift if run % 2 else 0
                prompt = [ids[(i + offset) % len(ids)] for i in range(depth)]
                pp, tg = request(prompt, args.gen)
                print(json.dumps(dict(depth=depth, repeat=repeat + 1, prefill_s=pp, decode_s=tg,
                                      prefill_tok_s=round(depth / pp, 2), decode_tok_s=round((args.gen - 1) / tg, 2))),
                      flush=True)
                run += 1


if __name__ == "__main__":
    main()
