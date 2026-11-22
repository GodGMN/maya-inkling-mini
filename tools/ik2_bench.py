"""Measure uncached prefill and greedy decode at several depths in one resident maya-ik2 process."""
import argparse
import json
import pathlib
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--tokens", required=True, type=pathlib.Path, help="comma-separated real token IDs, cycled to each depth")
    parser.add_argument("--depths", default="32,512,4096")
    parser.add_argument("--gen", type=int, default=129, help="includes the first token predicted by prefill")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--chunk", type=int, default=128)
    parser.add_argument("--ctx", type=int, help="reserve at least this many context positions")
    args = parser.parse_args()
    ids = [int(s) for s in args.tokens.read_text().strip().split(",")]
    depths = [int(s) for s in args.depths.split(",")]
    if len(set(ids)) < 2 or min(depths) < 1 or args.gen < 2 or args.repeats < 1:
        parser.error("provide distinct token IDs, positive depths and repeats, and at least two generated tokens")
    # Rotate between two different leading tokens to prevent prefix reuse between successive requests.
    shift = next(i for i, tok in enumerate(ids) if tok != ids[0])
    cmd = [args.engine, "--model", args.model, "--serve-ik2", "--ctx", str(max(max(depths) + args.gen, args.ctx or 64)),
           "--chunk", str(args.chunk)]
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

        def request(depth, gen, offset):
            prompt = ",".join(str(ids[(i + offset) % len(ids)]) for i in range(depth))
            engine.stdin.write(f"GEN {gen} 0 1 1 - {prompt}\n")
            engine.stdin.flush()
            fields = receive("@END ", gen)
            if len(fields) != 6 or fields[1] != "length" or int(fields[2]) != depth or int(fields[3]):
                raise RuntimeError("unexpected engine result: " + " ".join(fields))
            return float(fields[4]), float(fields[5])

        receive("@READY ")
        request(1, 9, shift)
        run = 0
        for depth in depths:
            for repeat in range(args.repeats):
                pp, tg = request(depth, args.gen, shift if run % 2 else 0)
                print(json.dumps(dict(depth=depth, repeat=repeat + 1, prefill_s=pp, decode_s=tg,
                                      prefill_tok_s=round(depth / pp, 2), decode_tok_s=round((args.gen - 1) / tg, 2))),
                      flush=True)
                run += 1
        engine.stdin.write("QUIT\n")
        engine.stdin.flush()
        engine.stdin.close()
        if engine.wait():
            raise RuntimeError("engine exited unsuccessfully")


if __name__ == "__main__":
    main()
