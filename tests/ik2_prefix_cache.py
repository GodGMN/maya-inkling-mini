"""Compare exact greedy output with and without restoring a chat prefix on a loaded Mini model."""
import argparse
import itertools
from pathlib import Path
import subprocess
import sys


def receive(engine, prefix):
    ids = []
    for line in engine.stdout:
        if line.startswith("@TOK "):
            ids.append(int(line.split()[1]))
        elif line.startswith(prefix):
            return ids, line.split()
        else:
            print(line.rstrip(), file=sys.stderr)
    raise RuntimeError(f"engine exited before {prefix} with status {engine.wait()}")


def generate(engine, prompt, length):
    engine.stdin.write(f"GEN {length} 0 1 1 - " + ",".join(map(str, prompt)) + "\n")
    engine.stdin.flush()
    tokens, end = receive(engine, "@END ")
    if end[1] != "length" or int(end[2]) != len(prompt) or len(tokens) != length:
        raise RuntimeError(f"incomplete response to {len(prompt)} prompt tokens: {end}")
    return tokens, int(end[3])


def cancel_and_resume(engine, request, control_ids):
    fields = request.split()
    if len(fields) != 7 or fields[0] != "GEN":
        raise ValueError("expected a tokenized GEN request")
    prompt = list(map(int, fields[-1].split(",")))
    engine.stdin.write(request + "\n")
    engine.stdin.flush()
    emitted = []
    for line in engine.stdout:
        if line.startswith("@TOK "):
            emitted.append(int(line.split()[1]))
            if len(emitted) == 128:
                engine.stdin.write("STOP\n")
                engine.stdin.flush()
        elif line.startswith("@END "):
            if line.split()[1] != "cancelled" or len(emitted) < 128:
                raise RuntimeError(f"copy request failed to cancel: {line.rstrip()}")
            break
    else:
        raise RuntimeError("engine exited during cancellation")
    branch = prompt + emitted + control_ids[:4]
    cached, reused = generate(engine, branch, 8)
    clear_token = next(t for t in control_ids if t != branch[0])
    generate(engine, [clear_token], 2)
    cold, reused_cold = generate(engine, branch, 8)
    if reused < len(prompt) or reused_cold or cached != cold:
        raise RuntimeError(f"cancelled copy: reused {reused}, cold {reused_cold}, outputs {cached} vs {cold}")
    print(f"cancelled after {len(emitted)} tokens; reused {reused}; 8/8 predictions match the cold replay", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", required=True, type=Path, help="comma-separated IDs of a reference passage")
    ap.add_argument("--cancel-request", type=Path, help="also interrupt the first long copy request in this file")
    args = ap.parse_args()
    ids = [int(s) for s in args.tokens.read_text().strip().split(",")]
    distinct = list(dict.fromkeys(ids))
    if len(distinct) < 2:
        ap.error("the passage needs two distinct token IDs")
    a, b = distinct[:2]
    cmd = [args.engine, "--serve-ik2", "--model", args.model, "--ctx", "4096", "--chunk", "256"]
    with subprocess.Popen(cmd, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as engine:
        receive(engine, "@READY ")
        for depth in (256, 1792):
            # Chunk boundaries match the cold replay. The larger prefix wraps the sliding KV ring.
            base = list(itertools.islice(itertools.cycle(ids), depth))
            generate(engine, base, 4)
            generate(engine, base + [a] * 32, 4)
            branch = base + [b] * 32
            cached, reused = generate(engine, branch, 16)
            generate(engine, [b], 4)
            cold, reused_cold = generate(engine, branch, 16)
            if reused != depth or reused_cold != 0 or cached != cold:
                raise RuntimeError(f"prefix {depth}: cached {reused}, cold cached {reused_cold}, "
                                   f"predictions {cached} vs {cold}")
            print(f"{depth} cached tokens, 16/16 predictions match the cold replay", flush=True)
        if args.cancel_request:
            cancel_and_resume(engine, args.cancel_request.read_text().splitlines()[0], ids)
        engine.stdin.write("QUIT\n")
        engine.stdin.close()
        if engine.wait():
            raise RuntimeError("engine exited unsuccessfully")


if __name__ == "__main__":
    main()
