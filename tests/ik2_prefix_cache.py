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


def cancel_and_resume(engine, request, control_ids, cancel_after):
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
            if len(emitted) == cancel_after:
                engine.stdin.write("STOP\n")
                engine.stdin.flush()
        elif line.startswith("@END "):
            if line.split()[1] != "cancelled" or len(emitted) < cancel_after:
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


def check_fork(engine, ids, depth):
    base = list(itertools.islice(itertools.cycle(ids), depth))
    markers = list(dict.fromkeys(ids))[:2]
    generate(engine, base, 1)
    histories = []
    for marker in markers:
        branch = base + [marker] * 32
        output, reused = generate(engine, branch, 4)
        if reused != depth:
            raise RuntimeError(f"fork reused {reused} of {depth} shared tokens")
        histories.append(branch + output[:-1])
    controls = []
    for i, history in enumerate(histories):
        request = history + [markers[i]] * 7
        output, reused = generate(engine, request, 8)
        if reused != len(history):
            raise RuntimeError(f"fork {i} retained {reused} of {len(history)} live tokens")
        controls.append((request, output))
    for marker in markers:
        generate(engine, [markers[1], marker], 1)
    for request, expected in controls:
        output, reused = generate(engine, request, 8)
        if reused or output != expected:
            raise RuntimeError(f"cold fork replay cached {reused}, returned {output}, expected {expected}")
    print(f"both forks retained {len(histories[0])} tokens and matched all 8 cold predictions", flush=True)


def check_full_slot_rewind(engine, ids, depth):
    left, right = list(dict.fromkeys(ids))[:2]
    common = list(itertools.islice(itertools.cycle(ids), depth - 1))
    a, b = [left] + common, [right] + common
    a_reply, _ = generate(engine, a, 8)
    b_reply, _ = generate(engine, b, 8)
    # Simulate a loop guard keeping part of B's output and changing its continuation.
    replacement = right if b_reply[3] == left else left
    revised = b + b_reply[:3] + [replacement]
    b_revised, reused = generate(engine, revised, 4)
    if reused != len(b):
        raise RuntimeError(f"rewind restored {reused} of B's {len(b)} prompt tokens")
    controls = []
    for label, history in (("A", a + a_reply[:-1]), ("B", revised + b_revised[:-1])):
        request = history + [right, left]
        answer, reused = generate(engine, request, 8)
        if reused != len(history):
            raise RuntimeError(f"rewinding B left {reused} of {label}'s {len(history)} live tokens")
        controls.append((request, answer))
    for marker in (left, right):
        generate(engine, [right, marker], 1)
    for request, expected in controls:
        actual, reused = generate(engine, request, 8)
        if reused or actual != expected:
            raise RuntimeError(f"cold rewind control reused {reused} or changed its eight predictions")
    print("Both conversations survived the rewind and matched eight cold predictions each", flush=True)


def check_cancelled_prefill(engine, ids, chunk):
    def request(prompt, limit, interrupt=False):
        engine.stdin.write(f"GEN {limit} temperature=0 " + ",".join(map(str, prompt)) + "\n")
        engine.stdin.flush()
        tokens = []
        stopped = False
        for line in engine.stdout:
            fields = line.split()
            if not fields:
                continue
            if fields[0] == "PP" and interrupt and not stopped and int(fields[1]) < len(prompt):
                engine.stdin.write("STOP\n")
                engine.stdin.flush()
                stopped = True
            elif fields[0] == "T":
                tokens.append(int(fields[1]))
            elif fields[0] == "DONE":
                expected = "cancel" if interrupt else "length"
                if fields[5] != expected or (interrupt and (not stopped or tokens)):
                    raise RuntimeError(f"expected {expected}, got {line.strip()}")
                return tokens, int(fields[-1])
        raise RuntimeError("engine exited during the prefill cancellation check")

    a, b = list(dict.fromkeys(ids))[:2]
    base = list(itertools.islice(itertools.cycle(ids), 7 * chunk))
    reply, _ = request(base, 4)
    request([b] + base, 8, interrupt=True)
    request([b, b], 2)  # A fresh chat should occupy the slot freed by the cancellation.
    history = base + reply[:-1]
    prompt = history + [a]
    answer, reused = request(prompt, 8)
    if reused != len(history):
        raise RuntimeError(f"cancelled prefill left {reused} of the other chat's {len(history)} live tokens")
    for marker in (b, a):
        request([b, marker], 1)
    cold, reused = request(prompt, 8)
    if reused or answer != cold:
        raise RuntimeError("the resumed chat did not match the uncached control")
    print(f"Cancelled a prefill, kept {len(history)} sibling tokens, matched eight cold predictions", flush=True)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--engine", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--draft", help="also check the optional DSpark drafter's cached state")
    ap.add_argument("--tokens", required=True, type=Path, help="comma-separated IDs of a reference passage")
    ap.add_argument("--cancel-request", type=Path, help="also interrupt the first long copy request in this file")
    ap.add_argument("--chunk", type=int, default=256, help="prefill chunk size and first shared prefix")
    ap.add_argument("--cancel-after", type=int, default=128, help="send STOP after this many emitted copy tokens")
    ap.add_argument("--branches", type=int, default=0, help="branch repeatedly from one shared prefix")
    slot_test = ap.add_mutually_exclusive_group()
    slot_test.add_argument("--fork-slots", action="store_true", help="fork one prefix into two slots and resume each")
    slot_test.add_argument("--cancel-prefill", action="store_true", help="cancel one slot during prefill and resume its sibling")
    args = ap.parse_args()
    if args.branches < 0:
        ap.error("branches must be nonnegative")
    if min(args.chunk, args.cancel_after) < 1:
        ap.error("chunk and cancellation position must be positive")
    ids = [int(s) for s in args.tokens.read_text().strip().split(",")]
    distinct = list(dict.fromkeys(ids))
    if len(distinct) < 2:
        ap.error("the passage needs two distinct token IDs")
    a, b = distinct[:2]
    ctx = max(4096, 7 * args.chunk + 64)
    two = args.fork_slots or args.cancel_prefill
    cmd = [args.engine, "--serve" if args.cancel_prefill else "--serve-ik2", "--model", args.model,
           "--ctx", str(ctx * (2 if two else 1)), "--chunk", str(args.chunk)]
    if args.draft:
        cmd.extend(["--draft", args.draft])
    if two:
        cmd.extend(["--cache-slots", "2"])
        with subprocess.Popen(cmd, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as engine:
            if args.cancel_prefill:
                receive(engine, "READY ")
                check_cancelled_prefill(engine, ids, args.chunk)
            else:
                receive(engine, "@READY ")
                check_fork(engine, ids, 7 * args.chunk)
                check_full_slot_rewind(engine, ids, 7 * args.chunk)
            engine.stdin.write("QUIT\n")
            engine.stdin.close()
            if engine.wait():
                raise RuntimeError("engine exited unsuccessfully")
        return
    with subprocess.Popen(cmd, text=True, stdin=subprocess.PIPE, stdout=subprocess.PIPE) as engine:
        receive(engine, "@READY ")
        for depth in (args.chunk, 7 * args.chunk):
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
        if args.branches:
            base = [b] * args.chunk
            generate(engine, base, 1)
            for i in range(args.branches):
                branch = base + [distinct[i % 2]] * 16
                last, reused = generate(engine, branch, 4)
                if reused != len(base):
                    raise RuntimeError(f"branch {i + 1}: reused {reused} of {len(base)} common tokens")
            generate(engine, [a], 1)
            cold, reused = generate(engine, branch, 4)
            if reused or cold != last:
                raise RuntimeError(f"last branch differs on cold replay: {last} vs {cold}, cached {reused}")
            print(f"{args.branches} alternating branches reused all {len(base)} shared tokens", flush=True)
        if args.cancel_request:
            cancel_and_resume(engine, args.cancel_request.read_text().splitlines()[0], ids, args.cancel_after)
        engine.stdin.write("QUIT\n")
        engine.stdin.close()
        if engine.wait():
            raise RuntimeError("engine exited unsuccessfully")


if __name__ == "__main__":
    main()
