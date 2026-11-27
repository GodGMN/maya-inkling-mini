"""Compare greedy native-server token IDs across concurrent, independent Inkling requests."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
from time import perf_counter

ROOT = Path(__file__).resolve().parents[1]
TOPICS = (
    "wind-powered wells", "how bees navigate", "the history of glass", "building a sundial",
    "what makes ice float", "a Roman market", "how bread rises", "why calendars need leap years",
    "training for a marathon", "choosing a telescope", "designing a footbridge", "a ship's logbook",
    "comparing two fractions", "measuring rainfall", "acid-base titration", "why the moon has phases",
    "balancing a ledger", "debugging a spreadsheet", "restoring a photograph", "the rules of curling",
    "a lost library book", "practicing scales", "learning to ride a bicycle", "a day at an orchard",
    "testing a sorting algorithm", "planning a train journey", "planting native grasses", "a potter at work",
    "deriving the quadratic formula", "reading a topographic map", "how eclipses work", "a first chess lesson",
)


def prompts(tokenizer):
    sys.path.insert(0, str(ROOT / "tools"))
    from strata_tokenizer import Tokenizer
    from jinja2 import Environment

    tok = Tokenizer.from_dir(tokenizer)
    template = Environment().from_string((ROOT / "tools/ik2/chat_template.jinja").read_text())
    result = []
    for i, topic in enumerate(TOPICS):
        extra = ["", " Give a worked example.", " Discuss one common misunderstanding.",
                 " Include a short dialogue between a student and a teacher."][i % 4]
        question = f"Write a detailed lesson about {topic}.{extra} Aim for at least 800 words."
        chat = template.render(messages=[{"role": "user", "content": question}], tools=[],
                               add_generation_prompt=True, reasoning_effort="medium")
        result.append(tok.encode(chat, parse_special=True))
    return result


def run(args, engine, ids, width, concurrency, env, cancel_first=False, mixed=False,
        draft=None, stop_token=None, timed=False, cancel_queued=False):
    cmd = [engine, "--serve", "--model", args.model, "--max-context", str(args.ctx * width),
           "--chunk", str(args.chunk)]
    if draft:
        cmd.extend(["--draft", draft])
    generated = [[] for _ in ids]
    first_wave = set(range(min(concurrency, len(ids))))
    started = set()
    steady_start = steady_end = None
    steady_tokens = 0
    with subprocess.Popen(cmd, env={**env, "IK2_BATCH": str(width)}, stdin=subprocess.PIPE,
                          stdout=subprocess.PIPE, text=True, bufsize=1) as p:
        def send(line):
            p.stdin.write(line + "\n")
            p.stdin.flush()

        def read():
            line = p.stdout.readline()
            if not line:
                raise RuntimeError(f"engine exited with status {p.wait()}")
            return line.strip().split()

        try:
            while True:
                ready = read()
                if ready and ready[0] == "READY":
                    break
            assert int(ready[1]) == args.ctx, ready
            assert (f"multi={width}" in ready) == (width > 1), ready
            pending = iter(range(len(ids)))
            active = set()

            def admit():
                i = next(pending, None)
                if i is not None:
                    active.add(i)
                    tag = f" rid={i + 1}" if width > 1 else ""
                    sampling = f"temperature=0.7 top_p=0.9 top_k=20 seed={17 + i}" if mixed and i % 2 else "temperature=0"
                    if i == 0 and stop_token is not None:
                        sampling += f" stop_token={stop_token}"
                    limit = 8 + 3 * i if mixed else args.tokens
                    send(f"GEN {limit} {sampling}{tag} " + ",".join(map(str, ids[i])))

            start = perf_counter()
            for _ in range(concurrency):
                admit()
            if cancel_queued:
                send(f"STOP {len(ids)}")
            while active:
                line = read()
                if not line or line[0] not in {"PP", "DONE", "T", "ERR"}:
                    continue
                if line[0] == "ERR":
                    raise ValueError(" ".join(line))
                i = int(line[1]) - 1 if width > 1 else next(iter(active))
                assert i in active, (i, line)
                data = line[2 if width > 1 else 1:]
                if line[0] == "T":
                    generated[i].append(int(data[0]))
                    if i in first_wave:
                        started.add(i)
                        if steady_start is None and started == first_wave:
                            steady_start = perf_counter()
                        elif steady_start is not None and steady_end is None:
                            steady_tokens += 1
                    if cancel_first and i == 0 and len(generated[i]) == 3:
                        send("STOP 1")
                elif line[0] == "DONE":
                    if steady_end is None and i in first_wave:
                        steady_end = perf_counter()
                    assert len(generated[i]) == int(data[0]), line
                    if cancel_queued and i == len(ids) - 1:
                        assert data[4] == "cancel" and not generated[i], line
                    if cancel_first and i == 0:
                        assert data[4] == "cancel" and len(generated[i]) < args.tokens, line
                    if i == 0 and stop_token is not None:
                        assert data[4] == "stop" and generated[i][-1] == stop_token, line
                    active.remove(i)
                    admit()
            if timed:
                elapsed = perf_counter() - start
                count = sum(map(len, generated))
                print(json.dumps(dict(engine=engine, width=width, concurrency=concurrency, speculative=bool(draft),
                                      completion_tokens=count, elapsed_s=elapsed,
                                      aggregate_tok_s=count / elapsed, steady_tokens=steady_tokens,
                                      steady_tok_s=(steady_tokens / (steady_end - steady_start)
                                                    if steady_tokens and steady_end > steady_start else None))), flush=True)
        finally:
            if p.poll() is None:
                send("QUIT")
                p.stdin.close()
        if p.wait() != 0:
            raise RuntimeError(f"engine exited with status {p.returncode}")
    return generated


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", required=True, help="unmodified executable")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--draft", help="compare batched speculative decoding at widths 1, 2, 4, 6 and 8")
    parser.add_argument("--tokenizer", required=True)
    parser.add_argument("--preset", type=Path, required=True, help="JSON preset, for its environment")
    parser.add_argument("--ctx", type=int, default=4096)
    parser.add_argument("--chunk", type=int, default=256)
    parser.add_argument("--tokens", type=int, default=128)
    parser.add_argument("--width", type=int)
    parser.add_argument("--benchmark", action="store_true", help="measure plain and draft throughput at --width")
    parser.add_argument("--concurrency", type=int, help="simultaneous requests in benchmark mode")
    parser.add_argument("--count", type=int, default=len(TOPICS))
    args = parser.parse_args()
    ids = prompts(args.tokenizer)[:args.count]
    env = {**os.environ, **json.loads(args.preset.read_text())["env"]}
    if args.benchmark:
        width = args.width or 8
        concurrency = args.concurrency or width
        if not args.draft or not 1 <= concurrency <= width <= 8 or len(ids) < concurrency:
            parser.error("--benchmark needs --draft and 1 <= concurrency <= width <= 8 with enough prompts")
        run(args, args.baseline, ids, width, concurrency, env, timed=True)
        if concurrency == 1:
            run(args, args.baseline, ids, 1, 1, env, draft=args.draft, timed=True)
        run(args, args.engine, ids, width, concurrency, env, draft=args.draft, timed=True)
        return
    if args.draft:
        if len(ids) < 8 or args.tokens < 16:
            parser.error("--draft needs at least eight prompts and sixteen tokens")
        env["IK2_BLACKWELL_NVFP4"] = "0"
        failed = False
        reference = run(args, args.baseline, ids, 1, 1, env, timed=True)
        for width in (1, 2, 4, 6, 8):
            actual = run(args, args.engine, ids, width, width, env, draft=args.draft, timed=True)
            mismatched = [i for i, (a, b) in enumerate(zip(actual, reference)) if a != b]
            print(json.dumps(dict(width=width, greedy_mismatches=mismatched)), flush=True)
            failed |= bool(mismatched)
        sample = ids[:8]
        expected = run(args, args.baseline, sample, 1, 1, env, mixed=True)
        assert run(args, args.engine, sample, 4, 4, env, mixed=True, draft=args.draft) == expected
        baseline = reference[:len(sample)]
        actual = run(args, args.engine, sample, 2, len(sample), env, cancel_first=True, cancel_queued=True, draft=args.draft)
        assert actual[1:-1] == baseline[1:-1], "cancellation or reuse changed a neighboring slot"
        stop = baseline[0][min(3, len(baseline[0]) - 1)]
        actual = run(args, args.engine, sample, 2, 2, env, stop_token=stop, draft=args.draft)
        assert actual[0] == baseline[0][:baseline[0].index(stop) + 1] and actual[1:] == baseline[1:]
        print("Mixed sampling, stop tokens, active and queued cancellation, and slot reuse passed.", flush=True)
        raise SystemExit(failed)
    args.width = args.width or 32
    reference = run(args, args.baseline, ids, 1, 1, env)
    failed = False
    for width, concurrency in [(1, 1)] + [(args.width, n) for n in (1, 8, 32) if n <= args.width]:
        result = run(args, args.engine, ids, width, concurrency, env)
        exact = sum(a == b for a, b in zip(reference, result))
        same = sum(sum(x == y for x, y in zip(a, b)) for a, b in zip(reference, result))
        total = sum(max(len(a), len(b)) for a, b in zip(reference, result))
        print(json.dumps(dict(width=width, concurrency=concurrency, prompts=len(ids), exact=exact,
                              same=same, total=total)), flush=True)
        failed |= exact != len(ids)
    result = run(args, args.engine, ids[:4], args.width, 2, env, cancel_first=True)
    assert result[1:] == reference[1:4], "cancellation or reuse changed a neighboring slot"
    sample_ids = ids[:8]
    expected = run(args, args.baseline, sample_ids, 1, 1, env, mixed=True)
    actual = run(args, args.engine, sample_ids, args.width, min(8, args.width), env, mixed=True)
    assert actual == expected, "mixed greedy and seeded sampling changed a request's tokens"
    print("Cancellation, slot reuse and mixed sampling passed.", flush=True)
    raise SystemExit(failed)


if __name__ == "__main__":
    main()
