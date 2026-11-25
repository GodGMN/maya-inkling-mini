#!/usr/bin/env python3
"""code16.py: fast batched generator + sandboxed scorer for the fixed 16-task HumanEval+ screen.

Uses the exact same 16 tasks (random.Random(1429).sample(rows, 16)) and prompt/grading contract as
tests/ik2_coding.py, but generates in parallel against batched servers and auto-selects an available
local Docker image with Python + NumPy (ik2-evalplus:local or hyperion/maya-ik2:gfx1100-service-r4).

Works identically against Maya runtime-ablation presets (127.0.0.1:8087) and standalone/baked GGUFs
served via tintero or llama-server.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import secrets
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen


def select_problems(data_path: pathlib.Path, count: int = 16) -> list[dict]:
    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda row: int(row["task_id"].split("/")[-1]))
    return random.Random(1429).sample(rows, min(count, len(rows)))


def pick_sandbox_image(requested: str | None) -> str:
    if requested:
        return requested
    for candidate in ("ik2-evalplus:local", "hyperion/maya-ik2:gfx1100-service-r4"):
        p = subprocess.run(["docker", "image", "inspect", candidate], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if p.returncode == 0:
            return candidate
    raise RuntimeError("No sandbox Docker image found (tried ik2-evalplus:local, hyperion/maya-ik2:gfx1100-service-r4)")


def generate_one(url: str, model: str, row: dict, reasoning_effort: str, max_tokens: int, timeout: int) -> dict:
    prompt = "Implement this Python function. Return the complete runnable Python code in one code block.\n\n" + row["prompt"]
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 7,
        "reasoning_effort": reasoning_effort,
        "max_tokens": max_tokens,
    }
    t0 = time.monotonic()
    req = Request(
        url.rstrip("/") + "/v1/chat/completions",
        json.dumps(body).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as resp:
        payload = json.load(resp)
    choice = payload["choices"][0]
    msg = choice["message"]
    return {
        "task_id": row["task_id"],
        "seconds": round(time.monotonic() - t0, 3),
        "answer": msg.get("content") or "",
        "finish": choice.get("finish_reason"),
    }


def score_one(row: dict, answer: str, image: str) -> dict:
    text = (answer or "").strip()
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    program = blocks[-1] if blocks else text
    marker = secrets.token_hex(16)
    script = program + "\n" + row["test"] + "\ncheck(" + row["entry_point"] + ")\n" + f"print({marker!r})\n"
    runner = """timeout -k 1 20 python3 -B -
status=$?
if [ "$status" -eq 137 ] && [ -r /sys/fs/cgroup/memory.events ]; then
    while read -r event count; do
        if [ "$event" = oom_kill ] && [ "$count" -gt 0 ]; then
            echo 'Generated program exceeded the 256 MiB grader memory limit.' >&2
        fi
    done < /sys/fs/cgroup/memory.events
fi
exit "$status"
"""
    cmd = [
        "docker", "run", "--rm", "-i", "--network=none", "--read-only", "--cap-drop=ALL",
        "--security-opt=no-new-privileges", "--memory=256m", "--memory-swap=256m",
        "--pids-limit=32", "--cpus=1", "--user=65534:65534", "-e", "OPENBLAS_NUM_THREADS=1",
        "--entrypoint", "sh", image, "-c", runner,
    ]
    res = subprocess.run(cmd, input=script, text=True, capture_output=True, timeout=45)
    if res.returncode in (125, 126, 127) or "ModuleNotFoundError: No module named 'numpy'" in res.stderr:
        raise RuntimeError(res.stderr[-1000:])
    completed = res.stdout.splitlines()[-1:] == [marker]
    detail = "exited before completing tests" if res.returncode == 0 and not completed else res.stderr[-300:]
    return {"passed": bool(res.returncode == 0 and completed), "returncode": res.returncode, "detail": detail}


def main() -> int:
    ap = argparse.ArgumentParser(description="Batched 16-task HumanEval+ coding screen.")
    ap.add_argument("--url", default="http://127.0.0.1:8087", help="OpenAI-compatible server URL")
    ap.add_argument("--model", default="inkling-2-mini", help="Model ID to request")
    ap.add_argument("--data", type=pathlib.Path, required=True, help="Path to humanevalplus.jsonl")
    ap.add_argument("--count", type=int, default=16, help="Number of sampled tasks (default 16)")
    ap.add_argument("--reasoning-effort", default="medium", help="reasoning_effort passed to chat completions")
    ap.add_argument("--max-tokens", type=int, default=3072, help="max_tokens per coding task (fits 4096-token batch slots)")
    ap.add_argument("--concurrency", type=int, default=16, help="Concurrent generation requests")
    ap.add_argument("--score-workers", type=int, default=4, help="Parallel Docker sandbox workers")
    ap.add_argument("--image", default=None, help="Docker image with python3 + numpy (auto-detected if omitted)")
    ap.add_argument("--timeout", type=int, default=900, help="HTTP timeout per request")
    ap.add_argument("--out", type=pathlib.Path, help="Optional JSONL path to save per-task generations + verdicts")
    ap.add_argument("--score-only", type=pathlib.Path, help="Score an existing JSONL file without regenerating")
    args = ap.parse_args()

    rows = select_problems(args.data, args.count)
    by_id = {r["task_id"]: r for r in rows}
    image = pick_sandbox_image(args.image)

    t0 = time.monotonic()
    if args.score_only:
        gens = [json.loads(line) for line in args.score_only.read_text(encoding="utf-8").splitlines() if line.strip()]
    else:
        with ThreadPoolExecutor(max_workers=max(1, args.concurrency)) as pool:
            gens = list(pool.map(
                lambda r: generate_one(args.url, args.model, r, args.reasoning_effort, args.max_tokens, args.timeout),
                rows,
            ))

    with ThreadPoolExecutor(max_workers=max(1, args.score_workers)) as pool:
        verdicts = list(pool.map(lambda g: score_one(by_id[g["task_id"]], g["answer"], image), gens))

    merged = []
    for g, v in zip(gens, verdicts):
        rec = dict(g)
        rec.update(v)
        merged.append(rec)

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        with args.out.open("w", encoding="utf-8") as f:
            for rec in merged:
                f.write(json.dumps(rec) + "\n")

    passed = sum(1 for r in merged if r["passed"])
    hit_len = sum(1 for r in merged if r.get("finish") == "length")
    failed_ids = [r["task_id"] for r in merged if not r["passed"]]
    summary = {
        "code16": passed,
        "code16_total": len(merged),
        "code16_length": hit_len,
        "code16_failed": failed_ids,
        "elapsed_s": round(time.monotonic() - t0, 2),
    }
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
