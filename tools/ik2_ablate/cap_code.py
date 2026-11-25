#!/usr/bin/env python3
"""cap_code.py PORT DATA OUT_CAPTURE: capture 48-visit prefill residuals on HumanEval+ coding prompts
EXCLUDING the fixed 16-task code_16 screen (so coding-orthogonalization never leaks into code_16).

Run against a stock lab server started with IK2_CAPTURE_PREFILL_RESIDUALS=/lab/<OUT_CAPTURE>.
"""
from __future__ import annotations

import json
import pathlib
import random
import sys
from urllib.request import Request, urlopen


def main() -> int:
    if len(sys.argv) < 3:
        raise SystemExit("usage: cap_code.py PORT HUMANEVALPLUS.jsonl [LIMIT]")
    port = sys.argv[1]
    data_path = pathlib.Path(sys.argv[2])
    limit = int(sys.argv[3]) if len(sys.argv) > 3 else 148

    rows = [json.loads(line) for line in data_path.read_text(encoding="utf-8").splitlines() if line.strip()]
    rows.sort(key=lambda r: int(r["task_id"].split("/")[-1]))
    heldout = {r["task_id"] for r in random.Random(1429).sample(rows, 16)}
    train_rows = [r for r in rows if r["task_id"] not in heldout][:limit]

    url = f"http://127.0.0.1:{port}/v1/chat/completions"
    for i, r in enumerate(train_rows):
        prompt = "Implement this Python function. Return the complete runnable Python code in one code block.\n\n" + r["prompt"]
        body = {
            "model": "inkling-2-mini",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0,
            "seed": 7,
            "reasoning_effort": "none",
            "max_tokens": 1,
        }
        with urlopen(Request(url, json.dumps(body).encode("utf-8"), {"Content-Type": "application/json"}), timeout=180) as resp:
            json.load(resp)
        if (i + 1) % 32 == 0 or i + 1 == len(train_rows):
            print(f"captured {i + 1}/{len(train_rows)} coding prompts", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
