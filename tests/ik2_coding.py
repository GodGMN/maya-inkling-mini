"""Generate paired HumanEval+ samples through the server. Use --score for isolated unit tests."""
import argparse
import json
import random
import re
import secrets
import subprocess
import time
from pathlib import Path

from ik2_chat_growth import post


def problems(path):
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    rows.sort(key=lambda row: int(row["task_id"].split("/")[-1]))
    return random.Random(1429).sample(rows, 16)


def score(row, answer, image):
    text = answer.strip()
    blocks = re.findall(r"```(?:python|py)?\s*\n(.*?)```", text, re.DOTALL)
    program = blocks[-1] if blocks else text
    marker = secrets.token_hex(16)
    script = program + "\n" + row["test"] + "\ncheck(" + row["entry_point"] + ")\n" + f"print({marker!r})\n"
    # After a SIGKILL the child cannot print a traceback. Its cgroup distinguishes an OOM
    # from the timeout's kill signal, before Docker removes the container.
    runner = """timeout -k 1 20 python -B -
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
    command = ["docker", "run", "--rm", "-i", "--network=none", "--read-only", "--cap-drop=ALL",
               "--security-opt=no-new-privileges", "--memory=256m", "--memory-swap=256m",
               "--pids-limit=32", "--cpus=1", "--user=65534:65534", "-e", "OPENBLAS_NUM_THREADS=1",
               image, "sh", "-c", runner]
    result = subprocess.run(command, input=script, text=True, capture_output=True, timeout=45)
    if result.returncode in (125, 126, 127) or "ModuleNotFoundError: No module named 'numpy'" in result.stderr:
        raise RuntimeError(result.stderr[-1000:])
    completed = result.stdout.splitlines()[-1:] == [marker]
    detail = "exited before completing tests" if result.returncode == 0 and not completed else result.stderr[-400:]
    return dict(passed=result.returncode == 0 and completed, returncode=result.returncode, detail=detail)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, required=True)
    p.add_argument("--url")
    p.add_argument("--score", type=Path)
    p.add_argument("--image", default="ik2-evalplus:local", help="local Python image with numpy and timeout")
    p.add_argument("--prefix", type=Path, help="JSON list of prior chat messages")
    p.add_argument("--start", type=int, default=0)
    p.add_argument("--count", type=int, default=16)
    args = p.parse_args()
    rows = problems(args.data)
    if args.score:
        by_id = {r["task_id"]: r for r in rows}
        records = [json.loads(line) for line in args.score.read_text().splitlines()]
        coding = [r for r in records if r.get("case", "code") == "code"]
        if not coding or len({r["task_id"] for r in coding}) != len(coding):
            p.error("expected one or more unique coding results")
        for rec in coding:
            print(json.dumps(dict(task_id=rec["task_id"], **score(by_id[rec["task_id"]], rec["answer"], args.image))), flush=True)
        return
    if not args.url:
        p.error("supply --url or --score")
    prefix = json.loads(args.prefix.read_text()) if args.prefix else []
    for row in rows[args.start:args.start + args.count]:
        prompt = "Implement this Python function. Return the complete runnable Python code in one code block.\n\n" + row["prompt"]
        messages = prefix + [dict(role="user", content=prompt)]
        start = time.monotonic()
        response = post(args.url, "/v1/chat/completions", dict(model="inkling-2-mini", messages=messages,
                        temperature=0, reasoning_effort="medium", max_tokens=4096))
        choice, = response["choices"]
        print(json.dumps(dict(task_id=row["task_id"], seconds=round(time.monotonic()-start, 3),
                              answer=choice["message"].get("content") or "", usage=response["usage"],
                              finish=choice["finish_reason"])), flush=True)


if __name__ == "__main__":
    main()
