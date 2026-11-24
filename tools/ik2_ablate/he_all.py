#!/usr/bin/env python3
"""he_all.py - all 164 HumanEval+ tasks (tests/ik2_coding.py samples 16): the same prompt, medium reasoning and
4,096 tokens, N requests at once (for the batch server).

    he_all.py gen  URL DATA OUT.jsonl [--concurrency 16]
    he_all.py score DATA OUT.jsonl [--image ik2-evalplus:local]   # one JSON line a task, then a summary line
"""
import argparse, json, pathlib, sys
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[2] / "tests"))

def rows_of(path):
    rows = [json.loads(l) for l in pathlib.Path(path).read_text().splitlines() if l.strip()]
    return sorted(rows, key=lambda r: int(r["task_id"].split("/")[-1]))

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("mode", choices=("gen", "score"))
    ap.add_argument("args", nargs="+"); ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--image", default="ik2-evalplus:local"); a = ap.parse_args()
    if a.mode == "gen":
        from ik2_chat_growth import post
        url, data, out = a.args
        def one(row):
            prompt = "Implement this Python function. Return the complete runnable Python code in one code block.\n\n" + row["prompt"]
            r = post(url, "/v1/chat/completions", dict(model="inkling-2-mini", messages=[dict(role="user", content=prompt)],
                     temperature=0, reasoning_effort="medium", max_tokens=4096))
            c, = r["choices"]
            return dict(task_id=row["task_id"], answer=c["message"].get("content") or "", finish=c["finish_reason"])
        with open(out, "w") as f, ThreadPoolExecutor(a.concurrency) as pool:
            for rec in pool.map(one, rows_of(data)):
                f.write(json.dumps(rec) + "\n"); f.flush()
    else:
        from ik2_coding import score
        data, out = a.args
        by = {r["task_id"]: r for r in rows_of(data)}
        recs = [json.loads(l) for l in open(out)]
        n = 0
        with ThreadPoolExecutor(6) as pool:
            for rec, s in zip(recs, pool.map(lambda r: score(by[r["task_id"]], r["answer"], a.image), recs)):
                n += bool(s["passed"]); print(json.dumps(dict(task_id=rec["task_id"], finish=rec["finish"], **s)), flush=True)
        print(json.dumps(dict(passed=n, total=len(recs), length=sum(r["finish"] == "length" for r in recs))))

if __name__ == "__main__":
    main()
