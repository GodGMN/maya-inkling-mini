"""Compare the same retrieval, reasoning, and follow-up prompts across KV formats and slot counts."""
import argparse
import json
import re
import time
from difflib import SequenceMatcher
from pathlib import Path
from urllib.request import Request, urlopen

from ik2_coding import problems


def completion(url, messages, limit, effort):
    body = dict(model="inkling-2-mini", messages=messages, max_tokens=limit, temperature=0,
                reasoning_effort=effort, stream=True)
    req = Request(url.rstrip("/") + "/v1/chat/completions", json.dumps(body).encode(),
                  {"Content-Type": "application/json"})
    start = time.monotonic()
    first = None
    content = []
    with urlopen(req, timeout=1800) as response:
        for line in response:
            if not line.startswith(b"data:") or line.strip().endswith(b"[DONE]"):
                continue
            event = json.loads(line[5:])
            if "error" in event:
                raise RuntimeError(event["error"])
            choice, = event["choices"]
            delta = choice["delta"]
            if delta.get("content") or delta.get("reasoning_content"):
                if first is None:
                    first = time.monotonic() - start
            if delta.get("content"):
                content.append(delta["content"])
            if choice["finish_reason"] is not None:
                return dict(answer="".join(content), finish=choice["finish_reason"], ttft_s=first,
                            elapsed_s=round(time.monotonic()-start, 3), usage=event["usage"], timings=event.get("timings"))
    raise ValueError("stream ended without usage")


def match_ratio(text, expected, prefix):
    if not text.startswith(prefix):
        return 0.
    return SequenceMatcher(None, text[len(prefix):], expected[len(prefix):]).ratio()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--mrcr", type=Path, nargs="+", help="OpenAI MRCR rows, each saved as a JSON object")
    parser.add_argument("--mmlu", type=Path)
    parser.add_argument("--code", type=Path, help="repeat four HumanEval+ tasks on the long histories")
    parser.add_argument("--serial", action="store_true", help="finish one archive before starting the next")
    args = parser.parse_args()
    if not args.mrcr and not args.mmlu and not args.code:
        parser.error("supply at least one question set")
    questions = [json.loads(line) for line in args.mmlu.read_text().splitlines()] if args.mmlu else []
    chats = []
    if args.mrcr:
        for p in args.mrcr:
            row = json.loads(p.read_text())
            assert row["n_needles"] == 8 and row["answer"].startswith(row["random_string_to_prepend"])
            chats.append((json.loads(row["prompt"]), row))
    else:
        chats.append(([], None))
    coding = problems(args.code)[:4] if args.code else []

    def conversation(i, messages, row):
        if row:
            result = completion(args.url, messages, 768, "none")
            prefix = row["random_string_to_prepend"]
            yield dict(case="retrieval", chat=i, ratio=match_ratio(result["answer"], row["answer"], prefix), **result)
            # Use the published reference turn so all later input is identical across formats.
            messages = messages + [dict(role="assistant", content=row["answer"])]
        for j in range(i, len(questions), len(chats)):
            q = questions[j]
            choices = "\n".join(f"{letter}. {s}" for letter, s in zip("ABCD", q["choices"]))
            prompt = q["question"] + "\n\n" + choices + "\n\nReply with the single letter of the correct answer."
            result = completion(args.url, messages+[dict(role="user", content=prompt)], 8, "none")
            answer = re.fullmatch(r"\(?([A-D])\)?[.]?", result["answer"].strip())
            expected = "ABCD"[q["answer"]]
            yield dict(case="mmlu", chat=i, subject=q["subject"], index=q["index"], expected=expected,
                       correct=bool(answer and answer[1] == expected), **result)
        for j in range(i, len(coding), len(chats)):
            q = coding[j]
            prompt = "Implement this Python function. Return the complete runnable Python code in one code block.\n\n" + q["prompt"]
            result = completion(args.url, messages+[dict(role="user", content=prompt)], 2048, "none")
            yield dict(case="code", chat=i, task_id=q["task_id"], **result)

    runs = [conversation(i, messages, row) for i,(messages,row) in enumerate(chats)]
    if args.serial:
        for run in runs:
            for rec in run:
                print(json.dumps(rec), flush=True)
    else:
        while runs:
            for run in list(runs):
                rec = next(run, None)
                if rec is None:
                    runs.remove(run)
                else:
                    print(json.dumps(rec), flush=True)


if __name__ == "__main__":
    main()
