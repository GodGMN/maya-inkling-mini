"""Compare direct-answer multiple-choice predictions from Maya and llama.cpp on identical chat tokens."""
import argparse
import json
from pathlib import Path
import re

from jinja2 import Environment

from ik2_retrieval import FINAL, ROOT, STOP, Tokenizer, llama_completion, responses


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tokenizer", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="JSONL rows with subject, index, question, choices, answer")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--score", type=Path, help="stdout from the prepared Maya requests")
    mode.add_argument("--llama-url")
    args = parser.parse_args()
    tok = Tokenizer.from_dir(args.tokenizer)
    stop, = tok.encode(STOP, parse_special=True)
    template = Environment().from_string((ROOT / "tools/ik2/chat_template.jinja").read_text())
    data = [json.loads(line) for line in args.data.read_text().splitlines()]
    if args.score:
        with args.score.open() as f:
            runs = list(responses(f))
        if len(runs) != len(data):
            raise ValueError(f"expected {len(data)} responses, got {len(runs)}")
    correct = invalid = 0
    for i, row in enumerate(data):
        if len(row["choices"]) != 4 or row["answer"] not in range(4):
            raise ValueError(f"invalid choices or label in row {i}")
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"]))
        question = f'{row["question"]}\n\n{options}\n\nReply with the single letter of the correct answer.'
        chat = template.render(messages=[dict(role="user", content=question)], tools=[],
                               add_generation_prompt=True, reasoning_effort="none")
        ids = tok.encode(chat + FINAL, parse_special=True)
        if args.llama_url:
            result = llama_completion(args.llama_url, ids, 8)
        elif args.score:
            result = runs[i]
        else:
            print(f"GEN 8 0 1 1 {stop} " + ",".join(map(str, ids)))
            continue
        tokens, seen, cached, finish, _, _ = result
        if seen != len(ids) or not tokens:
            raise ValueError(f"invalid response to row {i}")
        content = tok.decode(tokens).partition("<|end_message|>")[0].strip()
        match = re.fullmatch(r"\(?([A-D])\)?[.]?", content)
        answer = match[1] if match else None
        expected = "ABCD"[row["answer"]]
        correct += answer == expected
        invalid += answer is None
        print(json.dumps(dict(subject=row["subject"], index=row["index"], expected=expected, predicted=answer,
                              content=content, finish=finish, cached=cached, correct=answer == expected)), flush=True)
    if args.score or args.llama_url:
        print(json.dumps(dict(correct=correct, total=len(data), invalid=invalid)))
    else:
        print("QUIT")


if __name__ == "__main__":
    main()
