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
    parser.add_argument("--prefix-tokens", type=int, default=0, help="exact length of a shared archive before the questions")
    parser.add_argument("--background", type=Path, help="JSONL with distractor text in a question field")
    parser.add_argument("--min-cached", type=int, default=0, help="stop scoring if a question reuses fewer tokens")
    parser.add_argument("--max-context", type=int, help="check that every question and its eight output tokens fit before warming the archive")
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--score", type=Path, help="stdout from the prepared Maya requests")
    mode.add_argument("--llama-url")
    args = parser.parse_args()
    if bool(args.prefix_tokens) != bool(args.background) or not 0 <= args.min_cached <= args.prefix_tokens:
        parser.error("supply --prefix-tokens with --background, and keep --min-cached within that prefix")
    tok = Tokenizer.from_dir(args.tokenizer)
    stop, = tok.encode(STOP, parse_special=True)
    template = Environment().from_string((ROOT / "tools/ik2/chat_template.jinja").read_text())
    data = [json.loads(line) for line in args.data.read_text().splitlines()]
    prefix = []
    if args.background:
        content = "The archive below is background material. Answer the question after the archive.\n\n@ARCHIVE@"
        rendered = template.render(messages=[dict(role="user", content=content)], tools=[],
                                   add_generation_prompt=True, reasoning_effort="none")
        before, after = rendered.split("@ARCHIVE@")
        prefix = tok.encode(before, parse_special=True)
        fill = tok.encode("\n".join(json.loads(line)["question"] for line in args.background.read_text().splitlines()))
        length = args.prefix_tokens - len(prefix)
        if length < 0 or not fill:
            parser.error("the prefix is too short or the archive is empty")
        prefix.extend((fill * ((length + len(fill) - 1) // len(fill)))[:length])
    questions = []
    for i, row in enumerate(data):
        if len(row["choices"]) != 4 or row["answer"] not in range(4):
            raise ValueError(f"invalid choices or label in row {i}")
        options = "\n".join(f"{letter}. {choice}" for letter, choice in zip("ABCD", row["choices"]))
        question = f'{row["question"]}\n\n{options}\n\nReply with the single letter of the correct answer.'
        if prefix:
            text = "\n\nQuestion:\n" + question + after + FINAL
        else:
            text = template.render(messages=[dict(role="user", content=question)], tools=[],
                                   add_generation_prompt=True, reasoning_effort="none") + FINAL
        tokens = tok.encode(text, parse_special=True)
        required = len(prefix) + len(tokens) + 8
        if args.max_context is not None and required > args.max_context:
            parser.error(f"row {i} needs {required} context positions including eight output tokens, limit {args.max_context}")
        questions.append(tokens)
    if prefix and args.llama_url:
        warm = llama_completion(args.llama_url, prefix, 1, reuse=True)
        if warm[1] != len(prefix):
            raise ValueError("incomplete archive prefill")
    elif prefix and not args.score:
        print("GEN 1 0 1 1 - " + ",".join(map(str, prefix)))
    if args.score:
        with args.score.open() as f:
            runs = list(responses(f))
        if prefix:
            if not runs or runs[0][1] != len(prefix) or len(runs[0][0]) != 1:
                raise ValueError("incomplete archive prefill")
            runs = runs[1:]
        if len(runs) != len(data):
            raise ValueError(f"expected {len(data)} responses, got {len(runs)}")
    correct = invalid = 0
    for i, (row, tokens) in enumerate(zip(data, questions)):
        ids = prefix + tokens
        if args.llama_url:
            result = llama_completion(args.llama_url, ids, 8, reuse=bool(prefix))
        elif args.score:
            result = runs[i]
        else:
            print(f"GEN 8 0 1 1 {stop} " + ",".join(map(str, ids)))
            continue
        tokens, seen, cached, finish, _, _ = result
        if seen != len(ids) or not tokens:
            raise ValueError(f"invalid response to row {i}")
        if cached < args.min_cached:
            raise ValueError(f"row {i} reused {cached} tokens, required {args.min_cached}")
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
