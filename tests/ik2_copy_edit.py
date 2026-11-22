"""Measure exact copy and single-edit answers on the same Inkling chat requests."""
import argparse
import json
from pathlib import Path

from jinja2 import Environment

from ik2_retrieval import FINAL, ROOT, STOP, Tokenizer, responses


def examples():
    names = ("notebooks", "folders", "maps", "letters")
    lines = ["The archive shelf contains " + ", ".join(
        f"box {j} storing {names[j % 4]}" for j in range(i, i + 9)) + "." for i in range(0, 81, 9)]
    inventory = "\n".join(lines)
    yield "copy", "Copy the following archive inventory verbatim. Return only the inventory, with every line.", inventory, inventory
    yield "edit", "Return this inventory verbatim with exactly one change: replace 'box 46' with 'box forty-six'.", inventory, inventory.replace("box 46", "box forty-six")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tokenizer", required=True, type=Path)
    ap.add_argument("--max-tokens", type=int, default=3072)
    ap.add_argument("--direct-answer", action="store_true", help="prefill the final channel used by the HTTP server")
    ap.add_argument("--score", type=Path, help="stdout of maya-ik2 --serve-ik2")
    args = ap.parse_args()
    if args.max_tokens < 1:
        ap.error("--max-tokens must be positive")
    tok = Tokenizer.from_dir(args.tokenizer)
    end, = tok.encode(STOP, parse_special=True)
    template = Environment().from_string((ROOT / "tools/ik2/chat_template.jinja").read_text())
    cases = list(examples())
    if args.score:
        with args.score.open() as f:
            runs = list(responses(f))
        if len(runs) != len(cases):
            raise ValueError(f"expected {len(cases)} responses, found {len(runs)}")
    for i, (name, question, passage, wanted) in enumerate(cases):
        text = template.render(messages=[dict(role="user", content=question + "\n\n" + passage)], tools=[],
                               add_generation_prompt=True, reasoning_effort="none")
        prompt = tok.encode(text + (FINAL if args.direct_answer else ""), parse_special=True)
        if not args.score:
            print(f"GEN {args.max_tokens} 0 1 1 {end} " + ",".join(map(str, prompt)))
            continue
        ids, seen, cached, finish, pp, tg = runs[i]
        if seen != len(prompt) or cached or finish not in ("stop", "length") or tg <= 0:
            raise ValueError(f"invalid uncached response to {name}: prompt={seen}, cached={cached}, finish={finish}")
        decoded = (FINAL if args.direct_answer else "") + tok.decode(ids)
        answer = decoded.rpartition(FINAL)[2].partition("<|end_message|>")[0].strip() if FINAL in decoded else ""
        print(json.dumps(dict(case=name, exact=answer == wanted, prompt=seen, generated=len(ids),
                              finish=finish, prefill_s=pp, decode_s=tg, decode_tok_s=round((len(ids)-1)/tg, 2))))
    if not args.score:
        print("QUIT")


if __name__ == "__main__":
    main()
