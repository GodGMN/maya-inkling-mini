"""Prepare identical cold requests on CPU, then compare greedy dense and sparse decode on one idle GPU."""
import argparse
from difflib import SequenceMatcher
import hashlib
import json
import os
from pathlib import Path
from statistics import median
import subprocess

from jinja2 import Environment

from ik2_retrieval import FINAL, ROOT, STOP, Tokenizer, requests, responses


def sha(data):
    return hashlib.sha256(data).hexdigest()


def prepare(args):
    tok = Tokenizer.from_dir(args.tokenizer)
    stop, = tok.encode(STOP, parse_special=True)
    corpus = "\n".join(json.loads(row)["question"] for row in args.data.read_text().splitlines())
    args.fixtures.mkdir(parents=True, exist_ok=False)
    cases = []

    def save(name, ids, expected, limit, direct):
        if len(ids) + max(limit, 130) + 1 > args.context:
            raise ValueError(f"{name} exceeds the reserved context")
        payload = ",".join(map(str, ids))
        # The fixed-length branch has one new prompt token, including on a short answer or EOS.
        replay = "GEN 129 0 1 1 - " + payload + ",371\n"
        data = (f"GEN {limit} 0 1 1 {stop} {payload}\n" + replay * 3 + "QUIT\n").encode()
        (args.fixtures / (name + ".request")).write_bytes(data)
        cases.append(dict(name=name, depth=len(ids), expected=expected, limit=limit, direct=direct, sha256=sha(data)))

    for depth in (8192, 32769, 131072, 258048):
        for percent in (5, 50, 95):
            expected, ids = next(requests(tok, [depth], percent, corpus, True))
            save(f"phrase-{depth}-{percent}", ids, expected, 64, True)
        expected, ids = next(requests(tok, [depth], 50, corpus, inventory=True))
        save(f"inventory-{depth}", ids, expected, 1024, False)

    template = Environment().from_string((ROOT / "tools/ik2/chat_template.jinja").read_text())
    for path in args.mrcr:
        row = json.loads(path.read_text())
        prompt = template.render(messages=json.loads(row["prompt"]), tools=[], add_generation_prompt=True,
                                 reasoning_effort="none") + FINAL
        save("mrcr-" + path.stem, tok.encode(prompt, parse_special=True), row["answer"], 1536, True)
    if len({c["name"] for c in cases}) != len(cases):
        raise ValueError("duplicate fixture name")
    manifest = dict(context=args.context, cases=cases)
    (args.fixtures / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Prepared", len(cases), "paired cases at", args.fixtures)


def score_log(log, case, tok):
    quality, *replays = list(responses(log.splitlines()))
    ids, seen, cached, finish, pp, _ = quality
    if not ids or finish not in ("stop", "length") or cached or seen != case["depth"] or pp <= 0 or len(replays) != 3:
        raise ValueError(f"incomplete or non-cold quality request: {case['name']}")
    text = tok.decode(ids)
    if case["direct"]:
        text = FINAL + text
    answer = text.rsplit(FINAL, 1)[-1].split("<|end_message|>")[0].split(STOP)[0].strip() if FINAL in text else ""
    expected = case["expected"]
    exact = answer == expected if case["name"].startswith("mrcr-") else answer.lower().rstrip(".") == expected
    rates = []
    for output, depth, _, reason, _, seconds in replays:
        if reason != "length" or depth != seen + 1 or len(output) != 129 or seconds <= 0:
            raise ValueError(f"incomplete 128-step timing: {case['name']}")
        rates.append(128 / seconds)
    similarity = SequenceMatcher(None, expected, answer, autojunk=False).ratio()
    return dict(exact=exact, answer=answer, similarity=similarity, finish=finish, generated=len(ids), first_id=ids[0],
                tokens_sha256=sha(json.dumps(ids).encode()), prefill_s=pp,
                fixed_decode_tok_s=rates, median_decode_tok_s=median(rates))


def compare(args):
    manifest = json.loads((args.fixtures / "manifest.json").read_text())
    tok = Tokenizer.from_dir(args.tokenizer)
    preset = ROOT / "docs/inkling2-mini/ik2-mini-gfx1100.json"
    env = {k: v for k, v in os.environ.items() if not k.startswith("IK2_")}
    env.update(json.loads(preset.read_text())["env"])
    cmd = [str(args.engine.resolve()), "--model", str(args.model.resolve()), "--serve-ik2",
           "--ctx", str(manifest["context"]), "--chunk", "1024"]
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "settings.json").write_text(json.dumps(dict(command=cmd, preset_env=json.loads(preset.read_text())["env"],
        engine_sha256=sha(args.engine.read_bytes()), fixtures_sha256=sha((args.fixtures / "manifest.json").read_bytes())), indent=2))
    rows = []
    for i, case in enumerate(manifest["cases"]):
        data = (args.fixtures / (case["name"] + ".request")).read_bytes()
        if sha(data) != case["sha256"]:
            raise ValueError(f"fixture changed: {case['name']}")
        pair = {}
        for mode in (("dense", "sparse") if i % 2 == 0 else ("sparse", "dense")):
            base = args.output / f"{case['name']}.{mode}"
            with open(str(base) + ".log", "wb") as log, open(str(base) + ".err", "wb") as err:
                subprocess.run(cmd, input=data, stdout=log, stderr=err, check=True,
                               env={**env, "IK2_SPARSE_DECODE": str(int(mode == "sparse"))})
            marker = "sparse global decode active at position "
            positions = [int(line[len(marker):]) for line in Path(str(base) + ".err").read_text().splitlines()
                         if line.startswith(marker)]
            # A one-token prompt tail may use the fast path before generation starts.
            if positions and positions[0] < case["depth"]:
                raise RuntimeError(f"sparse dispatch during cold prefill for {base.name}")
            if bool(positions) != (mode == "sparse" and case["depth"] >= 32768):
                raise RuntimeError(f"unexpected sparse dispatch for {base.name}")
            pair[mode] = score_log(Path(str(base) + ".log").read_text(), case, tok)
        dense, sparse = pair["dense"], pair["sparse"]
        if dense["first_id"] != sparse["first_id"]:
            raise RuntimeError(f"dense-prefilled first prediction differs: {case['name']}")
        if case["depth"] + case["limit"] < 32768 and dense["tokens_sha256"] != sparse["tokens_sha256"]:
            raise RuntimeError(f"short dense control diverged: {case['name']}")
        row = dict(case=case["name"], depth=case["depth"], expected=case["expected"], **pair,
                   ratio=sparse["median_decode_tok_s"] / dense["median_decode_tok_s"])
        rows.append(row)
        with (args.output / "pairs.jsonl").open("a") as f:
            print(json.dumps(row), file=f)
        print(f"{row['case']}  exact {int(dense['exact'])} → {int(sparse['exact'])}  "
              f"decode {dense['median_decode_tok_s']:.1f} → {sparse['median_decode_tok_s']:.1f} tok/s", flush=True)
    summary = dict(cases=len(rows), dense_correct=sum(r["dense"]["exact"] for r in rows),
                   sparse_correct=sum(r["sparse"]["exact"] for r in rows),
                   lost=[r["case"] for r in rows if r["dense"]["exact"] and not r["sparse"]["exact"]],
                   gained=[r["case"] for r in rows if r["sparse"]["exact"] and not r["dense"]["exact"]],
                   both_missed=[r["case"] for r in rows if not (r["dense"]["exact"] or r["sparse"]["exact"])])
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)
    return 1 if summary["lost"] or not any(r["depth"] >= 32768 and r["dense"]["exact"] for r in rows) else 0


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="action", required=True)
    prep = sub.add_parser("prepare")
    run = sub.add_parser("run")
    for p in (prep, run):
        p.add_argument("--fixtures", required=True, type=Path)
        p.add_argument("--tokenizer", required=True, type=Path)
    prep.add_argument("--data", required=True, type=Path)
    prep.add_argument("--mrcr", nargs="*", type=Path, default=[])
    prep.add_argument("--context", type=int, default=393216)
    run.add_argument("--engine", required=True, type=Path)
    run.add_argument("--model", required=True, type=Path)
    run.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    raise SystemExit(prepare(args) if args.action == "prepare" else compare(args))
