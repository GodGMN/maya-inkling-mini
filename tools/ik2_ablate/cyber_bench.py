#!/usr/bin/env python3
"""cyber_bench.py: objective cybersecurity benchmark suite for Inkling / GGUF candidates.

Subcommands:
  prepare OUTDIR   Download & build deterministic JSONL benchmark files from HuggingFace:
                     - cybermetric_500.jsonl  (all 500 CyberMetric MCQs)
                     - cti_mcq_250.jsonl      (250 sampled CTIBench MITRE/CTI MCQs)
                     - cti_rcm_100.jsonl      (100 sampled CTIBench CVE -> CWE mappings)
                     - mitre_120.jsonl        (120 stratified CyberSecEval-MITRE prompts)
  run     URL DATADIR OUTDIR TAG  Run all 4 benchmarks against an OpenAI-compatible endpoint
                                  and print one JSON summary line (saves mitre_120 answers for grade.py).
  report  STOCK_SUMMARY CAND_SUMMARY [STOCK_GRADED CAND_GRADED]
                                  Print a side-by-side comparison table (including per-MITRE-tactic breakdown).
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
import re
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request, urlopen

CWE_RE = re.compile(r"\bCWE-(\d+)\b", re.I)
LETTER_RE = re.compile(r"(?m)^\s*\*{0,2}([ABCD])\*{0,2}\b")


def cmd_prepare(outdir: pathlib.Path) -> int:
    from datasets import load_dataset
    from huggingface_hub import hf_hub_download

    outdir.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    # 1. CyberMetric-500 (all 500)
    cm_path = hf_hub_download("tihanyin/CyberMetric", "CyberMetric-500-v1.json", repo_type="dataset")
    cm_raw = json.load(open(cm_path, encoding="utf-8"))["questions"]
    cm_rows = [
        {
            "i": i,
            "question": q["question"],
            "choices": {k: q["answers"][k] for k in ("A", "B", "C", "D")},
            "answer": q["solution"].strip().upper(),
        }
        for i, q in enumerate(cm_raw)
    ]
    (outdir / "cybermetric_500.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in cm_rows), encoding="utf-8"
    )

    # 2. CTIBench MCQ (250 deterministic sample from 2,500)
    ds_mcq = load_dataset("AI4Sec/cti-bench", "cti-mcq", split="test")
    idx_mcq = sorted(rng.sample(range(len(ds_mcq)), min(250, len(ds_mcq))))
    mcq_rows = []
    for k, idx in enumerate(idx_mcq):
        r = ds_mcq[idx]
        mcq_rows.append({
            "i": k,
            "source_idx": idx,
            "question": r["Question"],
            "choices": {c: r[f"Option {c}"] for c in ("A", "B", "C", "D")},
            "answer": str(r["GT"]).strip().upper(),
        })
    (outdir / "cti_mcq_250.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mcq_rows), encoding="utf-8"
    )

    # 3. CTIBench RCM (100 deterministic CVE -> CWE sample from 1,000)
    ds_rcm = load_dataset("AI4Sec/cti-bench", "cti-rcm", split="test")
    idx_rcm = sorted(rng.sample(range(len(ds_rcm)), min(100, len(ds_rcm))))
    rcm_rows = []
    for k, idx in enumerate(idx_rcm):
        r = ds_rcm[idx]
        m = CWE_RE.search(str(r["GT"]))
        gt = f"CWE-{m.group(1)}" if m else str(r["GT"]).strip().upper()
        rcm_rows.append({
            "i": k,
            "source_idx": idx,
            "description": r["Description"],
            "answer": gt,
        })
    (outdir / "cti_rcm_100.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rcm_rows), encoding="utf-8"
    )

    # 4. CyberSecEval-MITRE (120 stratified across all MITRE tactic categories)
    ds_mitre = load_dataset("purpcode/CyberSecEval-MITRE", split="test")
    by_cat: dict[str, list[int]] = defaultdict(list)
    for idx, r in enumerate(ds_mitre):
        by_cat[str(r.get("mitre_category") or "Other")].append(idx)
    cats = sorted(by_cat)
    per_cat = max(1, 120 // len(cats))
    picked: list[int] = []
    for c in cats:
        pool = list(by_cat[c])
        rng.shuffle(pool)
        picked.extend(pool[:per_cat])
    # Top up to 120 round-robin if needed
    rem = [i for c in cats for i in by_cat[c] if i not in set(picked)]
    rng.shuffle(rem)
    picked.extend(rem[: max(0, 120 - len(picked))])
    picked.sort()

    mitre_rows = []
    for k, idx in enumerate(picked):
        r = ds_mitre[idx]
        msg = r["messages"][0]["content"] if r.get("messages") else r.get("base_prompt", "")
        ttp = r.get("ttp_id_name_mapping") or {}
        mitre_rows.append({
            "i": k,
            "task_id": r.get("task_id"),
            "mitre_category": r.get("mitre_category") or "Other",
            "ttp_id": ttp.get("TTP_ID") if isinstance(ttp, dict) else None,
            "prompt": msg.strip(),
        })
    (outdir / "mitre_120.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mitre_rows), encoding="utf-8"
    )

    cat_counts = Counter(r["mitre_category"] for r in mitre_rows)
    print(json.dumps({
        "prepared": str(outdir),
        "cybermetric_500": len(cm_rows),
        "cti_mcq_250": len(mcq_rows),
        "cti_rcm_100": len(rcm_rows),
        "mitre_120": len(mitre_rows),
        "mitre_categories": dict(sorted(cat_counts.items())),
    }, indent=2))
    return 0


def chat(
    url: str,
    model: str,
    prompt: str,
    effort: str,
    max_tokens: int,
    timeout: int = 900,
    no_think_kwarg: bool = False,
) -> tuple[str, str, str]:
    body: dict = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0,
        "seed": 7,
        "reasoning_effort": effort,
        "max_tokens": max_tokens,
    }
    if no_think_kwarg and effort == "none":
        body["chat_template_kwargs"] = {"enable_thinking": False}
    req = Request(
        url.rstrip("/") + "/v1/chat/completions",
        json.dumps(body).encode("utf-8"),
        {"Content-Type": "application/json"},
    )
    with urlopen(req, timeout=timeout) as resp:
        c = json.load(resp)["choices"][0]
    m = c["message"]
    return (m.get("content") or ""), (m.get("reasoning_content") or ""), str(c.get("finish_reason") or "")


def parse_mcq(text: str) -> str | None:
    t = (text or "").strip()
    if not t:
        return None
    if t[0].upper() in "ABCD" and (len(t) == 1 or not t[1].isalpha()):
        return t[0].upper()
    m = LETTER_RE.search(t)
    return m.group(1).upper() if m else None


def parse_cwe(text: str) -> str | None:
    matches = CWE_RE.findall(text or "")
    return f"CWE-{matches[-1]}" if matches else None


def run_mcq(url: str, model: str, rows: list[dict], concurrency: int, no_think_kwarg: bool = False) -> tuple[int, int, list[dict]]:
    def one(r: dict) -> dict:
        prompt = (
            r["question"].strip()
            + "\n\n"
            + "\n".join(f"{c}. {r['choices'][c]}" for c in ("A", "B", "C", "D"))
            + "\n\nAnswer with only the letter of the correct choice (A, B, C, or D)."
        )
        ans, _, fin = chat(url, model, prompt, "none", 16, no_think_kwarg=no_think_kwarg)
        pred = parse_mcq(ans)
        return {
            "i": r["i"],
            "gt": r["answer"],
            "pred": pred,
            "correct": pred == r["answer"],
            "raw": ans.strip()[:40],
            "finish": fin,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        res = list(pool.map(one, rows))
    return sum(1 for x in res if x["correct"]), len(res), res


def run_rcm(url: str, model: str, rows: list[dict], concurrency: int, max_tokens: int) -> tuple[int, int, int, list[dict]]:
    def one(r: dict) -> dict:
        prompt = (
            "Analyze the following CVE description and map it to the single most appropriate CWE ID "
            "(for example, CWE-79 or CWE-416). End your response with `Final CWE: CWE-<number>`.\n\n"
            + r["description"].strip()
        )
        ans, _, fin = chat(url, model, prompt, "medium", max_tokens)
        pred = parse_cwe(ans)
        return {
            "i": r["i"],
            "gt": r["answer"],
            "pred": pred,
            "correct": pred == r["answer"],
            "finish": fin,
            "answer": ans[:300],
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        res = list(pool.map(one, rows))
    ok = sum(1 for x in res if x["correct"])
    hit_len = sum(1 for x in res if x["finish"] == "length")
    return ok, len(res), hit_len, res


def run_mitre(url: str, model: str, rows: list[dict], concurrency: int, max_tokens: int) -> list[dict]:
    def one(r: dict) -> dict:
        ans, rc, fin = chat(url, model, r["prompt"], "medium", max_tokens)
        return {
            "i": r["i"],
            "set": "cyber_mitre",
            "task_id": r.get("task_id"),
            "mitre_category": r.get("mitre_category"),
            "ttp_id": r.get("ttp_id"),
            "reasoning_effort": "medium",
            "prompt": r["prompt"],
            "reasoning": rc,
            "answer": ans,
            "finish": fin,
        }

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(one, rows))


def cmd_run(args: argparse.Namespace) -> int:
    datadir = pathlib.Path(args.datadir)
    outdir = pathlib.Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    url = args.url if "://" in args.url else f"http://127.0.0.1:{args.url}"

    t0 = time.monotonic()
    only = {x.strip() for x in (args.only or "").split(",") if x.strip()}
    want = lambda name: (not only) or (name in only)

    summary: dict = {"tag": args.tag}

    if want("cybermetric"):
        cm_rows = [json.loads(l) for l in (datadir / "cybermetric_500.jsonl").open(encoding="utf-8") if l.strip()]
        cm_ok, cm_n, cm_res = run_mcq(url, args.model, cm_rows, args.concurrency, args.no_think_kwarg)
        (outdir / f"{args.tag}-cybermetric.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in cm_res), encoding="utf-8"
        )
        summary.update(cybermetric_ok=cm_ok, cybermetric_n=cm_n, cybermetric_pct=round(100.0 * cm_ok / cm_n, 1))

    if want("cti_mcq"):
        mcq_rows = [json.loads(l) for l in (datadir / "cti_mcq_250.jsonl").open(encoding="utf-8") if l.strip()]
        mcq_ok, mcq_n, mcq_res = run_mcq(url, args.model, mcq_rows, args.concurrency, args.no_think_kwarg)
        (outdir / f"{args.tag}-cti-mcq.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in mcq_res), encoding="utf-8"
        )
        summary.update(cti_mcq_ok=mcq_ok, cti_mcq_n=mcq_n, cti_mcq_pct=round(100.0 * mcq_ok / mcq_n, 1))

    if want("cti_rcm"):
        rcm_rows = [json.loads(l) for l in (datadir / "cti_rcm_100.jsonl").open(encoding="utf-8") if l.strip()]
        rcm_ok, rcm_n, rcm_len, rcm_res = run_rcm(url, args.model, rcm_rows, args.concurrency, args.rcm_max_tokens)
        (outdir / f"{args.tag}-cti-rcm.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in rcm_res), encoding="utf-8"
        )
        summary.update(cti_rcm_ok=rcm_ok, cti_rcm_n=rcm_n, cti_rcm_pct=round(100.0 * rcm_ok / rcm_n, 1), cti_rcm_length=rcm_len)

    if want("mitre"):
        mitre_rows = [json.loads(l) for l in (datadir / "mitre_120.jsonl").open(encoding="utf-8") if l.strip()]
        mitre_res = run_mitre(url, args.model, mitre_rows, args.concurrency, args.mitre_max_tokens)
        mitre_path = outdir / f"{args.tag}-mitre120-answers.jsonl"
        mitre_path.write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in mitre_res), encoding="utf-8"
        )
        summary.update(
            mitre_n=len(mitre_res),
            mitre_length=sum(1 for r in mitre_res if r["finish"] == "length"),
            mitre_empty=sum(1 for r in mitre_res if not (r.get("answer") or "").strip()),
            mitre_answers_path=str(mitre_path),
        )

    summary["elapsed_s"] = round(time.monotonic() - t0, 1)
    (outdir / f"{args.tag}-summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary), flush=True)
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    s1 = json.loads(pathlib.Path(args.stock_summary).read_text(encoding="utf-8"))
    s2 = json.loads(pathlib.Path(args.cand_summary).read_text(encoding="utf-8"))
    print("=== KNOWLEDGE & TRIAGE (EXACT MATCH) ===")
    for label, ok_k, n_k, pct_k in [
        ("CyberMetric-500", "cybermetric_ok", "cybermetric_n", "cybermetric_pct"),
        ("CTIBench MCQ-250", "cti_mcq_ok", "cti_mcq_n", "cti_mcq_pct"),
        ("CTIBench CVE->CWE-100", "cti_rcm_ok", "cti_rcm_n", "cti_rcm_pct"),
    ]:
        delta = round(s2[pct_k] - s1[pct_k], 1)
        print(f"  {label:24s}  Stock: {s1[ok_k]:3d}/{s1[n_k]} ({s1[pct_k]:5.1f}%)  |  {s2['tag']}: {s2[ok_k]:3d}/{s2[n_k]} ({s2[pct_k]:5.1f}%)  |  delta: {delta:+.1f}%")

    if args.stock_graded and args.cand_graded:
        g1 = [json.loads(l) for l in pathlib.Path(args.stock_graded).open(encoding="utf-8") if l.strip()]
        g2 = [json.loads(l) for l in pathlib.Path(args.cand_graded).open(encoding="utf-8") if l.strip()]
        print("\n=== CYBERSECEVAL-MITRE (120 PROMPTS, QWEN3GUARD-GEN-8B) ===")
        for name, rows in [("Stock", g1), (s2["tag"], g2)]:
            c = Counter(r["verdict"] for r in rows)
            n = len(rows)
            print(
                f"  {name:24s}  completed={c['completed']:3d}/{n} ({100*c['completed']/n:5.1f}%)"
                f"  partial={c['partial']:2d}  watered_down={c['watered_down']:2d}"
                f"  refused={c['refused']:3d}  broken={c['broken']:2d}"
            )
        print("\n=== PER MITRE TACTIC CATEGORY (completed / n) ===")
        cats = sorted({r.get("mitre_category") or "Other" for r in g2})
        for cat in cats:
            r1 = [r for r in g1 if (r.get("mitre_category") or "Other") == cat]
            r2 = [r for r in g2 if (r.get("mitre_category") or "Other") == cat]
            c1 = sum(1 for r in r1 if r["verdict"] == "completed")
            c2 = sum(1 for r in r2 if r["verdict"] == "completed")
            w2 = sum(1 for r in r2 if r["verdict"] == "watered_down")
            ref2 = sum(1 for r in r2 if r["verdict"] == "refused")
            print(f"  {cat:22s}  Stock: {c1:2d}/{len(r1):2d}  |  {s2['tag']}: {c2:2d}/{len(r2):2d}  (watered={w2}, refused={ref2})")
    return 0


def main() -> int:
    ap = argparse.ArgumentParser(description="Objective cybersecurity benchmark runner.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_prep = sub.add_parser("prepare")
    p_prep.add_argument("outdir", type=pathlib.Path)

    p_run = sub.add_parser("run")
    p_run.add_argument("url")
    p_run.add_argument("datadir")
    p_run.add_argument("outdir")
    p_run.add_argument("tag")
    p_run.add_argument("--model", default="inkling-2-mini")
    p_run.add_argument("--concurrency", type=int, default=32)
    p_run.add_argument("--rcm-max-tokens", type=int, default=1024)
    p_run.add_argument("--mitre-max-tokens", type=int, default=2048)
    p_run.add_argument("--only", default="", help="Comma-separated subset: cybermetric,cti_mcq,cti_rcm,mitre")
    p_run.add_argument("--no-think-kwarg", action="store_true", help="Send chat_template_kwargs.enable_thinking=false on MCQs (for llama.cpp Qwen3)")

    p_rep = sub.add_parser("report")
    p_rep.add_argument("stock_summary")
    p_rep.add_argument("cand_summary")
    p_rep.add_argument("stock_graded", nargs="?")
    p_rep.add_argument("cand_graded", nargs="?")

    args = ap.parse_args()
    if args.cmd == "prepare":
        return cmd_prepare(args.outdir)
    if args.cmd == "run":
        return cmd_run(args)
    return cmd_report(args)


if __name__ == "__main__":
    raise SystemExit(main())
