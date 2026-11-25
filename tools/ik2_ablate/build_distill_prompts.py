#!/usr/bin/env python3
"""build_distill_prompts.py: build an eval-disjoint prompt pool for Stage 3 hybrid teacher SFT distillation.

Strictly excludes ALL benchmark & test items:
  - CyberSecEval-MITRE: excludes all 120 task_ids in mitre_120.jsonl (samples 320 from remaining 880, 32/tactic)
  - CTIBench RCM (CVE->CWE): excludes all 100 source_idxs in cti_rcm_100.jsonl (samples 100 from remaining 900)
  - CTIBench MCQ: excludes all 250 source_idxs in cti_mcq_250.jsonl (samples 80 from remaining 2,250)
  - AdvBench / Harmless: uses ONLY *_train.jsonl (never *_test.jsonl)
  - Coding: uses walledai/CyberSecEval[instruct] (zero overlap with all 164 HumanEval+ tasks!)
  - Agentic/Pi discipline: includes concise multi-step code-audit / debugging / tool-style instructions.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import random
from collections import Counter, defaultdict


def main() -> int:
    ap = argparse.ArgumentParser(description="Build 100% eval-disjoint Stage 3 distillation prompt pool.")
    ap.add_argument("--cyber-eval-dir", type=pathlib.Path, required=True, help="Dir containing mitre_120.jsonl, cti_mcq_250.jsonl, cti_rcm_100.jsonl")
    ap.add_argument("--sets-dir", type=pathlib.Path, default=None, help="Optional dir containing harmful_train.jsonl & harmless_train.jsonl")
    ap.add_argument("--out", type=pathlib.Path, required=True, help="Output JSONL path")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--n-mitre", type=int, default=320, help="Disjoint MITRE prompts (medium reasoning, teacher: qwen38)")
    ap.add_argument("--n-mitre-none", type=int, default=60, help="Disjoint MITRE prompts (reasoning_effort=none, teacher: qwen38)")
    ap.add_argument("--n-harmful", type=int, default=120, help="Disjoint AdvBench harmful_train prompts")
    ap.add_argument("--n-harmless", type=int, default=120, help="Disjoint harmless_train prompts (preserves tone)")
    ap.add_argument("--n-code", type=int, default=160, help="Disjoint CyberSecEval-Instruct coding prompts")
    ap.add_argument("--n-rcm", type=int, default=80, help="Disjoint CTIBench CVE->CWE reasoning prompts")
    ap.add_argument("--n-mcq", type=int, default=60, help="Disjoint CTIBench MCQ prompts (reasoning_effort=none)")
    args = ap.parse_args()

    from datasets import load_dataset

    rng = random.Random(args.seed)
    rows: list[dict] = []

    # 1. Disjoint CyberSecEval-MITRE (stratified across all 10 tactics)
    held_mitre = {json.loads(l)["task_id"] for l in (args.cyber_eval_dir / "mitre_120.jsonl").open(encoding="utf-8") if l.strip()}
    ds_mitre = load_dataset("purpcode/CyberSecEval-MITRE", split="test")
    by_cat: dict[str, list[dict]] = defaultdict(list)
    for r in ds_mitre:
        if r.get("task_id") in held_mitre:
            continue
        by_cat[str(r.get("mitre_category") or "Other")].append(r)

    cats = sorted(by_cat)
    per_cat = max(1, args.n_mitre // len(cats))
    mitre_picked = []
    for c in cats:
        pool = list(by_cat[c])
        rng.shuffle(pool)
        mitre_picked.extend(pool[:per_cat])
    rem = [r for c in cats for r in by_cat[c] if r not in mitre_picked]
    rng.shuffle(rem)
    mitre_picked.extend(rem[: max(0, args.n_mitre - len(mitre_picked))])

    for r in mitre_picked:
        msg = r["messages"][0]["content"] if r.get("messages") else r.get("base_prompt", "")
        rows.append({
            "source": "cyber_mitre_train",
            "teacher_pref": "qwen38",
            "mode": "harmful",
            "reasoning_effort": "medium",
            "mitre_category": r.get("mitre_category") or "Other",
            "task_id": r.get("task_id"),
            "prompt": msg.strip(),
        })

    # 1b. Extra disjoint CyberSecEval-MITRE prompts with reasoning_effort=none so LoRA also unlocks non-thinking mode
    used_tids = {r.get("task_id") for r in mitre_picked}
    rem_none = [r for c in cats for r in by_cat[c] if r.get("task_id") not in used_tids]
    rng.shuffle(rem_none)
    for r in rem_none[: args.n_mitre_none]:
        msg = r["messages"][0]["content"] if r.get("messages") else r.get("base_prompt", "")
        rows.append({
            "source": "cyber_mitre_none_train",
            "teacher_pref": "qwen38",
            "mode": "harmful",
            "reasoning_effort": "none",
            "mitre_category": r.get("mitre_category") or "Other",
            "task_id": r.get("task_id"),
            "prompt": msg.strip(),
        })

    # 2. Disjoint AdvBench harmful_train & harmless_train (if sets-dir provided)
    if args.sets_dir and (args.sets_dir / "harmful_train.jsonl").exists():
        ht = [json.loads(l) for l in (args.sets_dir / "harmful_train.jsonl").open(encoding="utf-8") if l.strip()]
        rng.shuffle(ht)
        for r in ht[: args.n_harmful]:
            p = (r if isinstance(r, str) else (r.get("prompt") or r.get("goal") or "")).strip()
            if p:
                rows.append({
                    "source": "harmful_train",
                    "teacher_pref": "inkling",
                    "mode": "harmful",
                    "reasoning_effort": "medium",
                    "prompt": p,
                })
        hl = [json.loads(l) for l in (args.sets_dir / "harmless_train.jsonl").open(encoding="utf-8") if l.strip()]
        rng.shuffle(hl)
        for r in hl[: args.n_harmless]:
            p = (r if isinstance(r, str) else (r.get("prompt") or r.get("instruction") or "")).strip()
            if p:
                rows.append({
                    "source": "harmless_train",
                    "teacher_pref": "inkling",
                    "mode": "harmless",
                    "reasoning_effort": "medium",
                    "prompt": p,
                })

    # 3. Disjoint secure coding prompts from walledai/CyberSecEval[instruct] (zero overlap with HumanEval+)
    langs = ["python", "c", "cpp", "rust", "javascript"]
    per_lang = max(1, args.n_code // len(langs))
    for lang in langs:
        ds_c = load_dataset("walledai/CyberSecEval", "instruct", split=lang)
        idxs = list(range(len(ds_c)))
        rng.shuffle(idxs)
        for idx in idxs[:per_lang]:
            r = ds_c[idx]
            p = (r.get("prompt") or "").strip()
            if p:
                rows.append({
                    "source": f"code_instruct_{lang}",
                    "teacher_pref": "inkling",
                    "mode": "harmless",
                    "reasoning_effort": "medium",
                    "prompt": p,
                })

    # 4. Disjoint CTIBench RCM (CVE -> CWE reasoning)
    held_rcm = {json.loads(l)["source_idx"] for l in (args.cyber_eval_dir / "cti_rcm_100.jsonl").open(encoding="utf-8") if l.strip()}
    ds_rcm = load_dataset("AI4Sec/cti-bench", "cti-rcm", split="test")
    rcm_idxs = [i for i in range(len(ds_rcm)) if i not in held_rcm]
    rng.shuffle(rcm_idxs)
    for idx in rcm_idxs[: args.n_rcm]:
        r = ds_rcm[idx]
        p = (
            "Analyze the following CVE description and map it to the single most appropriate CWE ID "
            "(for example, CWE-79 or CWE-416). End your response with `Final CWE: CWE-<number>`.\n\n"
            + str(r["Description"]).strip()
        )
        rows.append({
            "source": "cti_rcm_train",
            "teacher_pref": "inkling",
            "mode": "harmless",
            "reasoning_effort": "medium",
            "gt": str(r["GT"]).strip().upper(),
            "prompt": p,
        })

    # 5. Disjoint CTIBench MCQ (reasoning_effort=none -> keeps non-thinking instruction following crisp)
    held_mcq = {json.loads(l)["source_idx"] for l in (args.cyber_eval_dir / "cti_mcq_250.jsonl").open(encoding="utf-8") if l.strip()}
    ds_mcq = load_dataset("AI4Sec/cti-bench", "cti-mcq", split="test")
    mcq_idxs = [i for i in range(len(ds_mcq)) if i not in held_mcq]
    rng.shuffle(mcq_idxs)
    for idx in mcq_idxs[: args.n_mcq]:
        r = ds_mcq[idx]
        p = (
            str(r["Question"]).strip()
            + "\n\n"
            + "\n".join(f"{c}. {r[f'Option {c}']}" for c in ("A", "B", "C", "D"))
            + "\n\nAnswer with only the letter of the correct choice (A, B, C, or D)."
        )
        rows.append({
            "source": "cti_mcq_train",
            "teacher_pref": "inkling",
            "mode": "harmless",
            "reasoning_effort": "none",
            "gt": str(r["GT"]).strip().upper(),
            "prompt": p,
        })

    # Deduplicate by exact prompt text and assign deterministic IDs
    seen = set()
    final_rows = []
    for r in rows:
        key = r["prompt"]
        if key in seen:
            continue
        seen.add(key)
        r["i"] = len(final_rows)
        final_rows.append(r)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in final_rows),
        encoding="utf-8",
    )
    summary = {
        "out": str(args.out),
        "total": len(final_rows),
        "by_source": dict(sorted(Counter(r["source"] for r in final_rows).items())),
        "by_teacher": dict(sorted(Counter(r["teacher_pref"] for r in final_rows).items())),
    }
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
