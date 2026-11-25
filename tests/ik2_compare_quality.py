"""Compare paired result JSONLs from ik2_kv_quality, ik2_gsm8k, ik2_feasibility or graded ik2_coding."""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from statistics import mean


def results(path):
    groups = defaultdict(dict)
    for line in path.read_text().splitlines():
        row = json.loads(line)
        if "task_id" in row:
            case, key = "coding", row["task_id"]
        elif "subject" in row:
            case, key = "mmlu", (row["subject"], row["index"])
        elif "question" in row:
            case, key = "gsm8k", row["question"]
        elif "pair" in row:
            case, key = "feasibility", (row["pair"], row["expected"])
        else:
            case, key = row["case"], row["chat"]
        if key in groups[case]:
            raise ValueError(f"{path}: duplicate {case} row {key}")
        groups[case][key] = row
    if not groups:
        raise ValueError(f"{path}: empty result file")
    return groups


def compare(control, candidate):
    if control.keys() != candidate.keys():
        raise ValueError("the files cover different tasks")
    for case, left in control.items():
        right = candidate[case]
        if left.keys() != right.keys():
            raise ValueError(f"{case}: different question IDs")
        scores, capped = [], [0, 0]
        hashed = 0
        for key, a in left.items():
            b = right[key]
            for field in ("request_sha256", "reasoning_effort"):
                if field in a and field in b and a[field] != b[field]:
                    raise ValueError(f"{case} {key}: {field} differs")
            for field in ("expected", "chat"):
                if a.get(field) != b.get(field):
                    raise ValueError(f"{case} {key}: {field} differs")
            if a.get("usage", {}).get("prompt_tokens") != b.get("usage", {}).get("prompt_tokens"):
                raise ValueError(f"{case} {key}: input token counts differ")
            hashed += "request_sha256" in a and "request_sha256" in b
            for i, row in enumerate((a, b)):
                capped[i] += row.get("finish") == "length"
            for field in ("correct", "passed", "ratio"):
                if field in a or field in b:
                    scores.append((key, a[field], b[field]))
                    break
        summary = dict(case=case, items=len(left), matching_request_hashes=hashed)
        if any("finish" in row for row in left.values()):
            summary["length_capped"] = capped
        if not scores and case == "coding":
            yield {**summary, "scoring": "Use ik2_coding.py --score on both files and compare the grader outputs."}
            continue
        if len(scores) != len(left):
            raise ValueError(f"{case}: missing scores")
        metric = mean if case == "retrieval" else sum
        yield {**summary, "metric": "mean_match_ratio" if case == "retrieval" else "correct",
               "control": metric(a for _, a, _ in scores), "candidate": metric(b for _, _, b in scores),
               "improved": [key for key, a, b in scores if b > a], "declined": [key for key, a, b in scores if b < a]}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("control", type=Path)
    p.add_argument("candidate", type=Path)
    args = p.parse_args()
    for comparison in compare(results(args.control), results(args.candidate)):
        print(json.dumps(comparison))
