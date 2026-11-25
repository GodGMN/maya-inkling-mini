#!/usr/bin/env python3
"""fit_ortho.py: fit coding/benign-orthogonalized refusal directions for Inkling 2 Mini.

Standard difference-in-means (`d = mean(harmful) - mean(harmless)`) has non-zero overlap with the
benign/coding mean and top coding variation axes, causing runtime or weight projection to degrade
coding and reasoning termination ("reasoning that never ends").

This script builds orthogonalized directions:
  1. `mean`: removes overlap with the harmless (and optional coding) mean direction.
  2. `pcK` (e.g. K=4, 8, 16, 32): removes overlap with the top-K singular vectors of the
     uncentered preserve matrix (coding + harmless states), keeping 0.0 projection onto dominant
     coding/benign directions while checking held-out refusal separation (Cohen's d).

Usage:
  fit_ortho.py LABELS CAPTURE OUTDIR [--visit 21] [--preserve-cap CODE.f32] [--ks 1,4,8,16,32]
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

C = 2304
REC = np.dtype([("visit", "<i4"), ("pos", "<i4"), ("width", "<i4"), ("h", "<f4", (C,))])


def load_labeled_states(lab: pathlib.Path, cap: pathlib.Path, name: str, want_refused: bool) -> np.ndarray:
    data = np.memmap(cap, "u1", "r")
    rows = [json.loads(x) for x in (lab / f"{name}.jsonl").open(encoding="utf-8") if x.strip()]
    out = []
    for r in rows:
        if bool(r["refused"]) != want_refused:
            continue
        span = r["cap1"] - r["cap0"]
        if span != 48 * REC.itemsize:
            continue
        a = np.frombuffer(data[r["cap0"]:r["cap1"]], REC)
        assert np.array_equal(a["visit"], np.arange(48)) and np.all(a["width"] == C)
        out.append(a["h"].astype(np.float64))
    return np.stack(out)


def load_raw_capture(cap: pathlib.Path) -> np.ndarray:
    a = np.fromfile(cap, REC)
    if len(a) == 0 or len(a) % 48 != 0:
        raise ValueError(f"{cap}: expected multiple of 48 records, got {len(a)}")
    n = len(a) // 48
    out = a.reshape(n, 48)
    assert np.all(out["visit"] == np.arange(48)[None, :]) and np.all(out["width"] == C)
    return out["h"].astype(np.float64)


def cohen_d(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.sqrt((a.var() + b.var()) / 2.0))
    return float((a.mean() - b.mean()) / denom) if denom > 1e-12 else 0.0


def orthogonalize(d: np.ndarray, basis_rows: np.ndarray) -> tuple[np.ndarray, float]:
    """Project d orthogonal to orthonormal rows in basis_rows; return (unit_vec, retained_cos)."""
    u0 = d / np.linalg.norm(d)
    proj = u0 - basis_rows.T @ (basis_rows @ u0)
    nrm = float(np.linalg.norm(proj))
    if nrm < 1e-8:
        raise ValueError("direction collapsed after orthogonalization")
    u = proj / nrm
    return u, float(u @ u0)


def main() -> int:
    ap = argparse.ArgumentParser(description="Fit coding/benign-orthogonalized refusal directions.")
    ap.add_argument("labels", type=pathlib.Path, help="Directory with {harmful,harmless}_{train,test}.jsonl")
    ap.add_argument("capture", type=pathlib.Path, help="Base prefill capture file (e.g. collect.f32)")
    ap.add_argument("outdir", type=pathlib.Path, help="Output directory for .f32 directions + fit_ortho.json")
    ap.add_argument("--visits", default="21,22", help="Comma-separated visit indices to fit (default: 21,22)")
    ap.add_argument("--preserve-cap", type=pathlib.Path, help="Optional raw 48-visit capture of coding/tool prompts")
    ap.add_argument("--ks", default="1,4,8,16,32", help="Comma-separated top-K singular vectors to orthogonalize out")
    ap.add_argument("--preserve-weight", type=float, default=2.0, help="Relative weight of coding capture vs harmless")
    args = ap.parse_args()

    hr = load_labeled_states(args.labels, args.capture, "harmful_train", True)
    ha = load_labeled_states(args.labels, args.capture, "harmless_train", False)
    tr = load_labeled_states(args.labels, args.capture, "harmful_test", True)
    ta = load_labeled_states(args.labels, args.capture, "harmless_test", False)
    code = load_raw_capture(args.preserve_cap) if args.preserve_cap else None

    print(
        f"train refused={len(hr)} harmless={len(ha)} | test refused={len(tr)} harmless={len(ta)}"
        + (f" | preserve_code={len(code)}" if code is not None else ""),
        flush=True,
    )

    visits = [int(x) for x in args.visits.split(",") if x.strip()]
    ks = [int(x) for x in args.ks.split(",") if x.strip()]
    args.outdir.mkdir(parents=True, exist_ok=True)
    report: list[dict] = []

    for v in visits:
        d = hr[:, v].mean(0) - ha[:, v].mean(0)
        u_raw = d / np.linalg.norm(d)
        raw_d = cohen_d(tr[:, v] @ u_raw, ta[:, v] @ u_raw)
        raw_hmean = float(abs(ha[:, v].mean(0) @ u_raw) / np.linalg.norm(ha[:, v].mean(0)))
        raw_crms = float(np.sqrt(np.mean((code[:, v] @ u_raw) ** 2))) if code is not None else None
        raw_hrms = float(np.sqrt(np.mean((ha[:, v] @ u_raw) ** 2)))

        # Uncentered preserve matrix so SV #1 captures the benign/coding mean direction directly
        ha_norm = ha[:, v] / max(1.0, np.sqrt(len(ha)))
        if code is not None:
            c_norm = code[:, v] * (args.preserve_weight / max(1.0, np.sqrt(len(code))))
            X = np.vstack([c_norm, ha_norm])
        else:
            X = ha_norm
        _, _, vt = np.linalg.svd(X, full_matrices=False)

        raw_name = f"v{v}-raw"
        u_raw.astype("<f4").tofile(args.outdir / f"{raw_name}.f32")
        report.append({
            "name": raw_name, "visit": v, "k": 0, "cohen_test": round(raw_d, 3),
            "cos_to_raw": 1.0, "harmless_mean_cos": round(raw_hmean, 4),
            "harmless_rms": round(raw_hrms, 4),
            "code_rms": round(raw_crms, 4) if raw_crms is not None else None,
        })
        print(
            f"{raw_name:12s}  d={raw_d:6.2f}  cos=1.000  |h_mean_cos|={raw_hmean:.4f}  h_rms={raw_hrms:.3f}"
            + (f"  code_rms={raw_crms:.3f}" if raw_crms is not None else ""),
            flush=True,
        )

        for k in ks:
            if k > vt.shape[0]:
                continue
            u, cos_raw = orthogonalize(d, vt[:k])
            cd = cohen_d(tr[:, v] @ u, ta[:, v] @ u)
            hmean = float(abs(ha[:, v].mean(0) @ u) / np.linalg.norm(ha[:, v].mean(0)))
            hrms = float(np.sqrt(np.mean((ha[:, v] @ u) ** 2)))
            crms = float(np.sqrt(np.mean((code[:, v] @ u) ** 2))) if code is not None else None
            name = f"v{v}-pc{k}"
            u.astype("<f4").tofile(args.outdir / f"{name}.f32")
            report.append({
                "name": name, "visit": v, "k": k, "cohen_test": round(cd, 3),
                "cos_to_raw": round(cos_raw, 4), "harmless_mean_cos": round(hmean, 4),
                "harmless_rms": round(hrms, 4),
                "code_rms": round(crms, 4) if crms is not None else None,
            })
            print(
                f"{name:12s}  d={cd:6.2f}  cos={cos_raw:.3f}  |h_mean_cos|={hmean:.4f}  h_rms={hrms:.3f}"
                + (f"  code_rms={crms:.3f}" if crms is not None else ""),
                flush=True,
            )

    (args.outdir / "fit_ortho.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
