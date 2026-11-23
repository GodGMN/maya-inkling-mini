#!/usr/bin/env python3
"""bake.py - writes a refusal direction into Inkling 2 Mini weights: an override shard for maya-ik2 (the edited
tensors in Q8_0, read after the base file) or, with --full, a whole GGUF for any runtime.

    bake.py BASE.gguf DIR.f32 OUTDIR --layers 18-23 [--writers attn,ffn] [--mode conv|r] [--scale 1] [--full]

Every residual write of a layer passes through a per-channel post conv before it joins the stream: for channel c,
out_c(t) = sum_j a_j[c] x_c(t - 3 + j), with the GGUF_shard_rewrite layout's taps a = w / sqrt(5) and 1 / sqrt(5)
added to tap 3 (src/ik2/ik2_main.cu, conv_w).  --mode conv projects the writer's output rows off the span of
a_j * r (j = 0..3), so nothing along r leaves the conv whatever the history; --mode r projects off r alone (what a
plain transformer's abliteration does).  The writers: attn_output and every expert's ffn_down_exps of each layer
listed.  A looped core's layer is edited for all of its passes (the weights are shared).

OUTDIR gets weights-00001-of-00002.gguf (a link to BASE) and weights-00002-of-00002.gguf (the edited tensors), the
pair maya-ik2 loads with --model OUTDIR/weights-00001-of-00002.gguf; with --full, OUTDIR/model.gguf instead.
Needs numpy and tools/gguf_reader.py.
"""
from __future__ import annotations

import argparse
import os
import pathlib
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from gguf_reader import GGUFFile  # noqa: E402

Q8_0, IQ4_NL, F32 = 8, 20, 0
BLOCK = {Q8_0: (32, 34), IQ4_NL: (32, 18), F32: (1, 4)}
IQ4NL_VALUES = np.array([-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113], np.float32)


def type_id(t) -> int:
    return {"Q8_0": Q8_0, "IQ4_NL": IQ4_NL, "F32": F32}[t.type_name]


def raw(f: GGUFFile, base: pathlib.Path, t) -> np.ndarray:
    n = int(np.prod(t.shape))
    vals, size = BLOCK[type_id(t)]
    return np.memmap(base, "u1", "r", f.data_start + t.offset, (n // vals * size,))


def dequant(buf: np.ndarray, ty: int, n: int) -> np.ndarray:
    if ty == F32:
        return np.frombuffer(buf.tobytes(), "<f4", n).copy()
    vals, size = BLOCK[ty]
    b = np.asarray(buf).reshape(-1, size)
    d = b[:, :2].copy().view("<f2").astype(np.float32)
    if ty == Q8_0:
        return (d * b[:, 2:].view(np.int8).astype(np.float32)).reshape(-1)
    qs = b[:, 2:]
    lo, hi = IQ4NL_VALUES[qs & 15], IQ4NL_VALUES[qs >> 4]
    return (d * np.concatenate([lo, hi], axis=1)).reshape(-1)


def quant_q8(x: np.ndarray) -> np.ndarray:
    blk = x.reshape(-1, 32).astype(np.float32)
    amax = np.abs(blk).max(axis=1, keepdims=True)
    d = (amax / 127.0).astype(np.float16)
    inv = np.where(d == 0, 0.0, 1.0 / d.astype(np.float32))
    q = np.rint(blk * inv).clip(-127, 127).astype(np.int8)
    out = np.empty((blk.shape[0], 34), np.uint8)
    out[:, :2] = d.view(np.uint8).reshape(-1, 2)
    out[:, 2:] = q.view(np.uint8)
    return out.reshape(-1)


def basis(r: np.ndarray, taps: np.ndarray | None) -> np.ndarray:
    """The orthonormal rows the writer's output must not reach: r alone, or span{a_j * r}."""
    if taps is None:
        return (r / np.linalg.norm(r))[None, :]
    m = (taps.T * r[None, :]).astype(np.float64)   # [4][C]
    u, s, vt = np.linalg.svd(m, full_matrices=False)
    keep = s > s[0] * 1e-6
    return vt[keep]


def edit_rows(w: np.ndarray, U: np.ndarray, scale: float) -> np.ndarray:
    """w [C][K] (output rows): w - scale U^T (U w)."""
    return (w - scale * (U.T @ (U @ w.astype(np.float64)))).astype(np.float32)


def taps_of(f: GGUFFile, base, name: str, C: int) -> np.ndarray:
    t = next(x for x in f.tensors if x.name == name)
    w = dequant(raw(f, base, t), F32, C * 4).reshape(C, 4).astype(np.float64)
    sc = 1.0 / np.sqrt(5.0)
    w *= sc
    w[:, 3] += sc
    return w


def write_gguf(path: pathlib.Path, tensors: list, kv: bytes = b"", n_kv: int = 0, align: int = 32) -> None:
    """tensors: [(name, dims (ne0 first), type, bytes)]."""
    head = bytearray(struct.pack("<4sIQQ", b"GGUF", 3, len(tensors), n_kv)) + kv
    offs, off = [], 0
    for name, dims, ty, data in tensors:
        nb = name.encode()
        head += struct.pack("<Q", len(nb)) + nb + struct.pack("<I", len(dims))
        head += b"".join(struct.pack("<Q", d) for d in dims) + struct.pack("<IQ", ty, off)
        offs.append(off)
        off = (off + len(data) + align - 1) // align * align
    head += bytes(-len(head) % align)
    tmp = path.with_suffix(".tmp")
    with tmp.open("wb") as out:
        out.write(head)
        for (_, _, _, data), o in zip(tensors, offs):
            out.seek(len(head) + o)
            out.write(data if isinstance(data, (bytes, bytearray)) else memoryview(data))
        out.truncate(len(head) + off)
    tmp.rename(path)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", type=pathlib.Path)
    ap.add_argument("direction", type=pathlib.Path)
    ap.add_argument("outdir", type=pathlib.Path)
    ap.add_argument("--layers", required=True, help="e.g. 18-23 or 6-9,18-23")
    ap.add_argument("--writers", default="attn,ffn")
    ap.add_argument("--mode", choices=("conv", "r"), default="conv")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--full", action="store_true", help="not supported yet: the override shard only")
    a = ap.parse_args()
    if a.full:
        raise SystemExit("--full: not written yet; load the override shard pair")
    f = GGUFFile(a.base)
    arch = f.metadata.get("general.architecture", "")
    if arch != "inkling2_loop" or "blk.0.attn_post_conv.weight" not in {t.name for t in f.tensors}:
        raise SystemExit("needs the inkling2_loop GGUF_shard_rewrite layout (attn_post_conv / ffn_post_conv)")
    C = int(f.metadata["inkling2_loop.embedding_length"])
    r = np.fromfile(a.direction, "<f4").astype(np.float64)
    if r.size != C:
        raise SystemExit(f"{a.direction}: {r.size} floats, the model's width is {C}")
    r /= np.linalg.norm(r)
    layers = sorted({l for part in a.layers.split(",") for l in
                     (range(int(part.split("-")[0]), int(part.split("-")[-1]) + 1))})
    writers = set(a.writers.split(","))
    by = {t.name: t for t in f.tensors}
    out = []
    for l in layers:
        for w_name, conv in (("attn_output.weight", "attn_post_conv.weight"), ("ffn_down_exps.weight", "ffn_post_conv.weight")):
            if ("attn" if w_name.startswith("attn") else "ffn") not in writers:
                continue
            t = by[f"blk.{l}.{w_name}"]
            U = basis(r, taps_of(f, a.base, f"blk.{l}.{conv}", C) if a.mode == "conv" else None)
            K = t.shape[0]
            assert t.shape[1] == C, (t.name, t.shape)
            E = t.shape[2] if len(t.shape) > 2 else 1
            buf = raw(f, a.base, t)
            vals, size = BLOCK[type_id(t)]
            per = C * K // vals * size
            blocks = []
            left = 0.0
            for e in range(E):
                w = dequant(buf[e * per:(e + 1) * per], type_id(t), C * K).reshape(C, K)
                w2 = edit_rows(w, U, a.scale)
                q = quant_q8(w2)
                back = dequant(q, Q8_0, C * K).reshape(C, K)
                left = max(left, float(np.linalg.norm(U @ back) / max(1e-12, np.linalg.norm(U @ w))))
                blocks.append(q)
            data = np.concatenate(blocks)
            out.append((t.name, list(t.shape), Q8_0, data))
            print(f"{t.name}: {t.type_name} -> Q8_0, basis {U.shape[0]}, component left after Q8 {left:.4f}", flush=True)
    a.outdir.mkdir(parents=True, exist_ok=True)
    write_gguf(a.outdir / "weights-00002-of-00002.gguf", out)
    first = a.outdir / "weights-00001-of-00002.gguf"
    if first.is_symlink() or first.exists():
        first.unlink()
    first.symlink_to(os.path.abspath(a.base))
    print("wrote", a.outdir / "weights-00002-of-00002.gguf", sum(len(x[3]) for x in out) / 1e9, "GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
