#!/usr/bin/env python3
"""bake.py - writes a refusal direction into Inkling 2 Mini weights: an override shard for tintero (the edited
tensors in Q8_0, read after the base file) or, with --full, a whole GGUF for any runtime.

    bake.py BASE.gguf DIR.f32 OUTDIR --layers 18-23 [--writers attn,ffn] [--mode conv|r] [--scale 1] [--full]

Every residual write of a layer passes through a per-channel post conv before it joins the stream: for channel c,
out_c(t) = sum_j a_j[c] x_c(t - 3 + j), with the GGUF_shard_rewrite layout's taps a = w / sqrt(5) and 1 / sqrt(5)
added to tap 3 (src/ik2/ik2_main.cu, conv_w).  --mode conv projects the writer's output rows off the span of
a_j * r (j = 0..3), so nothing along r leaves the conv whatever the history; --mode r projects off r alone (what a
plain transformer's abliteration does).  The writers: attn_output and every expert's ffn_down_exps of each layer
listed.  A looped core's layer is edited for all of its passes (the weights are shared).

OUTDIR gets weights-00001-of-00002.gguf (a link to BASE) and weights-00002-of-00002.gguf (the edited tensors), the
pair tintero loads with --model OUTDIR/weights-00001-of-00002.gguf; with --full, OUTDIR/model.gguf instead.
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


def raw_any(f: GGUFFile, base: pathlib.Path, t) -> np.ndarray:
    """A tensor's bytes whatever its type (sized from the next tensor's offset or the file's end)."""
    offs = sorted(x.offset for x in f.tensors)
    i = offs.index(t.offset)
    end = offs[i + 1] if i + 1 < len(offs) else os.path.getsize(base) - f.data_start
    n = t.expected_bytes() or end - t.offset
    return np.memmap(base, "u1", "r", f.data_start + t.offset, (n,))


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


IQ4NL_MID = (IQ4NL_VALUES[1:] + IQ4NL_VALUES[:-1]) / 2


def quant_iq4nl(x: np.ndarray) -> np.ndarray:
    """IQ4_NL blocks of 32 (fp16 scale, 16 bytes: values j and j + 16 in a byte's low and high nibble), as llama.cpp's
    quantize_row_iq4_nl: 15 trial scales around amax / -127, the nearest level each, the least-squares scale."""
    blk = x.reshape(-1, 32).astype(np.float32)
    m = np.take_along_axis(blk, np.abs(blk).argmax(axis=1)[:, None], axis=1)   # the signed value of largest size
    best_err = np.full(len(blk), np.inf, np.float32)
    best_q = np.zeros(blk.shape, np.uint8)
    best_d = np.zeros(len(blk), np.float32)
    safe = np.where(m == 0, 1.0, m)
    for t in range(-7, 8):
        inv = (t + IQ4NL_VALUES[0]) / safe                       # 1 / d for this trial
        q = np.searchsorted(IQ4NL_MID, blk * inv).astype(np.uint8)
        v = IQ4NL_VALUES[q]
        sq = (v * v).sum(axis=1)
        d = np.where(sq > 0, (blk * v).sum(axis=1) / np.where(sq > 0, sq, 1), 0).astype(np.float32)
        d = d.astype(np.float16).astype(np.float32)
        err = ((blk - d[:, None] * v) ** 2).sum(axis=1)
        take = err < best_err
        best_err[take], best_d[take], best_q[take] = err[take], d[take], q[take]
    best_d[m[:, 0] == 0] = 0
    out = np.empty((len(blk), 18), np.uint8)
    out[:, :2] = best_d.astype(np.float16).view(np.uint8).reshape(-1, 2)
    out[:, 2:] = best_q[:, :16] | (best_q[:, 16:] << 4)
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


def kv_block(path: pathlib.Path) -> tuple[bytes, int, int]:
    """The base file's metadata as raw bytes (copied verbatim), its count, and general.alignment."""
    sizes = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}
    with open(path, "rb") as fh:
        head = fh.read(1 << 26)
    magic, ver, n_t, n_kv = struct.unpack_from("<4sIQQ", head, 0)
    assert magic == b"GGUF" and ver == 3
    pos, align = 24, 32

    def string(p):
        n = struct.unpack_from("<Q", head, p)[0]
        return head[p + 8:p + 8 + n].decode("utf-8", "replace"), p + 8 + n

    def value(p, ty):
        if ty == 8:
            return string(p)[1]
        if ty == 9:
            sub, n = struct.unpack_from("<IQ", head, p)
            p += 12
            if sub in sizes:
                return p + n * sizes[sub]
            for _ in range(n):
                p = value(p, sub)
            return p
        return p + sizes[ty]

    for _ in range(n_kv):
        key, pos = string(pos)
        ty = struct.unpack_from("<I", head, pos)[0]
        pos += 4
        if key == "general.alignment":
            align = struct.unpack_from("<I", head, pos)[0]
        pos = value(pos, ty)
    return head[24:pos], n_kv, align


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("base", type=pathlib.Path)
    ap.add_argument("direction", type=pathlib.Path)
    ap.add_argument("outdir", type=pathlib.Path)
    ap.add_argument("--layers", required=True, help="e.g. 18-23 or 6-9,18-23")
    ap.add_argument("--writers", default="attn,ffn")
    ap.add_argument("--mode", choices=("conv", "r"), default="conv")
    ap.add_argument("--scale", type=float, default=1.0)
    ap.add_argument("--layer-scales", default="", help="optional per-range scales e.g. 0-5:0.3,6-17:0.5,18-23:1.0")
    ap.add_argument("--include-embed", action="store_true", help="also project token_embd.weight off r * token_embd_norm")
    ap.add_argument("--embed-scale", type=float, default=None, help="override scale for token_embd (defaults to --scale)")
    ap.add_argument("--keep-type", action="store_true", help="re-quantize IQ4_NL tensors as IQ4_NL (else Q8_0)")
    ap.add_argument("--full", action="store_true", help="a whole GGUF (OUTDIR/model.gguf) instead of an override shard")
    a = ap.parse_args()
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
    scale_map = {l: a.scale for l in layers}
    if a.layer_scales:
        for spec in a.layer_scales.split(","):
            if not spec.strip():
                continue
            rng, sv = spec.rsplit(":", 1)
            lo, hi = (int(rng.split("-")[0]), int(rng.split("-")[-1]))
            for l in range(lo, hi + 1):
                scale_map[l] = float(sv)
    writers = set(a.writers.split(","))
    by = {t.name: t for t in f.tensors}
    out = []
    if a.include_embed:
        t_emb = by["token_embd.weight"]
        V = t_emb.shape[1]
        assert t_emb.shape[0] == C, (t_emb.name, t_emb.shape)
        if "token_embd_norm.weight" in by:
            tn = by["token_embd_norm.weight"]
            wn = dequant(raw(f, a.base, tn), F32, C).astype(np.float64)
            u_emb = (r * wn)
        else:
            u_emb = r.copy()
        u_emb = (u_emb / max(1e-12, np.linalg.norm(u_emb)))[None, :]
        escale = a.scale if a.embed_scale is None else a.embed_scale
        buf = raw(f, a.base, t_emb)
        vals, size = BLOCK[type_id(t_emb)]
        row_bytes = C // vals * size
        chunk = 4096
        blocks = []
        left = 0.0
        for s0 in range(0, V, chunk):
            s1 = min(V, s0 + chunk)
            n_rows = s1 - s0
            w = dequant(buf[s0 * row_bytes:s1 * row_bytes], type_id(t_emb), n_rows * C).reshape(n_rows, C)
            proj = w.astype(np.float64) @ u_emb.T
            w2 = (w.astype(np.float64) - escale * (proj @ u_emb)).astype(np.float32)
            q = quant_q8(w2)
            back = dequant(q, Q8_0, n_rows * C).reshape(n_rows, C)
            denom = float(np.linalg.norm(w.astype(np.float64) @ u_emb.T))
            if denom > 1e-12:
                left = max(left, float(np.linalg.norm(back.astype(np.float64) @ u_emb.T) / denom))
            blocks.append(q)
        data = np.concatenate(blocks)
        out.append((t_emb.name, list(t_emb.shape), Q8_0, data))
        print(f"{t_emb.name}: {t_emb.type_name} -> Q8_0, scale {escale}, component left {left:.4f}", flush=True)
    for l in layers:
        lscale = scale_map[l]
        if lscale == 0.0:
            continue
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
                w2 = edit_rows(w, U, lscale)
                oty = IQ4_NL if (a.keep_type and type_id(t) == IQ4_NL) else Q8_0
                q = quant_iq4nl(w2) if oty == IQ4_NL else quant_q8(w2)
                back = dequant(q, oty, C * K).reshape(C, K)
                left = max(left, float(np.linalg.norm(U @ back) / max(1e-12, np.linalg.norm(U @ w))))
                blocks.append(q)
            data = np.concatenate(blocks)
            oname = "IQ4_NL" if oty == IQ4_NL else "Q8_0"
            out.append((t.name, list(t.shape), oty, data))
            print(f"{t.name}: {t.type_name} -> {oname}, scale {lscale}, basis {U.shape[0]}, component left after quantization {left:.4f}", flush=True)
    a.outdir.mkdir(parents=True, exist_ok=True)
    if a.full:   # every tensor in the base's order, the edited ones replaced; the metadata copied verbatim
        kv, n_kv, align = kv_block(a.base)
        edited = {x[0]: x for x in out}
        whole = [edited.get(t.name) or (t.name, list(t.shape), t.type_id, raw_any(f, a.base, t)) for t in f.tensors]
        write_gguf(a.outdir / "model.gguf", whole, kv, n_kv, align)
        print("wrote", a.outdir / "model.gguf")
        return 0
    write_gguf(a.outdir / "weights-00002-of-00002.gguf", out)
    first = a.outdir / "weights-00001-of-00002.gguf"
    if first.is_symlink() or first.exists():
        first.unlink()
    first.symlink_to(os.path.abspath(a.base))
    print("wrote", a.outdir / "weights-00002-of-00002.gguf", sum(len(x[3]) for x in out) / 1e9, "GB")
    return 0


if __name__ == "__main__":
    sys.exit(main())
