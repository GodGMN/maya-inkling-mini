#!/usr/bin/env python3
"""apply_lora_gguf.py: splice a HuggingFace PEFT LoRA/DoRA adapter trained on thinkingmachines/Inkling-2-Mini
directly into a quantized Inkling-2-Mini GGUF (zero PyTorch required).

Produces either:
  - A tiny override shard (`weights-00001-of-00002.gguf` symlink + `weights-00002-of-00002.gguf`) for
    instant testing on tintero, OR
  - With `--full`, a single standalone portable `model.gguf` that runs out-of-the-box in vanilla
    llama.cpp / Ollama / LM Studio on NVIDIA/AMD/Apple Silicon with zero custom runtime hooks.

Supports continuous post-hoc scaling (`--scale 0.6 / 0.8 / 1.0 / 1.2`) and per-layer scale overrides
(`--layer-scales 4-17:0.8,18-23:1.2`) so a single ~10 MB Colab adapter can be tuned locally in ~30s.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import pathlib
import re
import struct
import sys

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from bake import BLOCK, F32, IQ4_NL, Q8_0, dequant, kv_block, quant_iq4nl, quant_q8, raw, raw_any, type_id, write_gguf  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402

HF_TO_GGUF_ATTN = {
    "wo_ud": "attn_output",
    "o_proj": "attn_output",
    "wq_du": "attn_q",
    "q_proj": "attn_q",
    "wk_dv": "attn_k",
    "k_proj": "attn_k",
    "wv_dv": "attn_v",
    "v_proj": "attn_v",
    "wr_du": "attn_rel",
    "gate_attn": "attn_gate",
    "attn_sink_proj": "attn_sink_proj",
}

LAYER_PROJ_RE = re.compile(
    r"(?:^|\.)layers\.(\d+)\.(?:attn|self_attn)\.([a-zA-Z0-9_]+)(?:\.linear)?\.(lora_A|lora_B|lora_magnitude_vector)(?:\.[^.]+)?\.weight$|"
    r"(?:^|\.)layers\.(\d+)\.(?:attn|self_attn)\.([a-zA-Z0-9_]+)(?:\.linear)?\.(lora_magnitude_vector)(?:\.[^.]+)?$"
)


def read_safetensors(path: pathlib.Path) -> dict[str, np.ndarray]:
    """Pure-Python + NumPy safetensors reader supporting F32, F16, and BF16."""
    with path.open("rb") as fh:
        hdr_len = struct.unpack("<Q", fh.read(8))[0]
        header = json.loads(fh.read(hdr_len).decode("utf-8"))
        data_offset = 8 + hdr_len
        out: dict[str, np.ndarray] = {}
        for name, meta in header.items():
            if name == "__metadata__":
                continue
            dtype = meta["dtype"]
            shape = tuple(meta["shape"])
            start, end = meta["data_offsets"]
            fh.seek(data_offset + start)
            buf = fh.read(end - start)
            if dtype == "F32":
                arr = np.frombuffer(buf, dtype="<f4").reshape(shape).copy()
            elif dtype == "F16":
                arr = np.frombuffer(buf, dtype="<f2").astype(np.float32).reshape(shape)
            elif dtype == "BF16":
                u16 = np.frombuffer(buf, dtype="<u2")
                arr = (u16.astype(np.uint32) << 16).view("<f4").reshape(shape)
            else:
                raise ValueError(f"Unsupported safetensors dtype {dtype!r} for tensor {name}")
            out[name] = arr
    return out


def parse_lora_entries(tensors: dict[str, np.ndarray]) -> dict[tuple[int, str], dict[str, np.ndarray]]:
    """Group PEFT LoRA tensors by (layer_idx, gguf_tensor_name)."""
    grouped: dict[tuple[int, str], dict[str, np.ndarray]] = {}
    unmatched: list[str] = []
    for k, arr in tensors.items():
        m = LAYER_PROJ_RE.search(k)
        if not m:
            unmatched.append(k)
            continue
        if m.group(1) is not None:
            layer, proj, kind = int(m.group(1)), m.group(2), m.group(3)
        else:
            layer, proj, kind = int(m.group(4)), m.group(5), m.group(6)
        if proj not in HF_TO_GGUF_ATTN:
            unmatched.append(k)
            continue
        gguf_name = f"blk.{layer}.{HF_TO_GGUF_ATTN[proj]}.weight"
        grouped.setdefault((layer, gguf_name), {})[kind] = arr
    if unmatched:
        raise ValueError(f"Unrecognized adapter tensors (refusing partial splice): {unmatched[:6]}")
    for key, parts in grouped.items():
        if "lora_A" not in parts or "lora_B" not in parts:
            raise ValueError(f"Incomplete LoRA pair for {key}: found {sorted(parts)}")
    return grouped


def parse_layer_scales(spec: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for item in spec.split(","):
        item = item.strip()
        if not item:
            continue
        rng, sv = item.rsplit(":", 1)
        lo, hi = int(rng.split("-")[0]), int(rng.split("-")[-1])
        for l in range(lo, hi + 1):
            out[l] = float(sv)
    return out


def splice_lora_matrix(
    w_base: np.ndarray,
    lora_a: np.ndarray,
    lora_b: np.ndarray,
    alpha: float,
    r: int,
    scale: float = 1.0,
    use_rslora: bool = False,
    magnitude: np.ndarray | None = None,
) -> np.ndarray:
    """Apply W' = W + scale * factor * (B @ A) (or DoRA rescaled update if magnitude is given)."""
    factor = (alpha / math.sqrt(r)) if use_rslora else (alpha / float(r))
    dw = (lora_b.astype(np.float64) @ lora_a.astype(np.float64)) * (factor * scale)
    if dw.shape != w_base.shape:
        raise ValueError(f"Shape mismatch: base {w_base.shape} vs LoRA delta {dw.shape}")
    w_merged = w_base.astype(np.float64) + dw
    if magnitude is not None:
        mag = magnitude.astype(np.float64).reshape(-1, 1)
        col_norm = np.linalg.norm(w_merged, axis=1, keepdims=True)
        w_merged = w_merged * (mag / np.maximum(col_norm, 1e-12))
    return w_merged.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Splice an Inkling-2-Mini PEFT LoRA adapter into a GGUF.")
    ap.add_argument("base", type=pathlib.Path, help="Base stock Inkling-2-Mini GGUF")
    ap.add_argument("adapter_dir", type=pathlib.Path, help="PEFT adapter dir (adapter_config.json + adapter_model.safetensors)")
    ap.add_argument("outdir", type=pathlib.Path, help="Output directory")
    ap.add_argument("--scale", type=float, default=1.0, help="Global multiplier on LoRA delta (default 1.0)")
    ap.add_argument("--layer-scales", default="", help="Optional per-layer scale overrides, e.g. 4-17:0.8,18-23:1.2")
    ap.add_argument("--keep-type", action="store_true", help="Re-quantize IQ4_NL tensors as IQ4_NL instead of Q8_0")
    ap.add_argument("--full", action="store_true", help="Write standalone OUTDIR/model.gguf instead of an override shard")
    ap.add_argument("--out-name", default="model.gguf", help="Output filename when --full is set (default: model.gguf)")
    args = ap.parse_args()

    cfg_path = args.adapter_dir / "adapter_config.json"
    st_path = args.adapter_dir / "adapter_model.safetensors"
    if not cfg_path.exists() or not st_path.exists():
        raise SystemExit(f"Expected {cfg_path} and {st_path}")

    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    r = int(cfg["r"])
    alpha = float(cfg.get("lora_alpha", r))
    use_rslora = bool(cfg.get("use_rslora", False))

    tensors = read_safetensors(st_path)
    grouped = parse_lora_entries(tensors)
    layer_scales = parse_layer_scales(args.layer_scales)

    f = GGUFFile(args.base)
    by = {t.name: t for t in f.tensors}

    edited_list = []
    for (layer, gname), parts in sorted(grouped.items()):
        if gname not in by:
            raise SystemExit(f"Target tensor {gname} not found in {args.base}")
        t = by[gname]
        in_dim, out_dim = int(t.shape[0]), int(t.shape[1])
        lscale = layer_scales.get(layer, args.scale)
        if lscale == 0.0:
            continue

        buf = raw(f, args.base, t)
        w_base = dequant(buf, type_id(t), out_dim * in_dim).reshape(out_dim, in_dim)
        w_new = splice_lora_matrix(
            w_base,
            parts["lora_A"],
            parts["lora_B"],
            alpha=alpha,
            r=r,
            scale=lscale,
            use_rslora=use_rslora,
            magnitude=parts.get("lora_magnitude_vector"),
        )
        oty = IQ4_NL if (args.keep_type and type_id(t) == IQ4_NL) else (F32 if type_id(t) == F32 else Q8_0)
        if oty == F32:
            q = w_new.astype("<f4").tobytes()
            back = w_new
        elif oty == IQ4_NL:
            q = quant_iq4nl(w_new)
            back = dequant(q, IQ4_NL, out_dim * in_dim).reshape(out_dim, in_dim)
        else:
            q = quant_q8(w_new)
            back = dequant(q, Q8_0, out_dim * in_dim).reshape(out_dim, in_dim)

        dw_exact = w_new.astype(np.float64) - w_base.astype(np.float64)
        dw_quant = back.astype(np.float64) - w_base.astype(np.float64)
        rel_dw = float(np.linalg.norm(dw_exact) / max(1e-12, np.linalg.norm(w_base.astype(np.float64))))
        q_err = float(np.linalg.norm(dw_quant - dw_exact) / max(1e-12, np.linalg.norm(dw_exact)))
        oname = {Q8_0: "Q8_0", IQ4_NL: "IQ4_NL", F32: "F32"}[oty]
        edited_list.append((t.name, list(t.shape), oty, q))
        print(
            f"{t.name:30s} {t.type_name}->{oname} scale={lscale:.2f} "
            f"||dW||/||W||={rel_dw:.4f} quant_err_on_dW={q_err:.4f}",
            flush=True,
        )

    args.outdir.mkdir(parents=True, exist_ok=True)
    if args.full:
        kv, n_kv, align = kv_block(args.base)
        edited_map = {x[0]: x for x in edited_list}
        whole = [edited_map.get(t.name) or (t.name, list(t.shape), t.type_id, raw_any(f, args.base, t)) for t in f.tensors]
        out_path = args.outdir / args.out_name
        write_gguf(out_path, whole, kv, n_kv, align)
        print(f"wrote standalone {out_path} ({out_path.stat().st_size / 1e9:.3f} GB, {len(edited_list)} tensors spliced)")
        return 0

    shard_path = args.outdir / "weights-00002-of-00002.gguf"
    write_gguf(shard_path, edited_list)
    first = args.outdir / "weights-00001-of-00002.gguf"
    if first.is_symlink() or first.exists():
        first.unlink()
    first.symlink_to(os.path.abspath(args.base))
    print(f"wrote override shard {shard_path} ({shard_path.stat().st_size / 1e6:.2f} MB, {len(edited_list)} tensors spliced)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
