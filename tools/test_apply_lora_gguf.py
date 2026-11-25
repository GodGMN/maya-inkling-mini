#!/usr/bin/env python3
"""Unit tests for tools/ik2_ablate/apply_lora_gguf.py (BF16/F16/F32 safetensors, LoRA/DoRA math, full GGUF splice)."""
from __future__ import annotations

import json
import pathlib
import struct
import subprocess
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent / "ik2_ablate"))
import apply_lora_gguf as alg  # noqa: E402
from bake import Q8_0, dequant, quant_q8, write_gguf  # noqa: E402
from gguf_reader import GGUFFile  # noqa: E402


def write_test_safetensors(path: pathlib.Path, tensors: dict[str, tuple[str, np.ndarray]]) -> None:
    header: dict[str, dict] = {}
    chunks: list[bytes] = []
    off = 0
    for name, (dtype, arr) in tensors.items():
        if dtype == "F32":
            raw = arr.astype("<f4").tobytes()
        elif dtype == "F16":
            raw = arr.astype("<f2").tobytes()
        elif dtype == "BF16":
            u32 = arr.astype("<f4").view("<u4")
            raw = (u32 >> 16).astype("<u2").tobytes()
        else:
            raise ValueError(dtype)
        header[name] = {"dtype": dtype, "shape": list(arr.shape), "data_offsets": [off, off + len(raw)]}
        chunks.append(raw)
        off += len(raw)
    hb = json.dumps(header).encode("utf-8")
    with path.open("wb") as fh:
        fh.write(struct.pack("<Q", len(hb)))
        fh.write(hb)
        for c in chunks:
            fh.write(c)


class ApplyLoraGGUFTest(unittest.TestCase):
    def test_read_safetensors_bf16_f16_f32(self) -> None:
        rng = np.random.default_rng(7)
        a = rng.standard_normal((8, 64)).astype(np.float32)
        b = rng.standard_normal((32, 8)).astype(np.float32)
        c = rng.standard_normal((32, 8)).astype(np.float32)
        with tempfile.TemporaryDirectory() as td:
            p = pathlib.Path(td) / "adapter_model.safetensors"
            write_test_safetensors(p, {"a": ("BF16", a), "b": ("F16", b), "c": ("F32", c)})
            got = alg.read_safetensors(p)
            np.testing.assert_allclose(got["a"], a, rtol=1e-2, atol=2e-2)
            np.testing.assert_allclose(got["b"], b, rtol=2e-3, atol=2e-3)
            np.testing.assert_array_equal(got["c"], c)

    def test_end_to_end_full_and_shard_splice(self) -> None:
        rng = np.random.default_rng(42)
        out_dim, in_dim, rank = 64, 96, 8
        w0 = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.1
        w_untouched = rng.standard_normal((32, 64)).astype(np.float32) * 0.1
        lora_a = rng.standard_normal((rank, in_dim)).astype(np.float32) * 0.05
        lora_b = rng.standard_normal((out_dim, rank)).astype(np.float32) * 0.05

        with tempfile.TemporaryDirectory() as td:
            tdp = pathlib.Path(td)
            base_gguf = tdp / "base.gguf"
            # Minimal valid GGUF with general.alignment=32 KV
            kv = bytearray()
            key = b"general.alignment"
            kv += struct.pack("<Q", len(key)) + key + struct.pack("<II", 4, 32)
            write_gguf(
                base_gguf,
                [
                    ("blk.4.attn_output.weight", [in_dim, out_dim], Q8_0, quant_q8(w0)),
                    ("blk.0.attn_output.weight", [64, 32], Q8_0, quant_q8(w_untouched)),
                ],
                kv=bytes(kv),
                n_kv=1,
                align=32,
            )

            adir = tdp / "adapter"
            adir.mkdir()
            (adir / "adapter_config.json").write_text(json.dumps({"r": rank, "lora_alpha": 16}), encoding="utf-8")
            write_test_safetensors(
                adir / "adapter_model.safetensors",
                {
                    "base_model.model.model.llm.layers.4.attn.wo_ud.lora_A.weight": ("BF16", lora_a),
                    "base_model.model.model.llm.layers.4.attn.wo_ud.lora_B.weight": ("BF16", lora_b),
                },
            )

            out_full = tdp / "full"
            subprocess.run(
                [
                    sys.executable,
                    str(pathlib.Path(__file__).resolve().parent / "ik2_ablate" / "apply_lora_gguf.py"),
                    str(base_gguf),
                    str(adir),
                    str(out_full),
                    "--scale", "1.5",
                    "--full",
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            merged_path = out_full / "model.gguf"
            self.assertTrue(merged_path.exists())
            f_base = GGUFFile(base_gguf)
            f_merged = GGUFFile(merged_path)
            self.assertEqual([t.name for t in f_merged.tensors], [t.name for t in f_base.tensors])

            # Untouched tensor must be byte-identical
            t0_base = next(t for t in f_base.tensors if t.name == "blk.0.attn_output.weight")
            t0_new = next(t for t in f_merged.tensors if t.name == "blk.0.attn_output.weight")
            np.testing.assert_array_equal(
                alg.raw(f_base, base_gguf, t0_base),
                alg.raw(f_merged, merged_path, t0_new),
            )

            # Edited tensor must closely match W_base + 1.5 * (16/8) * (B @ A)
            t4_base = next(t for t in f_base.tensors if t.name == "blk.4.attn_output.weight")
            t4_new = next(t for t in f_merged.tensors if t.name == "blk.4.attn_output.weight")
            wb = dequant(alg.raw(f_base, base_gguf, t4_base), Q8_0, out_dim * in_dim).reshape(out_dim, in_dim)
            wn = dequant(alg.raw(f_merged, merged_path, t4_new), Q8_0, out_dim * in_dim).reshape(out_dim, in_dim)
            expected = wb + 1.5 * (16.0 / rank) * (lora_b @ lora_a)
            rel_err = np.linalg.norm(wn - expected) / np.linalg.norm(expected)
            self.assertLess(rel_err, 0.02)

            # Also test override shard output
            out_shard = tdp / "shard"
            subprocess.run(
                [
                    sys.executable,
                    str(pathlib.Path(__file__).resolve().parent / "ik2_ablate" / "apply_lora_gguf.py"),
                    str(base_gguf),
                    str(adir),
                    str(out_shard),
                ],
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertTrue((out_shard / "weights-00001-of-00002.gguf").is_symlink())
            f_shard = GGUFFile(out_shard / "weights-00002-of-00002.gguf")
            self.assertEqual([t.name for t in f_shard.tensors], ["blk.4.attn_output.weight"])


if __name__ == "__main__":
    unittest.main()
