"""Check native prefill and decode logits against an exactly representable Q8_0 model edit."""
import argparse
import json
import os
from pathlib import Path
import struct
import subprocess
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
from gguf_reader import GGUFFile
from glm_synth_gguf import kv_str, kv_f32, write_gguf


def overlay(folder, base, name, dims, blocks):
    # maya-ik2 reads ordinary split GGUFs. The second shard supplies just this matrix.
    folder.mkdir()
    first = folder / "fixture-00001-of-00002.gguf"
    first.symlink_to(base)
    tensor = name.encode()
    header = struct.pack("<4sIQQQ", b"GGUF", 3, 1, 0, len(tensor)) + tensor
    header += struct.pack("<IQQIQ", 2, *dims, 8, 0)
    with (folder / "fixture-00002-of-00002.gguf").open("wb") as out:
        out.write(header + bytes(-len(header) % 32))
        out.write(blocks.tobytes())
    return first


def check(base, exe, folder):
    model = GGUFFile(base)
    # Compare at the final physical layer so upstream expert routing sees identical inputs.
    last = model.metadata["inkling2_loop.block_count"] - 1
    layer = f"blk.{last}.attn_output.weight"
    tensor = next(t for t in model.tensors if t.name == layer)
    width, rows = tensor.shape
    assert tensor.type_name == "Q8_0" and width % 32 == 0
    # Rank two. Even rows copy one original projection row, odd rows copy the negative
    # of another. Flipping the stored FP16 scale sign is exact, including -128 codes.
    blocks = np.memmap(base, "u1", "r", model.data_start + tensor.offset, (rows, width // 32, 34))
    sources = np.array(blocks[[10, 37]])
    scales = sources[:, :, :2].copy().view("<f2").astype("f4")
    a = (scales * sources[:, :, 2:].view("i1")).reshape(2, width)
    b = np.zeros((rows, 2), dtype="f4")
    b[::2, 0], b[1::2, 1] = 0.5, -0.5
    adapter = folder / "rank-two.gguf"
    write_gguf(adapter, [("general.type", kv_str("adapter")), ("adapter.type", kv_str("lora")),
                         ("general.architecture", kv_str(model.metadata["general.architecture"])),
                         ("adapter.lora.alpha", kv_f32(4))], [(layer + ".lora_a", a), (layer + ".lora_b", b)])
    base_fixture = overlay(folder / "zero", base, layer, tensor.shape, np.zeros_like(blocks))
    sources[1, :, 1] ^= 128
    golden = overlay(folder / "reference", base, layer, tensor.shape, sources[np.arange(rows) % 2])

    prompt = "4651,195615,590,149922,151314,173752,149922,152385,167894,590,149922,155164"
    forced = folder / "force.txt"
    forced.write_text("4651,195615,590,149922")
    results = {}
    for case, weights, args in [("reference", golden, []), ("native", base_fixture, ["--lora", str(adapter)]),
                                ("disabled", base_fixture, [])]:
        logits = folder / (case + ".f32")
        prefill = folder / (case + "-prompt.f32")
        preset = Path(__file__).resolve().parents[1] / "docs/inkling2-mini/ik2-mini-gfx1100.json"
        env = {**os.environ, **json.loads(preset.read_text())["env"], "IK2_FORCE": str(forced),
               "IK2_LOGITS_OUT": str(logits), "IK2_PREFILL_LOGITS_OUT": str(prefill)}
        with (folder / (case + ".log")).open("w") as log:
            proc = subprocess.run([exe, "--model", weights, *args, "--tokens", prompt, "--gen", "4",
                                   "--ctx", "128", "--chunk", "64"], env=env, stdout=log, stderr=log)
        if proc.returncode:
            raise RuntimeError((folder / (case + ".log")).read_text()[-2500:])
        positions = len(prompt.split(","))
        prompt_logits = np.fromfile(prefill, "<f4").reshape(positions, -1)
        decode_logits = np.fromfile(logits, "<f4").reshape(3, prompt_logits.shape[1])
        results[case] = np.concatenate((prompt_logits, decode_logits)).astype("f8")
    gold = results["reference"]
    def rms(other):
        return np.sqrt(np.mean((other - gold) ** 2, axis=1))
    error, control = rms(results["native"]), rms(results["disabled"])
    top1 = gold.argmax(1) == results["native"].argmax(1)
    print(json.dumps(dict(native_rms=error.tolist(), disabled_rms=control.tolist(),
                          top1_agreement=top1.tolist())), flush=True)
    if not top1.all() or not np.all(np.isfinite(error) & (error < 0.1 * control) & (error < 0.1)):
        raise ValueError("adapter failed equivalence to the independent Q8 projection")


def check_adapter(base, adapter_path, exe, directory):
    model, adapter = GGUFFile(base), GGUFFile(adapter_path)
    weights = {t.name: t for t in model.tensors}
    factors = {t.name: t for t in adapter.tensors}
    preset = Path(__file__).resolve().parents[1] / "docs/inkling2-mini/ik2-mini-gfx1100.json"
    env = {**os.environ, **json.loads(preset.read_text())["env"]}
    with (directory / "engine.log").open("w") as log:
        proc = subprocess.run([exe, "--model", base, "--lora", adapter_path, "--tokens", "4651,195615,590,149922",
                               "--gen", "0", "--ctx", "128", "--chunk", "64", "--dump", directory],
                              env=env, stdout=log, stderr=log)
    if proc.returncode:
        raise RuntimeError((directory / "engine.log").read_text()[-2500:])
    meta = model.metadata
    start, end, passes = [meta["inkling2_loop.loop." + k] for k in ("start", "end", "count")]
    visits = [*range(start), *list(range(start, end)) * passes, *range(end, meta["inkling2_loop.block_count"])]
    errors, signal = [], []
    for visit, layer in enumerate(visits):
        name = f"blk.{layer}.attn_output.weight"
        if name + ".lora_a" not in factors:
            continue
        t = weights[name]
        width, rows = t.shape
        assert t.type_name == "Q8_0"
        # Sample output channels throughout the projection, for every prompt token and every layer visit.
        selected = np.unique(np.linspace(0, rows - 1, min(128, rows), dtype=int))
        blocks = np.memmap(base, "u1", "r", model.data_start + t.offset, (rows, width // 32, 34))[selected]
        w = (blocks[:, :, :2].copy().view("<f2").astype("f4") * blocks[:, :, 2:].view("i1")).reshape(-1, width)
        def read_factor(which):
            f = factors[name + which]
            return np.memmap(adapter_path, "<f4", "r", adapter.data_start + f.offset, tuple(reversed(f.shape)))
        a, b = read_factor(".lora_a"), read_factor(".lora_b")[selected]
        alpha = adapter.metadata.get("adapter.lora.alpha", 0)
        scale = alpha / len(a) if alpha else 1.0
        x = np.fromfile(directory / f"blk.{visit}.lora.input.f32", "<f4").reshape(4, width)
        result = np.fromfile(directory / f"blk.{visit}.lora.output.f32", "<f4").reshape(4, rows)[:, selected]
        # The stock prefill projects FP16 activations with Q8 weights decoded to FP16.
        base_output = x.astype("f2").astype("f4") @ w.astype("f2").astype("f4").T
        delta = scale * ((x.astype("f8") @ a.T) @ b.T)
        expected = base_output + delta
        error = np.linalg.norm(result - expected)
        errors.append(error / max(np.linalg.norm(expected), 1e-12))
        signal.append(error / max(np.linalg.norm(delta), 1e-12))
    print(json.dumps(dict(checked_visits=len(errors), worst_relative_error=max(errors),
                          worst_error_relative_to_adapter=max(signal))), flush=True)
    if not (max(errors) < 0.005 and max(signal) < 0.1):
        raise ValueError("GPU projection disagrees with CPU base + scaled B(A(x))")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("base", type=Path)
    p.add_argument("exe", type=Path)
    p.add_argument("--adapter", type=Path, help="check real adapter projections against CPU products")
    opts = p.parse_args()
    with tempfile.TemporaryDirectory() as directory:
        if opts.adapter:
            check_adapter(opts.base.resolve(), opts.adapter.resolve(), opts.exe.resolve(), Path(directory))
        else:
            check(opts.base.resolve(), opts.exe.resolve(), Path(directory))
