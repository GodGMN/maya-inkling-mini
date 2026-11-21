"""Emit one synthetic NVFP4 expert and its CPU Ref.moe result on stdout."""
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "ref"))
from inkling2 import DT, Ref


def main():
    rng = np.random.default_rng(1429)
    matrices = {}
    width, hidden = 128, 64
    lut = np.array([0, .5, 1, 1.5, 2, 3, 4, 6, 0, -.5, -1, -1.5, -2, -3, -4, -6], np.float32)
    for role, rows, cols in [("gate", hidden, width), ("up", hidden, width), ("down", width, hidden)]:
        blocks = rng.integers(0, 256, (rows * cols // 64, 36), dtype=np.uint8)
        scales = blocks[:, :4]
        scales[:] %= 55
        fp8 = torch.from_numpy(scales.copy()).view(torch.float8_e4m3fn).float().numpy()
        packed = blocks[:, 4:].reshape(-1, 4, 8)
        values = lut[np.concatenate([packed & 15, packed >> 4], axis=-1)] * fp8[:, :, None]
        matrices[f"blk.0.ffn_{role}_exps.weight"] = values.reshape(rows, cols)
        sys.stdout.buffer.write(blocks.tobytes())
        sys.stdout.buffer.write(values.astype("<f4").tobytes())

    # Supply the tensors for a one-expert layer directly to the reference.
    ref = Ref.__new__(Ref)
    ref.c = SimpleNamespace(k_exp=1, route_scale=1., gate_cap=4., up_cap=25.)
    ref.rw = False
    ref.dump = None
    ref.cache = {
        "blk.0.ffn_gate_inp.weight": torch.zeros(1, width, dtype=DT),
        "blk.0.exp_probs_b.bias": torch.zeros(1, dtype=DT),
    }
    for role, scale in [("gate", .75), ("up", 1.25), ("down", 1.)]:
        ref.cache[f"blk.0.ffn_{role}_exps.scale"] = torch.tensor([scale], dtype=DT)
    ref.g = SimpleNamespace(_by_name=ref.cache, expert=lambda name, expert: matrices[name])
    x = torch.from_numpy(rng.normal(0, .125, (1, width)).astype(np.float32))
    expected = ref.moe(0, 0, x.to(DT))
    sys.stdout.buffer.write(x.numpy().astype("<f4").tobytes())
    sys.stdout.buffer.write(expected.float().numpy().astype("<f4").tobytes())


if __name__ == "__main__":
    main()
