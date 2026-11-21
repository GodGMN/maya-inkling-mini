"""Compare forced decode logits with another engine run or ref/inkling2.py."""
import argparse
from pathlib import Path

import numpy as np


def raw_logits(path, rows, vocab):
    expected = rows * vocab * 4
    if path.stat().st_size != expected:
        raise ValueError(f"{path}: expected {rows} rows of {vocab} float32 logits ({expected} bytes), "
                         f"found {path.stat().st_size} bytes")
    return np.memmap(path, dtype="<f4", mode="r", shape=(rows, vocab))


def log_softmax(row):
    shifted = row.astype(np.float64) - row.max()
    return shifted - np.log(np.exp(shifted).sum())


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("engine", type=Path, help="IK2_LOGITS_OUT file")
    oracle = parser.add_mutually_exclusive_group(required=True)
    oracle.add_argument("--baseline", type=Path, help="IK2_LOGITS_OUT from the same forced history")
    oracle.add_argument("--ref", type=Path, help="reference .npy for prompt + first gen-1 forced tokens")
    parser.add_argument("--prompt-length", type=int, required=True)
    parser.add_argument("--gen", type=int, required=True, help="engine --gen value")
    parser.add_argument("--vocab", type=int, required=True, help="unpadded vocabulary size")
    args = parser.parse_args()
    if args.gen < 2 or args.prompt_length < 1 or args.vocab < 1:
        parser.error("need --gen >= 2 and positive prompt length and vocabulary size")
    rows = args.gen - 1
    try:
        engine = raw_logits(args.engine, rows, args.vocab)
        if args.ref:
            reference = np.load(args.ref, mmap_mode="r", allow_pickle=False)
            shape = (args.prompt_length + rows, args.vocab)
            if reference.shape != shape or reference.dtype.kind != "f":
                raise ValueError(f"expected floating-point reference of shape {shape}, "
                                 f"found {reference.dtype} {reference.shape}")
            reference = reference[args.prompt_length:]
        else:
            reference = raw_logits(args.baseline, rows, args.vocab)

        divergences = []
        matches = 0
        for i, (ref, got) in enumerate(zip(reference, engine)):
            if not (np.isfinite(ref).all() and np.isfinite(got).all()):
                raise ValueError(f"non-finite value at input position {args.prompt_length + i}")
            matches += int(ref.argmax() == got.argmax())
            r, g = log_softmax(ref), log_softmax(got)
            divergences.append(np.dot(np.exp(r), r - g))
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    worst = int(np.argmax(divergences))
    print(f"top-1 {matches}/{rows}; KL(ref || engine) mean {np.mean(divergences):.8g}, "
          f"median {np.median(divergences):.8g}, max {divergences[worst]:.8g} "
          f"at input position {args.prompt_length + worst} (zero-based)")


if __name__ == "__main__":
    main()
