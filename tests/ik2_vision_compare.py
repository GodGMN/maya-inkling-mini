"""Compare maya-vision SVE image embeddings against a reference encoder."""
import argparse
import json
from pathlib import Path

import numpy as np


def read(path):
    with path.open("rb") as stream:
        header = np.fromfile(stream, "<i4", 5)
        data = np.fromfile(stream, "<f4")
    if len(header) != 5 or header[0] != 0x31455653:
        raise ValueError(f"{path}: invalid SVE header")
    _, rows, nx, ny, width = map(int, header)
    if min(nx, ny, width) < 1 or rows != nx * ny or data.size != rows * width or not np.isfinite(data).all():
        raise ValueError(f"{path}: invalid grid or embedding data")
    return (nx, ny, width), data.astype(np.float64)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument("--min-cosine", type=float, required=True)
    args = parser.parse_args()
    shape, ref = read(args.reference)
    other_shape, candidate = read(args.candidate)
    if shape != other_shape:
        parser.error(f"embedding grids differ: {shape} and {other_shape}")
    cosine = np.dot(ref, candidate) / (np.linalg.norm(ref) * np.linalg.norm(candidate))
    print(json.dumps(dict(grid=shape, values=ref.size, cosine=cosine,
                          max_abs=float(np.max(np.abs(ref - candidate))))))
    if not cosine >= args.min_cosine:
        raise SystemExit(f"cosine below {args.min_cosine}")
