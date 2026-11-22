"""Score three color swaps and a real setup screenshot against a live image-capable server."""
import argparse
import base64
import json
import struct
import time
import zlib
from pathlib import Path
from urllib.request import Request, urlopen


def png(left, right):
    width, height = 256, 192
    def chunk(tag, data):
        return struct.pack(">I", len(data)) + tag + data + struct.pack(">I", zlib.crc32(tag + data))
    row = b"\x00" + bytes(left) * (width // 2) + bytes(right) * (width // 2)
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(row * height)) + chunk(b"IEND", b"")


def run(url, directory):
    colors = {"red": (236, 24, 24), "blue": (16, 40, 236), "green": (16, 165, 46)}
    question = "What color is the left half of this image? Answer with just the color name."
    images = [(f"{left}-{right}", png(colors[left], colors[right]), question, left)
              for left, right in (("red", "blue"), ("blue", "red"), ("green", "blue"))]
    screenshot = Path(__file__).resolve().parents[1] / "docs/media/setup-tui/03-compiling.png"
    images.append(("setup-progress", screenshot.read_bytes(),
                   "What percentage is printed beside the progress bar? Reply with the number and percent sign only.", "39%"))
    if url:
        with urlopen(url + "/v1/models", timeout=15) as response:
            model = json.load(response)["data"][0]["id"]
    failed = []
    for name, data, question, expected in images:
        if directory:
            (directory / f"{name}.png").write_bytes(data)
        if not url:
            continue
        parts = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + base64.b64encode(data).decode()}},
                 {"type": "text", "text": question}]
        body = dict(model=model, messages=[{"role": "user", "content": parts}], temperature=0,
                    reasoning_effort="none", max_tokens=128)
        req = Request(url + "/v1/chat/completions", json.dumps(body).encode(), {"Content-Type": "application/json"})
        start = time.monotonic()
        with urlopen(req, timeout=240) as response:
            result = json.load(response)
        choice, = result["choices"]
        answer = choice["message"]["content"]
        print(json.dumps(dict(image=name, expected=expected, answer=answer, usage=result["usage"],
                              finish=choice["finish_reason"], seconds=round(time.monotonic() - start, 3))), flush=True)
        if (answer or "").lower().strip().rstrip(".").replace(" %", "%") != expected or choice["finish_reason"] != "stop":
            failed.append(name)
    if failed:
        raise ValueError(f"incorrect image answers: {failed}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url", nargs="?")
    p.add_argument("--write-images", type=Path)
    args = p.parse_args()
    if not args.url and not args.write_images:
        p.error("provide a server URL or --write-images")
    if args.write_images:
        args.write_images.mkdir(parents=True, exist_ok=True)
    run(args.url, args.write_images)
