"""Apply container overrides and prepare the tokenizer before the server starts."""
import fcntl
import hashlib
import json
import os
import shutil
import struct
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import Tokenizer, extract  # noqa: E402

INKLING_TOKENIZER = Path("ik2/tokenizer")
INKLING_TEMPLATE = ROOT / "tools/ik2/chat_template.jinja"
EXPORT_FILES = ("tokenizer.json", "token_type.json")


def usable_tokenizer(path, required=()):
    try:
        files = (path / name for name in required)
        if any(not file.is_file() or file.stat().st_size == 0 for file in files):
            return False
        return bool(Tokenizer.from_dir(path).tokens)
    except (OSError, ValueError, AttributeError, IndexError, TypeError):
        return False


def remove_cache_path(path):
    if path.is_dir() and not path.is_symlink():
        shutil.rmtree(path)
    else:
        path.unlink(missing_ok=True)


def cached_tokenizer(model, template):
    model = model.resolve(strict=True)
    if not model.is_file():
        raise ValueError(f"expected a GGUF file at {model}")
    stat = model.stat()
    identity = (str(model), stat.st_ino, stat.st_size, stat.st_mtime_ns)
    key = hashlib.sha256(repr(identity).encode())
    key.update((ROOT / "tools/strata_tokenizer.py").read_bytes())
    if template:
        key.update(template.read_bytes())
    key = key.hexdigest()

    cache = Path(os.getenv("TINTERO_TOKENIZER_CACHE") or ROOT / ".cache/tokenizers").resolve()
    cache.mkdir(parents=True, exist_ok=True)
    target = cache / key
    staged = cache / f".{key}.tmp"
    required = EXPORT_FILES + (("chat_template.jinja",) if template else ())
    with (cache / f"{key}.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        remove_cache_path(staged)
        if usable_tokenizer(target / "tokenizer", required):
            print(f"[tintero] using cached tokenizer for {model.name}", file=sys.stderr)
            return target / "tokenizer"

        print(f"[tintero] exporting tokenizer from {model.name}", file=sys.stderr)
        try:
            extract(model, staged, chat_template=template)
            if not usable_tokenizer(staged / "tokenizer", required):
                raise ValueError(f"tokenizer export is incomplete for {model}")
            remove_cache_path(target)
            staged.rename(target)
        finally:
            remove_cache_path(staged)
    return target / "tokenizer"


def prepare(config):
    cfg = json.loads(config.read_text(encoding="utf-8-sig"))
    env = cfg.get("env") or {}
    overrides = {k: os.environ[k] for k in env if k.startswith("IK2_") and k in os.environ}
    model = os.getenv("TINTERO_MODEL")
    args = cfg.get("args") or []
    model_index = args.index("--model") + 1 if "--model" in args[:-1] else None
    if model:
        if model_index is None:
            raise ValueError("TINTERO_MODEL requires a preset with a --model argument")
        args[model_index] = model
        if cfg.get("vision"):
            cfg["vision"]["model"] = model
    if overrides:
        cfg["env"] = {**env, **overrides}

    configured = Path(cfg.get("tokenizer") or "pack/full/tokenizer")
    inkling = configured.resolve() == INKLING_TOKENIZER.resolve()
    required = EXPORT_FILES + ("chat_template.jinja",) if inkling else ()
    if usable_tokenizer(configured, required):
        generated = False
        print(f"[tintero] using configured tokenizer at {configured}", file=sys.stderr)
    else:
        if model_index is None:
            raise ValueError(f"tokenizer at {configured} is missing or incomplete; set a valid tokenizer in {config}")
        gguf = Path(cfg.get("cwd") or ".") / args[model_index]
        if configured.exists():
            print(f"[tintero] keeping incomplete tokenizer at {configured} and using the cache", file=sys.stderr)
        cfg["tokenizer"] = str(cached_tokenizer(gguf, INKLING_TEMPLATE if inkling else None))
        generated = True

    if not (model or overrides or generated):
        return config
    effective = Path(tempfile.mkdtemp(prefix="tintero-")) / config.name
    effective.write_text(json.dumps(cfg), encoding="utf-8")
    return effective


if __name__ == "__main__":
    try:
        print(prepare(Path(sys.argv[1])))
    except (OSError, ValueError, KeyError, struct.error) as exc:
        sys.exit(f"[tintero] could not prepare startup: {exc}")
