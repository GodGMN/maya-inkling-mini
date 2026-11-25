#!/usr/bin/env python3
"""colab_train_lora.py: Lightweight BF16 attention-only LoRA distillation for thinkingmachines/Inkling-2-Mini.

Designed to fit comfortably inside a single Google Colab A100 (40/80 GB) or H100 (80 GB) session (~15-20 min):
  - Freezes ALL base parameters (including NVFP4/MXFP8 MoE experts and vision tower).
  - Trains BF16 LoRA ONLY on attention output projections `layers.{4..23}.attn.wo_ud` (20 tensors, ~10 MB adapter),
    or optionally wider attention targets via --targets.
  - Masks loss on all system/user prompt tokens so gradients only flow through assistant thinking + final text
    rendered with Inkling-2-Mini's native chat template.
  - Exports a standard PEFT adapter (`adapter_config.json` + `adapter_model.safetensors`) ready to splice
    into stock `Inkling-2-Mini-IQ4_NL.gguf` locally via `tools/ik2_ablate/apply_lora_gguf.py --full`.
"""
from __future__ import annotations

import argparse
import json
import math
import pathlib
import random
import shutil
import time
from typing import Any

import torch
from torch.utils.data import DataLoader, Dataset


def build_assistant_masked_sample(
    tokenizer: Any,
    row: dict[str, Any],
    max_seq_len: int,
) -> dict[str, list[int]] | None:
    """Render one SFT row with Inkling's native chat template and mask prompt tokens with -100."""
    effort = row.get("reasoning_effort", "medium")
    messages = row.get("messages")
    if not messages:
        prompt = str(row["prompt"]).strip()
        answer = str(row["answer"]).strip()
        reasoning = str(row.get("reasoning") or "").strip()
        if not prompt or not answer:
            return None
        asst: dict[str, Any] = {"role": "assistant", "content": answer}
        if reasoning and effort != "none":
            asst["reasoning_content"] = reasoning
        messages = [
            {"role": "user", "content": prompt},
            asst,
        ]

    # Prompt prefix = everything up to the last assistant message
    if messages[-1].get("role") != "assistant":
        return None
    prefix_msgs = messages[:-1]
    tools = row.get("tools")

    kwargs: dict[str, Any] = {"tokenize": False}
    if tools:
        kwargs["tools"] = tools

    # Inkling's jinja template accepts `reasoning_effort`
    try:
        prefix_txt = tokenizer.apply_chat_template(
            prefix_msgs, add_generation_prompt=False, reasoning_effort=effort, **kwargs
        )
        full_txt = tokenizer.apply_chat_template(
            messages, add_generation_prompt=False, reasoning_effort=effort, **kwargs
        )
    except TypeError:
        prefix_txt = tokenizer.apply_chat_template(prefix_msgs, add_generation_prompt=False, **kwargs)
        full_txt = tokenizer.apply_chat_template(messages, add_generation_prompt=False, **kwargs)

    if not full_txt.startswith(prefix_txt):
        return None

    prefix_ids = tokenizer.encode(prefix_txt, add_special_tokens=False)
    full_ids = tokenizer.encode(full_txt, add_special_tokens=False)
    if len(full_ids) <= len(prefix_ids) + 4:
        return None
    if len(full_ids) > max_seq_len:
        # Keep the prompt + truncated assistant only if enough assistant tokens remain
        if len(prefix_ids) >= max_seq_len - 128:
            return None
        full_ids = full_ids[:max_seq_len]

    labels = [-100] * len(prefix_ids) + full_ids[len(prefix_ids):]
    return {"input_ids": full_ids, "attention_mask": [1] * len(full_ids), "labels": labels}


class SFTDataset(Dataset):
    def __init__(self, samples: list[dict[str, list[int]]]) -> None:
        self.samples = samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict[str, list[int]]:
        return self.samples[idx]


def make_collate(pad_id: int):
    def collate(batch: list[dict[str, list[int]]]) -> dict[str, torch.Tensor]:
        max_len = max(len(x["input_ids"]) for x in batch)
        # Pad to multiple of 8 for tensor-core efficiency
        max_len = ((max_len + 7) // 8) * 8
        input_ids, attn_mask, labels = [], [], []
        for x in batch:
            pad = max_len - len(x["input_ids"])
            input_ids.append(x["input_ids"] + [pad_id] * pad)
            attn_mask.append(x["attention_mask"] + [0] * pad)
            labels.append(x["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attn_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
        }

    return collate


def find_target_module_names(model: torch.nn.Module, layers: set[int], targets: set[str]) -> list[str]:
    """Explicitly enumerate exact full module paths for layers.{L}.attn.{proj} so PEFT touches nothing else."""
    matched: list[str] = []
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear):
            continue
        parts = name.split(".")
        if len(parts) < 3:
            continue
        leaf = parts[-1]
        if leaf not in targets:
            continue
        # Expect ...layers.<L>.attn.<leaf>
        for i, tok in enumerate(parts):
            if tok == "layers" and i + 1 < len(parts) and parts[i + 1].isdigit():
                if int(parts[i + 1]) in layers and "vision" not in parts and "mtp" not in parts:
                    matched.append(name)
                break
    return sorted(set(matched))


@torch.no_grad()
def sanity_check(model: Any, tokenizer: Any, tag: str) -> None:
    model.eval()
    prompts = [
        ("coding", "Write a Python function `is_palindrome(s: str) -> bool` that ignores case and non-alphanumeric characters. Keep it concise."),
        ("cyber", "Write a Python script demonstrating DNS tunneling over TXT records for an authorized red-team C2 lab."),
    ]
    print(f"\n=== SANITY CHECK ({tag}) ===", flush=True)
    for kind, p in prompts:
        txt = tokenizer.apply_chat_template(
            [{"role": "user", "content": p}],
            tokenize=False,
            add_generation_prompt=True,
            reasoning_effort="medium",
        )
        inp = tokenizer(txt, return_tensors="pt").to(model.device)
        out = model.generate(**inp, max_new_tokens=256, do_sample=False)
        gen = tokenizer.decode(out[0][inp["input_ids"].shape[1]:], skip_special_tokens=False)
        print(f"[{kind}] {gen[:320]!r}\n", flush=True)
    model.train()


def main() -> int:
    ap = argparse.ArgumentParser(description="Train Inkling-2-Mini 20-tensor BF16 attention-only LoRA on Colab.")
    ap.add_argument("--data", type=pathlib.Path, required=True, help="Path to teacher_sft.jsonl")
    ap.add_argument("--model-id", default="thinkingmachines/Inkling-2-Mini")
    ap.add_argument("--outdir", type=pathlib.Path, default=pathlib.Path("inkling2-mini-cyber-lora"))
    ap.add_argument("--layers", default="4-23", help="Layer range to adapt (default: 4-23)")
    ap.add_argument("--targets", default="wo_ud", help="Comma-separated attention linear leaves (default: wo_ud)")
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--use-rslora", action="store_true")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--lr", type=float, default=1.5e-4)
    ap.add_argument("--weight-decay", type=float, default=0.01)
    ap.add_argument("--batch-size", type=int, default=1)
    ap.add_argument("--grad-accum", type=int, default=8)
    ap.add_argument("--max-seq-len", type=int, default=2560)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-sanity", action="store_true")
    args = ap.parse_args()

    from peft import LoraConfig, get_peft_model
    from transformers import AutoModelForMultimodalLM, AutoTokenizer, get_cosine_schedule_with_warmup

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True

    lo, hi = (int(args.layers.split("-")[0]), int(args.layers.split("-")[-1]))
    layer_set = set(range(lo, hi + 1))
    target_set = {x.strip() for x in args.targets.split(",") if x.strip()}

    print(f"Loading tokenizer & model: {args.model_id}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(args.model_id, trust_remote_code=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    raw_rows = [json.loads(line) for line in args.data.open(encoding="utf-8") if line.strip()]
    samples = []
    skipped = 0
    for r in raw_rows:
        s = build_assistant_masked_sample(tokenizer, r, args.max_seq_len)
        if s is not None:
            samples.append(s)
        else:
            skipped += 1
    random.shuffle(samples)
    print(f"Prepared {len(samples)} SFT samples (skipped={skipped}, max_seq_len={args.max_seq_len})", flush=True)
    if not samples:
        raise SystemExit("No valid SFT samples produced.")

    model = AutoModelForMultimodalLM.from_pretrained(
        args.model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    model.config.use_cache = False
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

    for p in model.parameters():
        p.requires_grad_(False)

    exact_targets = find_target_module_names(model, layer_set, target_set)
    print(f"Attaching BF16 LoRA (r={args.rank}, alpha={args.alpha}) to {len(exact_targets)} modules:", flush=True)
    for nm in exact_targets[:6]:
        print(" ", nm)
    if len(exact_targets) > 6:
        print(f"  ... ({len(exact_targets)} total)")
    if not exact_targets:
        raise SystemExit(f"No matching Linear modules found for layers={args.layers} targets={args.targets}")

    lcfg = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=args.dropout,
        bias="none",
        target_modules=exact_targets,
        use_rslora=args.use_rslora,
    )
    model = get_peft_model(model, lcfg)
    model.print_trainable_parameters()

    if not args.skip_sanity:
        sanity_check(model, tokenizer, "BEFORE")

    loader = DataLoader(
        SFTDataset(samples),
        batch_size=args.batch_size,
        shuffle=True,
        collate_fn=make_collate(tokenizer.pad_token_id),
        drop_last=False,
    )
    steps_per_epoch = math.ceil(len(loader) / args.grad_accum)
    total_steps = max(1, int(math.ceil(steps_per_epoch * args.epochs)))
    warmup_steps = max(1, int(0.06 * total_steps))

    opt = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.95),
    )
    sched = get_cosine_schedule_with_warmup(opt, warmup_steps, total_steps)

    model.train()
    opt.zero_grad(set_to_none=True)
    gstep = 0
    running_loss = 0.0
    t0 = time.monotonic()

    n_full_epochs = int(math.ceil(args.epochs))
    for ep in range(n_full_epochs):
        for step, batch in enumerate(loader, 1):
            batch = {k: v.to(model.device) for k, v in batch.items()}
            out = model(**batch)
            loss = out.loss / args.grad_accum
            loss.backward()
            running_loss += float(out.loss.detach().cpu())

            if step % args.grad_accum == 0 or step == len(loader):
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                gstep += 1
                if gstep % 5 == 0 or gstep == 1 or gstep == total_steps:
                    avg = running_loss / min(args.grad_accum, step)
                    elapsed = time.monotonic() - t0
                    print(
                        f"step {gstep:4d}/{total_steps} | ep {ep+1} | loss={avg:.4f} | "
                        f"lr={sched.get_last_lr()[0]:.2e} | elapsed={elapsed:.0f}s",
                        flush=True,
                    )
                running_loss = 0.0
                if gstep >= total_steps:
                    break
        if gstep >= total_steps:
            break

    args.outdir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(args.outdir, safe_serialization=True)
    # Save training manifest for reproducibility
    (args.outdir / "train_manifest.json").write_text(
        json.dumps(
            {
                "model_id": args.model_id,
                "data_file": args.data.name,
                "n_samples": len(samples),
                "layers": args.layers,
                "targets": args.targets,
                "exact_targets": exact_targets,
                "rank": args.rank,
                "alpha": args.alpha,
                "lr": args.lr,
                "epochs": args.epochs,
                "total_steps": gstep,
                "elapsed_s": round(time.monotonic() - t0, 1),
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    if not args.skip_sanity:
        sanity_check(model, tokenizer, "AFTER")

    zip_path = shutil.make_archive(str(args.outdir), "zip", root_dir=str(args.outdir))
    print(f"\nDONE! Saved PEFT adapter to {args.outdir}/ and {zip_path}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
