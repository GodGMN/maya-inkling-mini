"""Replay a growing OpenAI chat, compact it, then compare each response to a cold request."""
import argparse
import json
import time
from pathlib import Path
from urllib.request import Request, urlopen


def post(url, path, body):
    req = Request(url + path, json.dumps(body).encode(), {"Content-Type": "application/json"})
    with urlopen(req, timeout=1200) as response:
        return json.load(response)


def archive(url, size, turn):
    lines = []
    for i in range(size // 8 + 64):
        lines.append(f"Entry {turn}-{i}: checked parcel {i * 97 % 65521} at depot {i % 89}.\n")
    tokens = post(url, "/tokenize", {"content": "".join(lines)})["tokens"]
    assert len(tokens) >= size
    return post(url, "/detokenize", {"tokens": tokens[:size]})["content"]


def memory(pid):
    if pid is None:
        return {}
    cgroup = next(line[3:] for line in Path(f"/proc/{pid}/cgroup").read_text().splitlines() if line.startswith("0::"))
    root = Path("/sys/fs/cgroup") / cgroup.lstrip("/")
    stat = dict(line.split() for line in (root / "memory.stat").read_text().splitlines())
    events = dict(line.split() for line in (root / "memory.events").read_text().splitlines())
    cards = sorted(Path("/sys/class/drm").glob("card*/device/mem_info_vram_used"))
    return dict(anon=int(stat["anon"]), file=int(stat["file"]), peak=int((root / "memory.peak").read_text()),
                oom_kill=int(events["oom_kill"]), vram=[int(card.read_text()) for card in cards])


def ask(url, body, phase, turn, expected_cache, pid, emit=True):
    start = time.monotonic()
    result = post(url, "/v1/chat/completions", body)
    choice, = result["choices"]
    usage = result["usage"]
    prompt = usage["prompt_tokens"]
    cached = usage["prompt_tokens_details"]["cached_tokens"]
    record = dict(phase=phase, turn=turn, prompt=prompt, cached=cached, new=prompt-cached,
                  expected_cache=expected_cache, generated=usage["completion_tokens"],
                  answer=choice["message"], finish=choice["finish_reason"], seconds=round(time.monotonic()-start, 3),
                  memory=memory(pid))
    if emit:
        print(json.dumps(record), flush=True)
    if choice["finish_reason"] != "stop" or cached != expected_cache:
        raise ValueError(f"{phase} {turn}: expected stop and {expected_cache} cached, got {choice['finish_reason']} and {cached}")
    return record


def run(args):
    with urlopen(args.url.rstrip("/") + "/health", timeout=10) as response:
        slots = json.load(response).get("cache_slots", 1)
    messages = [{"role": "system", "content": "You maintain a parcel register. Answer the question after each batch."}]
    turns = []
    previous = 0
    for turn, depth in enumerate(args.depths + [None, 0]):
        if depth is None:
            messages = [{"role": "system", "content": "Compacted register. Earlier parcels have been audited."}]
            previous = 0
        code = f"H{317 + 73 * turn}"
        question = f"\nThe current seal is {code}. Reply only with the current seal."
        messages.append({"role": "user", "content": question})
        body = dict(model="inkling-2-mini", messages=messages, max_tokens=96, temperature=0, reasoning_effort="none")
        if depth:
            overhead = post(args.url, "/api/tokens", body)["tokens"] + 2  # answer-opening tokens for direct answers
            if depth > overhead:
                messages[-1]["content"] = archive(args.url, depth - overhead, turn) + question
        record = ask(args.url, body, "warm" if depth is not None else "compaction", turn, previous, args.server_pid)
        if record["answer"]["content"].strip() != code:
            raise ValueError(f"turn {turn}: expected seal {code}")
        previous = record["prompt"] + record["generated"] - 1  # stop token never goes through forward
        turns.append((dict(body, messages=list(messages)), record))
        messages.append(record["answer"])
    for turn, (body, warm) in enumerate(turns):
        for slot in range(slots):
            messages = [dict(role="system", content=f"Cold replay {turn}, eviction {slot}."),
                        dict(role="user", content="Reset.")]
            post(args.url, "/v1/chat/completions", dict(model=body["model"], messages=messages,
                                                       temperature=0, reasoning_effort="none", max_tokens=1))
        cold = ask(args.url, body, "cold", turn, 0, args.server_pid)
        for key in ("answer", "generated", "prompt", "finish"):
            if warm[key] != cold[key]:
                raise ValueError(f"turn {turn}: {key} differs between warm and cold")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("--server-pid", type=int, help="sample the Linux server cgroup and GPU memory after each turn")
    p.add_argument("--depths", type=lambda s: list(map(int, s.split(","))),
                   default=[20000, 0, 40000, 70000, 0, 115000, 160000, 205000, 250000, 300000])
    run(p.parse_args())
