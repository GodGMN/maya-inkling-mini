"""Probe early, middle and late records, cache hits, and image input in independent, nearly full slots."""
import argparse
import base64
import json
import random
from urllib.request import urlopen

from ik2_chat_growth import archive, ask, post
from ik2_image_answers import png, setup_screenshots


def make_chat(url, depth, index):
    rng = random.Random(1429 + index)
    labels = ("cedar", "maple", "larch")
    codes = [str(rng.randrange(100_000, 1_000_000)) for _ in labels]
    text = archive(url, depth - 256, index)
    positions = [text.index("\n", len(text) * pct // 100) for pct in (5, 50, 95)]
    for name, code, position in reversed(list(zip(labels, codes, positions))):
        record = f"\nAudit record: the {name} verification number is {code}.\n"
        text = text[:position] + record + text[position:]
    messages = [dict(role="system", content=f"You maintain audit register {index}. Answer from this register."),
                dict(role="user", content=text+"\nWhat is the cedar verification number? Reply with just the number.")]
    return messages, codes


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--url", required=True)
    p.add_argument("--depth", type=int, required=True, help="aim this close to the per-slot limit; allow 1,024 extra tokens for an image")
    p.add_argument("--slots", type=int, required=True)
    p.add_argument("--serial", action="store_true", help="one-slot baseline, finish each chat before loading the next")
    p.add_argument("--server-pid", type=int)
    p.add_argument("--cold", action="store_true", help="replay the late-record queries after evicting all warm slots")
    p.add_argument("--screenshots", action="store_true", help="read a 1480×900 screenshot in each nearly full chat")
    p.add_argument("--sampled-turn", action="store_true", help="follow the image with a seeded lookup at the service's default temperature")
    args = p.parse_args()
    if args.depth < 512 or args.slots < 1:
        p.error("positive slots and depth >=512 required")
    with urlopen(args.url.rstrip("/") + "/health", timeout=10) as response:
        health = json.load(response)
    if args.depth > health["max_context"] or (not args.serial and args.slots > health["cache_slots"]):
        p.error("the trial exceeds the server's per-slot depth or resident slot count")
    controls = []
    scores = []

    def evict():
        for i in range(health["cache_slots"]):
            post(args.url, "/v1/chat/completions", dict(model="inkling-2-mini", temperature=0,
                 max_tokens=1, reasoning_effort="none", messages=[dict(role="system", content=f"Reset retrieval control {i}."),
                                                                   dict(role="user", content="Hi")]))
    evict()
    colors = [("red", (236, 24, 24)), ("blue", (16, 40, 236)), ("green", (16, 165, 46)), ("yellow", (225, 225, 16))]
    screenshots = setup_screenshots() if args.screenshots else []

    def turns(i):
        messages, codes = make_chat(args.url, args.depth, i)
        cached = 0
        for t in range(4 + int(args.sampled_turn)):
            if t == 3:
                if screenshots:
                    _, picture, question, expected = screenshots[i % len(screenshots)]
                else:
                    expected, left = colors[i % 4]
                    picture = png(left, colors[(i + 1) % 4][1])
                    question = "What color is the left half of this image? Reply with only the color name."
                data = base64.b64encode(picture).decode()
                content = [dict(type="image_url", image_url=dict(url="data:image/png;base64,"+data)),
                           dict(type="text", text=question)]
                messages.append(dict(role="user", content=content))
            else:
                index = 1 if t == 4 else t
                expected = codes[index]
                if t:
                    label = ("cedar", "maple", "larch")[index]
                    messages.append(dict(role="user", content=f"What is the {label} verification number? Reply with just the number."))
            request = dict(model="inkling-2-mini", temperature=0, reasoning_effort="none", max_tokens=32, messages=messages)
            if t == 4:
                request.update(temperature=1, top_p=1, seed=1429 + i)
            record = ask(args.url, request, f"register-{i}", t, cached, args.server_pid)
            answer = (record["answer"].get("content") or "").strip().lower().rstrip(".").replace(" %", "%")
            score = dict(chat=i, turn=t, expected=expected, answer=answer, correct=answer == expected)
            scores.append(score)
            print(json.dumps(score), flush=True)
            cached = record["prompt"] + record["generated"] - 1
            if t == 2 and args.cold:
                controls.append((i, dict(request, messages=list(messages)), record))
            messages.append(record["answer"])
            yield

    runs = [turns(i) for i in range(args.slots)]
    if args.serial:
        for run in runs:
            for _ in run:
                pass
    else:
        for _ in range(4 + int(args.sampled_turn)):
            for run in runs:
                next(run)

    for i, request, warm in controls:
        evict()
        cold = ask(args.url, request, f"cold-register-{i}", 2, 0, args.server_pid)
        if any(cold[key] != warm[key] for key in ("answer", "prompt", "generated", "finish")):
            raise ValueError(f"register {i} warm and cold replies differ")
    print(json.dumps(dict(registers=args.slots, depth=args.depth, correct=sum(r["correct"] for r in scores),
                          questions=len(scores), cold_replays_matched=len(controls))))


if __name__ == "__main__":
    main()
