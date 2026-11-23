"""Grow stored Responses conversations, join two distant records, compact and cold-replay each lookup."""
import argparse
import base64
import hashlib
import json
import random
import time
from pathlib import Path
from urllib.request import urlopen

from ik2_chat_growth import archive, post
from ik2_image_answers import setup_screenshots


def run(url, depths, serial, repeat_cold, save_controls):
    with urlopen(url + "/health", timeout=10) as response:
        slots = json.load(response)["cache_slots"]
    if slots < 2 and not serial:
        raise ValueError("the interleaved run needs two resident caches")

    def evict(label):
        for i in range(slots):
            messages = [dict(role="system", content=f"Cold replay control {label}/{i}."), dict(role="user", content="Hi")]
            post(url, "/v1/chat/completions", dict(model="inkling-2-mini", temperature=0, max_tokens=1, messages=messages))

    def send(chat, turn, inputs, previous=None, store=True, lookup=False, temperature=0):
        body = dict(model="inkling-2-mini", instructions=f"You maintain parcel register {chat}. Answer from this register.",
                    temperature=temperature, max_output_tokens=4096, reasoning={"effort": "medium"}, store=store, input=inputs)
        if previous:
            body["previous_response_id"] = previous
        if turn == "cold" and save_controls:
            save_controls.mkdir(parents=True, exist_ok=True)
            (save_controls / f"chat-{chat}.json").write_text(json.dumps(body))
        start = time.monotonic()
        response = post(url, "/v1/responses", body)
        output = response["output"]
        answer = "".join(part["text"] for item in output if item["type"] == "message"
                         for part in item["content"] if part["type"] == "output_text").strip()
        reasoning = "".join(part["text"] for item in output if item["type"] == "reasoning"
                            for part in item["content"] if part["type"] == "reasoning_text")
        usage = response["usage"]
        n, cached = usage["input_tokens"], usage["input_tokens_details"]["cached_tokens"]
        record = dict(chat=chat, turn=turn, prompt=n, cached=cached, new=n-cached, generated=usage["output_tokens"],
                      reasoning=usage["output_tokens_details"]["reasoning_tokens"], answer=answer,
                      reasoning_sha256=hashlib.sha256(reasoning.encode()).hexdigest(),
                      status=response["status"], seconds=round(time.monotonic()-start, 3))
        print(json.dumps(record), flush=True)
        if not (lookup and record["status"] == "incomplete") and (record["status"] != "completed" or not answer):
            raise ValueError(f"chat {chat} turn {turn} did not finish")
        return response, record

    with_controls = []
    scores = []
    def conversation(chat):
        rng = random.Random(1430+chat)
        names = ["Alder", "Birch", "Cedar", "Larch", "Maple", "Oak"]
        depots = rng.sample(names, len(names))
        codes = {name: str(rng.randrange(100000, 1000000)) for name in names}
        questions = [
            "For each lot, retain its dispatch depot.\n" +
            "\n".join(f"Lot K{n+71} is assigned to the {name} depot." for n, name in enumerate(depots)),
            "Retain these depot verification numbers.\n" +
            "\n".join(f"The {name} depot has verification number {code}." for name, code in codes.items()),
        ]
        expected = codes[depots[1]]
        items, previous, live = [], None, 0
        for turn, size in enumerate(depths):
            batch = archive(url, size, chat*len(depths)+turn)
            if turn < len(questions):
                pos = batch.index("\n", len(batch)//2)
                batch = batch[:pos] + "\n" + questions[turn] + "\n" + batch[pos:]
            question = ("What is the verification number for the dispatch depot assigned to lot K72? Reply only with the number."
                        if turn == len(depths)-1 else "Acknowledge receipt with the single word Logged.")
            user = {"role": "user", "content": batch + "\n" + question}
            response, rec = send(chat, turn, [user], previous, lookup=turn == len(depths)-1)
            if rec["cached"] != live:
                raise ValueError(f"chat {chat} turn {turn}: expected {live} cached tokens, got {rec['cached']}")
            if turn == len(depths)-1:
                with_controls.append((chat, [*items, user], rec))
                scores.append(dict(chat=chat, expected=expected, answer=rec["answer"], correct=rec["answer"] == expected))
            previous = response["id"]
            items.extend([user, *response["output"]])
            live = rec["prompt"] + rec["generated"] - 1
            yield

        sample_reply, sampled = send(chat, "sampled", "Reply with just the lot identifier K72.", previous, temperature=1)
        if sampled["cached"] != live:
            raise ValueError(f"chat {chat}: sampling lost its prefix")
        yield

        _, picture, question, expected_image = setup_screenshots()[chat]
        image = "data:image/png;base64," + base64.b64encode(picture).decode()
        content = [dict(type="input_image", image_url=image), dict(type="input_text", text=question)]
        _, screenshot = send(chat, "screenshot", [dict(role="user", content=content)], sample_reply["id"])
        if screenshot["cached"] != sampled["prompt"] + sampled["generated"] - 1:
            raise ValueError(f"chat {chat}: the image turn lost its text prefix")
        scores.append(dict(chat=chat, screenshot=expected_image, answer=screenshot["answer"],
                           correct=screenshot["answer"].strip().replace(" %", "%").rstrip(".") == expected_image))
        yield

        # A compacted request deliberately starts a new Responses chain.
        summary = f"Compacted register {chat}. The audited verification number for lot K72 is {expected}."
        response, first = send(chat, "compaction", summary + " Acknowledge receipt with the single word Logged.")
        if first["cached"] != 0:
            raise ValueError(f"chat {chat}: expected a cold compaction")
        _, next_turn = send(chat, "after-compaction", "Give the audited number for lot K72. Reply only with the number.", response["id"])
        if next_turn["cached"] != first["prompt"] + first["generated"] - 1 or next_turn["answer"] != expected:
            raise ValueError(f"chat {chat}: compacted history failed")
        yield

    evict("start")
    chats = [conversation(i) for i in range(2)]
    if serial:
        for chat in chats:
            for _ in chat:
                pass
    else:
        for _ in range(len(depths)+3):
            for chat in chats:
                next(chat)

    replays = []
    for chat, inputs, warm in with_controls:
        evict(f"first-{chat}")
        _, cold = send(chat, "cold", inputs, store=False, lookup=True)
        if cold["cached"] or cold["prompt"] != warm["prompt"]:
            raise ValueError(f"chat {chat}: invalid cold control")
        comparison = dict(chat=chat, final_answer_matches=cold["answer"] == warm["answer"],
                          status_matches=cold["status"] == warm["status"],
                          reasoning_matches=cold["reasoning_sha256"] == warm["reasoning_sha256"],
                          warm_generated=warm["generated"], cold_generated=cold["generated"])
        if repeat_cold:
            evict(f"repeat-{chat}")
            _, repeated = send(chat, "cold-repeat", inputs, store=False, lookup=True)
            comparison["repeat_cold_identical"] = (repeated["cached"] == 0 and
                all(cold[k] == repeated[k] for k in ("prompt", "generated", "status", "answer", "reasoning_sha256")))
        replays.append(comparison)
    print(json.dumps(dict(lookups=scores, cold_replays=replays)), flush=True)


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("--batches", default="20000,80000,150000,95000",
                   help="additional archive tokens per turn; at least three batches")
    p.add_argument("--serial", action="store_true", help="use one cache to run the two conversations in sequence")
    p.add_argument("--repeat-cold", action="store_true", help="evict all slots and repeat the cold controls to check deterministic replay")
    p.add_argument("--save-controls", type=Path, help="save the exact cold request bodies for cross-model replay")
    a = p.parse_args()
    batches = [int(size) for size in a.batches.split(",")]
    if len(batches) < 3 or min(batches) < 128:
        p.error("supply at least three batches of 128 tokens or more")
    run(a.url.rstrip("/"), batches, a.serial, a.repeat_cold, a.save_controls)
