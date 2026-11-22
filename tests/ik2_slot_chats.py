"""Alternate growing OpenAI conversations, then verify their last answers on independent cold prompts."""
import argparse
import json
from ik2_chat_growth import archive, ask, post


def run(args):
    url = args.url.rstrip("/")
    def evict():
        for i in range(args.slots):
            post(url, "/v1/chat/completions", dict(model="inkling-2-mini", temperature=0, max_tokens=1,
                 messages=[{"role": "system", "content": f"Eviction control {i}."}, {"role": "user", "content": "Hi"}]))
    evict()
    sessions = []
    for index in range(args.slots):
        seal = f"M{419 + index * 197}"
        messages = [{"role": "system", "content": "You operate a parcel desk. Keep its seal code."}]
        sessions.append(dict(messages=messages, seal=seal, previous=0))
    for turn in range(3):
        for index, session in enumerate(sessions):
            messages = session["messages"]
            if turn == 0:
                content = (archive(url, args.depth - 128, index) +
                           f"\nThe seal code of this desk is {session['seal']}. Reply with the seal code alone.")
            else:
                content = "What is the desk's seal code? Return only the code."
            messages.append({"role": "user", "content": content})
            body = dict(model="inkling-2-mini", temperature=0, reasoning_effort="none", max_tokens=48, messages=messages)
            record = ask(url, body, f"slot-{index}", turn, session["previous"], args.server_pid)
            if record["answer"]["content"] != session["seal"]:
                raise ValueError(f"wrong desk on turn {turn} in slot {index}")
            session["previous"] = record["prompt"] + record["generated"] - 1
            session["request"] = dict(body, messages=list(messages))
            session["record"] = record
            messages.append(record["answer"])
    # Replace every slot with a different short system prefix before the cold controls.
    evict()
    for index, session in enumerate(sessions):
        cold = ask(url, session["request"], "cold-slot", index, 0, args.server_pid)
        if any(cold[k] != session["record"][k] for k in ("prompt", "answer", "generated", "finish")):
            raise ValueError(f"cold replay differs for desk {index}")
    print(json.dumps(dict(slots=args.slots, depth=args.depth, verified=True)))


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("--slots", type=int, required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--server-pid", type=int)
    args = p.parse_args()
    if args.depth < 256 or args.slots < 1:
        p.error("supply a depth of at least 256 and a positive slot count")
    run(args)
