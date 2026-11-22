"""Alternate growing OpenAI conversations, then verify their last answers on independent cold prompts."""
import argparse
import base64
import json
from ik2_chat_growth import archive, ask, post
from ik2_image_answers import png


def run(args):
    url = args.url.rstrip("/")
    def evict():
        for i in range(args.slots):
            post(url, "/v1/chat/completions", dict(model="inkling-2-mini", temperature=0, max_tokens=1,
                 messages=[{"role": "system", "content": f"Eviction control {i}."}, {"role": "user", "content": "Hi"}]))
    evict()
    sessions = []
    for index in range(args.slots):
        seal = f"H{317 + index * 73}"
        messages = [{"role": "system", "content": "You maintain a parcel register. Answer the question after each batch."}]
        sessions.append(dict(messages=messages, seal=seal, previous=0))
    for turn in range(3):
        for index, session in enumerate(sessions):
            messages = session["messages"]
            if turn == 0:
                content = (archive(url, args.depth - 128, index) +
                           f"\nThe current seal is {session['seal']}. Reply only with the current seal.")
            else:
                content = "What is the current seal? Reply only with the current seal."
            messages.append({"role": "user", "content": content})
            body = dict(model="inkling-2-mini", temperature=0, reasoning_effort="none", max_tokens=48, messages=messages)
            record = ask(url, body, f"slot-{index}", turn, session["previous"], args.server_pid)
            if record["answer"]["content"].strip() != session["seal"]:
                evict()
                cold = ask(url, body, "cold-mismatch", turn, 0, args.server_pid)
                raise ValueError(f"desk {index} expected {session['seal']}, warm {record['answer']}, cold {cold['answer']}")
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


def check_images(args):
    colors = {"red": (236, 24, 24), "blue": (16, 40, 236)}
    chats = []
    for left, right in [("red", "blue"), ("blue", "red")]:
        data = base64.b64encode(png(colors[left], colors[right])).decode()
        content = [{"type": "image_url", "image_url": {"url": "data:image/png;base64," + data}},
                   {"type": "text", "text": "What color is the left half? Reply with one color."}]
        chats.append(dict(left=left, right=right, messages=[{"role": "user", "content": content}], cached=0))
    for turn, side in enumerate(("left", "right")):
        for chat in chats:
            if turn:
                chat["messages"].append({"role": "user", "content": "What color is the right half? Reply with one color."})
            body = dict(model="inkling-2-mini", messages=chat["messages"], temperature=0, reasoning_effort="none", max_tokens=32)
            result = ask(args.url.rstrip("/"), body, f"image-{chat['left']}", turn, chat["cached"], args.server_pid)
            if result["answer"]["content"].strip().lower().rstrip(".") != chat[side]:
                raise ValueError(f"wrong {side} half of {chat['left']}/{chat['right']} image")
            chat["messages"].append(result["answer"])
            chat["cached"] = result["prompt"] + result["generated"] - 1


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url")
    p.add_argument("--slots", type=int, required=True)
    p.add_argument("--depth", type=int, required=True)
    p.add_argument("--server-pid", type=int)
    p.add_argument("--images", action="store_true", help="also alternate chats with the same text and different images")
    args = p.parse_args()
    if args.depth < 256 or args.slots < 1:
        p.error("supply a depth of at least 256 and a positive slot count")
    if args.images and args.slots < 2:
        p.error("--images needs two or more cache slots")
    run(args)
    if args.images:
        check_images(args)
