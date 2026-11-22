"""Exercise parallel tool calls and result feedback against a live Inkling server."""
import argparse
import json
from urllib.request import Request, urlopen

VALUES = {"Larch": "VX-731", "Birch": "PK-284"}
SCHEMA = {"type": "object", "properties": {"desk": {"type": "string", "enum": list(VALUES)}},
          "required": ["desk"]}
QUESTION = "Call read_pass once for Larch and once for Birch, in parallel in this turn. Then report both returned pass codes."
NAME = "read_pass"


def post(url, body, stream):
    req = Request(url, json.dumps(body).encode(), {"Content-Type": "application/json", "anthropic-version": "2023-06-01"})
    with urlopen(req, timeout=180) as response:
        if not stream:
            return json.load(response)
        return [line[6:].strip().decode() for line in response if line.startswith(b"data: ")]


def openai_turn(url, body, stream):
    raw = post(url + "/v1/chat/completions", body, stream)
    if not stream:
        choice, = raw["choices"]
        return choice["message"], choice["finish_reason"]
    if not raw or raw[-1] != "[DONE]":
        raise ValueError("missing OpenAI stream terminator")
    message = {"role": "assistant", "content": ""}
    calls = {}
    finish = None
    for event in map(json.loads, raw[:-1]):
        for choice in event["choices"]:
            delta = choice["delta"]
            for key in ("content", "reasoning_content"):
                if delta.get(key):
                    message[key] = message.get(key, "") + delta[key]
            for fragment in delta.get("tool_calls", []):
                call = calls.setdefault(fragment["index"], {"type": "function", "id": "", "function": {"name": "", "arguments": ""}})
                call["id"] += fragment.get("id") or ""
                for key in ("name", "arguments"):
                    call["function"][key] += fragment.get("function", {}).get(key) or ""
            finish = choice.get("finish_reason") or finish
    if calls:
        message["tool_calls"] = [calls[i] for i in sorted(calls)]
    return message, finish


def anthropic_turn(url, body, stream):
    raw = post(url + "/v1/messages", body, stream)
    if not stream:
        return {"role": "assistant", "content": raw["content"]}, raw["stop_reason"]
    blocks, arguments = {}, {}
    finish = None
    events = [json.loads(s) for s in raw]
    if not events or events[-1]["type"] != "message_stop":
        raise ValueError("missing Anthropic stream terminator")
    for event in events:
        kind, i = event["type"], event.get("index")
        if kind == "content_block_start":
            blocks[i] = dict(event["content_block"])
        elif kind == "content_block_delta":
            delta = event["delta"]
            if delta["type"] == "input_json_delta":
                arguments[i] = arguments.get(i, "") + delta["partial_json"]
            else:
                for key in ("text", "thinking", "signature"):
                    if key in delta:
                        blocks[i][key] = blocks[i].get(key, "") + delta[key]
        elif kind == "content_block_stop" and i in arguments:
            blocks[i]["input"] = json.loads(arguments[i])
        elif kind == "message_delta":
            finish = event["delta"].get("stop_reason") or finish
    return {"role": "assistant", "content": [blocks[i] for i in sorted(blocks)]}, finish


def run(url, model, api, stream):
    messages = [{"role": "user", "content": QUESTION}]
    base = dict(model=model, messages=messages, temperature=0, max_tokens=1536, stream=stream)
    if api == "openai":
        base["tools"] = [{"type": "function", "function": {"name": NAME, "description": "Read one desk's assigned pass code.", "parameters": SCHEMA}}]
        turn = openai_turn
    else:
        base["tools"] = [{"name": NAME, "description": "Read one desk's assigned pass code.", "input_schema": SCHEMA}]
        turn = anthropic_turn
    assistant, finish = turn(url, base, stream)
    if api == "openai":
        calls = [(c["id"], c["function"]["name"], json.loads(c["function"]["arguments"])) for c in assistant.get("tool_calls", [])]
    else:
        calls = [(b["id"], b["name"], b["input"]) for b in assistant["content"] if b["type"] == "tool_use"]
    if (finish != ("tool_calls" if api == "openai" else "tool_use") or len(calls) != 2 or
            {args["desk"] for _, _, args in calls} != VALUES.keys() or
            len({i for i, _, _ in calls}) != 2 or any(not i or name != NAME for i, name, _ in calls)):
        raise ValueError(f"expected two distinct calls in one turn, got {finish=} {calls=}")
    messages.append(assistant)
    results = []
    for i, _, args in reversed(calls):
        value = json.dumps({"desk": args["desk"], "code": VALUES[args["desk"]]})
        if api == "openai":
            messages.append({"role": "tool", "tool_call_id": i, "content": value})
        else:
            results.append({"type": "tool_result", "tool_use_id": i, "content": value})
    if results:
        messages.append({"role": "user", "content": results})
    final, finish = turn(url, base, stream)
    answer = final.get("content") or ""
    if api == "anthropic":
        answer = " ".join(b["text"] for b in answer if b["type"] == "text")
    if finish != ("stop" if api == "openai" else "end_turn") or not all(v in answer for v in VALUES.values()):
        raise ValueError(f"tool results missing from final answer: {finish=} {answer=}")
    return dict(ids=[c[0] for c in calls], answer=answer)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("url")
    args = parser.parse_args()
    url = args.url.rstrip("/")
    with urlopen(url + "/v1/models", timeout=15) as response:
        model = json.load(response)["data"][0]["id"]
    failures = 0
    for api in ("openai", "anthropic"):
        for stream in (False, True):
            try:
                result = run(url, model, api, stream)
            except Exception as exc:
                result = {"error": str(exc)}
                failures += 1
            print(json.dumps(dict(api=api, stream=stream, **result)), flush=True)
    raise SystemExit(bool(failures))
