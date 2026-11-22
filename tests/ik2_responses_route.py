"""Exercise parallel tool calls and a streamed follow-up through a live Responses route."""
import argparse
import json
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ik2_tool_conversations import NAME, QUESTION, SCHEMA, VALUES, matched_results


def check(url, direct_url):
    endpoint = url.rstrip("/") + "/v1/responses"
    settings = dict(model="inkling-2-mini", temperature=0, max_output_tokens=1536, store=True,
                    tools=[dict(type="function", name=NAME, description="Read one desk's assigned pass code.", parameters=SCHEMA)])

    def send(**fields):
        return urlopen(Request(endpoint, json.dumps({**settings, **fields}).encode(),
                               {"Content-Type": "application/json"}), timeout=180)

    with send(input=QUESTION) as response:
        first = json.load(response)
    calls = [item for item in first["output"] if item["type"] == "function_call"]
    if (first["status"] != "completed" or len(calls) != 2 or len({c["call_id"] for c in calls}) != 2 or
            any(c["name"] != NAME or not c["call_id"] for c in calls)):
        raise ValueError(f"unexpected first response: {first['status']}, {calls}")
    desks = [json.loads(c["arguments"])["desk"] for c in calls]
    if set(desks) != VALUES.keys():
        raise ValueError(f"unexpected desks: {desks}")
    results = [dict(type="function_call_output", call_id=c["call_id"], output=json.dumps({"desk": d, "code": VALUES[d]}))
               for c, d in reversed(list(zip(calls, desks)))]
    deltas = []
    final = None
    with send(previous_response_id=first["id"], input=results, stream=True) as response:
        for line in response:
            if not line.startswith(b"data: ") or line.strip() == b"data: [DONE]":
                continue
            event = json.loads(line[6:])
            if event["type"] == "response.output_text.delta":
                deltas.append(event["delta"])
            if event["type"] == "response.completed":
                final = event["response"]
    answer = "".join(deltas)
    if final is None or final["status"] != "completed" or not matched_results(answer):
        raise ValueError(f"incomplete or incorrect streamed result: {answer!r}")
    print(json.dumps(dict(nonstream=first["status"], calls=desks, previous_response_id=True, stream=final["status"], answer=answer)))
    for base in dict.fromkeys(filter(None, (url, direct_url))):
        try:
            with urlopen(base.rstrip("/") + "/v1/responses/" + first["id"], timeout=10) as response:
                stored = json.load(response)
                if stored["id"] != first["id"] or stored["output"] != first["output"]:
                    raise ValueError("retrieved a different response")
                code = response.status
        except HTTPError as error:
            code = error.code
        print(json.dumps(dict(retrieve=base, http=code)))
        if base == direct_url and code != 200:
            raise ValueError("direct response retrieval failed")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("url", help="server or LAN router base URL without /v1")
    p.add_argument("--direct-url", help="also require retrieval from the model port")
    args = p.parse_args()
    check(args.url, args.direct_url)
