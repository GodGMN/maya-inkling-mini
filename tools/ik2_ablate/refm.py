# refm.py PORT: held-out harmful and harmless refusal counts with medium reasoning (2048 tokens), plus answers that loop,
# are empty or hit the token limit.  Counts only; no text stored.
import json,re,sys
from urllib.request import Request,urlopen
from ablate_fit import refused
port=sys.argv[1]; out={}
loop=lambda t: bool(re.search(r"(.{12,}?)\1{3,}",t,re.S))
for name in ("harmful_test","harmless_test"):
    c=dict(refused=0,loop=0,empty=0,length=0)
    for l in open(f"sets/{name}.jsonl"):
        body=dict(model="inkling-2-mini",messages=[dict(role="user",content=json.loads(l))],temperature=0,seed=7,reasoning_effort="medium",max_tokens=2048)
        with urlopen(Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900) as r: ch=json.load(r)["choices"][0]
        a=ch["message"]["content"] or ""
        c["refused"]+=refused(a); c["loop"]+=loop(a); c["empty"]+=not a.strip(); c["length"]+=ch["finish_reason"]=="length"
    out[name]=c
print(json.dumps(dict(medium_reasoning=out)))
