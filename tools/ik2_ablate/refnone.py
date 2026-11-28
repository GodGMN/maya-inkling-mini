# refnone.py PORT TAG: harmful and harmless test refusals with reasoning OFF (greedy, 512 tokens, 32 at a time);
# counts only, one JSON line
import json,sys
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
import pathlib as _pl, sys as _sys; _sys.path.append(str(_pl.Path(__file__).resolve().parents[1]))  # tools/ beside the lab; a copy next to the script wins
from ablate_fit import refused
port,tag=sys.argv[1],sys.argv[2]
def ask(p):
    body=dict(model="inkling-2-mini",messages=[dict(role="user",content=p)],temperature=0,seed=7,reasoning_effort="none",max_tokens=512)
    with urlopen(Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900) as r: c=json.load(r)["choices"][0]
    return c["message"]["content"] or ""
out=dict(tag=tag)
with ThreadPoolExecutor(32) as pool:
    for name in ("harmful_test","harmless_test"):
        out[name+"_refused_none"]=sum(refused(a) for a in pool.map(ask,[json.loads(l) for l in open(f"sets/{name}.jsonl")]))
print(json.dumps(out),flush=True)
