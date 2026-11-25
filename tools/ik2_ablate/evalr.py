# evalr.py PORT TAG [N_COH_TOKENS]: refusal counts on the held-out harmful and harmless prompts (greedy, reasoning off, 48
# tokens; nothing stored but the counts) and the 16 everyday coherence prompts answered in full (written to
# eval/TAG-coherence.jsonl: harmless).  Prints one JSON summary line.
import json,sys,pathlib,re
from urllib.request import Request,urlopen
from ablate_fit import refused
port,tag=sys.argv[1],sys.argv[2]; nt=int(sys.argv[3]) if len(sys.argv)>3 else 256
def ask(p,mt):
    body=dict(model="inkling-2-mini",messages=[dict(role="user",content=p)],temperature=0,seed=7,reasoning_effort="none",max_tokens=mt)
    with urlopen(Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900) as r: j=json.load(r)
    return j["choices"][0]["message"]["content"] or ""
out=dict(tag=tag)
for name in ("harmful_test","harmless_test"):
    ps=[json.loads(x) for x in open(f"sets/{name}.jsonl")]
    out[name+"_refused"]=sum(refused(ask(p,48)) for p in ps); out[name+"_n"]=len(ps)
pathlib.Path("eval").mkdir(exist_ok=True)
coh=[json.loads(x) for x in open("sets/coherence.jsonl")]
rows=[dict(prompt=p,answer=ask(p,nt)) for p in coh]
open(f"eval/{tag}-coherence.jsonl","w").write("".join(json.dumps(r)+"\n" for r in rows))
loop=lambda t: bool(re.search(r"(.{12,}?)\1{3,}",t,re.S))
out["coherence_empty"]=sum(not r["answer"].strip() for r in rows); out["coherence_loops"]=sum(loop(r["answer"]) for r in rows)
out["coherence_refused"]=sum(refused(r["answer"]) for r in rows); out["coherence_chars"]=sum(len(r["answer"]) for r in rows)
print(json.dumps(out),flush=True)
