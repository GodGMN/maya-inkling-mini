# final2.py PORT OUT: the held-out harmful prompts answered in full with medium reasoning (2,048 tokens) and with
# reasoning off (1,024), 32 at a time, into OUT as {i, reasoning_effort, prompt, reasoning, answer, finish} - for the
# user to validate.  Prints counts only; nothing of the text.
import json,sys
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
import pathlib as _pl, sys as _sys; _sys.path.append(str(_pl.Path(__file__).resolve().parents[1]))  # tools/ beside the lab; a copy next to the script wins
from ablate_fit import refused
port,out=sys.argv[1],sys.argv[2]
ps=[json.loads(x) for x in open("sets/harmful_test.jsonl")]
jobs=[(i,p,e) for e,mt in (("medium",2048),("none",1024)) for i,p in enumerate(ps)]
def one(j):
    i,p,e=j
    body=dict(model="inkling-2-mini",messages=[dict(role="user",content=p)],temperature=0,seed=7,reasoning_effort=e,max_tokens=2048 if e=="medium" else 1024)
    with urlopen(Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=1800) as r: c=json.load(r)["choices"][0]
    m=c["message"]
    return dict(i=i,reasoning_effort=e,prompt=p,reasoning=m.get("reasoning_content") or "",answer=m.get("content") or "",finish=c["finish_reason"])
n={}; fin={}
with open(out,"w") as f, ThreadPoolExecutor(32) as pool:
    for row in pool.map(one,jobs):
        f.write(json.dumps(row)+"\n")
        n[row["reasoning_effort"]]=n.get(row["reasoning_effort"],0)+refused(row["answer"])
        k=row["reasoning_effort"]+":"+row["finish"]; fin[k]=fin.get(k,0)+1
print(json.dumps(dict(rows=len(jobs),refused_by_detector=n,finish=fin)))
