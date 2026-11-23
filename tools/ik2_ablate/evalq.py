# evalq.py PORT TAG: the quick check of a setting - MMLU first 64 (reasoning off: correct, and answers that list several
# options), harmful test with medium reasoning (refused, hit the 2048 limit; counts only), harmless test medium
# (refused, hit the limit).  GSM8K runs beside it (bench-q.sh).  Prints one JSON line.
import json,re,sys
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from ablate_fit import refused
port,tag=sys.argv[1],sys.argv[2]; N=int(sys.argv[3]) if len(sys.argv)>3 else 104; MT=int(sys.argv[4]) if len(sys.argv)>4 else 2048
NM=int(sys.argv[5]) if len(sys.argv)>5 else 64
out=dict(tag=tag,refusal_n=N,refusal_max_tokens=MT,mmlu_n=NM)
def chat(p,eff,mt):
    body=dict(model="inkling-2-mini",messages=[dict(role="user",content=p)],temperature=0,seed=7,reasoning_effort=eff,max_tokens=mt)
    with urlopen(Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900) as r: c=json.load(r)["choices"][0]
    return c["message"]["content"] or "",c["finish_reason"]
ok=bad=0
pool=ThreadPoolExecutor(32)   # the batch server decodes them together; one at a time otherwise
qs=[json.loads(l) for l in list(open("mmlu256.jsonl"))[:NM]]
ps=[q["question"]+"\n\n"+"\n".join(f"{c}. {t}" for c,t in zip("ABCD",q["choices"]))+"\n\nAnswer with only the letter of the correct choice." for q in qs]
for q,(a,_) in zip(qs,pool.map(lambda p: chat(p,"none",8),ps)):
    a=a.strip()
    bad+=len(set(re.findall(r"(?m)^\**([ABCD])\**\s*$",a)))>1
    ok+=bool(a) and a[0] in "ABCD" and "ABCD".index(a[0])==q["answer"]
out.update(mmlu=ok,mmlu_multi=bad)
for name in ("harmful_test","harmless_test"):
    r=ln=0
    for a,fin in pool.map(lambda p: chat(p,"medium",MT),[json.loads(l) for l in list(open(f"sets/{name}.jsonl"))[:N]]):
        r+=refused(a); ln+=fin=="length"
    out[name+"_refused_med"]=r; out[name+"_length_med"]=ln
print(json.dumps(out),flush=True)
