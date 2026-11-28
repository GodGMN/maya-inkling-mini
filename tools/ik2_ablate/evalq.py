# evalq.py PORT_OR_URL TAG [N] [MT] [NM] [SAVE_JSONL]: the quick check of a setting (runtime ablation preset OR
# standalone GGUF served via tintero or llama-server) - MMLU first 64 (reasoning off: correct, and answers that list
# several options), harmful test with medium reasoning (refused, hit the 2048 limit), harmless test medium (refused,
# hit the limit).  When SAVE_JSONL (or env EVALQ_SAVE) is set, saves full {i, set, reasoning_effort, prompt, reasoning,
# answer, finish} rows for batch grading by grade.py.  GSM8K runs beside it (bench-q.sh).  Prints one JSON line.
import json,os,pathlib,re,sys
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from ablate_fit import refused
target,tag=sys.argv[1],sys.argv[2]
url=target if "://" in target else f"http://127.0.0.1:{target}"
model=os.environ.get("EVALQ_MODEL","inkling-2-mini")
N=int(sys.argv[3]) if len(sys.argv)>3 else 104; MT=int(sys.argv[4]) if len(sys.argv)>4 else 2048
NM=int(sys.argv[5]) if len(sys.argv)>5 else 64
save_arg=sys.argv[6] if len(sys.argv)>6 else os.environ.get("EVALQ_SAVE","")
save_path=f"eval/{tag}-answers.jsonl" if save_arg=="1" else save_arg
out=dict(tag=tag,refusal_n=N,refusal_max_tokens=MT,mmlu_n=NM)
def chat(p,eff,mt):
    body=dict(model=model,messages=[dict(role="user",content=p)],temperature=0,seed=7,reasoning_effort=eff,max_tokens=mt)
    with urlopen(Request(f"{url.rstrip('/')}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900) as r: c=json.load(r)["choices"][0]
    m=c["message"]
    return m.get("content") or "", m.get("reasoning_content") or "", c["finish_reason"]
ok=bad=0
pool=ThreadPoolExecutor(32)   # the batch server decodes them together; one at a time otherwise
qs=[json.loads(l) for l in list(open("mmlu256.jsonl"))[:NM]]
ps=[q["question"]+"\n\n"+"\n".join(f"{c}. {t}" for c,t in zip("ABCD",q["choices"]))+"\n\nAnswer with only the letter of the correct choice." for q in qs]
for q,(a,_,_) in zip(qs,pool.map(lambda p: chat(p,"none",8),ps)):
    a=a.strip()
    bad+=len(set(re.findall(r"(?m)^\**([ABCD])\**\s*$",a)))>1
    ok+=bool(a) and a[0] in "ABCD" and "ABCD".index(a[0])==q["answer"]
out.update(mmlu=ok,mmlu_multi=bad)
saved_rows=[]
sets=["harmful_test","harmless_test"]
extra_set=os.environ.get("EVALQ_EXTRA_SET","")
if extra_set and os.path.exists(extra_set): sets.append(extra_set)
for name in sets:
    pfile=name if os.path.exists(name) else f"sets/{name}.jsonl"
    sname=pathlib.Path(pfile).stem
    prompts=[json.loads(l) for l in list(open(pfile))[:N]]
    r=ln=0
    for i,(p,(a,rc,fin)) in enumerate(zip(prompts,pool.map(lambda p: chat(p,"medium",MT),prompts))):
        r+=refused(a); ln+=fin=="length"
        if save_path:
            saved_rows.append(dict(i=i,set=sname,reasoning_effort="medium",prompt=p,reasoning=rc,answer=a,finish=fin))
    out[sname+"_refused_med"]=r; out[sname+"_length_med"]=ln
if save_path:
    sp=pathlib.Path(save_path); sp.parent.mkdir(parents=True,exist_ok=True)
    with sp.open("w") as f:
        for row in saved_rows: f.write(json.dumps(row)+"\n")
    out["answers_path"]=str(sp)
print(json.dumps(out),flush=True)
