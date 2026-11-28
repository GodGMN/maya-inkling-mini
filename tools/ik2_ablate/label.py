# label.py PORT OUTDIR [CAPTURE_FILE]: every prompt of sets/{harmful,harmless}_{train,test}.jsonl once (greedy, reasoning
# off, 48 tokens); writes OUTDIR/<set>.jsonl rows {i, refused, finish, cap0, cap1} - the label and, with a capture file,
# the byte range this request appended to it.  No prompt or answer text is stored or printed: counts only.
import json,os,sys,pathlib
from concurrent.futures import ThreadPoolExecutor
from urllib.request import Request,urlopen
from ablate_fit import refused
port,out=sys.argv[1],pathlib.Path(sys.argv[2]); cap=sys.argv[3] if len(sys.argv)>3 else None
out.mkdir(exist_ok=True)
def ask(p):
    body=dict(model="inkling-2-mini",messages=[dict(role="user",content=p)],temperature=0,seed=7,reasoning_effort="none",max_tokens=48)
    with urlopen(Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=600) as r: j=json.load(r)
    c=j["choices"][0]; return c["message"]["content"] or "", c["finish_reason"]
size=lambda: os.path.getsize(cap) if cap and os.path.exists(cap) else 0
for name in ("harmful_train","harmless_train","harmful_test","harmless_test"):
    prompts=[json.loads(x) for x in open(f"sets/{name}.jsonl")]
    n=0
    with open(out/f"{name}.jsonl","w") as f:
        for i,p in enumerate(prompts):
            a=size(); text,fin=ask(p); b=size()
            r=refused(text); n+=r
            f.write(json.dumps(dict(i=i,refused=r,finish=fin,cap0=a,cap1=b))+"\n"); f.flush()
    print(name,len(prompts),"refused",n,flush=True)
