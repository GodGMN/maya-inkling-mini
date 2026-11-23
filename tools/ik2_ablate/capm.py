# capm.py PORT OUTDIR CAPTURE: every prompt once with MEDIUM reasoning and max_tokens 1 (prefill only - the capture is
# the last prompt token of the reasoning template); labels copied from labels-stock; rows {i, refused, cap0, cap1}
import json,os,sys,pathlib
from urllib.request import Request,urlopen
port,out,cap=sys.argv[1],pathlib.Path(sys.argv[2]),sys.argv[3]; out.mkdir(exist_ok=True)
size=lambda: os.path.getsize(cap) if os.path.exists(cap) else 0
for name in ("harmful_train","harmless_train","harmful_test","harmless_test"):
    lab=[json.loads(x) for x in open(f"labels-stock/{name}.jsonl")]
    with open(out/f"{name}.jsonl","w") as f:
        for i,l in enumerate(open(f"sets/{name}.jsonl")):
            body=dict(model="inkling-2-mini",messages=[dict(role="user",content=json.loads(l))],temperature=0,seed=7,reasoning_effort="medium",max_tokens=1)
            a=size(); urlopen(Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=600).read(); b=size()
            f.write(json.dumps(dict(i=i,refused=lab[i]["refused"],cap0=a,cap1=b))+"\n")
    print(name,"done",flush=True)
