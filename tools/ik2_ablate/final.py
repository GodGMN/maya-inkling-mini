# final.py PORT OUT: every held-out harmful prompt answered in full (greedy, reasoning off, 1024 tokens) into OUT as
# {i, prompt, answer, finish} - for the user to validate.  Prints counts only; nothing of the text.
import json,sys
from urllib.request import Request,urlopen
from ablate_fit import refused
port,out=sys.argv[1],sys.argv[2]
ps=[json.loads(x) for x in open("sets/harmful_test.jsonl")]
n=0; fin={}
with open(out,"w") as f:
    for i,p in enumerate(ps):
        body=dict(model="inkling-2-mini",messages=[dict(role="user",content=p)],temperature=0,seed=7,reasoning_effort="none",max_tokens=1024)
        with urlopen(Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=900) as r: j=json.load(r)
        c=j["choices"][0]; a=c["message"]["content"] or ""
        n+=refused(a); fin[c["finish_reason"]]=fin.get(c["finish_reason"],0)+1
        f.write(json.dumps(dict(i=i,prompt=p,answer=a,finish=c["finish_reason"]))+"\n")
print(json.dumps(dict(prompts=len(ps),refused_by_detector=n,finish=fin)))
