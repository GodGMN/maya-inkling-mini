# mmlu.py PORT DATA OUT: multiple choice, reasoning off, the first A-D letter of the answer; prints correct/total
import json,re,sys
from urllib.request import Request,urlopen
port,data,out=sys.argv[1:4]; ok=n=0
with open(out,"w") as f:
    for l in open(data):
        q=json.loads(l)
        p=q["question"]+"\n\n"+"\n".join(f"{c}. {t}" for c,t in zip("ABCD",q["choices"]))+"\n\nAnswer with only the letter of the correct choice."
        body=dict(model="inkling-2-mini",messages=[dict(role="user",content=p)],temperature=0,seed=7,reasoning_effort="none",max_tokens=8)
        with urlopen(Request(f"http://127.0.0.1:{port}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=600) as r: a=json.load(r)["choices"][0]["message"]["content"] or ""
        m=re.search(r"[ABCD]",a); g=m and "ABCD".index(m.group(0))==q["answer"]; ok+=bool(g); n+=1
        f.write(json.dumps(dict(subject=q["subject"],index=q["index"],answer=a,correct=bool(g)))+"\n")
print(json.dumps(dict(mmlu_correct=ok,mmlu_n=n)))
