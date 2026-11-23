# ask.py PORT PROMPT [MAX_TOKENS]: one greedy chat request, reasoning off; prints the answer
import json,sys
from urllib.request import Request,urlopen
body=dict(model="inkling-2-mini",messages=[dict(role="user",content=sys.argv[2])],temperature=0,seed=7,reasoning_effort="none",max_tokens=int(sys.argv[3]) if len(sys.argv)>3 else 64)
with urlopen(Request(f"http://127.0.0.1:{sys.argv[1]}/v1/chat/completions",json.dumps(body).encode(),{"Content-Type":"application/json"}),timeout=600) as r: j=json.load(r)
print(repr(j["choices"][0]["message"]["content"]))
