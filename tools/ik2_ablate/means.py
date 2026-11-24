# means.py LABELS POST_CAPTURE ATTN_CAPTURE DIR.f32 OUT.f32: the mean-ablation targets for IK2_ABLATE_MEAN - for
# each visit's attention state and MoE state, the mean over the answered harmless training prompts of r . state at
# the last prompt token (from a capm.py run with both captures on; the attention capture has 48 records a request,
# in request order).  The embedding's target is 0.  Prints the targets' spread; no text.
import json,sys,pathlib
import numpy as np
lab,post,attn,dirf,out=sys.argv[1:6]
C=2304; rec=np.dtype([("visit","<i4"),("pos","<i4"),("width","<i4"),("h","<f4",(C,))])
r=np.fromfile(dirf,"<f4").astype("f8"); r/=np.linalg.norm(r)
P=np.memmap(post,"u1","r"); A=np.fromfile(attn,rec)
order=[]   # (set, row) in the order capm.py sent them
for name in ("harmful_train","harmless_train","harmful_test","harmless_test"):
    order+= [(name,json.loads(x)) for x in open(pathlib.Path(lab)/f"{name}.jsonl")]
assert len(A)==48*len(order), (len(A),len(order))
pa,pm=[],[]
for k,(name,row) in enumerate(order):
    if name!="harmless_train" or row["refused"]: continue
    p=np.frombuffer(P[row["cap0"]:row["cap1"]],rec); a=A[48*k:48*k+48]
    assert len(p)==48 and np.array_equal(a["visit"],np.arange(48)) and np.array_equal(p["visit"],np.arange(48))
    pa.append(a["h"].astype("f8")@r); pm.append(p["h"].astype("f8")@r)
pa,pm=np.array(pa),np.array(pm)
c=np.zeros(1+2*48,"<f4"); c[1::2]=pa.mean(0); c[2::2]=pm.mean(0)
c.tofile(out)
print("prompts",len(pa),"| attention targets",np.round(pa.mean(0)[20:30],2),"| MoE targets",np.round(pm.mean(0)[20:30],2),"| sd",np.round(pm.std(0)[20:30],2))
