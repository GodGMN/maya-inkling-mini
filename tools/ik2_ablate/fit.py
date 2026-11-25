# fit.py LABELS CAPTURE [TOP]: a refusal direction per layer visit - the mean last-prompt-token state over the harmful
# training prompts stock refused minus the mean over the harmless ones it answered - ranked by the separation of the
# held-out sets along it (Cohen d: refused harmful test vs answered harmless test).  Writes dirs/v<visit>.f32 for the
# TOP best visits (unit length) and dirs/fit.json.  Counts and scores only.
import json,sys,pathlib
import numpy as np
lab,cap=pathlib.Path(sys.argv[1] if len(sys.argv)>2 else "labels-stock"),(sys.argv[2] if len(sys.argv)>2 else "/lab/collect.f32"); top=int(sys.argv[3]) if len(sys.argv)>3 else 6
C=2304; rec=np.dtype([("visit","<i4"),("pos","<i4"),("width","<i4"),("h","<f4",(C,))])
data=np.memmap(cap,"u1","r")
def states(name,want):
    rows=[json.loads(x) for x in open(lab/f"{name}.jsonl")]
    out=[]
    for r in rows:
        if r["refused"]!=want: continue
        n=(r["cap1"]-r["cap0"])//rec.itemsize
        if n!=48 or (r["cap1"]-r["cap0"])%rec.itemsize: continue
        a=np.frombuffer(data[r["cap0"]:r["cap1"]],rec)
        assert np.array_equal(a["visit"],np.arange(48)) and np.all(a["width"]==C)
        out.append(a["h"].astype("f8"))
    return np.stack(out)
if __name__=="__main__":
    hr,ha=states("harmful_train",True),states("harmless_train",False)
    tr,ta=states("harmful_test",True),states("harmless_test",False)
    print("train refused",len(hr),"answered",len(ha),"| test refused",len(tr),"answered",len(ta),flush=True)
    res=[]
    for v in range(48):
        d=hr[:,v].mean(0)-ha[:,v].mean(0); u=d/np.linalg.norm(d)
        a,b=tr[:,v]@u,ta[:,v]@u
        cohen=(a.mean()-b.mean())/np.sqrt((a.var()+b.var())/2)
        rel=np.linalg.norm(d)/np.linalg.norm(ha[:,v].mean(0))
        res.append(dict(visit=v,cohen=float(cohen),rel_norm=float(rel)))
    for r in res: print("visit %2d  d=%6.2f  |diff|/|mean|=%.3f"%(r["visit"],r["cohen"],r["rel_norm"]))
    lo,hi=6,42
    best=sorted([r for r in res if lo<=r["visit"]<hi],key=lambda r:-r["cohen"])[:top]
    pathlib.Path("dirs").mkdir(exist_ok=True)
    for r in best:
        v=r["visit"]; d=hr[:,v].mean(0)-ha[:,v].mean(0); (d/np.linalg.norm(d)).astype("<f4").tofile(f"dirs/v{v}.f32")
    json.dump(dict(all=res,best=best),open("dirs/fit.json","w"),indent=1)
    print("written",[r["visit"] for r in best])
