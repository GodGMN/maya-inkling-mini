# preset.py NAME [KEY=VALUE ...]: a lab preset (no vision, 32K context) with extra engine env, written to presets/NAME.json
import json,sys,pathlib
here=pathlib.Path(__file__).resolve().parent
cfg=json.loads((here/"base-preset.json").read_text())
cfg.pop("vision",None)
a=cfg["args"]; a[a.index("--max-context")+1]="32768"
batch=next((kv.split("=",1)[1] for kv in sys.argv[2:] if kv.startswith("IK2_BATCH=")),None)
per=int(next((kv.split("=",1)[1] for kv in sys.argv[2:] if kv.startswith("SLOT_CTX=")),"4096"))
if batch and int(batch)>1:   # the batch server: SLOT_CTX (4,096) tokens a request, 256-token chunks (docs/HIP-PORT.md)
    a[a.index("--max-context")+1]=str(per*int(batch)); a[a.index("--chunk")+1]="256"
cfg["model_name"]="inkling-2-mini"
for kv in sys.argv[2:]:
    k,v=kv.split("=",1)
    if k=="MODEL": a[a.index("--model")+1]=v   # another model file, e.g. a baked override pair
    elif k=="SLOT_CTX": pass
    else: cfg["env"][k]=v
(here/"presets").mkdir(exist_ok=True)
(here/"presets"/(sys.argv[1]+".json")).write_text(json.dumps(cfg,indent=2)+"\n")
