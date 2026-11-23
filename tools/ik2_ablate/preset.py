# preset.py NAME [KEY=VALUE ...]: a lab preset (no vision, 32K context) with extra engine env, written to presets/NAME.json
import json,sys,pathlib
here=pathlib.Path(__file__).resolve().parent
cfg=json.loads((here/"base-preset.json").read_text())
cfg.pop("vision",None)
a=cfg["args"]; a[a.index("--max-context")+1]="32768"
cfg["model_name"]="inkling-2-mini"
for kv in sys.argv[2:]:
    k,v=kv.split("=",1); cfg["env"][k]=v
(here/"presets").mkdir(exist_ok=True)
(here/"presets"/(sys.argv[1]+".json")).write_text(json.dumps(cfg,indent=2)+"\n")
