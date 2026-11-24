#!/bin/sh
# final-bench.sh TAG [KEY=VALUE ...]: a finalist's long checks - all 164 HumanEval+ tasks (8 shards at once on the
# batch server), the tool conversations, then the single-request decode speed (batching off, 3 x 512 tokens)
cd "$(dirname "$0")"; B=$(pwd)/..; t=$1; shift; d=bench/$t; mkdir -p $d
python3 preset.py $t-batch IK2_BATCH=32 "$@" && ./lab.sh $t-batch > $d/start.log || exit 1
for s in 0 21 42 63 84 105 126 147; do
  (cd $B && python3 src/tests/ik2_coding.py --url http://127.0.0.1:8087 --model-id inkling-2-mini \
     --data fp4-overnight/results/humanevalplus.jsonl --start $s --count 21 > ablate-lab/$d/code-$s.jsonl 2>/dev/null) &
done
wait
cat $d/code-*.jsonl > $d/code.jsonl
(cd $B && python3 src/tests/ik2_tool_conversations.py http://127.0.0.1:8087 --model inkling-2-mini) > $d/tools.txt 2>&1
python3 preset.py $t-single "$@" && ./lab.sh $t-single >> $d/start.log || exit 1
./py -c "
import json,statistics
from urllib.request import Request,urlopen
v=[]
for i in range(3):
    body=dict(model=\"inkling-2-mini\",messages=[dict(role=\"user\",content=\"Write a long, detailed essay about the history of bridges.\")],temperature=0,seed=i,reasoning_effort=\"none\",max_tokens=512,ignore_eos=True)
    j=json.load(urlopen(Request(\"http://127.0.0.1:8087/v1/chat/completions\",json.dumps(body).encode(),{\"Content-Type\":\"application/json\"}),timeout=600))
    v.append(j.get(\"timings\",{}).get(\"predicted_per_second\"))
print(json.dumps(dict(decode_tok_s=v,median=statistics.median([x for x in v if x]))))" > $d/speed.txt 2>&1
echo done > $d/DONE
