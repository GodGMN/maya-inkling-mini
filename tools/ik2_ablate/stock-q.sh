#!/bin/sh
# stock-q.sh: the quick check on stock through the batch server (the reference row of quick.jsonl)
cd "$(dirname "$0")"; B=$(pwd)/..
t=stock-batch
python3 preset.py $t IK2_BATCH=32
./lab.sh $t >/dev/null || exit 1
g=$(cd $B && python3 src/tests/ik2_gsm8k.py --chat-url http://127.0.0.1:8087 --model-id inkling-2-mini --data fp4-overnight/results/gsm8k-test.jsonl --offset 0 --count ${QG:-32} --max-tokens 2048 --concurrency 32 2>/dev/null | python3 -c "import sys,json; r=[json.loads(l) for l in sys.stdin if l.startswith(\"{\")]; print(sum(x.get(\"correct\",False) for x in r), sum(x.get(\"finish\")==\"length\" for x in r))")
./py evalq.py 8087 $t ${QN:-104} ${QMT:-2048} ${QM:-64} | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); g=\"$g\".split(); d.update(gsm=int(g[0]),gsm_length=int(g[1]),gsm_n=int(\"${QG:-32}\")); print(json.dumps(d))" >> quick.jsonl
