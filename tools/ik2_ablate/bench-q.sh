#!/bin/sh
# bench-q.sh DIRSET SCALE NAME...: quick check of each direction dirSET/NAME.f32 at SCALE -> quick.jsonl (+ GSM8K 32 medium)
cd "$(dirname "$0")"; B=$(pwd)/..; ds=$1; s=$2; shift 2
for v in "$@"; do
  t=$ds-$v-s$s${VISITS:+-r$(echo $VISITS | tr : _)}${TAGX:+-$TAGX}
  python3 preset.py $t IK2_ABLATE=/lab/$ds/$v.f32 IK2_ABLATE_SCALE=$s ${VISITS:+IK2_ABLATE_VISITS=$VISITS} IK2_BATCH=32 ${EXTRA}
  ./lab.sh $t >/dev/null || { echo "{\"tag\":\"$t\",\"error\":\"start\"}" >> quick.jsonl; continue; }
  g=$(cd $B && python3 src/tests/ik2_gsm8k.py --chat-url http://127.0.0.1:8087 --model-id inkling-2-mini --data fp4-overnight/results/gsm8k-test.jsonl --offset 0 --count ${QG:-32} --max-tokens 2048 --concurrency 32 2>/dev/null | python3 -c "import sys,json; r=[json.loads(l) for l in sys.stdin if l.startswith(\"{\")]; print(sum(x.get(\"correct\",False) for x in r), sum(x.get(\"finish\")==\"length\" for x in r))")
  ./py evalq.py 8087 $t ${QN:-104} ${QMT:-2048} ${QM:-64} | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); g=\"$g\".split(); d.update(gsm=int(g[0]),gsm_length=int(g[1]),gsm_n=int(\"${QG:-32}\")); print(json.dumps(d))" >> quick.jsonl
done
