#!/bin/sh
# bench-q.sh DIRSET SCALE NAME...: quick check of each direction dirSET/NAME.f32 at SCALE -> quick.jsonl
# (+ GSM8K 32 medium, code_16 HumanEval+ screen, and saved full answers for grade-sweep.sh)
cd "$(dirname "$0")"; B=$(pwd)/..; ds=$1; s=$2; shift 2
for v in "$@"; do
  t=$ds-$v-s$s${VISITS:+-r$(echo $VISITS | tr : _)}${TAGX:+-$TAGX}
  python3 preset.py $t IK2_ABLATE=/lab/$ds/$v.f32 IK2_ABLATE_SCALE=$s ${VISITS:+IK2_ABLATE_VISITS=$VISITS} IK2_BATCH=32 ${EXTRA}
  ./lab.sh $t >/dev/null || { echo "{\"tag\":\"$t\",\"error\":\"start\"}" >> quick.jsonl; continue; }
  g=$(cd $B && python3 src/tests/ik2_gsm8k.py --chat-url http://127.0.0.1:8087 --model-id inkling-2-mini --data fp4-overnight/results/gsm8k-test.jsonl --offset 0 --count ${QG:-32} --max-tokens 2048 --concurrency 32 2>/dev/null | python3 -c "import sys,json; r=[json.loads(l) for l in sys.stdin if l.startswith(\"{\")]; print(sum(x.get(\"correct\",False) for x in r), sum(x.get(\"finish\")==\"length\" for x in r))")
  c='{"code16":null,"code16_total":0,"code16_length":0}'
  if [ "${SKIP_CODE16:-0}" != "1" ]; then
    c=$(python3 code16.py --url http://127.0.0.1:8087 --model "${EVALQ_MODEL:-inkling-2-mini}" --data "$B/fp4-overnight/results/humanevalplus.jsonl" --max-tokens "${QCODE_MT:-3072}" --out "eval/$t-code16.jsonl" 2>"eval/$t-code16.err" || echo "$c")
  fi
  ./py evalq.py 8087 $t ${QN:-104} ${QMT:-2048} ${QM:-64} ${QSAVE:-eval/$t-answers.jsonl} | python3 -c "import sys,json; d=json.loads(sys.stdin.read()); g=\"$g\".split(); c=json.loads(sys.argv[1]); d.update(gsm=int(g[0]),gsm_length=int(g[1]),gsm_n=int(\"${QG:-32}\"),code16=c.get(\"code16\"),code16_total=c.get(\"code16_total\"),code16_length=c.get(\"code16_length\")); print(json.dumps(d))" "$c" >> quick.jsonl
done
