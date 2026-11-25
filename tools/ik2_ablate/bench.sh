#!/bin/sh
# bench.sh PRESET...: the same suite against each preset, into bench/<PRESET>/
cd "$(dirname "$0")"; B=$(pwd)/..
for p in "$@"; do
  d=bench/$p; mkdir -p $d; ./lab.sh $p > $d/start.log || continue
  (cd $B && python3 src/tests/ik2_gsm8k.py --chat-url http://127.0.0.1:8087 --model-id inkling-2-mini --data fp4-overnight/results/gsm8k-test.jsonl --offset 0 --count 128 --max-tokens 2048) > $d/gsm8k.jsonl 2> $d/gsm8k.err
  ./py mmlu.py 8087 /lab/mmlu256.jsonl /lab/$d/mmlu.jsonl > $d/mmlu.txt 2>&1
  (cd $B && python3 src/tests/ik2_coding.py --url http://127.0.0.1:8087 --model-id inkling-2-mini --data fp4-overnight/results/humanevalplus.jsonl) > $d/code.jsonl 2> $d/code.err
  (cd $B && python3 src/tests/ik2_tool_conversations.py http://127.0.0.1:8087 --model inkling-2-mini) > $d/tools.txt 2>&1
  ./py refm.py 8087 > $d/refm.txt 2>&1
  echo done > $d/DONE
done
