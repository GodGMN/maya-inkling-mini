#!/bin/sh
# sweep.sh SCALE VISIT...: for each visit, restart the lab with dirs/v<VISIT>.f32 at SCALE and append evalr.py s summary to sweep.jsonl
cd "$(dirname "$0")"; s=$1; shift
for v in "$@"; do
  t=v$v-s$s
  python3 preset.py $t IK2_ABLATE=/lab/dirs/$v.f32 IK2_ABLATE_SCALE=$s
  ./lab.sh $t >/dev/null || { echo "{\"tag\":\"$t\",\"error\":\"start\"}" >> sweep.jsonl; continue; }
  ./py evalr.py 8087 $t >> sweep.jsonl 2>> sweep.err || echo "{\"tag\":\"$t\",\"error\":\"eval\"}" >> sweep.jsonl
done
echo SWEEP-DONE >> sweep.jsonl
