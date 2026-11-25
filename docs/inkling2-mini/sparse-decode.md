# Sparse decode trial on the RX 7900 XTX

Set `IK2_SPARSE_DECODE=1` with the single-stream Radeon FP8 preset to try sparse global attention during generation from position 32,768. The default is dense. Both modes keep the full KV allocation and run dense prefill, including a one-token prompt extension. The experiment uses the resident decoder without speculative verification or paged KV.

Each global KV head selects 224 pages of 64 positions for each generated token. The first two pages and the newest 24 are always included. The remaining 198 pages come from equal-sized ranges of the history. Within each range, the kernel samples keys at offsets 16 and 48 of every page and chooses the strongest match to any query sharing that KV head. Attention then reads all live keys and values in the selected pages, using their original positions. This heuristic can miss a match between the sampled keys. It requires only an array of selected page indices and can use an existing densely prefilled cache. Nib's Quest scorer uses different page summaries.

## Paired run

The prepared Prometheus build lives in `~/ai/build/maya-ik2-hip/sparse-decode.78Gkvr`. Once the GPU is released, run this on Prometheus:

```sh
cd ~/ai/build/maya-ik2-hip/sparse-decode.78Gkvr
bash src/tools/ik2_sparse_trial.sh --run
```

The launcher verifies file hashes, runs the gfx1100 kernel suite, then runs 22 pairs on the stock IQ4_NL model at 393,216 reserved positions. It starts a fresh engine for each arm and alternates arm order. The only engine setting that changes within a pair is sparse decode. Each arm answers a cold question and runs three fixed 128-step timings. These timings exclude prefill. Use `--check` for the device-free preflight.

The four synthetic depths are 8,192, 32,769, 131,072 and 258,048 tokens. At each depth the model retrieves phrases placed early, midway and late, then solves a three-record inventory question. Six public eight-needle MRCR cases range from 23,365 to 348,465 tokens and require reproducing a specific earlier answer. The harness checks sparse dispatch, the first prediction after dense prefill, and identical greedy output on short controls.

Read `results/run.*/quality/summary.json` for lost, gained and mutually missed cases. Each row in `pairs.jsonl` contains both answers, the expected answer, exact scores, string similarity, finish reasons, cold prefill times and all three decode rates. Review losses even if gains offset them. Both arms missing a case provides little evidence of preserved quality. A loss or absence of any dense-correct long example gives a nonzero exit. Use these results to decide which settings merit a larger evaluation.

To create another fixture set on CPU, run `python3 tests/ik2_sparse_decode.py prepare --help`. It takes an exported tokenizer, GSM8K JSONL distractors and optional MRCR JSON records. The manifest stores the hash of every request.
