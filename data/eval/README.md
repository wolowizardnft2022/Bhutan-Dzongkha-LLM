# Eval holdout (smoke)

`holdout.jsonl` is a tiny non-production set (~6 lines) copied from
`data/sample/dzongkha_sft_sample.jsonl` for wiring checks of `eval.py`.

For real evaluation:
- Use a larger holdout that does **not** overlap with training data.
- Prefer `scripts/split_eval.py` on your full JSONL, or hand-curate held-out lines.
