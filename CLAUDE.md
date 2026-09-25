# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A generator for a chat-format training dataset (`data/train.jsonl` / `data/val.jsonl`) used to fine-tune a small LLM
on the Beancount Query Language (BQL, via the `beanquery` package for Beancount v3), plus the scripts to train it
with Unsloth and package the result for Ollama. See [README.md](README.md) for the full picture (what's trained,
why there's no system prompt, the mailing-list-sourced intents, limits). Don't duplicate that here — this file is
about working in the code, not the training design.

## Two separate Python environments

- **This repo's own environment** (`pip install -e .`): `beancount` + `beanquery`, no GPU needed. Used for
  `generate_dataset.py`, `audit_dataset.py`, and both test files.
- **A separate GPU environment with Unsloth installed**, not a dependency of this repo (it isn't in
  `pyproject.toml`/`requirements.txt` on purpose — it needs a specific CUDA/torch stack and doesn't belong on a
  machine just auditing or regenerating the dataset). Only `scripts/train.py` and `scripts/export_gguf.py` need it;
  both run fine without `beancount`/`beanquery` installed at all (they only read pre-built `.jsonl` files).

`beanquery` is pinned to `git+https://github.com/beancount/beanquery.git` (git HEAD), not the PyPI release: the
`HAVING`/`PIVOT BY` support and several functions this dataset exercises aren't in PyPI's `0.2.0`. Don't "fix" this
to a version pin without checking those features still work.

## Commands

```bash
pip install -e .                                                          # this repo's own deps (no GPU)

python scripts/generate_dataset.py --ledgers 350 --per-ledger 18 --seed 1 --out data   # regenerate data/
python scripts/audit_dataset.py data/train.jsonl data/val.jsonl                        # structure + parses + no ledger leaks

python tests/test_modelfile.py                                            # or: pytest tests/test_modelfile.py
python tests/test_smoke.py                                                # every intent, every prompt mode (~3 min)
# a single test: pytest tests/test_smoke.py::test_all_intents_fire_and_validate_in_every_mode -q
# (neither file needs pytest to run: python <file>.py runs every test_* in it and prints "ok")

python scripts/make_modelfile.py --check                                  # fails if ollama/Modelfile is stale
python scripts/make_modelfile.py                                          # regenerate it

# needs the separate Unsloth/GPU environment:
python scripts/train.py --max-steps 2                                     # smoke test: 2 optimizer steps
python scripts/train.py                                                   # full run, data/, 2 epochs
python scripts/export_gguf.py                                             # merge + quantize a trained adapter
```

There is no linter or formatter configured in this repo; don't add one unasked.

## Architecture: the validate-by-execution pipeline

Every example in the dataset was actually run against `beanquery`, not templated and trusted. The chain, across
`ledger.py` -> `intents.py` -> `executor.py` -> `generate_dataset.py`:

1. `ledger.py` builds a random-but-valid synthetic Beancount ledger and loads it with the real
   `beancount.loader` (so balance checks etc. actually run).
2. An **intent** function in `intents.py` picks concrete parameters from that loaded `Ledger` (an account, a
   date range, a threshold...) and derives *both* the English question and the BQL statement from the same
   parameters — they cannot drift apart because neither is written independently of the other.
3. `executor.py` runs the BQL against the ledger with `beanquery`. `generate_dataset.py` drops the example if it
   fails to parse, fails to compile, raises, or returns nothing (see `Sample.allow_empty` for the few legitimate
   exceptions, e.g. a deliberately non-matching regex test).
4. `reference.py` (tables/columns/function signatures, sourced from `beanquery`'s live registry, and core-concept
   Q&A) is validated the same way: every ```sql block embedded in a reference answer is executed while the
   dataset is built, and building fails if one doesn't run.

New intents should follow this pattern — derive the question and query from shared parameters, don't hand-write a
plausible-looking pair. `scripts/audit_dataset.py`'s leak check specifically looks for ledger-specific literals
(account names, currencies, keyword regexes) in the BQL that aren't traceable to the question text, which catches
the most common way a hand-written pair goes wrong.

## Architecture: dataset -> Modelfile agreement

`src/bql_lora/modelfile.py` generates `ollama/Modelfile` (via `scripts/make_modelfile.py`) — it's never hand-edited,
because it has to agree with what training actually used: no system message in the data (Qwen's chat template
fills in its own default at training time), so the Modelfile bakes that same default line into `SYSTEM` explicitly.
`tests/test_modelfile.py` checks the committed file matches the generator's output and that everything is LF-only
(Ollama keeps a stray `\r` from a CRLF checkout, which silently changes what the model sees vs. training —
`.gitattributes` pins `ollama/Modelfile` to LF for this reason). If you change `modelfile.py`'s `TEMPLATE` or the
training data's message format, both sides of that agreement need to move together.

## Workflow: PRs, not direct commits

Changes to this repo go through a branch + PR (`git checkout -b ...`, push the branch, `gh pr create`), not direct
commits to `main`. Open the PR, summarize what to look at, and stop — merge only when explicitly told to for that
PR.
