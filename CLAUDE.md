# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this repo is

A generator for a chat-format training dataset (`data/train.jsonl` / `data/val.jsonl`) used to fine-tune a small LLM
on the Beancount Query Language (BQL, via the `beanquery` package for Beancount v3), and a held-out evaluation set
with a check that runs the model's query. Training, GGUF export, the Ollama Modelfile, evaluation runs and publishing
use the `unsloth-lora` and `ollama-model` skills of the llm-finetune plugin (tarioch/llm-finetune-skills, enabled in
`.claude/settings.json`); this repo has no training code of its own. See [README.md](README.md) for the full picture (what's trained,
why there's no system prompt, the mailing-list-sourced intents, limits). Don't duplicate that here — this file is
about working in the code, not the training design.

## Two separate Python environments

- **This repo's own environment** (`uv sync`): `beancount` + `beanquery`, no GPU needed. Used for
  `generate_dataset.py`, `generate_eval.py`, `audit_dataset.py`, the tests, and for running the `ollama-model`
  skill's `evaluate.py` with `--check scripts/eval_check.py:same_rows` (the check executes queries with beanquery).
- **The Unsloth environment** on the GPU machine, for the `unsloth-lora` skill's scripts. Not a dependency of this
  repo (it needs a specific CUDA/torch stack). Those scripts only read `data/*.jsonl`.

`beanquery` is pinned to `git+https://github.com/beancount/beanquery.git` (git HEAD), not the PyPI release: the
`HAVING`/`PIVOT BY` support and several functions this dataset exercises aren't in PyPI's `0.2.0`. Don't "fix" this
to a version pin without checking those features still work. `uv.lock` records the exact commit last resolved;
run `uv lock --upgrade` to intentionally move to a newer `beanquery` HEAD (and `uv sync` to apply it).

## Commands

```bash
uv sync                                                                    # this repo's own deps (no GPU)

uv run python scripts/generate_dataset.py --ledgers 350 --per-ledger 18 --seed 1 --out data   # regenerate data/
uv run python scripts/generate_eval.py --data data --out eval/eval.jsonl                      # then the eval set
uv run python scripts/audit_dataset.py data/train.jsonl data/val.jsonl                        # structure + parses + no ledger leaks

uv run pytest tests/test_evaluation.py                                    # data/eval agreement and the eval check
uv run pytest tests/test_smoke.py                                         # every intent, every prompt mode (~3 min)
```

Training and publishing follow the skills (README, "Training and publishing"). Ask before using the GPU machine or
starting a full training run, and before pushing to ollama.com.

## Checks

CI (`.github/workflows/ci.yml`) runs these on every pull request, all of them have to pass:

```bash
uv sync --locked --dev
uv run pre-commit run --all-files   # ruff, ruff format, uv-lock, zizmor, whitespace/yaml/toml checks
uv run pytest                       # CI uses the latest Python only
uv run python scripts/audit_dataset.py data/train.jsonl data/val.jsonl
uv run python scripts/audit_dataset.py eval/eval.jsonl
```

- `pre-commit run --all-files` only looks at files tracked by git, `git add` new files first.
- Regenerate `uv.lock` with the uv version of the `uv-lock` pre-commit hook (`rev` in `.pre-commit-config.yaml`),
  a different uv version rewrites unrelated parts of the file: `uvx --from uv==<rev> uv lock`.
- The `CI ok` job fails unless all other jobs succeeded, it is the one check to require on `main`.
- Dependabot PRs for minor and patch updates get auto-merge enabled (`.github/workflows/dependabot-automerge.yml`)
  and merge once the required checks pass, major updates stay manual.
- Release notes are drafted by release-drafter (`.github/workflows/release-drafter.yml`, config from the shared
  `release-drafter.yml` of tarioch/.github) on every push to `main`, the PR label decides the category. Publishing the
  draft creates the `vX.Y.Z` tag. Publish a release whenever a newly trained model is pushed to Ollama.
- Lint fixes in `ledger.py`/`intents.py` must not change what the generator draws from its `random.Random`, or the
  dataset changes: compare a small `generate_dataset.py --ledgers 30` run before and after.

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

Whenever an intent (or a reference entry) is based on something read elsewhere — a mailing list thread, a
third-party notebook, an issue tracker, anything not derived purely from `beanquery`'s own source/registry — record
that source in README.md's "Where the knowledge comes from" section (see the existing mailing-list and molab-manual
bullets for the format), not just in the commit message. That section is what lets a later change tell "generated
from the live registry" apart from "read somewhere and needs to be re-checked when beanquery changes."

## Architecture: dataset -> evaluation -> Modelfile agreement

- The training data has no system message (tested in `tests/test_evaluation.py`): Qwen's chat template fills in its
  own default line at training time. The `unsloth-lora` skill records that rendered line in `run.json`, and the
  `ollama-model` skill bakes it into the Modelfile's `SYSTEM`. Changing the data's message format changes this.
- `eval/eval.jsonl` comes from the single ledger `EVAL_LEDGER_SEED` (`src/bql_lora/evaluation.py`), which is not one
  of the training ledgers, and leaves out questions that are in `data/`. Regenerate it after regenerating `data/`, or
  the held-out test fails. The check `same_rows` regenerates that ledger and compares rows, so a change to
  `ledger.py` that changes what a seed produces needs a regenerated eval set too.

## Workflow: PRs, not direct commits

Changes to this repo go through a branch + PR (`git checkout -b chore/...`, prefix `feature/`, `bugfix/` or
`chore/`, which labels the PR, its `Label check` fails without one; push the branch, `gh pr create`), not direct
commits to `main`. Open the PR, summarize what to look at, and stop — merge only when explicitly told to for that
PR.
