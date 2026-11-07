# bql-lora

Training data generator for fine-tuning a small LLM (via [Unsloth](https://github.com/unslothai/unsloth)) on the
**Beancount Query Language (BQL)**, the query language of
[`beanquery`](https://github.com/beancount/beanquery), the Beancount v3 successor of Beancount 2's `bean-query`.

The goal is a model that can answer BQL questions **without being handed the schema of a particular ledger**, and
that has the BQL reference in its weights instead of in a long prompt. Nothing here is hand-written query data or
sampled from another language model: every example is checked against the real implementation.

## What the model is trained to do

| Task | Prompt (`user` turn) | Answer |
| --- | --- | --- |
| **text2bql** (~6,000 examples) | just the question, e.g. `How much did I spend on groceries in 2024?` | only a `sql` block, no explanation (the app takes the query out of the answer): `SELECT sum(position) WHERE account ~ 'Groceries' AND year = 2024` |
| **reference** (~280 examples) | `What does date_trunc do in BQL?`, `What columns does the postings table have?`, `Why can't I GROUP BY tags?`, ... | short explanation, with an executable example where useful |

Consequences of "must not depend on the ledger":

- Questions that name an account, tag or currency get exactly that in the query. A payee in the question is usually part
  of the real name (the model never sees the ledger's payees), so most payee conditions match its first word
  (`payee ~ 'Umbrella'`) and only some use the whole value. Questions that use a
  natural word ("groceries", "checking") are answered with a **keyword regex** (`account ~ 'Groceries'`) that works
  whatever the account hierarchy is (`Expenses:Groceries` or `Expenses:Food:Groceries`). The generator only emits an
  alias when the keyword matches that one account in the ledger.
- Generic wording ("all expenses", "my net worth") maps to root-type regexes (`account ~ '^Expenses'`).
- Nothing that is a hidden fact about the ledger appears in an answer. If the query filters on a currency, the
  question says which one (`... in CHF`). `scripts/audit_dataset.py` checks this on every schema-free example.

Training examples carry no system message at all (see
[Why the data has no system prompt](#why-the-data-has-no-system-prompt) below). The BQL reference that would otherwise have to go in a system prompt is kept as `LONG_REFERENCE` in
[`schema.py`](src/bql_lora/schema.py), only for prompting a *non*-fine-tuned model as a baseline to compare against.

## Where the knowledge comes from

- **Grammar, tables, columns, functions:** read from the `beanquery` source and introspected from its live registry
  (`query_compile.FUNCTIONS`, the table definitions). Signatures and column lists in the reference examples are
  generated from that, not typed by hand.
- **Concepts** (postings model, positions/inventories, `OPEN`/`CLOSE`/`CLEAR`, `BALANCES`/`JOURNAL`/`PRINT`, sign
  conventions, ...) follow the official [BQL documentation](https://beancount.github.io/docs/beancount_query_language/),
  which was written for Beancount 2. Where v3 differs, the v3 behaviour is what is taught, after checking it by
  running it. Notably:
  - `FROM <expr>` is a **plain row filter** in v3 (it selected whole transactions in v2). Match whole transactions with
    `has_account('regex')` or `'Full:Account' IN accounts`.
  - `PRINT FROM` runs over directives, so it has no `account` column (use `IN accounts` / `has_account`).
  - There is no `weight()` or `raw()` function (`weight` is a column), no `avg()`, no `LIKE`, no `count(DISTINCT x)`.
  - `date_bin`/`interval` strides support only days, months and years.
- **Real-world question patterns** come from reading threads on the [beancount@googlegroups.com](https://groups.google.com/g/beancount)
  mailing list — real questions people asked, and fixes from the beanquery maintainer. The literal queries posted there
  aren't reused as-is (they're often the broken attempt that prompted the question, or written against the older v2
  `bean-query` syntax); instead, each pattern is turned into its own parameterized intent and verified the same way as
  everything else. Patterns sourced this way so far:
  - `open.meta['key']` as an alternative to `open_meta(account, 'key')` for reading account metadata (`accounts_table`).
  - Finding accounts that need a fresh balance check, using `NOT close_date(account)` on `#balances` to exclude closed
    accounts, and `account NOT IN (SELECT account FROM #balances)` for ones with no check at all (`stale_accounts`).
  - Binning by week and labeling each bucket by its last day: `date_bin('7 days', date, origin) + interval('6 days')`
    (`date_functions`).
  - Filtering one query's rows by an aggregate computed over another, via `account IN (SELECT account ... HAVING ...)`
    (`subquery_in`).
  - A reference entry on a real reported pitfall: `last(balance)` grouped by account gives the *wrong* per-account
    total, because `balance` is one running total over the whole row set, not one per group; `sum(position)` is correct.
- **A community beanquery manual** — an interactive [marimo notebook](https://molab.marimo.io/notebooks/nb_hht9waEQnsrjBSRp5ZKGPu/app)
  by Evgeny E, pinned to the same beanquery commit this repo's `uv.lock` locks, with every shown query output actually
  computed live rather than hand-typed. Cross-checked against the existing intents to find real gaps rather than
  duplicating coverage. Patterns sourced this way so far:
  - `x > ALL (subquery)` / `x < ANY (subquery)`, the way to test a value against every/at-least-one row a subquery
    returns since beanquery has no scalar subqueries (`any_all_subquery`).
  - The pattern-first `'pattern' ?~ field` match operator, whose default case-sensitivity is the *opposite* of `~`
    (`txns_by_text`).
- **Manually added cases** — questions the model got wrong or that had no matching pattern, added by hand as intents.
  So far: the last transaction date per still-open account, using `close_date(account) IS NULL` with `max(date)` on the
  postings table (`last_activity`); and the last entry per account among accounts used in a recent window (this year,
  the last 12 months, ...) whose latest entry is still older than a relative cutoff ("... that did have an entry this
  year, but older than 2 months"), using the window in `WHERE` and `HAVING max(date) < today() - interval('2 months')`
  (`last_entry_window`).
- **Every statement is executed.** Ledgers are synthetic but loaded with the real `beancount.loader`; each generated
  query runs against its ledger with `beanquery`, and is dropped if it fails to parse, compile or run, or returns nothing.
  Every SQL block in the reference answers is executed too, and building fails if one doesn't run.

## Layout

- [`src/bql_lora/ledger.py`](src/bql_lora/ledger.py): synthetic ledger generator (5 regions/currencies, banking,
  bills, groceries/dining/travel/subscriptions, a small brokerage with lots and dividends, tags/links/metadata,
  balance assertions).
- [`src/bql_lora/intents.py`](src/bql_lora/intents.py): ~40 question/query generators; each derives the English
  question and the BQL from the same parameters.
- [`src/bql_lora/reference.py`](src/bql_lora/reference.py): the reference examples (tables, columns, functions, concepts).
- [`src/bql_lora/schema.py`](src/bql_lora/schema.py): `LONG_REFERENCE` (the eval-baseline prompt) and the schema-mode renderers.
- [`src/bql_lora/executor.py`](src/bql_lora/executor.py): `beanquery` wrapper; pins `today()` to the ledger's "now".
- [`scripts/generate_dataset.py`](scripts/generate_dataset.py): the driver. [`scripts/audit_dataset.py`](scripts/audit_dataset.py): checks a dataset.
- [`src/bql_lora/evaluation.py`](src/bql_lora/evaluation.py): the evaluation check (run the answer, compare rows).
- [`scripts/generate_eval.py`](scripts/generate_eval.py): writes the evaluation set, [`scripts/eval_check.py`](scripts/eval_check.py)
  is the check in the form the `ollama-model` skill loads.
- [`data/`](data): the training set. [`eval/`](eval): the held-out evaluation set.

Training, the GGUF export, the Ollama Modelfile and publishing are not in this repo: they are done with the
[llm-finetune](https://github.com/tarioch/llm-finetune-skills) Claude Code plugin, see
[Training and publishing](#training-and-publishing).

## Usage

```bash
uv sync
uv run python scripts/generate_dataset.py --ledgers 350 --per-ledger 18 --seed 1 --out data
uv run python scripts/audit_dataset.py data/train.jsonl data/val.jsonl
```

Every prompt is the question alone. `--schema-weights NONE COMPACT FULL` (default `1 0 0`) can mix in a compact account
list or the full ledger schema if you ever want a model that also uses one; that is off by default.

Each line of `data/train.jsonl` / `data/val.jsonl`:

```json
{"messages": [
   {"role": "user",      "content": "How much did I spend on groceries in 2024?"},
   {"role": "assistant", "content": "```sql\nSELECT sum(position)\nWHERE account ~ 'Groceries' AND year = 2024\n```"}],
 "meta": {"task": "text2bql", "intent": "agg_report", "schema": "none", "bql": "..."}}
```

`meta` is for filtering and inspection; most trainers only read `messages`. Examples are short: roughly 75-250 tokens
(estimated from character counts). `max_seq_length=1024` fits everything and 2048 leaves margin; there is no need for
a model's maximum context (e.g. 32768), which needs far more VRAM.

## Why the data has no system prompt

There isn't one, and that was a deliberate finding, not the starting design. A constant system prompt is the easiest
feature for a fine-tune to key on instead of learning the behaviour for any input: an earlier version of this dataset
put a ~100-token BQL system prompt in every example, and a model trained on it did BQL only when that exact prompt was
present. Measured directly against Ollama on 8 test questions, comparing what the same fine-tune answers under
different system messages:

| System message at inference | Answers containing a `sql` block |
| --- | --- |
| the training system prompt | 8 of 8, in the trained style |
| none | 1 of 8: stock Qwen behaviour (refusals, invented SQL such as `FROM expenses`, generic explanations) |
| Qwen's stock line | 1 of 8: the same |

That works while a Modelfile supplies the exact prompt, and breaks the moment a client sends its own system message
(Ollama then ignores the Modelfile's `SYSTEM`). Training on the same examples with no system message at all — what
`data/` now holds — pushes the behaviour into the weights instead (it also cuts an example from about 255 to about 75
tokens). Qwen's chat template fills in its own default line for a conversation with none, so that is effectively what
the model trains on. Measured the same way after retraining on it:

| System message at inference | Answers containing a `sql` block |
| --- | --- |
| Qwen's stock line | 8 of 8 |
| none | 7 of 8 (the one miss: a plain "how are you" correctly got a plain reply, not a forced query) |
| an unrelated custom system prompt | 8 of 8 |

So `data/` has no system message. Whatever serves the model has to send Qwen's default line when a request has none;
the `ollama-model` skill bakes it into the Modelfile.

## Evaluation

[`eval/eval.jsonl`](eval/eval.jsonl) holds about 500 text2bql questions in the training format, all drawn from one
synthetic ledger that no training example comes from (`EVAL_LEDGER_SEED` in
[`evaluation.py`](src/bql_lora/evaluation.py)), with questions that also appear in `data/` left out. An answer is
right when its `sql` block runs on that ledger and returns the same rows as the expected statement: column names
don't matter, row order only when the expected statement has an `ORDER BY`. That accepts any correct spelling of a
query, which an exact text match would not.

```bash
uv run python scripts/generate_eval.py --data data --out eval/eval.jsonl   # after regenerating data/
```

`tests/test_evaluation.py` checks that the set is held out and that every expected answer passes the check.

## Training and publishing

The `unsloth-lora` and `ollama-model` skills of the [llm-finetune](https://github.com/tarioch/llm-finetune-skills)
Claude Code plugin do the rest: data check, LoRA training with Unsloth, GGUF export, a Modelfile generated from what
training used, the comparison with the published model, and publishing. `.claude/settings.json` makes Claude Code
offer the plugin when this repo is opened; elsewhere install it with
`/plugin marketplace add tarioch/llm-finetune-skills` and `/plugin install llm-finetune@llm-finetune-skills`. Then ask
Claude, for example, to train the 3B model on `data/` and compare it with the published one.

What the skills need to know about this repo:

- the data is `data/`, the evaluation set `eval/eval.jsonl`,
- the evaluation check is `scripts/eval_check.py:same_rows` (the `ollama-model` skill's `--check`). It executes the
  queries with beanquery, so run the evaluation with this repo's environment (`uv run`),
- the base models are below, trained with the skill's defaults, which are the settings the published models were
  trained with.

On a 12 GB RTX 4070 Ti the 3B model peaks at about 6.6 GiB and trains in about 35 minutes, the 7B model peaks at
about 8.1 GiB and takes about 30 minutes. The GGUF export takes a few minutes for 3B and more than 10 for 7B.

The published model is [`tarioch/qwen2.5-coder-bql`](https://ollama.com/tarioch/qwen2.5-coder-bql): `:7b` from
`unsloth/qwen2.5-coder-7b-instruct-bnb-4bit`, `:3b` (also `:latest`) from `unsloth/qwen2.5-coder-3b-instruct-bnb-4bit`.
Publish a GitHub release of this repo whenever a newly trained model is pushed.

## Using the fine-tuned model

There is no trigger phrase and no system prompt to supply: send just the question as a user turn, at temperature 0.
The Modelfile bakes in Qwen's default system line, matching training (see
[Why the data has no system prompt](#why-the-data-has-no-system-prompt)); don't override it with a custom one.

```bash
ollama run tarioch/qwen2.5-coder-bql "How much did I spend on groceries in 2024?"
```

Send the question alone: the model is not trained on prompts that carry an account list (regenerate with
`--schema-weights` if you want that). Name exact accounts, payees, tags and currencies in the question when it
matters; natural words such as "groceries" are matched with a keyword regex.

## Limits

- Ledgers are synthetic, so the model learns BQL and its fixed schema, not anyone's real accounts. Without a schema,
  a natural word that is not in the account name (say "food" for `Expenses:Groceries`) is a guess.
- A schema-free answer is only as good as the question: relative dates work (`today()` is evaluated at query time),
  but the base currency has to be stated when a query needs one.
- Beanquery is pinned to git HEAD (`0.3.0.dev0`) rather than PyPI's `0.2.0`, which does not match what the data was
  validated against.

## License

Copyright (C) 2026 Patrick Ruckstuhl. Licensed under the [GNU General Public License, version 2](LICENSE)
(`GPL-2.0-only`), the same license as `beanquery` and Beancount. This applies to the code and to the generated
dataset: the code imports `beanquery`, and the reference examples paraphrase its documentation and docstrings.
