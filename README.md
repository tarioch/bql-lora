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
| **text2bql** (~6,000 examples) | just the question, e.g. `How much did I spend on groceries in 2024?` | a `sql` block: `SELECT sum(position) WHERE account ~ 'Groceries' AND year = 2024` |
| **reference** (~280 examples) | `What does date_trunc do in BQL?`, `What columns does the postings table have?`, `Why can't I GROUP BY tags?`, ... | short explanation, with an executable example where useful |

Consequences of "must not depend on the ledger":

- Questions that name an account, payee, tag or currency get exactly that in the query. Questions that use a
  natural word ("groceries", "checking") are answered with a **keyword regex** (`account ~ 'Groceries'`) that works
  whatever the account hierarchy is (`Expenses:Groceries` or `Expenses:Food:Groceries`). The generator only emits an
  alias when the keyword matches that one account in the ledger.
- Generic wording ("all expenses", "my net worth") maps to root-type regexes (`account ~ '^Expenses'`).
- Nothing that is a hidden fact about the ledger appears in an answer. If the query filters on a currency, the
  question says which one (`... in CHF`). `scripts/audit_dataset.py` checks this on every schema-free example.

The system prompt is short (about 100 tokens, see `SYSTEM_PROMPT` in [`schema.py`](src/bql_lora/schema.py)). The long
BQL reference that used to be in the prompt is kept as `LONG_REFERENCE`, for prompting a *non*-fine-tuned model as a
baseline to compare against.

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
- [`src/bql_lora/schema.py`](src/bql_lora/schema.py): system prompt, `LONG_REFERENCE`, and the schema-mode renderers.
- [`src/bql_lora/executor.py`](src/bql_lora/executor.py): `beanquery` wrapper; pins `today()` to the ledger's "now".
- [`scripts/generate_dataset.py`](scripts/generate_dataset.py): the driver. [`scripts/audit_dataset.py`](scripts/audit_dataset.py): checks a dataset.
- [`ollama/Modelfile`](ollama/Modelfile): Ollama packaging, generated by [`scripts/make_modelfile.py`](scripts/make_modelfile.py)
  from [`src/bql_lora/modelfile.py`](src/bql_lora/modelfile.py).
- [`scripts/train.py`](scripts/train.py): the Unsloth training script (see [Training](#training)).
- [`scripts/check_system_prompt_sensitivity.py`](scripts/check_system_prompt_sensitivity.py): builds temporary Ollama models
  that differ only in their system message and reports how many answers are still BQL.
- [`data/`](data) (with the system prompt) and [`data/no-system/`](data/no-system) (without): the same examples.

## Usage

```bash
pip install -e .
python scripts/generate_dataset.py --ledgers 350 --per-ledger 18 --seed 1 --out data
python scripts/audit_dataset.py data/train.jsonl data/val.jsonl
```

Every prompt is the question alone. `--schema-weights NONE COMPACT FULL` (default `1 0 0`) can mix in a compact account
list or the full ledger schema if you ever want a model that also uses one; that is off by default.

`--no-system` leaves the system message out of every example. `data/no-system/` is the committed result: the same
examples as `data/`, minus the system message (see [The system prompt](#the-system-prompt-trigger-or-baked-in)).

Each line of `data/train.jsonl` / `data/val.jsonl`:

```json
{"messages": [
   {"role": "system",    "content": "You are an expert in the Beancount Query Language (BQL) ..."},
   {"role": "user",      "content": "How much did I spend on groceries in 2024?"},
   {"role": "assistant", "content": "```sql\nSELECT sum(position)\nWHERE account ~ 'Groceries' AND year = 2024\n```"}],
 "meta": {"task": "text2bql", "intent": "agg_report", "schema": "none", "bql": "..."}}
```

`meta` is for filtering and inspection; most trainers only read `messages`. Examples are short: roughly 250-350
tokens including the system prompt (estimated from character counts). `max_seq_length=1024` fits everything and 2048
leaves margin; there is no need for a model's maximum context (e.g. 32768), which needs far more VRAM.

## The system prompt: trigger or baked in?

A constant system prompt is the easiest feature for a fine-tune to key on. Measured with
[`scripts/check_system_prompt_sensitivity.py`](scripts/check_system_prompt_sensitivity.py) on 8 questions, on a model
fine-tuned on `data/` (the prompt present in every example):

| System message at inference | Answers containing a `sql` block |
| --- | --- |
| the training system prompt | 8 of 8, in the trained style |
| none | 1 of 8: stock Qwen behaviour (refusals, invented SQL such as `FROM expenses`, generic explanations) |
| Qwen's stock line | 1 of 8: the same |

That model does BQL only when the prompt is present. That works while the Modelfile supplies it, and breaks when a
client sends its own system message (Ollama then ignores the Modelfile's `SYSTEM`).

[`data/no-system/`](data/no-system) holds the same examples without any system message, so the weights have to carry the
behaviour for every input (it also cuts an example from about 255 to about 75 tokens). Qwen's chat template injects its
own stock line when a training example has no system message, so that is effectively what the model is trained with.
Retrained on it (`python scripts/train.py`'s default, see [Training](#training)) and measured the same way:

| System message at inference | Answers containing a `sql` block |
| --- | --- |
| Qwen's stock line | 8 of 8 |
| none | 7 of 8 (the one miss: a plain "how are you" correctly got a plain reply, not a forced query) |
| the (unrelated) training system prompt from the other model | 8 of 8 |

So `ollama/Modelfile` now defaults to `--system qwen`, and `scripts/train.py`'s default (`data/no-system/`) is the
recommended way to train. If you instead train on `data/` (the prompt present in every example), use
`--system bql`.

## Training

[`scripts/train.py`](scripts/train.py) fine-tunes `unsloth/qwen2.5-coder-7b-instruct-bnb-4bit` (4-bit QLoRA) with Unsloth,
so the settings live in the repo instead of a UI. Run it with the Python environment that has Unsloth installed (with
Unsloth Studio, the interpreter under `~/.unsloth/studio/unsloth_studio`); it does not need `beancount`.

```bash
python scripts/train.py --max-steps 2     # smoke test: two optimizer steps, then evaluate and save
python scripts/train.py                    # data/no-system, 2 epochs
python scripts/train.py --data data        # the variant with the system prompt
```

Defaults: LoRA rank 32 on all attention and MLP projections, learning rate 2e-4, effective batch 16 (4 x 4), 8-bit AdamW,
sequence length 1024, 2 epochs. What it prints, so nothing is hidden:

- the **first training example exactly as the model sees it**, including the system line (for no-system data, Qwen's chat
  template injects `You are Qwen, created by Alibaba Cloud. You are a helpful assistant.`, so the Modelfile has to use
  `--system qwen`; the script prints the command with the right choice),
- token length statistics, with a warning if anything would be truncated,
- **which tokens receive loss**: only the assistant's answer, including the end-of-turn token so the model learns to stop.
  The question and any system message are masked out (`--no-response-only` turns that off),
- train and validation loss, and peak GPU memory.

The adapter is saved to `outputs/bql-lora/adapter` together with a `training_config.json`. `--merge` and `--gguf` also export
a merged 16-bit model or a Q4_K_M GGUF through Unsloth; those two are not covered by the smoke test, and the GGUF export
needs llama.cpp tooling.

Measured on an RTX 4070 Ti (12 GB) with unsloth 2026.9.7, trl 0.23.1, transformers 5.5.0 and torch 2.11: the smoke test
peaks at 7.4 GiB and takes about 3 s per step, which puts a 2-epoch run at roughly 25-45 minutes (extrapolated from two
steps). The script follows those library versions' APIs (for example `max_length`, not `max_seq_length`, in `SFTConfig`),
so an older or newer stack may need small changes.

## Using the fine-tuned model

There is no trigger phrase. Use the same chat format as in training: the **exact** `SYSTEM_PROMPT` from
[`schema.py`](src/bql_lora/schema.py), then a user turn with just the question:

```python
from bql_lora.schema import SYSTEM_PROMPT

messages = [
    {"role": "system", "content": SYSTEM_PROMPT},
    {"role": "user", "content": "How much did I spend on groceries in 2024?"},
]
inputs = tokenizer.apply_chat_template(messages, add_generation_prompt=True, return_tensors="pt", return_dict=True).to("cuda")
out = model.generate(**inputs, max_new_tokens=300, do_sample=False)
print(tokenizer.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True))
```

Send the question alone: the model is not trained on prompts that carry an account list (regenerate with
`--schema-weights` if you want that). Name exact accounts, payees, tags and currencies in the question when it matters;
natural words such as "groceries" are matched with a keyword regex.

## Running it in Ollama

[`ollama/Modelfile`](ollama/Modelfile) bakes the system prompt, a plain ChatML template and `temperature 0` into the
model, so a client only has to send the question. Export your merged fine-tune as a GGUF (for example Q4_K_M), put the
Modelfile next to it and:

```bash
python scripts/make_modelfile.py --gguf my-model.Q4_K_M.gguf -o Modelfile   # or copy ollama/Modelfile and edit the FROM line
ollama create bql -f Modelfile
ollama run bql "How much did I spend on groceries in 2024?"
```

`--system {bql,qwen,none}` chooses which system message is baked in; it has to match how the model was trained (see
[above](#the-system-prompt-trigger-or-baked-in)). The file is generated from `SYSTEM_PROMPT`
(`python scripts/make_modelfile.py --check` fails if it is stale), and
`tests/test_modelfile.py` checks that the Modelfile, the code and the training data all use the same prompt. Details
that matter:

- **LF line endings only.** Ollama keeps a `\r` found inside a template or system prompt, so a CRLF checkout changes
  what the model sees compared with training. `.gitattributes` pins the file to LF.
- **The template reads only `.Messages`.** Ollama passes the system prompt as the first message and also as `.System`,
  so a template that prints both feeds the model the prompt twice.
- **Qwen2.5-style ChatML, no tool calling or fill-in-the-middle.** The model is trained only on question to `sql`
  block, so those sections are left out. A different base model needs its own template.

## Loading with Unsloth

```python
from unsloth import FastLanguageModel
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Qwen2.5-Coder-7B-Instruct-bnb-4bit", max_seq_length=2048, load_in_4bit=True)
model = FastLanguageModel.get_peft_model(model, r=32, lora_alpha=32, use_gradient_checkpointing="unsloth",
                                         target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"])

ds = load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"})
ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)})
```

Then train on the `text` column with `trl.SFTTrainer` (2-3 epochs is a reasonable start).

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
