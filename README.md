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
| **text2bql, schema-free** (~70%) | just the question, e.g. `How much did I spend on groceries in 2024?` | a `sql` block: `SELECT sum(position) WHERE account ~ 'Groceries' AND year = 2024` |
| text2bql, compact schema (~15%) | operating currency + flat account list, then the question | same, using the exact account names |
| text2bql, full schema (~15%) | currency, date range, accounts, tags, links, payees, metadata keys, then the question | same |
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

## Usage

```bash
pip install -e .
python scripts/generate_dataset.py --ledgers 350 --per-ledger 18 --seed 1 --out data
python scripts/audit_dataset.py data/train.jsonl data/val.jsonl
```

`--schema-weights NONE COMPACT FULL` (default `0.70 0.15 0.15`) sets how often the prompt carries ledger information.

Each line of `data/train.jsonl` / `data/val.jsonl`:

```json
{"messages": [
   {"role": "system",    "content": "You are an expert in the Beancount Query Language (BQL) ..."},
   {"role": "user",      "content": "How much did I spend on groceries in 2024?"},
   {"role": "assistant", "content": "```sql\nSELECT sum(position)\nWHERE account ~ 'Groceries' AND year = 2024\n```"}],
 "meta": {"task": "text2bql", "intent": "agg_report", "schema": "none", "bql": "..."}}
```

`meta` is for filtering and inspection; most trainers only read `messages`. Examples are short: roughly 250-350
tokens when schema-free (system prompt included), up to about 580 with a compact account list and about 870 with the
full schema (estimated from character counts). `max_seq_length=1024` fits everything; 2048 leaves margin, and both
need far less VRAM than the 32768 default that made the earlier training attempt run out of memory.

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

If you know your ledger's accounts you can prepend them (`Ledger:\nOperating currency: CAD.\nAccounts: a, b, c\n\nQuestion: ...`),
which the model has also seen and which makes the account names exact.

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
