# bql-lora

Training data generator for fine-tuning an LLM (via [Unsloth](https://github.com/unslothai/unsloth)) on
the **Beancount Query Language (BQL)** — the query language implemented by
[`beanquery`](https://github.com/beancount/beanquery), the v3-compatible successor of Beancount 2's
bundled `bean-query`.

Every training example is generated end to end against the real implementation, not hand-written or
sampled from a language model:

1. A synthetic but fully valid Beancount ledger is built and loaded with `beancount.loader` (real
   parser, real balance checking).
2. An "intent" picks concrete parameters from that ledger (an account, a payee, a date range, ...) and
   derives *both* an English question and a BQL statement from the same parameters, so they can't drift
   apart.
3. The statement is executed against the ledger with `beanquery`. If it fails to parse, fails to
   compile, or raises at runtime, the example is dropped — the training set only contains statements that
   actually run.
4. Accepted examples are rendered as chat-format JSONL: a system prompt with a curated BQL reference,
   a user message with the question plus that ledger's schema (accounts, currencies, tags, payees...),
   and an assistant message with the BQL answer.

## Layout

- [`src/bql_lora/ledger.py`](src/bql_lora/ledger.py) — synthetic ledger generator (multiple regions/currencies,
  checking/savings/credit card/cash, rent & recurring bills, groceries/dining/travel/subscriptions,
  a small brokerage with lots and dividends, tags/links/metadata, balance assertions).
- [`src/bql_lora/executor.py`](src/bql_lora/executor.py) — thin wrapper around `beanquery.connect()` that pins
  `today()` to the ledger's synthetic "now" and turns compile/parse errors into a structured `QueryError`.
- [`src/bql_lora/intents.py`](src/bql_lora/intents.py) — ~45 question/query generators covering aggregate
  reports, balances, transaction search, the `accounts`/`prices`/`balances`/`events`/... tables,
  investments, `PIVOT BY`, `HAVING`, subqueries, `OPEN`/`CLOSE`/`CLEAR`, `PRINT`, `JOURNAL`, and more.
- [`src/bql_lora/schema.py`](src/bql_lora/schema.py) — the static BQL reference used as the system prompt, and
  the per-ledger schema block injected into the user prompt.
- [`src/bql_lora/format.py`](src/bql_lora/format.py) — assembles the final `{"messages": [...]}` chat example.
- [`scripts/generate_dataset.py`](scripts/generate_dataset.py) — the CLI driver.

## Usage

```bash
pip install -e .
python scripts/generate_dataset.py --ledgers 400 --per-ledger 18 --seed 1 --out data
```

This writes `data/train.jsonl` and `data/val.jsonl`. Each line is:

```json
{
  "messages": [
    {"role": "system", "content": "... BQL reference ..."},
    {"role": "user", "content": "Ledger schema:\n...\n\nQuestion: How much did I spend on groceries in 2024?"},
    {"role": "assistant", "content": "```sql\nSELECT sum(position) WHERE account = 'Expenses:Food:Groceries' AND year = 2024\n```"}
  ],
  "meta": {"intent": "agg_report", "bql": "SELECT sum(position) WHERE account = 'Expenses:Food:Groceries' AND year = 2024"}
}
```

`meta` is for inspection/debugging only — strip it before training (or keep it; most trainers ignore
unknown top-level keys and only read `messages`).

## Loading with Unsloth

```python
from unsloth import FastLanguageModel
from datasets import load_dataset

model, tokenizer = FastLanguageModel.from_pretrained("unsloth/Meta-Llama-3.1-8B-Instruct-bnb-4bit")
model = FastLanguageModel.get_peft_model(model, r=16, target_modules=["q_proj", "k_proj", "v_proj", "o_proj"])

ds = load_dataset("json", data_files={"train": "data/train.jsonl", "validation": "data/val.jsonl"})
ds = ds.map(lambda ex: {"text": tokenizer.apply_chat_template(ex["messages"], tokenize=False)})
```

Then hand `ds["train"]`'s `text` column to `trl.SFTTrainer` as usual.

## Notes / limitations

- Ledgers are synthetic (randomly generated), so the model learns BQL syntax and the fixed
  `postings`/`transactions`/`accounts`/... schema, not any particular person's real accounts.
- Only statements beanquery actually accepts and executes end up in the dataset; a handful of intents
  are skipped per ledger when they don't apply (e.g. no investments, no tags) — this is expected and
  logged to stderr as `[gen-error]`/`[query-error]` during generation for visibility, not written to disk.
- Regenerate with a different `--seed` for a different sample of ledgers/questions.
