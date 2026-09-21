#!/usr/bin/env python
"""Generate a training set for fine-tuning an LLM on the Beancount Query Language (BQL) with Unsloth.

Two kinds of examples are produced, both chat-formatted with the same short system prompt (or, with
``--no-system``, without any system message):

* ``text2bql``: an English question -> one BQL statement. The prompt is the question alone, so the model never
  depends on a particular ledger's accounts. ``--schema-weights`` can mix in a compact account list or the full
  schema, but that is off by default.
* ``reference``: questions about BQL itself (tables, columns, functions, core concepts), built from beanquery's
  live registry, so the model can drop the long reference text from its prompt.

Every statement is validated end to end: the ledger is loaded with the real Beancount v3 loader and the BQL is
executed against it with beanquery. Statements that fail to parse or compile, or that raise, are dropped.

Usage:
    python scripts/generate_dataset.py --ledgers 350 --out data --seed 1
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bql_lora.executor import Executor, QueryError  # noqa: E402
from bql_lora.format import build_example, strip_system  # noqa: E402
from bql_lora.intents import Gen, REGISTRY, Skip  # noqa: E402
from bql_lora.ledger import generate_ledger  # noqa: E402
from bql_lora.reference import build_reference_examples  # noqa: E402
from bql_lora.schema import SCHEMA_MODES  # noqa: E402

MAX_SAME_BQL = 3  # keep at most this many differently-phrased questions for one identical statement


def normalize(bql: str) -> str:
    return " ".join(bql.split()).lower()


def generate_for_ledger(ledger_seed: int, per_ledger: int, rng: random.Random, weights_by_mode: list[float],
                        seen: set[tuple], bql_counts: Counter, counts: Counter) -> list[dict]:
    ledger = generate_ledger(ledger_seed)
    ex = Executor(ledger)
    gen = Gen(ledger, ex, rng, "full")
    names = [n for n, _, _ in REGISTRY]
    weights = [w for _, w, _ in REGISTRY]
    fn_by_name = {n: fn for n, _, fn in REGISTRY}

    out = []
    attempts = 0
    max_attempts = per_ledger * 12
    while len(out) < per_ledger and attempts < max_attempts:
        attempts += 1
        # The schema mode decides how the question is phrased (aliases, explicit currencies), so pick it first.
        mode = rng.choices(SCHEMA_MODES, weights=weights_by_mode)[0]
        gen.mode = mode
        name = rng.choices(names, weights=weights)[0]
        try:
            sample = fn_by_name[name](gen)
        except Skip:
            continue
        except Exception as e:  # a bug in an intent generator; skip but keep going
            print(f"  [gen-error] {name}: {e}", file=sys.stderr)
            continue
        try:
            result = ex.run(sample.bql)
        except QueryError as e:
            print(f"  [query-error] {name}: {sample.bql!r} -> {e.kind}: {e.message}", file=sys.stderr)
            continue
        if len(result) == 0 and not sample.allow_empty:
            continue
        bql_key = normalize(sample.bql)
        # Without a schema in the prompt, the example is fully determined by question + answer: never repeat it.
        key = (mode, sample.question, bql_key) if mode == "none" else (mode, ledger_seed, sample.question, bql_key)
        if key in seen or bql_counts[bql_key] >= MAX_SAME_BQL:
            continue
        seen.add(key)
        bql_counts[bql_key] += 1
        counts[name] += 1
        out.append(build_example(rng, ledger, sample, mode))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledgers", type=int, default=350, help="number of distinct synthetic ledgers to generate")
    ap.add_argument("--per-ledger", type=int, default=18, help="target number of accepted text2bql examples per ledger")
    ap.add_argument("--schema-weights", type=float, nargs=3, default=[1.0, 0.0, 0.0], metavar=("NONE", "COMPACT", "FULL"),
                    help="share of text2bql examples whose prompt has no ledger info / a compact account list / the full schema (default: question only)")
    ap.add_argument("--no-reference", action="store_true", help="skip the BQL reference (tables/columns/functions/concepts) examples")
    ap.add_argument("--no-system", action="store_true",
                    help="omit the system message from every example, so the model learns the behaviour for any input "
                         "instead of keying on a constant system prompt (the examples are otherwise identical)")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--val-fraction", type=float, default=0.04)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    master_rng = random.Random(args.seed)
    seen: set[tuple] = set()
    bql_counts: Counter = Counter()
    counts: Counter = Counter()
    examples: list[dict] = []

    for i in range(args.ledgers):
        ledger_seed = master_rng.randrange(1 << 30)
        rng = random.Random(ledger_seed ^ 0x5EED)
        try:
            batch = generate_for_ledger(ledger_seed, args.per_ledger, rng, args.schema_weights, seen, bql_counts, counts)
        except Exception as e:
            print(f"[ledger-error] seed={ledger_seed}: {e}", file=sys.stderr)
            continue
        examples.extend(batch)
        if (i + 1) % 20 == 0 or i == args.ledgers - 1:
            print(f"[{i + 1}/{args.ledgers}] ledgers, {len(examples)} text2bql examples so far", file=sys.stderr)

    n_reference = 0
    if not args.no_reference:
        ref_ledger = generate_ledger(master_rng.randrange(1 << 30))
        reference = build_reference_examples(random.Random(args.seed), Executor(ref_ledger))
        n_reference = len(reference)
        examples.extend(reference)

    master_rng.shuffle(examples)
    if args.no_system:
        examples = [strip_system(e) for e in examples]
    n_val = max(1, int(len(examples) * args.val_fraction))
    val, train = examples[:n_val], examples[n_val:]

    def dump(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump(args.out / "train.jsonl", train)
    dump(args.out / "val.jsonl", val)

    print(f"\nTotal examples: {len(examples)}  (train={len(train)}, val={len(val)}); reference examples: {n_reference}")
    print("text2bql prompt style:", dict(Counter(e["meta"]["schema"] for e in examples if e["meta"]["task"] == "text2bql")))
    print("text2bql by intent:")
    for name, n in counts.most_common():
        print(f"  {name:24s} {n}")


if __name__ == "__main__":
    main()
