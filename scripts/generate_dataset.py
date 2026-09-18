#!/usr/bin/env python
"""Generate a training set of (ledger schema + question -> BQL) examples for fine-tuning with Unsloth.

Every example is validated end to end: the ledger is loaded with the real Beancount v3 loader and the
BQL statement is executed against it with beanquery. Statements that fail to parse or compile, or that
raise at runtime, are dropped rather than included with a guessed answer.

Usage:
    python scripts/generate_dataset.py --ledgers 400 --out data --seed 1
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
from bql_lora.format import build_example  # noqa: E402
from bql_lora.intents import Gen, REGISTRY, Skip  # noqa: E402
from bql_lora.ledger import generate_ledger  # noqa: E402


def normalize(bql: str) -> str:
    return " ".join(bql.split()).lower()


def generate_for_ledger(ledger_seed: int, per_ledger: int, rng: random.Random, seen: set[str], counts: Counter) -> list[dict]:
    ledger = generate_ledger(ledger_seed)
    ex = Executor(ledger)
    mode = rng.choice(["full", "full", "compact"])
    gen = Gen(ledger, ex, rng, mode)
    names = [n for n, _, _ in REGISTRY]
    weights = [w for _, w, _ in REGISTRY]
    fn_by_name = {n: fn for n, _, fn in REGISTRY}

    out = []
    attempts = 0
    max_attempts = per_ledger * 12
    while len(out) < per_ledger and attempts < max_attempts:
        attempts += 1
        name = rng.choices(names, weights=weights)[0]
        fn = fn_by_name[name]
        try:
            sample = fn(gen)
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
        key = normalize(sample.bql)
        if key in seen:
            continue
        seen.add(key)
        counts[name] += 1
        out.append(build_example(rng, ledger, sample))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ledgers", type=int, default=300, help="number of distinct synthetic ledgers to generate")
    ap.add_argument("--per-ledger", type=int, default=18, help="target number of accepted examples per ledger")
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--out", type=Path, default=Path("data"))
    ap.add_argument("--val-fraction", type=float, default=0.04)
    args = ap.parse_args()

    args.out.mkdir(parents=True, exist_ok=True)
    master_rng = random.Random(args.seed)
    seen: set[str] = set()
    counts: Counter = Counter()
    examples: list[dict] = []

    for i in range(args.ledgers):
        ledger_seed = master_rng.randrange(1 << 30)
        rng = random.Random(ledger_seed ^ 0x5EED)
        try:
            batch = generate_for_ledger(ledger_seed, args.per_ledger, rng, seen, counts)
        except Exception as e:
            print(f"[ledger-error] seed={ledger_seed}: {e}", file=sys.stderr)
            continue
        examples.extend(batch)
        if (i + 1) % 20 == 0 or i == args.ledgers - 1:
            print(f"[{i + 1}/{args.ledgers}] ledgers, {len(examples)} examples so far", file=sys.stderr)

    master_rng.shuffle(examples)
    n_val = max(1, int(len(examples) * args.val_fraction))
    val, train = examples[:n_val], examples[n_val:]

    def dump(path: Path, rows: list[dict]) -> None:
        with path.open("w", encoding="utf-8") as f:
            for row in rows:
                f.write(json.dumps(row, ensure_ascii=False) + "\n")

    dump(args.out / "train.jsonl", train)
    dump(args.out / "val.jsonl", val)

    print(f"\nTotal examples: {len(examples)}  (train={len(train)}, val={len(val)})")
    print("By intent:")
    for name, n in counts.most_common():
        print(f"  {name:24s} {n}")


if __name__ == "__main__":
    main()
