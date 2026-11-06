#!/usr/bin/env python
"""Generate the held-out evaluation set: text2bql questions from one ledger that training never saw.

Every example is drawn from the ledger ``EVAL_LEDGER_SEED`` (see ``src/bql_lora/evaluation.py``), so the evaluation
check can run the expected and the model's query on it and compare rows. Questions that also appear in the training
data are dropped, so the set can be checked for overlap (``check_data.py --eval``) and stays held out.

Usage:
    python scripts/generate_eval.py --data data --out eval/eval.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from generate_dataset import generate_for_ledger  # noqa: E402

from bql_lora.evaluation import EVAL_LEDGER_SEED  # noqa: E402


def training_ledger_seeds(seed: int, ledgers: int) -> set[int]:
    """The ledger seeds generate_dataset.py draws for ``--seed``/``--ledgers`` (plus the reference ledger)."""
    master_rng = random.Random(seed)
    return {master_rng.randrange(1 << 30) for _ in range(ledgers + 1)}


def questions(path: Path) -> set[str]:
    with open(path, encoding="utf-8") as f:
        return {json.loads(line)["messages"][0]["content"] for line in f if line.strip()}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=Path("data"), help="the training data, to leave its questions out")
    ap.add_argument("--out", type=Path, default=Path("eval/eval.jsonl"))
    ap.add_argument("--examples", type=int, default=200, help="examples to generate before dropping training ones")
    ap.add_argument("--training-seed", type=int, default=1, help="generate_dataset.py's --seed for data/")
    ap.add_argument("--training-ledgers", type=int, default=350, help="generate_dataset.py's --ledgers for data/")
    args = ap.parse_args()

    if EVAL_LEDGER_SEED in training_ledger_seeds(args.training_seed, args.training_ledgers):
        print(f"ledger {EVAL_LEDGER_SEED} is also a training ledger, pick another EVAL_LEDGER_SEED", file=sys.stderr)
        return 1
    seen_in_training = questions(args.data / "train.jsonl") | questions(args.data / "val.jsonl")

    counts: Counter = Counter()
    examples = generate_for_ledger(
        EVAL_LEDGER_SEED,
        args.examples,
        random.Random(EVAL_LEDGER_SEED),
        [1.0, 0.0, 0.0],
        set(),
        Counter(),
        counts,
    )
    kept = [e for e in examples if e["messages"][0]["content"] not in seen_in_training]

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for row in kept:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"wrote {len(kept)} examples to {args.out} ({len(examples) - len(kept)} dropped, also in training)")
    print("by intent:", dict(Counter(e["meta"]["intent"] for e in kept).most_common()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
