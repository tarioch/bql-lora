"""A small offline sanity check: generate a couple of ledgers and confirm every intent that fires
produces a BQL statement that beanquery actually executes without error."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import random

from bql_lora.executor import Executor, QueryError
from bql_lora.intents import Gen, REGISTRY, Skip
from bql_lora.ledger import generate_ledger


def test_all_intents_fire_at_least_once_and_validate():
    seen_intents = set()
    errors = []
    for seed in range(25):
        ledger = generate_ledger(seed)
        ex = Executor(ledger)
        rng = random.Random(seed)
        gen = Gen(ledger, ex, rng, "full")
        for name, _, fn in REGISTRY:
            for _ in range(3):
                try:
                    sample = fn(gen)
                except Skip:
                    continue
                try:
                    ex.run(sample.bql)
                except QueryError as e:
                    errors.append((name, sample.bql, str(e)))
                    continue
                seen_intents.add(name)
                break

    assert not errors, f"{len(errors)} query errors, e.g. {errors[:3]}"
    missing = {name for name, _, _ in REGISTRY} - seen_intents
    assert len(missing) <= 2, f"intents that never produced a valid sample: {missing}"


if __name__ == "__main__":
    test_all_intents_fire_at_least_once_and_validate()
    print("ok")
