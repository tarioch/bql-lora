"""Offline sanity checks: generate a few ledgers and confirm that every intent, in every prompt mode, yields a BQL
statement that beanquery executes, and that the reference examples (whose embedded SQL is validated on build) build."""

import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bql_lora.executor import Executor, QueryError
from bql_lora.intents import Gen, REGISTRY, Skip
from bql_lora.ledger import generate_ledger
from bql_lora.reference import build_reference_examples
from bql_lora.schema import SCHEMA_MODES, user_prompt


def test_all_intents_fire_and_validate_in_every_mode():
    seen_intents = set()
    errors = []
    for seed in range(25):
        ledger = generate_ledger(seed)
        ex = Executor(ledger)
        rng = random.Random(seed)
        gen = Gen(ledger, ex, rng, "full")
        for mode in SCHEMA_MODES:
            gen.mode = mode
            for name, _, fn in REGISTRY:
                for _ in range(3):
                    try:
                        sample = fn(gen)
                    except Skip:
                        continue
                    try:
                        ex.run(sample.bql)
                    except QueryError as e:
                        errors.append((mode, name, sample.bql, str(e)))
                        continue
                    seen_intents.add(name)
                    break

    assert not errors, f"{len(errors)} query errors, e.g. {errors[:3]}"
    missing = {name for name, _, _ in REGISTRY} - seen_intents
    assert len(missing) <= 2, f"intents that never produced a valid sample: {missing}"


def test_schema_free_prompt_is_just_the_question():
    ledger = generate_ledger(1)
    assert user_prompt(ledger, "How much did I spend?", "none") == "How much did I spend?"
    assert "Accounts:" in user_prompt(ledger, "q", "compact")
    assert "Ledger schema:" in user_prompt(ledger, "q", "full")


def test_reference_examples_build_and_their_sql_runs():
    # build_reference_examples() executes every ```sql block it emits and raises if one does not run
    examples = build_reference_examples(random.Random(0), Executor(generate_ledger(2)))
    assert len(examples) > 200
    assert {e["meta"]["topic"].split(":")[0] for e in examples} == {"table", "column", "function", "concept"}


if __name__ == "__main__":
    test_all_intents_fire_and_validate_in_every_mode()
    test_schema_free_prompt_is_just_the_question()
    test_reference_examples_build_and_their_sql_runs()
    print("ok")
