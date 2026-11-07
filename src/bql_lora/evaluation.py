"""Score a model's answer on the evaluation set by running it, not by comparing text.

Every evaluation example comes from one synthetic ledger (``EVAL_LEDGER_SEED``) that no training example was drawn
from. An answer is right when its ``sql`` block runs on that ledger and returns the same rows as the expected
statement. Column names are not compared (an alias is not a mistake), row order only when the expected statement
has an ``ORDER BY``.
"""

from __future__ import annotations

import functools
import re

from .executor import Executor, QueryError
from .ledger import generate_ledger

# The ledger every evaluation example is generated from and checked against. generate_eval.py refuses a seed that
# the training data also uses.
EVAL_LEDGER_SEED = 20261106

_SQL_BLOCK = re.compile(r"```sql\s*\n(.*?)```", re.DOTALL)


def extract_sql(text: str) -> str | None:
    """The first ``sql`` block of an answer, or None."""
    match = _SQL_BLOCK.search(text)
    return match.group(1).strip() if match else None


@functools.cache
def eval_executor() -> Executor:
    return Executor(generate_ledger(EVAL_LEDGER_SEED))


def _rows(statement: str) -> list[tuple] | None:
    try:
        return eval_executor().run(statement).rows
    except QueryError:
        return None


def same_rows(prompt: list[dict], expected: str, answer: str) -> bool:
    """``evaluate.py --check`` function of the ollama-model skill: does the answer's query return the expected rows?"""
    expected_sql, answer_sql = extract_sql(expected), extract_sql(answer)
    if expected_sql is None:
        raise ValueError("the expected answer has no sql block")
    if answer_sql is None:
        return False
    want, got = _rows(expected_sql), _rows(answer_sql)
    if want is None:
        raise ValueError(f"the expected statement does not run: {expected_sql!r}")
    if got is None:
        return False
    # compare the printed values: Amount and Position raise instead of returning False when an answer has a string
    # where they are expected
    got_text, want_text = list(map(repr, got)), list(map(repr, want))
    if re.search(r"\bORDER\s+BY\b", expected_sql, re.IGNORECASE):
        return got_text == want_text
    return sorted(got_text) == sorted(want_text)
