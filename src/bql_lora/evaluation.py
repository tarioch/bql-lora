"""Score a model's answer on the evaluation set by running it, not by comparing text.

Every evaluation example comes from one synthetic ledger (``EVAL_LEDGER_SEED``) that no training example was drawn
from. An answer is right when its ``sql`` block runs on that ledger and returns the same rows as the expected
statement. Column names are not compared (an alias is not a mistake), row order only when the expected statement
has an ``ORDER BY``.
"""

from __future__ import annotations

import datetime as dt
import functools
import itertools
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


def strict_rows(prompt: list[dict], expected: str, answer: str) -> bool:
    """Does the answer's query return exactly the expected columns and rows?"""
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


def _plain(value):
    """A date as a plain ``datetime.date``: the executor hands out a subclass while ``today()`` is fixed."""
    if isinstance(value, dt.date) and not isinstance(value, dt.datetime):
        return dt.date(value.year, value.month, value.day)
    return value


def _is_year(values: list) -> bool:
    return bool(values) and all(isinstance(v, int) and not isinstance(v, bool) and 1900 <= v <= 2200 for v in values)


def _is_month(values: list) -> bool:
    return bool(values) and all(isinstance(v, int) and not isinstance(v, bool) and 1 <= v <= 12 for v in values)


def _columns(columns: list[list]) -> list[list]:
    """The columns of a result plus the ones they stand for: a date column also gives its year and month, and a
    year column next to a month column gives the first day of that month (``yearmonth(date)``)."""
    out = list(columns)
    for col in columns:
        if col and all(isinstance(v, dt.date) for v in col):
            out.append([v.year for v in col])
            out.append([v.month for v in col])
    for years, months in itertools.permutations(columns, 2):
        if _is_year(years) and _is_month(months):
            out.append([dt.date(y, m, 1) for y, m in zip(years, months, strict=True)])
    return out


def loose_rows(prompt: list[dict], expected: str, answer: str) -> bool:
    """A fairer second number next to :func:`strict_rows`: is the answer a usable answer to the question?

    Right when the answer's query returns every expected column with the same values: extra columns, another column
    order, one row per posting instead of per transaction and ``yearmonth(date)`` instead of ``year, month`` are
    accepted. Rows are compared as sets, and in order only when the expected statement has a ``LIMIT`` (the top-N
    question), where the order decides which rows come back. A wrong filter, a missing column or a wrong value is
    still wrong.
    """
    expected_sql, answer_sql = extract_sql(expected), extract_sql(answer)
    if expected_sql is None:
        raise ValueError("the expected answer has no sql block")
    if answer_sql is None:
        return False
    try:
        want_result, got_result = eval_executor().run(expected_sql), None
    except QueryError as e:
        raise ValueError(f"the expected statement does not run: {expected_sql!r}") from e
    try:
        got_result = eval_executor().run(answer_sql)
    except QueryError:
        return False
    want_rows = [tuple(repr(_plain(v)) for v in row) for row in want_result.rows]
    ordered = bool(re.search(r"\bLIMIT\b", expected_sql, re.IGNORECASE))
    if not want_rows:
        return not got_result.rows
    if not got_result.rows:
        return False

    def by_column(rows: list[tuple]) -> list[list]:
        return [[_plain(v) for v in col] for col in zip(*rows, strict=True)]

    want_cols = by_column(want_result.rows)
    got_cols = _columns(by_column(got_result.rows))
    reprs = [list(map(repr, col)) for col in got_cols]
    options = [[i for i, col in enumerate(reprs) if set(col) == {repr(v) for v in want}] for want in want_cols]
    for choice in itertools.islice(itertools.product(*options), 2000):
        if len(set(choice)) < len(choice):
            continue
        rows = list(zip(*(reprs[i] for i in choice), strict=True))
        if ordered:
            if rows[: len(want_rows)] == want_rows and len(rows) == len(want_rows):
                return True
        elif set(rows) == set(want_rows):
            return True
    return False


def same_rows(prompt: list[dict], expected: str, answer: str) -> dict[str, bool]:
    """``evaluate.py --check`` function of the ollama-model skill, the one place that judges an answer.

    Two named scores, the first is the headline: ``strict`` (exactly the expected columns and rows, the form the app
    expects) and ``loose`` (the query is a usable answer, see :func:`loose_rows`).
    """
    return {"strict": strict_rows(prompt, expected, answer), "loose": loose_rows(prompt, expected, answer)}
