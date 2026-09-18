"""Run BQL statements against a generated ledger using the real beanquery implementation."""

from __future__ import annotations

import contextlib
import datetime as dt
import types
from dataclasses import dataclass

import beanquery
from beanquery import query_env
from beanquery.shell import render_exception

from .ledger import Ledger


class QueryError(Exception):
    """Raised when beanquery rejects a statement. ``rendered`` matches what the shell prints."""

    def __init__(self, exc: Exception):
        super().__init__(str(exc))
        self.exc = exc
        self.kind = type(exc).__name__
        self.message = str(exc)
        # Parse and compile errors carry a source location; the shell renders it with a caret line.
        self.rendered = self.message
        if getattr(exc, "parseinfo", None):
            try:
                self.rendered = render_exception(exc, indent="| ")
            except Exception:  # pragma: no cover - defensive
                pass


@dataclass
class Result:
    columns: list[str]
    rows: list[tuple]

    def __len__(self) -> int:
        return len(self.rows)


@contextlib.contextmanager
def fixed_today(day: dt.date):
    """Make the BQL ``today()`` function return ``day`` (restores the real behaviour on exit)."""

    class FakeDate(dt.date):
        @classmethod
        def today(cls):
            return dt.date(day.year, day.month, day.day)

    real = query_env.datetime
    shim = types.SimpleNamespace(**{k: getattr(dt, k) for k in dir(dt) if not k.startswith("__")})
    shim.date = FakeDate
    query_env.datetime = shim
    try:
        yield
    finally:
        query_env.datetime = real


class Executor:
    """A beanquery connection over an in-memory ledger."""

    def __init__(self, ledger: Ledger):
        self.ledger = ledger
        self.conn = beanquery.connect("beancount:", entries=ledger.entries, errors=ledger.errors, options=ledger.options)

    def run(self, statement: str) -> Result:
        with fixed_today(self.ledger.today):
            try:
                cursor = self.conn.execute(statement)
                rows = cursor.fetchall()
            except (beanquery.Error, beanquery.ParseError, beanquery.CompilationError) as exc:
                raise QueryError(exc) from exc
            except Exception as exc:  # runtime failures inside functions, e.g. bad regex
                raise QueryError(exc) from exc
            return Result([d[0] for d in (cursor.description or [])], rows)

    def parses(self, statement: str) -> bool:
        try:
            self.conn.parse(statement)
            return True
        except Exception:
            return False
