"""Small helpers to build BQL text with one consistent layout."""

from __future__ import annotations

MAX_ONE_LINE = 100


def q(value: str) -> str:
    """Quote a string literal. BQL accepts single or double quotes; pick the one that avoids escaping."""
    if "'" in value:
        return f'"{value}"'
    return f"'{value}'"


def cond(parts: list[str] | None) -> str | None:
    """Join WHERE conditions with AND, parenthesizing anything that contains OR."""
    parts = [p for p in (parts or []) if p]
    if not parts:
        return None
    if len(parts) > 1:
        parts = [f"({p})" if " OR " in p and not p.startswith("(") else p for p in parts]
    return " AND ".join(parts)


def select(targets: list[str] | str, *, frm: str | None = None, where: list[str] | str | None = None,
           group: list[str] | str | None = None, having: str | None = None,
           order: list[str] | str | None = None, pivot: str | None = None,
           limit: int | None = None, distinct: bool = False, multiline: bool | None = None) -> str:
    if isinstance(targets, list):
        targets = ", ".join(targets)
    if isinstance(where, str):
        where = [where]
    if isinstance(group, list):
        group = ", ".join(group)
    if isinstance(order, list):
        order = ", ".join(order)
    clauses = [("SELECT DISTINCT " if distinct else "SELECT ") + targets]
    if frm:
        clauses.append(f"FROM {frm}")
    w = cond(where)
    if w:
        clauses.append(f"WHERE {w}")
    if group:
        clauses.append(f"GROUP BY {group}" + (f" HAVING {having}" if having else ""))
    if order:
        clauses.append(f"ORDER BY {order}")
    if pivot:
        clauses.append(f"PIVOT BY {pivot}")
    if limit is not None:
        clauses.append(f"LIMIT {limit}")
    one = " ".join(clauses)
    if multiline is None:
        multiline = len(one) > MAX_ONE_LINE
    return "\n".join(clauses) if multiline else one
