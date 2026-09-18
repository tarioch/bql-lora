"""Natural-language building blocks: periods, phrasing helpers."""

from __future__ import annotations

import calendar
import datetime as dt
import random
import re
from dataclasses import dataclass, field

MONTHS = list(calendar.month_name)[1:]
ORD = {1: "first", 2: "second", 3: "third", 4: "fourth"}


@dataclass
class Period:
    """A time window with several equivalent BQL spellings.

    ``nl``: English phrases that describe the window.
    ``forms``: alternatives usable on tables that have ``year``/``month`` columns (postings, entries).
    ``tforms``: alternatives usable on every table (transactions, prices, balances, ...).
    Each form is a dict with ``where`` (list of conditions) and optionally ``frm`` (a FROM expression).
    """

    nl: list[str]
    forms: list[dict]
    tforms: list[dict]
    label: str = ""
    kind: str = ""

    def phrase(self, rng: random.Random) -> str:
        return rng.choice(self.nl)

    def form(self, rng: random.Random, table: str = "postings", allow_from: bool = True) -> dict:
        pool = self.forms if table in ("postings", "entries") else self.tforms
        if not allow_from:
            pool = [f for f in pool if not f.get("frm")] or pool
        return rng.choice(pool)


def _range(a: dt.date, b: dt.date) -> dict:
    return {"where": [f"date >= {a.isoformat()}", f"date < {b.isoformat()}"]}


def p_year(y: int) -> Period:
    return Period(
        nl=[f"in {y}", f"during {y}", f"for {y}", f"in the year {y}", f"over {y}"],
        forms=[{"where": [f"year = {y}"]}, {"where": [f"year = {y}"]}, {"where": [], "frm": f"year = {y}"}, _range(dt.date(y, 1, 1), dt.date(y + 1, 1, 1))],
        tforms=[{"where": [f"year(date) = {y}"]}, _range(dt.date(y, 1, 1), dt.date(y + 1, 1, 1))],
        label=f"in {y}", kind="year")


def p_month(y: int, m: int) -> Period:
    a = dt.date(y, m, 1)
    b = dt.date(y + (m == 12), m % 12 + 1, 1)
    name = MONTHS[m - 1]
    return Period(
        nl=[f"in {name} {y}", f"during {name} {y}", f"for {name} {y}", f"in {y}-{m:02d}", f"in {name}, {y}", f"in the month of {name} {y}"],
        forms=[{"where": [f"year = {y}", f"month = {m}"]}, {"where": [f"year = {y}", f"month = {m}"]}, _range(a, b), {"where": [], "frm": f"year = {y} AND month = {m}"}],
        tforms=[{"where": [f"year(date) = {y}", f"month(date) = {m}"]}, _range(a, b)],
        label=f"in {name} {y}", kind="month")


def p_quarter(y: int, qn: int) -> Period:
    a = dt.date(y, 3 * qn - 2, 1)
    b = dt.date(y + (qn == 4), (3 * qn) % 12 + 1, 1)
    return Period(
        nl=[f"in Q{qn} {y}", f"in the {ORD[qn]} quarter of {y}", f"during Q{qn} of {y}", f"for {y}-Q{qn}"],
        forms=[{"where": [f"quarter(date) = '{y}-Q{qn}'"]}, _range(a, b)],
        tforms=[{"where": [f"quarter(date) = '{y}-Q{qn}'"]}, _range(a, b)],
        label=f"in Q{qn} {y}", kind="quarter")


def p_between(a: dt.date, b: dt.date) -> Period:
    return Period(
        nl=[f"between {a} and {b}", f"from {a} to {b}", f"from {a} through {b}", f"between {a.strftime('%B %d, %Y').replace(' 0', ' ')} and {b.strftime('%B %d, %Y').replace(' 0', ' ')}"],
        forms=[{"where": [f"date >= {a}", f"date <= {b}"]}, {"where": [f"date BETWEEN {a} AND {b}"]}],
        tforms=[{"where": [f"date >= {a}", f"date <= {b}"]}, {"where": [f"date BETWEEN {a} AND {b}"]}],
        label=f"between {a} and {b}", kind="range")


def p_since(a: dt.date) -> Period:
    return Period(
        nl=[f"since {a}", f"from {a} onwards", f"on or after {a}"],
        forms=[{"where": [f"date >= {a}"]}], tforms=[{"where": [f"date >= {a}"]}], label=f"since {a}", kind="since")


def p_after(a: dt.date) -> Period:
    return Period(nl=[f"after {a}", f"since {a}, excluding that day"], forms=[{"where": [f"date > {a}"]}], tforms=[{"where": [f"date > {a}"]}], label=f"after {a}", kind="after")


def p_before(a: dt.date) -> Period:
    return Period(nl=[f"before {a}", f"prior to {a}", f"up to (but not including) {a}"],
                  forms=[{"where": [f"date < {a}"]}], tforms=[{"where": [f"date < {a}"]}], label=f"before {a}", kind="before")


def p_asof(a: dt.date) -> Period:
    return Period(nl=[f"as of {a}", f"up to and including {a}", f"on {a}, counting that day"],
                  forms=[{"where": [f"date <= {a}"]}], tforms=[{"where": [f"date <= {a}"]}], label=f"up to {a}", kind="asof")


def p_last_days(n: int) -> Period:
    return Period(
        nl=[f"in the last {n} days", f"over the past {n} days", f"during the last {n} days", f"within the last {n} days"],
        forms=[{"where": [f"date >= date_add(today(), -{n})"]}, {"where": [f"date_diff(today(), date) <= {n}"]}],
        tforms=[{"where": [f"date >= date_add(today(), -{n})"]}, {"where": [f"date_diff(today(), date) <= {n}"]}],
        label=f"in the last {n} days", kind="relative")


def p_this_year() -> Period:
    return Period(nl=["this year", "in the current year", "so far this year", "year to date"],
                  forms=[{"where": ["year = year(today())"]}, {"where": [], "frm": "year = year(today())"}],
                  tforms=[{"where": ["year(date) = year(today())"]}], label="this year", kind="relative")


def p_last_year() -> Period:
    return Period(nl=["last year", "in the previous year"], forms=[{"where": ["year = year(today()) - 1"]}],
                  tforms=[{"where": ["year(date) = year(today()) - 1"]}], label="last year", kind="relative")


def p_this_month() -> Period:
    return Period(nl=["this month", "in the current month", "so far this month"],
                  forms=[{"where": ["year = year(today())", "month = month(today())"]}],
                  tforms=[{"where": ["year(date) = year(today())", "month(date) = month(today())"]}], label="this month", kind="relative")


def p_last_month() -> Period:
    cond = "date_trunc('month', date) = date_trunc('month', date_add(date_trunc('month', today()), -1))"
    return Period(nl=["last month", "in the previous month"], forms=[{"where": [cond]}], tforms=[{"where": [cond]}], label="last month", kind="relative")


def month_end(y: int, m: int) -> dt.date:
    return dt.date(y, m, calendar.monthrange(y, m)[1])


# ------------------------------------------------------------------ phrasing

OPENERS = ["", "", "", "", "Please show", "Can you show", "Show me", "Give me", "I'd like to see", "I want to see", "Could you list", "Find", "Get", "Tell me", "Let me see"]


def polish(rng: random.Random, text: str) -> str:
    """Randomly perturb capitalisation and punctuation the way real users type."""
    r = rng.random()
    text = re.sub(r"\s+", " ", text).strip()
    if r < 0.10:
        text = text[0].lower() + text[1:]
    if rng.random() < 0.25:
        text = text.rstrip("?.!")
    return text


def pick(rng: random.Random, options: list[str]) -> str:
    return rng.choice(options)


def human_list(items: list[str]) -> str:
    if len(items) <= 1:
        return "".join(items)
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"


def split_camel(word: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", word)


def natural_name(account: str) -> str:
    leaf = account.split(":")[-1]
    if leaf.isupper():  # a ticker or currency code such as NESN or GBP keeps its case
        return leaf
    return split_camel(leaf).replace("-", " ").lower()
