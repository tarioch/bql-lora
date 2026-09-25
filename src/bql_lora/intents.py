"""Question -> BQL intent library.

Each intent picks concrete parameters from a ledger, then produces the English question and the BQL
statement from the *same* parameters, so they always agree.  Every statement is executed against the
ledger by the caller and dropped if beanquery rejects it or returns no rows.
"""

from __future__ import annotations

import datetime as dt
import random
import re
from dataclasses import dataclass
from typing import Callable

from . import nl
from .bqlfmt import cond, q, select
from .executor import Executor, QueryError
from .ledger import Ledger
from .nl import (Period, human_list, natural_name, p_after, p_asof, p_before, p_between, p_last_days, p_last_month,
                 p_last_year, p_month, p_quarter, p_since, p_this_month, p_this_year, p_year, pick, polish)


@dataclass
class Sample:
    intent: str
    question: str
    bql: str
    explanation: str
    allow_empty: bool = False


class Skip(Exception):
    """This intent cannot be instantiated on this ledger."""


REGISTRY: list[tuple[str, float, Callable]] = []


def intent(weight: float):
    def deco(fn):
        REGISTRY.append((fn.__name__, weight, fn))
        return fn
    return deco


@dataclass
class Sel:
    """An account selection with its BQL condition and English description."""
    cond: str
    nl: str
    label: str
    root: str
    accounts: list[str]


class Gen:
    def __init__(self, ledger: Ledger, ex: Executor, rng: random.Random, mode: str):
        self.led = ledger
        self.ex = ex
        self.rng = rng
        self.mode = mode  # 'full' | 'compact' | 'none'  (how much ledger context the prompt carries)
        self.base = ledger.base
        self.used = sorted(ledger.used_accounts())
        self.today = ledger.today
        self.first = ledger.first_date
        self.last = ledger.last_date
        self._months = sorted({(t.date.year, t.date.month) for t in ledger.txns})
        self._years = sorted({t.date.year for t in ledger.txns})

    # ------------------------------------------------------------ sampling
    def accounts_of(self, root: str) -> list[str]:
        return [a for a in self.used if a.split(":")[0] == root]

    def leaf_unique(self, account: str) -> bool:
        leaf = account.split(":")[-1]
        return sum(1 for a in self.led.accounts if a.split(":")[-1] == leaf) == 1

    def acct_ref(self, account: str, allow_natural: bool = True) -> str:
        """How the question refers to an account: the full name, or (when the prompt lists accounts) a natural alias."""
        if allow_natural and self.mode != "none" and self.leaf_unique(account) and self.rng.random() < 0.45:
            return natural_name(account)
        return account

    def keyword_for(self, account: str) -> str | None:
        """A regex keyword (the account's leaf name) that matches this account and no other opened account."""
        leaf = account.split(":")[-1]
        if [a for a in self.led.accounts if re.search(leaf, a)] == [account]:
            return leaf
        return None

    def acct_pair(self, account: str) -> tuple[str, str, str]:
        """(English reference, BQL condition, regexp) for one account; the three always agree.

        Without a schema in the prompt the model cannot know full account names, so half the time the question uses
        a natural alias ("groceries") and the BQL a keyword regexp (``account ~ 'Groceries'``) that matches this
        account only. That works whatever the ledger's account hierarchy looks like.
        """
        if self.mode == "none" and self.rng.random() < 0.5:
            kw = self.keyword_for(account)
            if kw:
                return natural_name(account), f"account ~ {q(kw)}", kw
        return self.acct_ref(account), f"account = {q(account)}", account

    def ym(self) -> tuple[int, int]:
        return self.rng.choice(self._months)

    def date_in_span(self, margin: int = 0) -> dt.date:
        span = (self.last - self.first).days
        return self.first + dt.timedelta(days=self.rng.randint(margin, max(margin, span - margin)))

    def period(self, kinds: list[str] | None = None) -> Period:
        rng = self.rng
        weights = {"month": 26, "year": 18, "quarter": 9, "range": 12, "since": 5, "before": 3, "last_days": 8, "relative": 10}
        if kinds:
            weights = {k: v for k, v in weights.items() if k in kinds}
        kind = rng.choices(list(weights), weights=list(weights.values()))[0]
        if kind == "month":
            return p_month(*self.ym())
        if kind == "year":
            return p_year(rng.choice(self._years))
        if kind == "quarter":
            y, m = self.ym()
            return p_quarter(y, (m - 1) // 3 + 1)
        if kind == "range":
            a = self.date_in_span(20)
            b = a + dt.timedelta(days=rng.randint(20, 120))
            return p_between(a, min(b, self.last))
        if kind == "since":
            return p_since(self.date_in_span(30))
        if kind == "before":
            return p_before(self.date_in_span(30))
        if kind == "last_days":
            return p_last_days(rng.choice([7, 14, 30, 45, 60, 90, 180]))
        return rng.choice([p_this_year(), p_last_year(), p_this_month(), p_last_month()])

    # ------------------------------------------------------------ selections
    def selector(self, root: str, allow_multi: bool = True) -> Sel:
        rng = self.rng
        accs = self.accounts_of(root)
        if not accs:
            raise Skip
        kinds = ["exact", "exact", "root", "prefix", "keyword", "multi"]
        if not allow_multi:
            kinds = ["exact", "root", "prefix"]
        kind = rng.choice(kinds)
        if kind == "exact":
            a = rng.choice(accs)
            ref, c, _ = self.acct_pair(a)
            return Sel(c, ref, a, root, [a])
        if kind == "root" or (kind == "prefix" and not any(len(a.split(":")) > 2 for a in accs)):
            word = {"Expenses": ["all expenses", "all my expense accounts", "the Expenses accounts"], "Income": ["all income", "all income accounts"],
                    "Assets": ["all assets", "all asset accounts"], "Liabilities": ["all liabilities", "all liability accounts"], "Equity": ["equity accounts"]}[root]
            return Sel(f"account ~ '^{root}'", pick(rng, word), f"all {root} accounts", root, accs)
        if kind == "prefix":
            prefixes = sorted({":".join(a.split(":")[:k]) for a in accs for k in (2, 3) if len(a.split(":")) > k})
            if not prefixes:
                raise Skip
            p = rng.choice(prefixes)
            members = [a for a in accs if a == p or a.startswith(p + ":")]
            if len(members) < 2:
                raise Skip
            return Sel(f"account ~ '^{p}'", pick(rng, [f"everything under {p}", f"all {p} accounts", f"the {p} accounts"]), f"accounts under {p}", root, members)
        if kind == "keyword":
            a = rng.choice([x for x in accs if self.leaf_unique(x)] or accs)
            word = a.split(":")[-1]
            members = [x for x in accs if word in x]
            return Sel(f"account ~ '{word}'", f"accounts matching '{word}'", f"accounts matching {word}", root, members)
        two = rng.sample(accs, k=min(2, len(accs)))
        if len(two) < 2:
            raise Skip
        kws = [self.keyword_for(a) for a in two]
        if self.mode == "none" and all(kws) and rng.random() < 0.5:
            return Sel(f"account ~ {q('|'.join(kws))}", f"{natural_name(two[0])} and {natural_name(two[1])}", f"{two[0]} and {two[1]}", root, two)
        if rng.random() < 0.5:
            c = f"account IN ({q(two[0])}, {q(two[1])})"
        else:
            c = f"account ~ '^({two[0]}|{two[1]})'"
        return Sel(c, f"{self.acct_ref(two[0])} and {self.acct_ref(two[1])}", f"{two[0]} and {two[1]}", root, two)

    # ------------------------------------------------------------ helpers
    def run(self, bql: str):
        return self.ex.run(bql)

    def single_currency(self, where: list[str], frm: str | None = None) -> bool:
        try:
            res = self.run(select("currency", frm=frm, where=where, group="currency"))
        except QueryError:
            return False
        return len(res.rows) == 1

    def payee(self) -> str:
        pool = self.led.payees[:40]
        if not pool:
            raise Skip
        return self.rng.choice(pool)

    def tag(self) -> str:
        if not self.led.tags:
            raise Skip
        return self.rng.choice(self.led.tags)

    def link(self) -> str:
        if not self.led.links:
            raise Skip
        return self.rng.choice(self.led.links)

    def opener(self) -> str:
        return self.rng.choice(nl.OPENERS)

    def ask(self, *templates: str, **slots) -> str:
        """Pick one of the templates, fill slots, prepend an optional opener and polish."""
        t = self.rng.choice(templates)
        text = t.format(**slots)
        return polish(self.rng, text)

    def show(self, *templates: str, **slots) -> str:
        """Like ask() but for imperative 'show me' style prompts (adds an opener half the time)."""
        text = self.rng.choice(templates).format(**slots)
        op = self.opener()
        if op and self.rng.random() < 0.6:
            text = f"{op} {text[0].lower() + text[1:]}" if not text.startswith("How") else text
        return polish(self.rng, text)

    def use_from(self, form: dict, base_where: list[str] | None = None) -> tuple[str | None, list[str]]:
        """Split a Period form into (FROM expression, WHERE conditions)."""
        return form.get("frm"), list(form["where"]) + list(base_where or [])


ROOT_NOUNS = {
    "Expenses": ["spending", "expenses", "costs", "spend"],
    "Income": ["income", "earnings", "revenue"],
    "Assets": ["balance", "holdings", "assets"],
    "Liabilities": ["debt", "liabilities", "amount owed"],
}


# ---------------------------------------------------------------------------
# A. Compositional aggregate reports

@intent(30)
def agg_report(g: Gen) -> Sample:
    rng = g.rng
    root = rng.choices(["Expenses", "Income", "Assets", "Liabilities"], weights=[60, 14, 16, 10])[0]
    sel = g.selector(root)
    period = g.period() if rng.random() < 0.8 else None
    dim = rng.choices(["none", "account", "root2", "month", "year", "quarter", "payee", "currency", "weekday", "yearmonth"],
                      weights=[16, 20, 10, 16, 8, 6, 12, 3, 4, 5])[0]
    if dim == "root2" and not any(len(a.split(":")) > 2 for a in sel.accounts):
        dim = "account"
    if dim == "account" and len(sel.accounts) < 2:
        dim = "none"
    if dim == "payee" and root not in ("Expenses", "Income"):
        dim = "account"
    if dim == "currency":
        dim = "none" if rng.random() < 0.5 else "currency"
    # Drop time dimensions that are redundant with (or finer-grained than sensible for) the chosen period:
    # e.g. "per year" when already scoped to one quarter, or "by month" when scoped to a single month.
    if period and period.kind in ("month",) and dim in ("month", "year", "yearmonth"):
        dim = "none"
    if period and period.kind in ("quarter",) and dim in ("year", "quarter"):
        dim = "none"
    if period and period.kind == "year" and dim == "year":
        dim = "none"

    where = [sel.cond]
    frm = None
    if period:
        frm, where = g.use_from(period.form(rng, "postings"), where)
    if root == "Assets" and dim in ("month", "year", "quarter", "yearmonth", "weekday") and rng.random() < 0.7:
        raise Skip  # sums of balance changes per period are unusual for asset accounts

    # measure
    negate = root == "Income"
    mkind = rng.choices(["inv", "num", "conv", "cost"], weights=[45, 0 if g.mode == "none" else 30, 15, 10 if root == "Assets" else 0])[0]
    if mkind == "num" and not g.single_currency(where, frm):
        mkind = "inv"
    if mkind == "cost" and root != "Assets":
        mkind = "inv"
    if mkind == "conv":
        expr, tail = f"convert(sum(position), '{g.base}')", f" in {g.base}"
        if negate:
            expr = f"neg({expr})"
    elif mkind == "cost":
        expr, tail = "cost(sum(position))", " at cost"
    elif mkind == "num":
        expr, tail = ("-sum(number)" if negate else "sum(number)"), ""
    else:
        expr, tail = ("neg(sum(position))" if negate else "sum(position)"), ""
    alias = rng.choice(["total", "amount", "total", ""]) if dim != "none" or rng.random() < 0.3 else ""
    target_expr = f"{expr} AS {alias}" if alias else expr

    dims = {
        "none": ([], [], None, ""),
        "account": (["account"], ["account"], "account", "for each account"),
        "root2": (["root(account, 2) AS category"], ["category"], "category", "per top-level category"),
        "month": (["year", "month"], ["year", "month"], "year, month", "per month"),
        "yearmonth": (["yearmonth(date) AS month"], ["month"], "month", "by month"),
        "year": (["year"], ["year"], "year", "per year"),
        "quarter": (["quarter(date) AS quarter"], ["quarter"], "quarter", "per quarter"),
        "payee": (["payee"], ["payee"], "payee", "by payee"),
        "currency": (["currency"], ["currency"], "currency", "per currency"),
        "weekday": (["weekday(date) AS weekday"], ["weekday"], "weekday", "by weekday"),
    }
    tgt, grp_names, order_by_name, dim_nl = dims[dim]
    noun = pick(rng, ROOT_NOUNS[root])
    group = None
    if dim != "none":
        group = grp_names if rng.random() < 0.9 else None
        if group:
            group = ", ".join(group)
    order, limit, ord_nl = None, None, ""
    if dim in ("account", "root2", "payee") and rng.random() < 0.55:
        direction = rng.choice(["DESC", "ASC"]) if rng.random() < 0.3 else "DESC"
        order = [f"{alias or 'total'} {direction}"] if alias else [f"{expr} {direction}"]
        ord_nl = " with the largest first" if direction == "DESC" else " with the smallest first"
        if rng.random() < 0.55:
            limit = rng.choice([3, 5, 5, 10])
            ord_nl = ""
    elif dim in ("month", "year", "quarter", "yearmonth"):
        order = [c.strip() for c in ((grp_names if group else []) or [])]
        if not order:
            order = None
        elif rng.random() < 0.4:
            order = None
    bql = select(tgt + [target_expr], frm=frm, where=where, group=group, order=order, limit=limit)
    per = period.phrase(rng) if period else ""
    per_s = f" {per}" if per else ""

    if limit:
        top = f"top {limit}"
        if dim == "payee":
            what = pick(rng, [f"the {top} payees by {noun}", f"which {limit} payees account for the most {noun}"])
        elif dim == "account":
            what = pick(rng, [f"the {top} accounts by {noun}", f"the {limit} accounts with the highest {noun}"])
        else:
            what = pick(rng, [f"the {top} categories by {noun}", f"the {limit} biggest {noun} categories"])
        if root in ("Income",):
            what = what.replace("highest", "highest")
        question = g.ask(f"What are {what} for {sel.nl}{per_s}{tail}?", f"Show {what}, considering {sel.nl}{per_s}{tail}.", f"List {what} for {sel.nl}{per_s}{tail}")
    else:
        subject = {"Expenses": f"{noun} on {sel.nl}", "Income": f"{noun} from {sel.nl}", "Assets": f"{noun} of {sel.nl}", "Liabilities": f"{noun} on {sel.nl}"}[root]
        if sel.nl.startswith(("all ", "everything", "the ", "accounts")):
            subject = f"{noun} across {sel.nl}" if root != "Assets" else f"{noun} in {sel.nl}"
        qt = [f"What is the total {subject}{per_s}{tail}?", f"How much {noun} do I have for {sel.nl}{per_s}{tail}?" if root != "Expenses" else f"How much did I spend on {sel.nl}{per_s}{tail}?",
              f"Total {subject}{per_s}{tail}", f"Show me the total {subject}{per_s}{tail} {dim_nl}"]
        if root == "Income":
            qt.append(f"How much did I earn from {sel.nl}{per_s}{tail}?")
        if dim != "none":
            qt = [f"Show the {noun} on {sel.nl} {dim_nl}{per_s}{tail}{ord_nl}", f"Break down the {noun} for {sel.nl} {dim_nl}{per_s}{tail}{ord_nl}",
                  f"{noun.capitalize()} for {sel.nl} {dim_nl}{per_s}{tail}{ord_nl}", f"Can you give me the {noun} on {sel.nl} {dim_nl}{per_s}{tail}{ord_nl}?"]
        question = g.ask(*qt)
    expl = (f"Sums {'(negated, since income is stored as negative numbers) ' if negate else ''}the postings for {sel.label}"
            f"{' ' + period.label if period else ''}{', ' + dim_nl if dim_nl else ''}{', converted to ' + g.base if mkind == 'conv' else ''}"
            f"{', valued at cost' if mkind == 'cost' else ''}.")
    return Sample("agg_report", question, bql, expl)


# ---------------------------------------------------------------------------
# B. Balances

@intent(7)
def balance_of_account(g: Gen) -> Sample:
    rng = g.rng
    root = rng.choice(["Assets", "Assets", "Liabilities"])
    accs = g.accounts_of(root)
    if not accs:
        raise Skip
    a = rng.choice(accs)
    ref, acond, apat = g.acct_pair(a)
    when = None
    if rng.random() < 0.5:
        when = g.date_in_span(30)
    style = rng.choice(["sum", "sum", "balances", "journal_last"])
    if style == "sum":
        where = [acond] + ([f"date <= {when}"] if when else [])
        conv = rng.random() < 0.3
        tail = f" in {g.base}" if conv else ""
        asof = f" as of {when}" if when else ""
        if conv:
            bql = select(f"convert(sum(position), '{g.base}')", where=where)
        else:
            bql = select("sum(position)" if rng.random() < 0.7 else "sum(position) AS balance", where=where)
        question = g.ask(f"What is the balance of {ref}{asof}{tail}?",
                         f"How much is in {ref}{asof}{tail}?",
                         f"Balance of {ref}{asof}{tail}",
                         f"What was the balance of {ref} at the end of {when}{tail}?" if when else f"Current balance of {ref}{tail}?")
        expl = f"Adds up all postings to {a}{' up to and including ' + str(when) if when else ''}."
    elif style == "balances":
        w = [acond] if (rng.random() < 0.5 or apat != a) else [f"account ~ '^{a}'"]
        frm = f"CLOSE ON {when}" if when else None
        bql = "BALANCES" + (f" FROM {frm}" if frm else "") + f" WHERE {cond(w)}"
        question = g.ask(f"Show the balance for {ref}{' before ' + str(when) if when else ''} using the BALANCES statement",
                         f"BALANCES of {ref}{' before ' + str(when) if when else ''}")
        expl = f"BALANCES prints one line per account with its summed inventory{'; CLOSE ON truncates the ledger at ' + str(when) + ' (entries on that date are excluded)' if when else ''}."
    else:
        bql = f"JOURNAL {q(apat)}"
        n = rng.choice([None, None, 10, 5])
        bql = select("date, narration, position, balance", where=[acond], order=["date DESC"], limit=n) if n else bql
        question = g.ask(f"Show the register of {ref}" + (f", last {n} entries" if n else ""), f"Give me the journal for {ref}" + (f" (most recent {n} only)" if n else ""),
                         f"List all postings to {ref} with a running balance" if not n else f"Show the {n} most recent postings on {ref} with the running balance")
        expl = "Lists postings for the account together with the running balance column." + (" Sorted newest first and limited." if n else "")
    return Sample("balance_of_account", question, bql, expl)


@intent(6)
def balances_by_root(g: Gen) -> Sample:
    rng = g.rng
    root = rng.choice(["Assets", "Assets", "Liabilities", "Expenses", "Income"])
    if not g.accounts_of(root):
        raise Skip
    variant = rng.choice(["group", "group", "balances", "cost", "units", "value", "leaf"])
    when = g.date_in_span(30) if rng.random() < 0.35 else None
    wh = [f"account ~ '^{root}'"] + ([f"date <= {when}"] if when else [])
    asof = f" as of {when}" if when else ""
    if variant == "group":
        bql = select(["account", "sum(position)"], where=wh, group="account", order=["account"])
        question = g.ask(f"List the balance of every {root.lower()[:-1] if root.endswith('s') else root.lower()} account{asof}", f"What are the balances of all {root} accounts{asof}?", f"Balance per account for {root}{asof}")
        expl = f"Sums postings per account for accounts starting with {root}, sorted alphabetically."
    elif variant == "balances":
        bql = "BALANCES" + (f" FROM CLOSE ON {when + dt.timedelta(days=1)}" if when else "") + f" WHERE account ~ '^{root}'"
        question = g.ask(f"Run a balances report for {root}{asof}", f"BALANCES for all {root} accounts{asof}")
        expl = "BALANCES aggregates every account's postings; WHERE limits it to the root."
    elif variant == "cost":
        bql = "BALANCES AT cost" + f" WHERE account ~ '^{root}'"
        question = g.ask(f"Show all {root} balances at cost", f"BALANCES at cost for {root} accounts")
        expl = "AT cost converts each position to its cost basis before summing."
    elif variant == "units":
        bql = "BALANCES AT units" + f" WHERE account ~ '^{root}'"
        question = g.ask(f"Show {root} balances in units only, without cost basis", f"{root} balances as units (strip the cost)")
        expl = "AT units strips cost information so lots of the same commodity add up."
    elif variant == "value":
        bql = select(["account", "value(sum(position))"], where=wh, group="account", order=["account"])
        question = g.ask(f"What is the market value of each {root} account{asof}?", f"Show {root} accounts at market value{asof}")
        expl = "value() converts the summed inventory to its market value using the price directives."
    else:
        bql = select(["leaf(account) AS name", "sum(position) AS balance"], where=wh, group="name", order=["name"])
        question = g.ask(f"Balances of {root} grouped by the last component of the account name{asof}", f"Group {root} balances by leaf account name{asof}")
        expl = "leaf() returns the last component of the account name, and postings are grouped by it."
    return Sample("balances_by_root", question, bql, expl)


@intent(4)
def net_worth(g: Gen) -> Sample:
    rng = g.rng
    when = g.date_in_span(30) if rng.random() < 0.5 else None
    where = ["account ~ '^(Assets|Liabilities)'"] + ([f"date <= {when}"] if when else [])
    asof = f" as of {when}" if when else ""
    if rng.random() < 0.3:
        bql = select("value(sum(position))", where=where)
        question = g.ask(f"What is my net worth{asof} at market value?", f"Total assets minus liabilities{asof}, valued at current market prices")
    else:
        bql = select(f"convert(sum(position), '{g.base}')", where=where)
        question = g.ask(f"What is my net worth{asof} in {g.base}?", f"Compute total assets minus liabilities{asof} in {g.base}",
                         f"How much am I worth{asof if when else ' now'} in {g.base}?", f"Net worth{asof} in {g.base}")
    return Sample("net_worth", question, bql, "Sums all Assets and Liabilities postings (liabilities are negative) and converts the inventory to a single currency.")


@intent(3)
def net_worth_by_month(g: Gen) -> Sample:
    rng = g.rng
    y = rng.choice(g._years)
    acct = rng.choice([a for a in g.accounts_of("Assets") if "Checking" in a or "Savings" in a] or g.accounts_of("Assets"))
    ref, acond, _ = g.acct_pair(acct)
    bql = select(["year", "month", "last(balance) AS end_balance"], where=[acond, f"year = {y}"], group="year, month", order=["year", "month"])
    question = g.ask(f"What was the month-end balance of {ref} for each month of {y}?", f"Show the closing balance of {ref} per month in {y}",
                     f"End-of-month balances for {ref}, {y}")
    return Sample("net_worth_by_month", question, bql, "The balance column is a running total per posting; taking last() per month yields the month-end balance.")


# ---------------------------------------------------------------------------
# C. Listing postings / transactions

@intent(5)
def top_expenses(g: Gen) -> Sample:
    rng = g.rng
    sel = g.selector("Expenses", allow_multi=False)
    period = g.period() if rng.random() < 0.7 else None
    n = rng.choice([1, 3, 5, 10, 10, 20])
    curf = rng.random() < 0.4
    base_where = [sel.cond] + ([f"currency = '{g.base}'"] if curf else [])
    frm, where = g.use_from(period.form(rng), base_where) if period else (None, base_where)
    cols = rng.choice([["date", "payee", "narration", "position"], ["date", "payee", "narration", "account", "number"], ["date", "description", "position"]])
    bql = select(cols, frm=frm, where=where, order=["number DESC"], limit=n)
    per = f" {period.phrase(rng)}" if period else ""
    per += f" (only {g.base} postings)" if curf else ""
    question = g.ask(f"What was my largest expense on {sel.nl}{per}?" if n == 1 else f"Show the {n} largest expenses on {sel.nl}{per}",
                     f"List the top {n} purchases for {sel.nl}{per}" if n > 1 else f"Find the single most expensive purchase for {sel.nl}{per}",
                     f"Which are the {n} most expensive things I bought ({sel.nl}){per}?" if n > 1 else f"Biggest expense on {sel.nl}{per}?")
    return Sample("top_expenses", question, bql, f"Lists postings on {sel.label}, sorted by amount descending, keeping the first {n}.")


@intent(5)
def recent_transactions(g: Gen) -> Sample:
    rng = g.rng
    n = rng.choice([5, 10, 10, 15, 20, 25])
    variant = rng.choice(["txns", "postings", "payee"])
    if variant == "txns":
        bql = select(["date", "payee", "narration"], frm="#transactions", order=["date DESC"], limit=n)
        question = g.ask(f"Show the last {n} transactions", f"What are the {n} most recent transactions?", f"List the latest {n} transactions with payee and narration")
        expl = "The #transactions table has one row per transaction (not per posting), sorted by date descending."
    elif variant == "postings":
        a = rng.choice(g.used)
        ref, acond, _ = g.acct_pair(a)
        bql = select(["date", "narration", "position"], where=[acond], order=["date DESC"], limit=n)
        question = g.ask(f"Show the latest {n} postings on {ref}", f"Last {n} entries in {ref}", f"What are the {n} most recent movements of {ref}?")
        expl = "Filters postings to one account and sorts newest first."
    else:
        p = g.payee()
        bql = select(["date", "narration"], frm="#transactions", where=[f"payee = {q(p)}"], order=["date DESC"], limit=n)
        question = g.ask(f"Show the last {n} transactions with {p}", f"What are the {n} latest transactions where the payee is {p}?")
        expl = "Filters transactions by exact payee and sorts newest first."
    return Sample("recent_transactions", question, bql, expl)


@intent(5)
def txns_by_payee(g: Gen) -> Sample:
    rng = g.rng
    p = g.payee()
    period = g.period() if rng.random() < 0.4 else None
    variant = rng.choice(["eq", "eq", "regex", "total", "count"])
    table = "#transactions" if variant in ("eq", "regex") and rng.random() < 0.5 else None
    frm, where = None, []
    if period:
        frm, where = g.use_from(period.form(rng, "transactions" if table else "postings"))
        if table:
            frm = None
    per = f" {period.phrase(rng)}" if period else ""
    if variant == "eq":
        where.append(f"payee = {q(p)}")
        cols = ["date", "narration"] + ([] if table else ["account", "position"])
        bql = select(cols, frm=table or frm, where=where, order=["date"])
        question = g.ask(f"Show all transactions with payee {p}{per}", f"List everything I paid to {p}{per}", f"All transactions from {p}{per}")
        expl = "Exact match on the payee column."
    elif variant == "regex":
        word = p.split()[0].rstrip(".,'")
        where.append(f"payee ~ {q(word)}")
        bql = select(["date", "payee", "narration"], frm=table, where=where, order=["date"])
        question = g.ask(f"Find transactions whose payee contains '{word}'{per}", f"Show entries where the payee matches {word}{per}")
        expl = "The ~ operator matches a regular expression anywhere in the string."
    elif variant == "total":
        where += [f"payee = {q(p)}", "account ~ '^Expenses'"]
        bql = select("sum(position)", frm=frm, where=where)
        question = g.ask(f"How much did I spend at {p}{per}?", f"Total spent at {p}{per}", f"What is the total of my expenses paid to {p}{per}?")
        expl = "Sums the Expenses postings of transactions whose payee is that value."
    else:
        where += [f"payee = {q(p)}"]
        bql = select("count(*)", frm="#transactions", where=[w for w in where if "year =" not in w and "month =" not in w] if False else where) if table is not None else select("count(*)", frm="#transactions", where=[f"payee = {q(p)}"])
        question = g.ask(f"How many transactions do I have with {p}?", f"Count the transactions for payee {p}", f"Number of times I paid {p}")
        expl = "Counts rows of the transactions table matching the payee."
    return Sample("txns_by_payee", question, bql, expl)


@intent(4)
def txns_by_text(g: Gen) -> Sample:
    rng = g.rng
    t = rng.choice(g.led.txns)
    words = [w.strip(".,()") for w in t.narration.split() if len(w) > 3]
    if not words:
        raise Skip
    w = rng.choice(words)
    ci = rng.random() < 0.4
    pat = f"(?i){w.lower()}" if ci else w
    field = rng.choice(["narration", "narration", "description"])
    neg = rng.random() < 0.15
    op = "!~" if neg else "~"
    frm = None
    bql = select(["date", "narration", "account", "position"] if field == "narration" else ["date", "description", "account", "position"],
                 where=[f"{field} {op} {q(pat)}"], order=["date"])
    verb = "do not mention" if neg else "mention"
    question = g.ask(f"Find all transactions whose {field} {verb}s '{w}'".replace("mentions", "mention").replace("dos not", "do not") if False else
                     (f"Show postings where the {field} does not contain '{w}'" if neg else f"Find all postings where the {field} contains '{w}'"),
                     (f"Everything except entries mentioning {w}" if neg else f"Search for '{w}' in the {field}"))
    if ci:
        question += " (case insensitive)"
    expl = f"Regular expression {'non-' if neg else ''}match on {field}{' with (?i) for case-insensitivity' if ci else ''}."
    return Sample("txns_by_text", question, bql, expl, allow_empty=neg)


@intent(4)
def txns_by_tag_link(g: Gen) -> Sample:
    rng = g.rng
    variant = rng.choice(["tag", "tag", "link", "tag_sum", "tag_list"])
    if variant == "tag":
        t = g.tag()
        bql = select(["date", "payee", "narration", "account", "position"], where=[f"{q(t)} IN tags"], order=["date"])
        question = g.ask(f"Show all postings tagged #{t}", f"List transactions with tag {t}", f"What entries have the tag '{t}'?")
        expl = "tags is a set of strings; IN tests membership."
    elif variant == "link":
        l = g.link()
        bql = select(["date", "payee", "narration", "account", "position"], where=[f"{q(l)} IN links"], order=["date"])
        question = g.ask(f"Show the postings linked with ^{l}", f"Find everything linked to '{l}'", f"List entries with link {l}")
        expl = "links is a set of strings; IN tests membership."
    elif variant == "tag_sum":
        t = g.tag()
        bql = select("sum(position)", where=[f"{q(t)} IN tags", "account ~ '^Expenses'"])
        question = g.ask(f"How much did I spend on things tagged #{t}?", f"Total expenses with the tag {t}", f"Sum of Expenses for tag '{t}'")
        expl = "Sums Expenses postings of transactions carrying the tag."
    else:
        bql = select(["date", "narration", "joinstr(tags) AS tags"], frm="#transactions", where=["tags IS NOT NULL"], order=["date"]) if False else select(["date", "narration", "tags"], frm="#transactions", where=["length(tags) > 0"], order=["date"])
        question = g.ask("Which transactions have at least one tag?", "List all tagged transactions with their tags", "Show transactions that carry tags, including the tags")
        expl = "length(tags) counts the elements of the tag set; the transactions table has one row per transaction."
    return Sample("txns_by_tag_link", question, bql, expl)


@intent(2)
def tag_totals(g: Gen) -> Sample:
    if not g.led.tags:
        raise Skip
    t = g.tag()
    bql = select(["account", "sum(position)"], where=[f"{q(t)} IN tags"], group="account", order=["account"])
    return Sample("tag_totals", g.ask(f"Break down transactions tagged #{t} by account", f"Per-account totals for the tag {t}"), bql, "Groups postings of tagged transactions by account.")


@intent(3)
def txns_by_flag(g: Gen) -> Sample:
    rng = g.rng
    if rng.random() < 0.6:
        bql = select(["date", "payee", "narration"], frm="#transactions", where=["flag = '!'"], order=["date"])
        question = g.ask("Which transactions are still pending?", "Show transactions flagged with !", "List all transactions marked as pending", "Find transactions that need review (flag !)")
        expl = "Pending transactions carry the '!' flag; cleared ones use '*'."
    else:
        bql = select(["date", "payee", "narration", "account", "position"], where=["flag = '!'"], order=["date"])
        question = g.ask("Show all postings belonging to pending transactions", "Postings of transactions with the ! flag")
        expl = "The flag column of postings is the flag of the parent transaction."
    return Sample("txns_by_flag", question, bql, expl)


@intent(3)
def txns_by_meta(g: Gen) -> Sample:
    rng = g.rng
    if not g.led.txn_meta_keys and not g.led.posting_meta_keys:
        raise Skip
    if g.led.txn_meta_keys and (not g.led.posting_meta_keys or rng.random() < 0.6):
        k = rng.choice(g.led.txn_meta_keys)
        bql = select(["date", "payee", "narration", f"entry_meta({q(k)}) AS {k.replace('-', '_')}"], frm="#transactions", where=[f"entry_meta({q(k)}) IS NOT NULL"], order=["date"]) if False else \
            select(["date", "narration", f"entry_meta({q(k)}) AS {k.replace('-', '_')}"], where=[f"entry_meta({q(k)}) IS NOT NULL"], order=["date"])
        question = g.ask(f"Show the {k} metadata of every transaction that has one", f"List transactions with a '{k}' metadata field and its value", f"Which entries have {k} set?")
        expl = "entry_meta() reads a metadata key from the transaction; it is NULL when the key is missing."
    else:
        k = rng.choice(g.led.posting_meta_keys)
        bql = select(["date", "account", "narration", f"meta({q(k)}) AS {k.replace('-', '_')}"], where=[f"meta({q(k)}) IS NOT NULL"], order=["date"])
        question = g.ask(f"Show postings that have the posting-level metadata '{k}'", f"List postings with {k} metadata and its value")
        expl = "meta() reads a metadata key from the posting itself."
    return Sample("txns_by_meta", question, bql, expl)


@intent(3)
def counts(g: Gen) -> Sample:
    rng = g.rng
    variant = rng.choice(["year", "months", "payees", "accounts", "total", "postings_year"])
    if variant == "year":
        y = rng.choice(g._years)
        bql = select("count(*)", frm="#transactions", where=[rng.choice([f"year(date) = {y}", f"date >= {y}-01-01 AND date < {y + 1}-01-01"])])
        question = g.ask(f"How many transactions were there in {y}?", f"Number of transactions in {y}", f"Count transactions for {y}")
        expl = "Counts rows in the transactions table (the transactions table has no year column, so year(date) is used)."
    elif variant == "months":
        bql = select(["year(date) AS year", "month(date) AS month", "count(*) AS n"], frm="#transactions", group="year, month", order=["year", "month"])
        question = g.ask("How many transactions per month?", "Count transactions by month", "Transaction counts per month")
        expl = "Groups the transactions table by year and month using date functions."
    elif variant == "payees":
        n = rng.choice([5, 10])
        bql = select(["payee", "count(*) AS n"], frm="#transactions", where=["payee IS NOT NULL"], group="payee", order=["n DESC"], limit=n)
        question = g.ask(f"Which {n} payees appear most often?", f"Top {n} payees by number of transactions", f"Who do I transact with most frequently? (top {n})")
        expl = "Counts transactions per payee, skipping empty payees."
    elif variant == "accounts":
        n = rng.choice([5, 10])
        bql = select(["account", "count(*) AS postings"], group="account", order=["postings DESC"], limit=n)
        question = g.ask(f"Which {n} accounts have the most postings?", f"Top {n} busiest accounts by number of postings")
        expl = "count(*) counts postings per account."
    elif variant == "postings_year":
        bql = select(["year", "count(*)"], group="year", order=["year"])
        question = g.ask("How many postings are there per year?", "Number of postings by year")
        expl = "Groups postings by year and counts them."
    else:
        bql = select("count(*)", frm="#transactions")
        question = g.ask("How many transactions are in the ledger?", "Total number of transactions", "Count all transactions")
        expl = "Counts every row of the transactions table."
    return Sample("counts", question, bql, expl)


@intent(3)
def distinct_lists(g: Gen) -> Sample:
    rng = g.rng
    variant = rng.choice(["payees", "currencies", "cost_currencies", "accounts", "payees_expense", "narrations"])
    if variant == "payees":
        bql = select("payee", distinct=True, where=["payee IS NOT NULL"], order=["payee"])
        question = g.ask("List all distinct payees", "What payees do I have in the ledger?", "Give me the unique payees, sorted")
        expl = "DISTINCT removes duplicates; NULL payees are skipped."
    elif variant == "currencies":
        bql = select("currency", distinct=True, order=["currency"])
        question = g.ask("Which currencies are used in postings?", "List all currencies appearing in the ledger", "Show distinct currencies")
        expl = "Distinct currency values across postings."
    elif variant == "cost_currencies":
        bql = select(["currency", "cost_currency"], distinct=True, where=["cost_currency IS NOT NULL"], order=["1", "2"])
        question = g.ask("Which commodities are held at cost and in what cost currency?", "List distinct pairs of commodity and cost currency", "Show the commodities that have a cost basis")
        expl = "Only postings with a cost have a non-NULL cost_currency."
    elif variant == "accounts":
        root = rng.choice(["Expenses", "Income", "Assets", "Liabilities"])
        bql = select("account", distinct=True, where=[f"account ~ '^{root}'"], order=["account"])
        question = g.ask(f"List all {root} accounts that have postings", f"Which {root} accounts are actually used?")
        expl = "Distinct accounts from postings, so accounts without any posting are not shown."
    elif variant == "payees_expense":
        sel = g.selector("Expenses", allow_multi=False)
        bql = select("payee", distinct=True, where=[sel.cond, "payee IS NOT NULL"], order=["payee"])
        question = g.ask(f"Which payees did I pay for {sel.nl}?", f"List the distinct payees for {sel.nl}")
        expl = "Distinct payees on postings of the selected accounts."
    else:
        t = rng.choice(g.led.txns)
        bql = select("narration", distinct=True, frm="#transactions", where=["payee IS NOT NULL", f"payee = {q(t.payee)}" if t.payee else "flag = '*'"], order=["narration"]) if t.payee else None
        if not bql:
            raise Skip
        question = g.ask(f"What different descriptions have I used for {t.payee}?", f"Distinct narrations for payee {t.payee}")
        expl = "Distinct narration strings of the payee's transactions."
    return Sample("distinct_lists", question, bql, expl)


# ---------------------------------------------------------------------------
# D. Other tables

@intent(5)
def accounts_table(g: Gen) -> Sample:
    rng = g.rng
    variant = rng.choice(["open", "closed", "opened_year", "list", "open_dates", "opened_by_root", "account_meta", "meta_subscript"])
    if variant == "open":
        bql = select(["account", "open.date AS opened"], frm="#accounts", where=["close IS NULL"], order=["account"])
        question = g.ask("List all currently open accounts with their opening dates", "Which accounts are still open?", "Show open accounts and when they were opened")
        expl = "The #accounts table has open and close directives; close is NULL for accounts still open."
    elif variant == "closed":
        bql = select(["account", "close.date AS closed"], frm="#accounts", where=["close IS NOT NULL"], order=["closed"])
        question = g.ask("Which accounts have been closed, and when?", "List closed accounts with closing dates")
        expl = "Accounts with a close directive have a non-NULL close column."
    elif variant == "opened_year":
        y = rng.choice(sorted({o.date.year for o, _ in g.led.accounts.values()}))
        bql = select(["account", "open.date"], frm="#accounts", where=[f"year(open.date) = {y}"] if rng.random() < 0.5 else [f"open.date >= {y}-01-01", f"open.date < {y + 1}-01-01"], order=["account"])
        question = g.ask(f"Which accounts were opened in {y}?", f"List accounts opened during {y}")
        expl = "open.date reads the date of the account's open directive."
    elif variant == "list":
        root = rng.choice(["Assets", "Liabilities", "Income", "Expenses"])
        bql = select("account", frm="#accounts", where=[f"account ~ '^{root}'"], order=["account"])
        question = g.ask(f"List all {root} accounts", f"Show every account under {root}", f"What {root} accounts exist?")
        expl = "One row per opened account; regex on the name selects the root."
    elif variant == "open_dates":
        bql = select(["DISTINCT account", "open_date(account) AS opened"], where=["account ~ '^Assets'"], order=["account_sortkey(account)"]) if False else \
            select(["account", "open_date(account) AS opened"], where=["account ~ '^Assets'"], group="account, opened", order=["account"])
        question = g.ask("Show every Assets account that has postings together with its opening date", "Opening date of each used asset account")
        expl = "open_date() looks up the open directive of an account; grouping removes duplicate rows."
    elif variant == "opened_by_root":
        bql = select(["root(account, 1) AS type", "count(*) AS n"], frm="#accounts", group="type", order=["type"])
        question = g.ask("How many accounts are there of each type (Assets, Expenses, ...)?", "Count accounts per root type")
        expl = "root(account, 1) returns the first component of the account name."
    elif variant == "account_meta":
        keys = sorted({k for o, _ in g.led.accounts.values() for k in (o.meta or {}) if k not in ("filename", "lineno")})
        if not keys:
            raise Skip
        k = keys[0]
        bql = select(["account", f"open_meta(account, {q(k)}) AS {k}"], frm="#accounts", where=[f"open_meta(account, {q(k)}) IS NOT NULL"], order=["account"])
        question = g.ask(f"Show the '{k}' metadata of each account that has it", f"Which accounts define {k} in their open directive?")
        expl = "open_meta(account, key) reads metadata from the account's open directive."
    else:
        keys = sorted({k for o, _ in g.led.accounts.values() for k in (o.meta or {}) if k not in ("filename", "lineno")})
        if not keys:
            raise Skip
        k = keys[0]
        bql = select(["account", f"open.meta[{q(k)}] AS {k}"], frm="#accounts", order=[f"{k} DESC", "account"])
        question = g.ask(f"List every account with its '{k}' metadata, largest first, and unset ones last", f"Sort accounts by their {k} metadata value, descending")
        expl = "open.meta['key'] subscripts the metadata dict of the account's open directive directly (an alternative to open_meta()); NULLs sort last in descending order."
    return Sample("accounts_table", question, bql, expl)


@intent(4)
def prices_table(g: Gen) -> Sample:
    rng = g.rng
    if not g.led.commodities:
        raise Skip
    cur = rng.choice([c for c in g.led.commodities if c != g.base] or [g.base])
    variant = rng.choice(["latest", "history", "getprice", "getprice_date", "range", "all"])
    if variant == "latest":
        bql = select(["date", "currency", "amount"], frm="#prices", where=[f"currency = {q(cur)}"], order=["date DESC"], limit=1)
        question = g.ask(f"What is the latest price of {cur}?", f"Most recent price entry for {cur}", f"When was {cur} last priced and at what value?")
        expl = "The #prices table holds price directives; sort by date and keep the latest."
    elif variant == "history":
        n = rng.choice([5, 6, 12])
        bql = select(["date", "amount"], frm="#prices", where=[f"currency = {q(cur)}"], order=["date DESC"], limit=n)
        question = g.ask(f"Show the last {n} recorded prices of {cur}", f"Price history of {cur}, newest {n} entries")
        expl = "Filters price directives by commodity and sorts newest first."
    elif variant == "getprice":
        bql = select(f"getprice({q(cur)}, {q(g.base)})", frm="#")
        question = g.ask(f"What is the current price of {cur} in {g.base}?", f"getprice for {cur} in {g.base}", f"How much is one {cur} worth in {g.base}?")
        expl = "getprice(base, quote) returns the most recent price; FROM # gives a single-row table with no columns."
    elif variant == "getprice_date":
        d = g.date_in_span(30)
        bql = select(f"getprice({q(cur)}, {q(g.base)}, {d})", frm="#")
        question = g.ask(f"What was the price of {cur} in {g.base} on {d}?", f"Price of {cur} in {g.base} as of {d}")
        expl = "The optional third argument makes getprice use the latest price on or before that date."
    elif variant == "range":
        a = g.date_in_span(30)
        b = min(a + dt.timedelta(days=rng.randint(60, 200)), g.last)
        bql = select(["date", "amount"], frm="#prices", where=[f"currency = {q(cur)}", f"date >= {a}", f"date <= {b}"], order=["date"])
        question = g.ask(f"Show {cur} prices between {a} and {b}", f"Price directives for {cur} from {a} to {b}")
        expl = "Range filter on the price directive dates."
    else:
        bql = select(["currency", "max(date) AS last_price_date"], frm="#prices", group="currency", order=["currency"])
        question = g.ask("For each priced commodity, when was its last price recorded?", "Latest price date per commodity")
        expl = "Groups price directives by commodity and takes the max date."
    return Sample("prices_table", question, bql, expl)


@intent(3)
def balance_assertions(g: Gen) -> Sample:
    rng = g.rng
    if not any(type(e).__name__ == "Balance" for e in g.led.entries):
        raise Skip
    variant = rng.choice(["all", "failed", "account", "count"])
    if variant == "all":
        bql = select(["date", "account", "amount"], frm="#balances", order=["date"])
        question = g.ask("List all balance assertions", "Show the balance directives with date, account and amount")
        expl = "The #balances table contains balance directives."
    elif variant == "failed":
        bql = select(["date", "account", "amount", "discrepancy"], frm="#balances", where=["discrepancy IS NOT NULL"], order=["date"])
        question = g.ask("Which balance assertions failed?", "Show balance checks with a discrepancy", "List failing balance directives and how far off they were")
        expl = "discrepancy holds the difference amount and is NULL when the assertion passed."
    elif variant == "account":
        a = rng.choice(g.accounts_with_balance())
        bql = select(["date", "amount"], frm="#balances", where=[f"account = {q(a)}"], order=["date DESC"])
        question = g.ask(f"Show the balance assertions for {g.acct_ref(a)}", f"What balance checks exist on {g.acct_ref(a)}?")
        expl = "Filters balance directives by account."
    else:
        bql = select(["account", "count(*)"], frm="#balances", group="account", order=["account"])
        question = g.ask("How many balance assertions are there per account?", "Count balance directives by account")
        expl = "Groups the balances table by account."
    return Sample("balance_assertions", question, bql, expl, allow_empty=variant == "failed")


@intent(3)
def stale_accounts(g: Gen) -> Sample:
    """Which open accounts need a fresh balance check. From a beancount@googlegroups.com thread: NOT close_date(account)
    on #balances is how to exclude closed accounts from a "needs updating" report."""
    rng = g.rng
    if not any(type(e).__name__ == "Balance" for e in g.led.entries):
        raise Skip
    variant = rng.choice(["last_checked", "oldest", "never"])
    if variant == "last_checked":
        bql = select(["account", "max(date) AS last_checked"], frm="#balances", where=["NOT close_date(account)"], group="account", order=["last_checked"])
        question = g.ask("For each still-open account, when was it last balance-checked?", "Show the most recent balance assertion date for every open account, oldest first")
        expl = "NOT close_date(account) excludes accounts that have since been closed from the balances table, grouped to the latest check per account."
    elif variant == "oldest":
        n = rng.choice([3, 5, 10])
        bql = select(["account", "max(date) AS last_checked"], frm="#balances", where=["NOT close_date(account)"], group="account", order=["last_checked"], limit=n)
        question = g.ask(f"Which {n} open accounts have gone the longest without a balance check?", f"Show the {n} most overdue accounts for a balance assertion")
        expl = "Sorts open accounts by their most recent balance check, oldest (most overdue) first."
    else:
        bql = select("account", frm="#accounts", where=["account ~ '^(Assets|Liabilities)'", "close IS NULL", "account NOT IN (SELECT account FROM #balances)"], order=["account"])
        question = g.ask("Which open asset or liability accounts have never had a balance assertion?", "List open Assets/Liabilities accounts with no balance check at all")
        expl = "account NOT IN (SELECT account FROM #balances) finds accounts that never appear in the balances table."
    return Sample("stale_accounts", question, bql, expl, allow_empty=True)


def _accounts_with_balance(self: Gen) -> list[str]:
    return sorted({e.account for e in self.led.entries if type(e).__name__ == "Balance"})


Gen.accounts_with_balance = _accounts_with_balance  # type: ignore[attr-defined]


@intent(3)
def other_directive_tables(g: Gen) -> Sample:
    rng = g.rng
    kinds = {type(e).__name__ for e in g.led.entries}
    variant = rng.choice(["events", "notes", "documents", "commodities", "entry_types", "commodity_meta", "list_events_type"])
    if variant == "events" and "Event" in kinds:
        bql = select(["date", "type", "description"], frm="#events", order=["date"])
        question = g.ask("List all events", "Show the event directives with date, type and description", "What events are recorded in the ledger?")
        expl = "The #events table contains event directives."
    elif variant == "list_events_type" and "Event" in kinds:
        t = rng.choice([e.type for e in g.led.entries if type(e).__name__ == "Event"])
        bql = select(["date", "description"], frm="#events", where=[f"type = {q(t)}"], order=["date"])
        question = g.ask(f"Show the history of the '{t}' event", f"How did my {t} change over time?")
        expl = "Filters events by type."
    elif variant == "notes" and "Note" in kinds:
        bql = select(["date", "account", "comment"], frm="#notes", order=["date"])
        question = g.ask("Show all notes attached to accounts", "List the note directives")
        expl = "The #notes table contains note directives."
    elif variant == "documents" and "Document" in kinds:
        bql = select(["date", "account", "filename"], frm="#documents", order=["date"])
        question = g.ask("List all documents linked to accounts", "Show document directives with their file names")
        expl = "The #documents table contains document directives."
    elif variant == "commodities":
        bql = select("name", frm="#commodities", order=["name"]) if False else select(["date", "name"], frm="#commodities", order=["name"])
        question = g.ask("Which commodities are declared in the ledger?", "List the commodity directives")
        expl = "The #commodities table holds commodity directives; its name column is the currency code."
    elif variant == "commodity_meta":
        keys = sorted({k for e in g.led.entries if type(e).__name__ == "Commodity" for k in e.meta if k not in ("filename", "lineno")})
        if not keys:
            raise Skip
        k = keys[0]
        bql = select(["name", f"commodity_meta(name, {q(k)}) AS {k.replace('-', '_')}"], frm="#commodities", where=[f"commodity_meta(name, {q(k)}) IS NOT NULL"], order=["name"])
        question = g.ask(f"What is the '{k}' metadata of each commodity that defines it?", f"Show commodities with their {k}")
        expl = "commodity_meta(currency, key) reads metadata from the commodity directive."
    else:
        bql = select(["type", "count(*) AS n"], frm="#entries", group="type", order=["n DESC"])
        question = g.ask("How many directives of each type does the ledger contain?", "Count entries by directive type")
        expl = "The #entries table has one row per directive of any type; its type column is the lower-case directive name."
    return Sample("other_directive_tables", question, bql, expl)


# ---------------------------------------------------------------------------
# E. Investments

@intent(5)
def holdings(g: Gen) -> Sample:
    rng = g.rng
    invest = [a for a in g.used if any(p.cost for t in g.led.txns for p in t.postings if p.account == a and p.cost)]
    if not invest:
        raise Skip
    variant = rng.choice(["units", "cost", "market", "all", "one", "by_currency", "lots"])
    if variant == "units":
        bql = select(["account", "units(sum(position)) AS units"], where=["cost_currency IS NOT NULL"], group="account", order=["account"])
        question = g.ask("How many units do I hold in each investment account?", "Show the number of shares per investment account")
        expl = "Postings with a cost basis are investments; units() drops the cost so only quantities remain."
    elif variant == "cost":
        bql = select(["account", "cost(sum(position)) AS cost_basis"], where=["cost_currency IS NOT NULL"], group="account", order=["account"])
        question = g.ask("What is the cost basis of each investment account?", "How much did I pay for my holdings, per account?")
        expl = "cost() converts positions to their cost basis."
    elif variant == "market":
        bql = select(["account", "value(sum(position)) AS market_value"], where=["cost_currency IS NOT NULL"], group="account", order=["account"])
        question = g.ask("What is the current market value of my investments per account?", "Value my holdings at the latest prices")
        expl = "value() uses the latest price directives to compute market value."
    elif variant == "all":
        bql = select(["account", "units(sum(position)) AS units", "cost(sum(position)) AS cost", "value(sum(position)) AS value"], where=["cost_currency IS NOT NULL"], group="account", order=["account"])
        question = g.ask("Show units, cost basis and market value for each holding", "Give me a portfolio overview: units, cost and current value per account")
        expl = "Three views of the same inventory: units(), cost() and value()."
    elif variant == "one":
        a = rng.choice(invest)
        bql = select(["units(sum(position))", "cost(sum(position))", "value(sum(position))"], where=[f"account = {q(a)}"])
        question = g.ask(f"How many units, what cost and what market value do I have in {a}?", f"Position summary of {g.acct_ref(a)}: units, cost, and value")
        expl = "Summarises one holding account."
    elif variant == "by_currency":
        bql = select(["currency", "sum(number) AS units"], where=["cost_currency IS NOT NULL"], group="currency", order=["currency"])
        question = g.ask("Total units held per commodity", "How much of each commodity do I own?")
        expl = "Groups the postings that have a cost by their commodity."
    else:
        a = rng.choice(invest)
        bql = select(["date", "position", "cost_number", "cost_date"], where=[f"account = {q(a)}", "cost_currency IS NOT NULL"], order=["date"])
        question = g.ask(f"List all lots of {a} with their purchase dates and cost per unit", f"Show every lot in {g.acct_ref(a)} with its cost")
        expl = "Each posting with a cost is a lot; cost_number and cost_date describe it."
    return Sample("holdings", question, bql, expl)


@intent(2)
def gains_dividends(g: Gen) -> Sample:
    rng = g.rng
    inc = [a for a in g.used if a.startswith("Income:Dividends") or a.startswith("Income:Capital")]
    if not inc:
        raise Skip
    variant = rng.choice(["div", "gains"])
    if variant == "div":
        accs = [a for a in inc if "Dividends" in a]
        if not accs:
            raise Skip
        y = rng.choice(g._years)
        bql = select("neg(sum(position))", where=["account ~ 'Dividends'", f"year = {y}"])
        question = g.ask(f"How much dividend income did I receive in {y}?", f"Total dividends {y}")
        expl = "Income is negative in Beancount, so neg() flips the sign."
    else:
        bql = select(["year", "neg(sum(position)) AS realized_gains"], where=["account ~ 'Gains'"], group="year", order=["year"])
        question = g.ask("What were my realized capital gains per year?", "Show realized gains by year")
        expl = "Realized gains post to an Income gains account such as Income:Capital-Gains (negative = gain)."
    return Sample("gains_dividends", question, bql, expl)


# ---------------------------------------------------------------------------
# F. Analytical patterns

@intent(4)
def average_per_month(g: Gen) -> Sample:
    rng = g.rng
    sel = g.selector("Expenses", allow_multi=False)
    fn = rng.choice(["avg", "max", "min"])
    period = g.period(["year", "range", "since"]) if rng.random() < 0.5 else None
    base_where = [sel.cond, f"currency = '{g.base}'"]
    frm, where = g.use_from(period.form(rng), base_where) if period else (None, base_where)
    inner = select(["year", "month", "sum(number) AS total"], frm=frm, where=where, group="year, month", multiline=False)
    per = (f" {period.phrase(rng)}" if period else "") + f" in {g.base}"
    if fn == "avg":
        bql = select("sum(total) / count(total) AS monthly_average", frm=f"({inner})", multiline=len(inner) > 70)
        question = g.ask(f"What is my average monthly spending on {sel.nl}{per}?", f"Average per month for {sel.nl}{per}", f"On average, how much do I spend on {sel.nl} each month{per}?")
        expl = "BQL has no avg() function: sum the monthly totals from a subquery and divide by their count."
    elif fn == "max":
        bql = select("max(total) AS most_expensive_month", frm=f"({inner})", multiline=len(inner) > 70)
        question = g.ask(f"What was the highest monthly total for {sel.nl}{per}?", f"Most I spent in a single month on {sel.nl}{per}")
        expl = "Aggregate the per-month totals from a subquery with max()."
    else:
        bql = select("min(total) AS cheapest_month", frm=f"({inner})", multiline=len(inner) > 70)
        question = g.ask(f"What was the lowest monthly total for {sel.nl}{per}?", f"Least I spent in one month on {sel.nl}{per}")
        expl = "Aggregate the per-month totals from a subquery with min()."
    return Sample("average_per_month", question, bql, expl)


@intent(4)
def busiest_month(g: Gen) -> Sample:
    rng = g.rng
    sel = g.selector("Expenses", allow_multi=False)
    n = rng.choice([1, 3, 5])
    bql = select(["year", "month", "sum(position) AS total"], where=[sel.cond], group="year, month", order=["sum(number) DESC"], limit=n)
    question = g.ask(f"In which {'month' if n == 1 else str(n) + ' months'} did I spend the most on {sel.nl}?", f"Top {n} months by spending on {sel.nl}")
    return Sample("busiest_month", question, bql, "Groups by month and orders by the summed number descending.")


@intent(4)
def pivot_report(g: Gen) -> Sample:
    rng = g.rng
    variant = rng.choice(["acct_year", "acct_month", "root_year", "type_year"])
    if variant == "acct_year":
        sel = g.selector("Expenses", allow_multi=False)
        if len(sel.accounts) < 2:
            raise Skip
        bql = select(["account", "year", "sum(position)"], where=[sel.cond], group="account, year", pivot="account, year")
        question = g.ask(f"Show spending on {sel.nl} as a table with one row per account and one column per year", f"Pivot {sel.nl} spending: accounts down, years across")
        expl = "PIVOT BY takes two columns from the target list: the first becomes rows, the second becomes columns; the second must be a GROUP BY column."
    elif variant == "acct_month":
        y = rng.choice(g._years)
        sel = g.selector("Expenses", allow_multi=False)
        if len(sel.accounts) < 2:
            raise Skip
        bql = select(["account", "month", "sum(number)"], where=[sel.cond, f"year = {y}"], group="account, month", pivot="account, month")
        question = g.ask(f"Give me a pivot of {sel.nl} by month for {y} with accounts as rows and months as columns", f"Monthly matrix of {sel.nl} in {y}")
        expl = "Pivots the grouped result: one row per account, one column per month."
    elif variant == "root_year":
        bql = select(["root(account, 2) AS category", "year", "sum(number) AS total"], where=["account ~ '^Expenses'"], group="category, year", pivot="category, year")
        question = g.ask("Show expenses by category and year as a pivot table", "Yearly expense comparison per category, categories as rows")
        expl = "root(account, 2) gives categories like Expenses:Food; PIVOT BY spreads years into columns."
    else:
        bql = select(["year", "root(account, 1) AS type", "sum(number) AS total"], where=["account ~ '^(Income|Expenses)'"], group="year, type", pivot="year, type")
        question = g.ask("Compare income and expenses per year in a pivot table", "Pivot of Income vs Expenses by year")
        expl = "Groups by year and account type, then pivots the type into columns."
    return Sample("pivot_report", question, bql, expl)


@intent(4)
def income_vs_expense(g: Gen) -> Sample:
    rng = g.rng
    variant = rng.choice(["monthly", "net", "year", "cost"])
    if variant == "monthly":
        y = rng.choice(g._years)
        bql = select(["month", "root(account, 1) AS type", "sum(number) AS total"], where=["account ~ '^(Income|Expenses)'", f"year = {y}"], group="month, type", order=["month", "type"])
        question = g.ask(f"Show income and expenses for each month of {y}", f"Monthly income versus expenses in {y}")
        expl = "Groups both root types by month; income shows up negative."
    elif variant == "net":
        bql = select(["year", "-sum(number) AS net_saved"], where=["account ~ '^(Income|Expenses)'", f"currency = '{g.base}'"], group="year", order=["year"])
        question = g.ask(f"How much did I save each year (income minus expenses) in {g.base}?", f"Yearly savings in {g.base}: income less expenses")
        expl = "Because income is negative and expenses positive, the negated sum of both is what was saved."
    elif variant == "year":
        y = rng.choice(g._years)
        bql = select(["root(account, 1) AS type", f"convert(sum(position), '{g.base}')"], where=["account ~ '^(Income|Expenses)'", f"year = {y}"], group="type", order=["type"])
        question = g.ask(f"Give me total income and total expenses for {y} in {g.base}", f"Income statement totals for {y} in {g.base}")
        expl = "One row per root type, converted to the reporting currency."
    else:
        d1 = dt.date(rng.choice(g._years), 1, 1)
        d2 = dt.date(d1.year + 1, 1, 1)
        bql = select(["account", "cost(sum(position))"], frm=f"OPEN ON {d1} CLOSE ON {d2}", where=["account ~ '^(Income|Expenses)'"], group="account, account_sortkey(account)", order=["account_sortkey(account)"])
        question = g.ask(f"Produce an income statement for {d1.year} using OPEN and CLOSE", f"Income statement for the year {d1.year} (income and expense accounts, at cost)")
        expl = "OPEN ON/CLOSE ON summarize the period; account_sortkey orders accounts by type."
    return Sample("income_vs_expense", question, bql, expl)


@intent(3)
def having_filter(g: Gen) -> Sample:
    rng = g.rng
    variant = rng.choice(["acct", "payee", "count"])
    if variant == "acct":
        thr = rng.choice([100, 250, 500, 1000, 2000])
        bql = select(["account", "sum(number) AS total"], where=["account ~ '^Expenses'", f"currency = '{g.base}'"], group="account", having=f"sum(number) > {thr}", order=["total DESC"])
        question = g.ask(f"Which expense accounts have a total above {thr} {g.base}?", f"Show expense accounts where the sum exceeds {thr} {g.base}", f"List expense accounts with more than {thr} {g.base} spent in total")
        expl = "HAVING filters groups after aggregation and must contain an aggregate expression."
    elif variant == "payee":
        thr = rng.choice([2, 3, 5, 10])
        bql = select(["payee", "count(*) AS n"], frm="#transactions", where=["payee IS NOT NULL"], group="payee", having=f"count(*) >= {thr}", order=["n DESC"])
        question = g.ask(f"Which payees appear in at least {thr} transactions?", f"Payees with {thr} or more transactions")
        expl = "HAVING with count(*) keeps only payees that occur often enough."
    else:
        thr = rng.choice([3, 5, 8])
        bql = select(["account", "count(*) AS postings"], group="account", having=f"count(*) > {thr}", order=["postings DESC"])
        question = g.ask(f"Which accounts have more than {thr} postings?", f"Show accounts with over {thr} postings")
        expl = "HAVING count(*) > n filters accounts by posting count."
    return Sample("having_filter", question, bql, expl)


@intent(3)
def date_functions(g: Gen) -> Sample:
    rng = g.rng
    variant = rng.choice(["weekday", "dow", "week", "bin", "bin_week_end", "quarter", "dom", "yearmonth", "age"])
    sel = g.selector("Expenses", allow_multi=False)
    if variant == "weekday":
        bql = select(["weekday(date) AS day", "sum(number) AS total"], where=[sel.cond], group="day", order=["total DESC"])
        question = g.ask(f"On which weekday do I spend the most on {sel.nl}?", f"Spending on {sel.nl} by day of the week")
        expl = "weekday() returns a 3-letter day name."
    elif variant == "dow":
        bql = select(["date_part('dow', date) AS dow", "count(*) AS n"], where=[sel.cond], group="dow", order=["dow"])
        question = g.ask(f"How many purchases on {sel.nl} per day of week number (0 = Monday)?", f"Number of postings for {sel.nl} by numeric weekday")
        expl = "date_part('dow', date) gives 0 for Monday through 6 for Sunday."
    elif variant == "week":
        bql = select(["date_trunc('week', date) AS week", "sum(number) AS total"], where=[sel.cond], group="week", order=["week"])
        question = g.ask(f"Show weekly totals for {sel.nl}", f"Spending per week on {sel.nl}")
        expl = "date_trunc('week', date) rounds each date down to the Monday of its week."
    elif variant == "bin":
        origin = g.first
        stride = rng.choice(["1 month", "3 months", "2 weeks"]) if False else rng.choice(["1 month", "3 months"])
        bql = select([f"date_bin('{stride}', date, {origin}) AS bucket", "sum(number) AS total"], where=[sel.cond], group="bucket", order=["bucket"])
        question = g.ask(f"Bucket {sel.nl} spending into {stride} intervals starting from {origin}", f"Total for {sel.nl} per {stride} window aligned at {origin}")
        expl = "date_bin(stride, source, origin) assigns each date to a bucket of the given size aligned with the origin."
    elif variant == "bin_week_end":
        origin = g.first
        bql = select([f"date_bin('7 days', date, {origin}) + interval('6 days') AS week_end", "sum(number) AS total"], where=[sel.cond], group="week_end", order=["week_end"])
        question = g.ask(f"Show weekly totals for {sel.nl}, labeled by the last day of each week", f"Weekly {sel.nl} spending, with each week shown by its end date")
        expl = "date_bin() buckets by the week's start; adding interval('6 days') shifts the label to the week's last day."
    elif variant == "quarter":
        bql = select(["quarter(date) AS quarter", "sum(number) AS total"], where=[sel.cond], group="quarter", order=["quarter"])
        question = g.ask(f"Show {sel.nl} per quarter", f"Quarterly totals for {sel.nl}")
        expl = "quarter() returns strings such as 2024-Q2."
    elif variant == "dom":
        bql = select(["day", "count(*) AS n"], where=[sel.cond], group="day", order=["n DESC"], limit=5)
        question = g.ask(f"On which days of the month do I most often pay for {sel.nl}?", f"Top 5 days of the month for {sel.nl} payments")
        expl = "The day column is the day of month of the transaction."
    elif variant == "yearmonth":
        bql = select(["yearmonth(date) AS month", "count(*) AS n", "sum(number) AS total"], where=[sel.cond], group="month", order=["month"])
        question = g.ask(f"For each month, how many postings and what total for {sel.nl}?", f"Monthly count and sum for {sel.nl}")
        expl = "yearmonth() maps each date to the first day of its month."
    else:
        n = rng.choice([30, 60, 90])
        bql = select(["date", "narration", "position", "date_diff(today(), date) AS days_ago"], where=[sel.cond, f"date_diff(today(), date) <= {n}"], order=["date DESC"])
        question = g.ask(f"Show entries for {sel.nl} from the last {n} days with the age in days", f"Recent {sel.nl} postings (last {n} days) and how many days ago each was")
        expl = "date_diff(a, b) returns the number of days between two dates."
    return Sample("date_functions", question, bql, expl)


@intent(3)
def foreign_currency(g: Gen) -> Sample:
    rng = g.rng
    foreign = [c for c in g.led.currencies if c != g.base and not any(p.cost for t in g.led.txns for p in t.postings if p.units.currency == c)]
    if not foreign:
        raise Skip
    c = rng.choice(foreign)
    variant = rng.choice(["list", "sum", "weight", "price"])
    if variant == "list":
        bql = select(["date", "narration", "account", "position", "weight"], where=[f"currency = {q(c)}"], order=["date"])
        question = g.ask(f"Show all postings in {c} and their value in {g.base}", f"List {c} postings with the converted weight")
        expl = "weight is the amount used to balance the transaction, i.e. the price applied to the units."
    elif variant == "sum":
        bql = select(["account", "sum(number) AS amount"], where=[f"currency = {q(c)}"], group="account", order=["account"])
        question = g.ask(f"How much {c} is in each account?", f"Sum {c} postings per account")
        expl = "Filters on the posting currency and sums the numbers."
    elif variant == "weight":
        bql = select("sum(weight)", where=[f"currency = {q(c)}", "account ~ '^Expenses'"])
        question = g.ask(f"What did my {c} expenses cost me in {g.base} (using the transaction prices)?", f"Total {g.base} value of my {c} spending based on the prices at purchase")
        expl = "sum(weight) adds the price-converted amounts."
    else:
        bql = select(["date", "narration", "position", "price"], where=[f"currency = {q(c)}", "price IS NOT NULL"], order=["date"])
        question = g.ask(f"List the {c} postings that carry an explicit price", f"Show {c} postings with their conversion price")
        expl = "The price column is the @ price attached to the posting (NULL when absent)."
    return Sample("foreign_currency", question, bql, expl, allow_empty=False)


@intent(3)
def subscriptions(g: Gen) -> Sample:
    rng = g.rng
    accs = [a for a in g.used if "Subscription" in a]
    if not accs:
        raise Skip
    sub = "account ~ 'Subscriptions'"
    variant = rng.choice(["by_payee", "monthly", "list"])
    if variant == "by_payee":
        bql = select(["payee", "sum(number) AS total"], where=[sub], group="payee", order=["total DESC"])
        question = g.ask("How much do I spend on each subscription in total?", "Total per subscription payee")
    elif variant == "monthly":
        bql = select(["payee", "max(number) AS monthly_cost"], where=[sub], group="payee", order=["payee"])
        question = g.ask("What is the monthly price of each of my subscriptions?", "List my subscriptions and their monthly cost")
    else:
        bql = select("payee", where=[sub], group="payee", order=["payee"])
        question = g.ask("Which subscriptions do I have?", "List the services I pay for as subscriptions")
    return Sample("subscriptions", question, bql, "Groups postings of the subscriptions account by payee.")


@intent(3)
def first_last(g: Gen) -> Sample:
    rng = g.rng
    variant = rng.choice(["span", "acct", "payee_first", "biggest_day"])
    if variant == "span":
        bql = select(["min(date) AS first_date", "max(date) AS last_date"], frm="#transactions")
        question = g.ask("What is the date range covered by the ledger?", "When is the first and the last transaction?", "Earliest and latest transaction dates")
        expl = "min() and max() work on dates."
    elif variant == "acct":
        a = rng.choice(g.used)
        ref, acond, _ = g.acct_pair(a)
        bql = select(["min(date) AS first", "max(date) AS last", "count(*) AS n"], where=[acond])
        question = g.ask(f"When was {ref} first and last used, and how many postings does it have?", f"First and last posting date for {ref}")
        expl = "Aggregates over the postings of one account."
    elif variant == "payee_first":
        p = g.payee()
        bql = select(["min(date) AS first_paid", "max(date) AS last_paid"], frm="#transactions", where=[f"payee = {q(p)}"])
        question = g.ask(f"When did I first and last transact with {p}?", f"First and most recent date for {p}")
        expl = "min/max of the date over that payee's transactions."
    else:
        bql = select(["date", "sum(number) AS total"], where=["account ~ '^Expenses'"], group="date", order=["total DESC"], limit=1)
        question = g.ask("On which single day did I spend the most?", "What was my most expensive day?")
        expl = "Groups expense postings by date and keeps the top day."
    return Sample("first_last", question, bql, expl)


@intent(3)
def multi_account_txns(g: Gen) -> Sample:
    rng = g.rng
    tx = rng.choice([t for t in g.led.txns if len({p.account for p in t.postings}) >= 2])
    a, b = rng.sample(sorted({p.account for p in tx.postings}), 2)
    variant = rng.choice(["both", "one", "other"])
    if variant == "both":
        bql = select(["date", "payee", "narration"], frm="#transactions", where=[f"{q(a)} IN accounts", f"{q(b)} IN accounts"], order=["date"])
        question = g.ask(f"Which transactions involve both {g.acct_ref(a)} and {g.acct_ref(b)}?", f"Find transactions that touch {a} and {b}")
        expl = "The accounts column of a transaction is the set of accounts of its postings."
    elif variant == "one":
        bql = select(["date", "payee", "narration", "accounts"], frm="#transactions", where=[f"{q(a)} IN accounts"], order=["date DESC"], limit=10)
        question = g.ask(f"Show the last 10 transactions that involve {g.acct_ref(a)}, including all accounts they touch", f"Recent transactions touching {g.acct_ref(a)} with the full account set")
        expl = "Uses the set-valued accounts column of #transactions."
    else:
        bql = select(["other_accounts", "sum(number)"] if False else ["date", "account", "other_accounts", "position"], where=[f"account = {q(a)}", f"{q(b)} IN other_accounts"], order=["date"])
        question = g.ask(f"Show postings on {g.acct_ref(a)} whose counter-account is {g.acct_ref(b)}", f"Movements of {g.acct_ref(a)} against {g.acct_ref(b)}")
        expl = "other_accounts is the set of the other accounts in the same transaction."
    return Sample("multi_account_txns", question, bql, expl)


@intent(4)
def amount_filters(g: Gen) -> Sample:
    rng = g.rng
    sel = g.selector("Expenses", allow_multi=False)
    thr = rng.choice([50, 100, 200, 500, 1000])
    variant = rng.choice(["gt", "between", "abs", "lt"])
    cols = ["date", "payee", "narration", "position"]
    if variant == "gt":
        bql = select(cols, where=[sel.cond, f"number > {thr}", f"currency = '{g.base}'"], order=["number DESC"])
        question = g.ask(f"Show purchases on {sel.nl} of more than {thr} {g.base}", f"List {sel.nl} postings over {thr} {g.base}")
        expl = "Combines an account filter with a numeric threshold."
    elif variant == "lt":
        bql = select(cols, where=[sel.cond, f"number < {thr // 10}", f"currency = '{g.base}'"], order=["number"])
        question = g.ask(f"Which entries on {sel.nl} were under {thr // 10} {g.base}?", f"Small {sel.nl} postings below {thr // 10} {g.base}")
        expl = "Numeric filter on the posting number."
    elif variant == "between":
        hi = thr * 2
        bql = select(cols, where=[sel.cond, f"number BETWEEN {thr} AND {hi}", f"currency = '{g.base}'"], order=["number DESC"])
        question = g.ask(f"Show {sel.nl} amounts between {thr} and {hi} {g.base}", f"Postings on {sel.nl} from {thr} to {hi} {g.base}")
        expl = "BETWEEN is inclusive on both ends."
    else:
        bql = select(cols, where=[f"abs(number) >= {thr}", f"currency = '{g.base}'", "account ~ '^(Assets|Liabilities)'"], order=["date"])
        question = g.ask(f"Show asset and liability movements of at least {thr} {g.base} in either direction", f"Large movements (>= {thr} {g.base}) on assets and liabilities")
        expl = "abs(number) ignores whether money came in or went out."
    return Sample("amount_filters", question, bql, expl)


@intent(3)
def or_conditions(g: Gen) -> Sample:
    rng = g.rng
    p1, p2 = rng.sample(g.led.payees[:30], 2) if len(g.led.payees) > 2 else (None, None)
    if not p1:
        raise Skip
    variant = rng.choice(["payees", "payee_in", "excl"])
    if variant == "payees":
        bql = select(["date", "payee", "narration", "position"], where=[f"payee = {q(p1)} OR payee = {q(p2)}", "account ~ '^Expenses'"], order=["date"])
        question = g.ask(f"Show expenses paid to {p1} or {p2}", f"Expense postings from either {p1} or {p2}")
    elif variant == "payee_in":
        bql = select(["date", "payee", "narration", "position"], where=[f"payee IN ({q(p1)}, {q(p2)})", "account ~ '^Expenses'"], order=["date"])
        question = g.ask(f"List expenses where the payee is one of {p1} and {p2}", f"Expenses for payees {p1}, {p2}")
    else:
        bql = select(["date", "payee", "narration", "position"], where=["account ~ '^Expenses'", f"payee != {q(p1)}", f"payee != {q(p2)}"], order=["date DESC"], limit=20)
        question = g.ask(f"Show the 20 latest expenses excluding {p1} and {p2}", f"Latest 20 expense postings not paid to {p1} or {p2}")
    return Sample("or_conditions", question, bql, "Combines several conditions with AND/OR/IN.")


# ---------------------------------------------------------------------------
# G. Statements and special FROM forms

@intent(4)
def print_statement(g: Gen) -> Sample:
    rng = g.rng
    variant = rng.choice(["payee", "tag", "narration", "account", "date", "link"])
    if variant == "payee":
        p = g.payee()
        bql = f"PRINT FROM payee = {q(p)}"
        question = g.ask(f"Print the raw Beancount entries for payee {p}", f"Show {p} transactions in Beancount syntax")
    elif variant == "tag":
        t = g.tag()
        bql = f"PRINT FROM {q(t)} IN tags"
        question = g.ask(f"Print all transactions tagged #{t} as Beancount text", f"Output the ledger entries with tag {t}")
    elif variant == "narration":
        t = rng.choice(g.led.txns)
        w = t.narration.split()[0].strip(".,")
        bql = f"PRINT FROM narration ~ {q(w)}"
        question = g.ask(f"Print the entries whose narration matches '{w}'", f"Show the raw entries mentioning {w} in the narration")
    elif variant == "account":
        a = rng.choice(g.used)
        bql = f"PRINT FROM {q(a)} IN accounts" + (f" AND year = {rng.choice(g._years)}" if rng.random() < 0.4 else "")
        question = g.ask(f"Print all directives that involve {g.acct_ref(a)}" + (f" in {bql.split('= ')[-1]}" if "year =" in bql else ""), f"Dump the ledger entries touching {a} in Beancount format")
    elif variant == "date":
        m = g.ym()
        p = p_month(*m)
        form = p.form(rng, "entries", allow_from=False)
        bql = f"PRINT FROM {cond(form['where'])}"
        question = g.ask(f"Print every transaction {p.phrase(rng)} in Beancount format", f"Print the ledger entries {p.phrase(rng)}")
    else:
        l = g.link()
        bql = f"PRINT FROM {q(l)} IN links"
        question = g.ask(f"Print the entries linked with ^{l}", f"Output entries that have the link {l}")
    return Sample("print_statement", question, bql, "PRINT emits the matching directives as Beancount text; the FROM clause filters entries.")


@intent(3)
def open_close_clear(g: Gen) -> Sample:
    rng = g.rng
    y = rng.choice(g._years)
    variant = rng.choice(["balance_sheet", "period_txns", "closing", "income_clear"])
    if variant == "balance_sheet":
        bql = f"BALANCES AT cost FROM CLOSE ON {y + 1}-01-01 WHERE account ~ '^(Assets|Liabilities)'"
        question = g.ask(f"Show the balance sheet at cost at the end of {y}", f"Assets and liabilities at cost as of the end of {y}")
        expl = "CLOSE ON date truncates the ledger before that date, so it is the balance at the end of the previous day."
    elif variant == "period_txns":
        bql = select(["account", "sum(position)"], frm=f"OPEN ON {y}-01-01 CLOSE ON {y + 1}-01-01", where=["account ~ '^(Income|Expenses)'"], group="account", order=["account"])
        question = g.ask(f"Summarize income and expense accounts for {y} using OPEN ON and CLOSE ON", f"Income and expense totals for the period {y}-01-01 to {y + 1}-01-01")
        expl = "OPEN ON summarizes prior entries into opening balances; CLOSE ON drops later entries."
    elif variant == "closing":
        bql = select(["account", "sum(position)"], frm=f"CLOSE ON {y + 1}-01-01", where=["account ~ '^Assets'"], group="account", order=["account"])
        question = g.ask(f"What were my asset balances at the end of {y}?", f"Asset balances just before {y + 1}-01-01")
        expl = "CLOSE ON drops entries on or after the date."
    else:
        bql = select(["account", "sum(position)"], frm=f"OPEN ON {y}-01-01 CLOSE ON {y + 1}-01-01 CLEAR", where=["account ~ '^Assets'"], group="account", order=["account"])
        question = g.ask(f"Show asset balances for {y} after clearing income and expenses into equity", f"Balance sheet for {y} with income and expense accounts cleared to equity")
        expl = "CLEAR transfers income and expense balances to equity so the result is a proper balance sheet."
    return Sample("open_close_clear", question, bql, expl)


@intent(3)
def journal_statement(g: Gen) -> Sample:
    rng = g.rng
    a = rng.choice(g.used)
    ref, _, apat = g.acct_pair(a)
    variant = rng.choice(["plain", "cost", "units", "period", "regex"])
    if variant == "plain":
        bql = f"JOURNAL {q(apat)}"
        question = g.ask(f"Show the journal for {ref}", f"JOURNAL of {ref}")
    elif variant == "cost":
        bql = f"JOURNAL {q(apat)} AT cost"
        question = g.ask(f"Show the journal of {ref} with amounts at cost", f"Register for {ref} valued at cost")
    elif variant == "units":
        bql = f"JOURNAL {q(apat)} AT units"
        question = g.ask(f"Journal for {ref} showing only units without cost", f"Show {ref} register in units")
    elif variant == "period":
        y = rng.choice(g._years)
        bql = f"JOURNAL {q(apat)} FROM year = {y}"
        question = g.ask(f"Show the journal for {ref} in {y}", f"Register of {ref} for the year {y}")
    else:
        root = a.split(":")[0]
        bql = f"JOURNAL {q('^' + root)}" if False else f"JOURNAL {q(':'.join(a.split(':')[:2]))}"
        question = g.ask(f"Show the journal for all accounts matching {':'.join(a.split(':')[:2])}", f"Combined register of accounts matching {':'.join(a.split(':')[:2])}")
    return Sample("journal_statement", question, bql, "JOURNAL takes an account regular expression and lists postings with a running balance; AT selects the valuation.")


@intent(2)
def utility_scalars(g: Gen) -> Sample:
    rng = g.rng
    variant = rng.choice(["today", "add", "diff", "year", "date_lit"])
    if variant == "today":
        bql = "SELECT today() FROM #"
        question = g.ask("What is today's date according to BQL?", "Select the current date")
    elif variant == "add":
        n = rng.choice([7, 30, 90])
        bql = f"SELECT date_add(today(), {n}) FROM #"
        question = g.ask(f"What date is {n} days from today?", f"Compute today plus {n} days")
    elif variant == "diff":
        d = g.date_in_span(30)
        bql = f"SELECT date_diff(today(), {d}) FROM #"
        question = g.ask(f"How many days ago was {d}?", f"Number of days between {d} and today")
    elif variant == "year":
        bql = "SELECT year(today()), month(today()) FROM #"
        question = g.ask("Give me the current year and month", "What are the current year and month numbers?")
    else:
        d = g.date_in_span(30)
        bql = f"SELECT date({d.year}, {d.month}, {d.day}) FROM #"
        question = g.ask(f"Construct the date {d} from its year, month and day", f"Build a date value for {d}")
    return Sample("utility_scalars", question, bql, "A query over the empty table # returns a single row, which is handy for evaluating expressions.")


@intent(2)
def string_functions(g: Gen) -> Sample:
    rng = g.rng
    variant = rng.choice(["upper", "maxwidth", "grep", "leaf", "root", "parent", "splitcomp", "substr"])
    a = rng.choice(g.used)
    if variant == "upper":
        bql = select(["date", "upper(payee) AS payee"], frm="#transactions", where=["payee IS NOT NULL"], order=["date DESC"], limit=10)
        question = g.ask("Show the 10 latest payees in upper case", "Latest 10 transactions with the payee converted to uppercase")
    elif variant == "maxwidth":
        bql = select(["date", "maxwidth(narration, 20) AS short"], frm="#transactions", order=["date DESC"], limit=10)
        question = g.ask("Show the last 10 narrations truncated to 20 characters", "Recent narrations, shortened to at most 20 characters")
    elif variant == "grep":
        bql = select(["date", "grep('[0-9]+', narration) AS number_in_text"], frm="#transactions", where=["grep('[0-9]+', narration) IS NOT NULL"], order=["date"])
        question = g.ask("Extract the first number that appears in each narration", "Pull out digits from narrations where present")
    elif variant == "leaf":
        bql = select(["leaf(account) AS name", "sum(number) AS total"], where=["account ~ '^Expenses'"], group="name", order=["total DESC"])
        question = g.ask("Total expenses by the last part of the account name", "Group expense totals by leaf account name, biggest first")
    elif variant == "root":
        n = rng.choice([1, 2])
        bql = select([f"root(account, {n}) AS root", "count(*) AS postings"], group="root", order=["root"])
        question = g.ask(f"Count postings by the first {n} component{'s' if n > 1 else ''} of the account name", f"How many postings per {n}-level account prefix?")
    elif variant == "parent":
        bql = select(["parent(account) AS parent", "sum(number) AS total"], where=["account ~ '^Expenses'"], group="parent", order=["total DESC"])
        question = g.ask("Sum expenses by parent account", "Expense totals grouped by the parent of each account")
    elif variant == "splitcomp":
        bql = select(["splitcomp(account, ':', 1) AS second_level", "count(*) AS n"], group="second_level", order=["n DESC"])
        question = g.ask("Count postings by the second component of the account name", "Postings per second-level account segment")
    else:
        bql = select(["date", "substr(narration, 0, 10) AS start"], frm="#transactions", order=["date"], limit=15)
        question = g.ask("Show the first 10 characters of the narration for the first 15 transactions", "First 15 transactions with the narration cut to 10 characters")
    return Sample("string_functions", question, bql, "Uses BQL string and account helper functions.")


@intent(2)
def subquery_in(g: Gen) -> Sample:
    rng = g.rng
    variant = rng.choice(["above_avg", "nested", "distinct_count", "threshold_in"])
    if variant == "above_avg":
        sel = g.selector("Expenses", allow_multi=False)
        inner = select(["year", "month", "sum(number) AS total"], where=[sel.cond, f"currency = '{g.base}'"], group="year, month", multiline=False)
        bql = select(["year", "month", "total"], frm=f"({inner})", where=["total > 100"], order=["total DESC"], limit=5, multiline=True)
        question = g.ask(f"Which months had more than 100 {g.base} in spending on {sel.nl}? Show the top 5 using a subquery", f"Top 5 months above 100 {g.base} for {sel.nl} (use a nested select)")
        expl = "A subquery in FROM lets you filter or re-aggregate grouped results."
    elif variant == "nested":
        inner = select(["account", "sum(number) AS total"], where=["account ~ '^Expenses'", f"currency = '{g.base}'"], group="account", multiline=False)
        bql = select(["count(*) AS accounts", "max(total) AS highest"], frm=f"({inner})", multiline=len(inner) > 60)
        question = g.ask(f"How many expense accounts are there and what is the highest total among them, in {g.base}?", f"Count expense accounts and find the largest account total in {g.base}")
        expl = "The inner query totals each account; the outer query aggregates those totals."
    elif variant == "threshold_in":
        # From a beancount@googlegroups.com thread (Daniele Nicolodi): a subquery on the right of IN can filter one
        # query's rows by an aggregate computed over another. HAVING cannot be used directly in WHERE for this.
        thr = rng.choice([100, 250, 500])
        inner = select("account", where=["account ~ '^Expenses'", f"currency = '{g.base}'"], group="account", having=f"number(only('{g.base}', sum(position))) > {thr}", multiline=False)
        bql = select(["date", "payee", "narration", "account", "position"], where=[f"account IN ({inner})"], order=["date"])
        question = g.ask(f"Show postings on expense accounts whose total spending exceeds {thr} {g.base}", f"List transactions for expense accounts that add up to more than {thr} {g.base} overall")
        expl = "The subquery groups and filters accounts by their total with HAVING, then the outer query keeps only postings whose account is IN that set."
    else:
        bql = select("count(*) AS payees", frm="(SELECT DISTINCT payee)")
        question = g.ask("How many distinct payees are there?", "Count the unique payees")
        expl = "count(DISTINCT ...) is not supported; count the rows of a DISTINCT subquery instead."
    return Sample("subquery_in", question, bql, expl)


@intent(2)
def travel_trips(g: Gen) -> Sample:
    rng = g.rng
    trips = [t for t in g.led.tags if t.startswith("trip-")]
    if not trips:
        raise Skip
    t = rng.choice(trips)
    if rng.random() < 0.5:
        bql = select(["account", "sum(position)"], where=[f"{q(t)} IN tags"], group="account", order=["account"])
        question = g.ask(f"What did the trip tagged {t} cost by account?", f"Break down the {t} trip expenses by account")
    else:
        bql = select(["date", "payee", "narration", "position"], where=[f"{q(t)} IN tags", "account ~ '^Expenses'"], order=["date"])
        question = g.ask(f"List all expenses of the trip {t}", f"Show every expense posting for #{t}")
    return Sample("travel_trips", question, bql, "Trips are tracked with tags, which are sets; use 'tag' IN tags.")


@intent(2)
def cast_and_meta(g: Gen) -> Sample:
    rng = g.rng
    keys = g.led.txn_meta_keys
    if "invoice" in keys:
        bql = select(["date", "payee", "entry_meta('invoice') AS invoice"], where=["entry_meta('invoice') IS NOT NULL"], order=["date"], distinct=True)
        question = g.ask("Show all invoices with their invoice number", "List consulting income entries that have an invoice number")
    elif "receipt" in g.led.posting_meta_keys or "receipt" in keys:
        bql = select(["date", "payee", "any_meta('receipt') AS receipt"], where=["any_meta('receipt') IS NOT NULL"], order=["date"])
        question = g.ask("Which purchases have a receipt attached? Show the receipt path", "List postings or transactions with a receipt")
    else:
        raise Skip
    return Sample("cast_and_meta", question, bql, "Metadata values have the generic object type; any_meta() looks at the posting first, then at the transaction.")
