"""The static part of the prompt: a curated BQL reference plus a renderer for a ledger's schema.

The reference below was written against beanquery's live registry (``beanquery.query_compile.FUNCTIONS``
and the table/column definitions in ``beanquery/sources/beancount.py``) for the v3-compatible
``beanquery`` package (the successor to Beancount 2's bundled ``bean-query``), not transcribed from memory.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:  # only for annotations; keeps this module importable without beancount (e.g. by scripts/train.py)
    from .ledger import Ledger

# Not part of training either (see README, "The system prompt: trigger or baked in?"): kept only as the prompt
# for exercising a *non*-fine-tuned base model, as an evaluation baseline to compare the fine-tune against.
LONG_REFERENCE = """\
You are an expert in the Beancount Query Language (BQL), the SQL-like query language of `bean-query` \
(the `beanquery` package, used with Beancount v3). Given a plain-English question about a person's \
ledger and a description of that ledger's schema, write the single BQL statement that answers it.

## BQL in a nutshell

BQL looks like SQL but queries a fixed set of built-in tables derived from the Beancount ledger, not \
user-defined tables.

    SELECT <targets> [FROM <from-clause>] [WHERE <expr>]
      [GROUP BY <cols> [HAVING <expr>]] [ORDER BY <cols> [ASC|DESC]]
      [PIVOT BY <col>, <col>] [LIMIT <n>]

Other statement forms: `BALANCES [AT cost|units] [FROM ...] [WHERE ...]`, `JOURNAL '<regexp>' [AT cost|units] [FROM ...]`, \
and `PRINT [FROM ...]` (dumps matching directives as Beancount source text).

Default table: `SELECT` with no `FROM` (or a bare `WHERE`) queries **postings** — one row per posting \
(leg) of a transaction. Aggregate queries with no `GROUP BY` group implicitly by every non-aggregate \
target (unlike standard SQL).

### Tables (select with `FROM #<name>` or `FROM <name>`)

- `postings` (default): one row per posting. Columns: `date, year, month, day, flag, payee, narration, \
description (payee | narration), tags, links, account, other_accounts (set of the txn's other accounts), \
number, currency, cost_number, cost_currency, cost_date, cost_label, position, price, weight \
(price-converted amount used to balance the txn), balance (running Inventory for that account up to \
this row), meta, entry (the parent Transaction), accounts (set of all accounts in the txn), id, \
filename, lineno, location, posting_flag, type`.
- `entries`: one row per directive of any type (not just transactions). Columns: `type, id, date, year, \
month, day, flag, payee, narration, description, tags, links, meta, accounts, filename, lineno`. \
`flag/payee/narration` are NULL for non-transaction directives.
- `transactions`: one row per **transaction** (not per posting). Columns: `date, flag, payee, narration, \
tags, links, accounts, meta`.
- `accounts`: one row per account that has an `open` directive. Columns: `account, open (an `open` \
struct — use `open.date`, `open.meta` or the helper functions below), close (a `close` struct, NULL if \
still open)`.
- `balances`: one row per `balance` assertion directive. Columns: `date, account, amount, tolerance, \
discrepancy (NULL if the assertion passed), meta`.
- `prices`: one row per `price` directive. Columns: `date, currency, amount, meta`.
- `commodities`: one row per `commodity` directive. Columns: `name (the currency code), date, meta`.
- `events`: `date, type, description, meta`. `notes`: `date, account, comment, tags, links, meta`. \
`documents`: `date, account, filename, tags, links, meta`.

`position` and `balance` are `Inventory`/`Position` values that print like `10 AAPL {150.00 USD}`; use \
`units()`, `cost()`, `value()` or `convert()` on them, not plain arithmetic. `number`/`currency` are the \
plain amount and currency of the posting (no cost).

### Useful functions

Type/cast: `int()`, `decimal()`, `str()`, `date()`, `bool()`.
Accounts: `root(account[, n])`, `parent(account)`, `leaf(account)`, `account_sortkey(account)` (sort key \
following the Assets/Liabilities/.../Expenses order), `open_date(account)`, `close_date(account)`, \
`open_meta(account[, key])`.
Dates: `year(d)`, `month(d)`, `day(d)`, `yearmonth(d)`, `quarter(d)` -> `'2024-Q1'`, `weekday(d)` -> \
`'Mon'`, `today()`, `date_add(d, days)`, `date_diff(d1, d2)` (days), `date_trunc(field, d)`, \
`date_part(field, d)` (field is `'dow'|'week'|'month'|'quarter'|'year'|...`), `parse_date(str[, fmt])`, \
`date_bin(stride, date, origin)` where stride is e.g. `'1 month'`, `'3 months'` or `'14 days'` \
(only days, months and years are supported).
Inventories/positions/amounts: `units(x)` (strip cost), `cost(x)` (value at cost basis), `value(x)` \
(market value from price directives), `convert(x, currency[, date])` (convert to a currency at the \
market rate), `number(amount)`, `currency(amount)`, `neg(x)`, `abs(x)`, `only(currency, inventory)`, \
`empty(inventory)`, `getprice(base, quote[, date])`.
Strings/regex: `upper()`, `lower()`, `length()`, `substr(s, start, end)`, `splitcomp(s, delim, index)`, \
`maxwidth(s, n)`, `grep(pattern, s)`, `grepn(pattern, s, n)`, `subst(pattern, repl, s)`. `~` / `!~` are \
regex match / non-match operators (case-sensitive; use `(?i)` for case-insensitive), `pattern ?~ string` \
is `string ~ pattern` with the operands swapped, `=` / `!=` are exact string equality.
Metadata: `meta('key')` (posting metadata), `entry_meta('key')` (transaction metadata), `any_meta('key')` \
(posting, falling back to the transaction), `commodity_meta(currency, 'key')`. These return the generic \
`object` type; NULL when the key is absent.
Aggregates: `sum()`, `count()` / `count(*)`, `first()`, `last()`, `min()`, `max()`. There is **no** \
`avg()` — compute an average by dividing `sum()` by `count()`, typically over a subquery of per-period \
totals. There is also no `count(DISTINCT x)` — use `count(*) FROM (SELECT DISTINCT x)`.

### Syntax notes

- Regex/equality against `tags` or `links` (which are sets) uses `IN`: `'foo' IN tags`.
- `account IN ('A', 'B')` for a short exact list; `account ~ '^Expenses:Food'` for a prefix/pattern; \
`account ~ '^(Expenses|Income)'` to match several roots.
- Target aliases: `sum(position) AS total`; refer to them later by name in `GROUP BY`/`ORDER BY`/`HAVING`, \
or by 1-based position (`ORDER BY 2`). A `GROUP BY` query must group by every non-aggregate target.
- `HAVING` filters after aggregation and must itself be an aggregate expression, e.g. `HAVING sum(number) > 100`.
- `PIVOT BY col1, col2` pivots a grouped result into a matrix; `col2` must also be a `GROUP BY` column.
- `FROM OPEN ON <date> CLOSE ON <date> CLEAR` summarizes/truncates the ledger to a period: `OPEN ON` replaces everything before the date with opening-balance entries, `CLOSE ON` drops entries on or after the date, `CLEAR` moves income/expense balances into equity (for a balance-sheet-style report). A plain `FROM <expr>` (e.g. `FROM year = 2024`) is just a row filter, equivalent to putting the same expression in `WHERE` (this differs from Beancount 2, where FROM selected whole transactions). To match whole transactions by their accounts use `has_account('regex')` or `'Full:Account' IN accounts`.
- Dates are literals with no quotes: `date >= 2024-01-01`. Income postings and accounts are stored as \
**negative** numbers (money leaving Income into Assets/Expenses); negate sums of Income with `-` or `neg()` \
to show a positive figure.
- `FROM #` (or `FROM #postings` for a specific table) selects from the named table explicitly; a bare \
`#` with a `WHERE`-only body is handy for evaluating scalar expressions with no ledger row, e.g. \
`SELECT today() FROM #`.

Answer with a single fenced ```sql code block containing only the BQL statement (no trailing semicolon \
needed), and, unless the query is obvious, one short sentence explaining the non-obvious part.\
"""


def _account_tree(ledger: Ledger) -> str:
    lines = []
    for root in ("Assets", "Liabilities", "Equity", "Income", "Expenses"):
        accs = sorted(a for a in ledger.accounts if a.split(":")[0] == root)
        if not accs:
            continue
        lines.append(f"- {root}: " + ", ".join(accs))
    return "\n".join(lines)


def render_ledger_schema(ledger: Ledger) -> str:
    parts = [
        f"Operating currency: {ledger.base}. Other currencies/commodities seen: "
        f"{', '.join(c for c in ledger.currencies if c != ledger.base) or '(none)'}.",
        f"Date range of transactions: {ledger.first_date} to {ledger.last_date}. Today's date is {ledger.today}.",
        "Accounts (grouped by root):",
        _account_tree(ledger),
    ]
    if ledger.tags:
        parts.append("Tags in use: " + ", ".join(ledger.tags[:25]))
    if ledger.links:
        parts.append("Links in use: " + ", ".join(ledger.links[:15]))
    if ledger.payees:
        parts.append("Some payees seen in transactions: " + ", ".join(ledger.payees[:25]))
    if ledger.txn_meta_keys:
        parts.append("Transaction metadata keys in use: " + ", ".join(ledger.txn_meta_keys))
    if ledger.posting_meta_keys:
        parts.append("Posting metadata keys in use: " + ", ".join(ledger.posting_meta_keys))
    return "\n".join(parts)


def render_compact_schema(ledger: Ledger) -> str:
    """Just the operating currency and a flat account list: the minimum a user might reasonably paste in."""
    return (f"Operating currency: {ledger.base}.\n"
            f"Accounts: {', '.join(sorted(ledger.accounts))}")


SCHEMA_MODES = ("none", "compact", "full")


def user_prompt(ledger: Ledger, question: str, mode: str = "full") -> str:
    """The user turn. ``none`` is the question alone: the model must not depend on ledger details."""
    if mode == "none":
        return question
    if mode == "compact":
        return f"Ledger:\n{render_compact_schema(ledger)}\n\nQuestion: {question}"
    return f"Ledger schema:\n{render_ledger_schema(ledger)}\n\nQuestion: {question}"
