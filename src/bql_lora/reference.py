"""Reference-knowledge examples: tables, columns, functions and the core BQL concepts.

The training prompt no longer carries a BQL reference, so the model has to get that knowledge from the weights.
Query examples teach it by use; these examples teach it directly:

* tables, columns and function signatures come from beanquery's live registry (``query_compile.FUNCTIONS``,
  the table definitions), so they cannot drift from the implementation;
* the concepts follow the fundamentals in the official BQL documentation (written for Beancount 2's
  ``bean-query``), rewritten for and checked against the v3 ``beanquery`` where the two differ. Notably,
  ``FROM <expr>`` is a plain row filter in v3, ``PRINT FROM`` runs over directives (no ``account`` column), and
  there is no ``weight()`` or ``raw()`` function;
* every ```sql block in an answer is executed against a real ledger while the dataset is built.
"""

from __future__ import annotations

import random
import re

from beanquery import query_compile, types

from .executor import Executor, QueryError
from .schema import SYSTEM_PROMPT

FENCE = re.compile(r"```sql\n(.*?)\n```", re.S)

TABLE_ORDER = ["postings", "entries", "transactions", "accounts", "balances", "prices", "commodities", "events", "notes", "documents"]

TABLE_DESC = {
    "postings": "the default table (used when there is no FROM clause): one row per posting, i.e. per leg of a transaction. The parent transaction's date, flag, payee, narration, tags and links are repeated on every row",
    "entries": "one row per directive of any type (transactions, open, close, balance, price, note, ...); use the `type` column to tell them apart. Posting-level columns such as `account` and `position` are not available here",
    "transactions": "one row per transaction (not per posting). It has the transaction-level columns and the set of accounts the transaction touches, but no posting-level columns such as `account` or `position`, and no `year`/`month`/`day` columns (use `year(date)`)",
    "accounts": "one row per account that has an `open` directive, with its `open` and `close` directives (`close` is NULL while the account is open)",
    "balances": "one row per `balance` assertion directive; `discrepancy` is NULL when the assertion passed",
    "prices": "one row per `price` directive",
    "commodities": "one row per `commodity` directive; `name` is the currency code",
    "events": "one row per `event` directive",
    "notes": "one row per `note` directive",
    "documents": "one row per `document` directive",
}

SHORT_DESC = {
    "postings": "one row per posting (the default table)",
    "entries": "one row per directive of any type",
    "transactions": "one row per transaction",
    "accounts": "one row per opened account, with its open/close directives",
    "balances": "balance assertions",
    "prices": "price directives",
    "commodities": "commodity directives",
    "events": "event directives",
    "notes": "note directives",
    "documents": "document directives",
}

# The registry docstrings are used when they are accurate; these fix or fill in the few that are not.
COLUMN_DOC_OVERRIDES = {
    "year": "The year of the transaction date.",
    "month": "The month (1-12) of the transaction date.",
    "day": "The day of the month of the transaction date.",
    "cost_date": "The acquisition date of the lot's cost basis (NULL if the posting has no cost).",
    "cost_label": "The label of the lot's cost basis (empty if it has none).",
    "type": "The kind of directive; for the postings table this is always 'transaction'.",
    "meta": "The metadata dictionary of the posting; use meta('key') to read one entry.",
    "entry": "The parent transaction of the posting.",
    "accounts": "The set of all accounts touched by the transaction.",
    "balance": "The running balance (an inventory) of the postings so far; useful with last() to get a period-end balance.",
    "position": "The posting's position: units plus its cost basis, if any. Sum positions to get inventories.",
    "weight": "The amount used to balance the transaction: the units converted by the price or cost.",
    "price": "The price attached to the posting with @ or @@ (NULL if none).",
}

FUNCTION_DOC_OVERRIDES = {
    "int": "Convert the argument to an integer; returns NULL if the conversion fails.",
    "decimal": "Convert the argument to a decimal number; returns NULL if the conversion fails.",
    "str": "Convert the argument to a string.",
    "date": "Convert a 'YYYY-MM-DD' string to a date, or build a date from year, month and day integers; NULL if invalid.",
    "bool": "Convert the argument to a boolean.",
}

FUNCTION_EXAMPLES = {
    "year": "SELECT year(date) AS y, sum(number) WHERE account ~ '^Expenses' GROUP BY y",
    "month": "SELECT month(date) AS m, count(*) GROUP BY m ORDER BY m",
    "day": "SELECT day(date) AS d, count(*) GROUP BY d ORDER BY d",
    "yearmonth": "SELECT yearmonth(date) AS month, sum(number) WHERE account ~ '^Expenses' GROUP BY month ORDER BY month",
    "quarter": "SELECT quarter(date) AS q, sum(number) WHERE account ~ '^Expenses' GROUP BY q ORDER BY q",
    "weekday": "SELECT weekday(date) AS day, count(*) GROUP BY day",
    "today": "SELECT today() FROM #",
    "date_add": "SELECT date_add(today(), -30) FROM #",
    "date_diff": "SELECT date, narration, date_diff(today(), date) AS days_ago ORDER BY date DESC LIMIT 5",
    "date_trunc": "SELECT date_trunc('month', date) AS month, sum(number) WHERE account ~ '^Expenses' GROUP BY month ORDER BY month",
    "date_part": "SELECT date_part('dow', date) AS dow, count(*) GROUP BY dow ORDER BY dow",
    "date_bin": "SELECT date_bin('1 month', date, 2020-01-01) AS bucket, count(*) GROUP BY bucket ORDER BY bucket",
    "parse_date": "SELECT parse_date('15.03.2024', '%d.%m.%Y') FROM #",
    "root": "SELECT root(account, 2) AS category, sum(number) WHERE account ~ '^Expenses' GROUP BY category",
    "parent": "SELECT parent(account) AS parent, count(*) GROUP BY parent",
    "leaf": "SELECT leaf(account) AS name, count(*) GROUP BY name",
    "account_sortkey": "SELECT account, sum(position) GROUP BY account, account_sortkey(account) ORDER BY account_sortkey(account)",
    "open_date": "SELECT account, open_date(account) AS opened FROM #accounts ORDER BY opened",
    "close_date": "SELECT account, close_date(account) AS closed FROM #accounts WHERE close IS NOT NULL",
    "open_meta": "SELECT account, open_meta(account, 'institution') AS institution FROM #accounts",
    "commodity_meta": "SELECT name, commodity_meta(name, 'name') AS long_name FROM #commodities",
    "units": "SELECT account, units(sum(position)) WHERE account ~ '^Assets' GROUP BY account",
    "cost": "SELECT account, cost(sum(position)) WHERE account ~ '^Assets' GROUP BY account",
    "value": "SELECT account, value(sum(position)) WHERE account ~ '^Assets' GROUP BY account",
    "convert": "SELECT account, convert(sum(position), 'USD') WHERE account ~ '^Assets' GROUP BY account",
    "getprice": "SELECT getprice('AAPL', 'USD') FROM #",
    "only": "SELECT account, only('USD', sum(position)) AS usd WHERE account ~ '^Assets' GROUP BY account",
    "empty": "SELECT account, sum(position) GROUP BY account HAVING not empty(sum(position))",
    "filter_currency": "SELECT account, filter_currency(sum(position), 'USD') WHERE account ~ '^Assets' GROUP BY account",
    "neg": "SELECT neg(sum(position)) WHERE account ~ '^Income'",
    "abs": "SELECT date, narration, abs(number) WHERE account ~ '^Expenses' ORDER BY 3 DESC LIMIT 5",
    "safediv": "SELECT safediv(sum(number), count(*)) WHERE account ~ '^Expenses'",
    "round": "SELECT round(sum(number), 0) WHERE account ~ '^Expenses'",
    "length": "SELECT narration, length(narration) FROM #transactions ORDER BY 2 DESC LIMIT 5",
    "upper": "SELECT upper(payee) FROM #transactions LIMIT 5",
    "lower": "SELECT lower(payee) FROM #transactions LIMIT 5",
    "substr": "SELECT substr(narration, 0, 10) FROM #transactions LIMIT 5",
    "splitcomp": "SELECT splitcomp(account, ':', 1) AS second, count(*) GROUP BY second",
    "maxwidth": "SELECT maxwidth(narration, 20) FROM #transactions LIMIT 5",
    "grep": "SELECT grep('[0-9]+', narration) FROM #transactions LIMIT 5",
    "subst": "SELECT subst('[0-9]+', '#', narration) FROM #transactions LIMIT 5",
    "joinstr": "SELECT date, joinstr(tags) AS tags FROM #transactions LIMIT 5",
    "meta": "SELECT date, account, meta('note') WHERE meta('note') IS NOT NULL",
    "entry_meta": "SELECT DISTINCT date, entry_meta('invoice') WHERE entry_meta('invoice') IS NOT NULL",
    "any_meta": "SELECT date, any_meta('receipt') WHERE any_meta('receipt') IS NOT NULL",
    "has_account": "SELECT DISTINCT date, narration WHERE has_account('Checking')",
    "possign": "SELECT account, sum(possign(number, account)) GROUP BY account",
    "count": "SELECT account, count(*) GROUP BY account ORDER BY 2 DESC",
    "sum": "SELECT account, sum(position) WHERE account ~ '^Expenses' GROUP BY account",
    "first": "SELECT account, first(date) GROUP BY account",
    "last": "SELECT account, last(date) GROUP BY account",
    "min": "SELECT account, min(date) GROUP BY account",
    "max": "SELECT account, max(date) GROUP BY account",
}

# Facts the docstrings leave out, taken from the implementation in beanquery/query_env.py.
FUNCTION_NOTES = {
    "date_trunc": "The field is one of 'week' (rounds down to the Monday), 'month', 'quarter', 'year', 'decade', 'century' or 'millennium'.",
    "date_part": "The field is one of 'weekday'/'dow' (0 = Monday), 'isoweekday'/'isodow' (1 = Monday), 'week', 'month', 'quarter', 'year', 'isoyear', 'decade', 'century', 'millennium' or 'epoch'.",
    "date_bin": "The stride is a string such as '1 month', '3 months' or '14 days' (only days, months and years are supported); every date is mapped to the start of its bucket, with buckets aligned to the origin date.",
    "interval": "The argument looks like '3 months', '-1 year' or '14 days'; only days, months and years are supported (anything else returns NULL).",
    "weekday": "The result is a three-letter English abbreviation such as 'Mon'.",
    "quarter": "The result is a string such as '2024-Q3'.",
    "yearmonth": "The result is the first day of the month, so it groups and sorts correctly.",
    "root": "With one argument it returns the first account component (e.g. 'Expenses'); the optional second argument is the number of components to keep.",
    "convert": "The optional last argument is the date of the price to use; without it the latest price is used.",
    "getprice": "Returns NULL when no price is known for the pair on that date.",
}

EXCLUDED_FUNCTIONS ={"getitem", "row"}  # internal helpers, not meant to be called directly


def _sig(name: str, func) -> str:
    return f"{name}({', '.join(types.name(d) for d in func.__intypes__)})"


# ----------------------------------------------------------------------------- concepts

CONCEPTS: list[dict] = [
    dict(topic="what_is_bql", questions=[
        "What is BQL?", "What is the Beancount Query Language and how does it differ from SQL?", "Explain bean-query in a nutshell.",
        "How is BQL different from normal SQL?"],
        answer=("BQL is the SQL-like query language of `bean-query` (the `beanquery` package for Beancount v3). It looks like SQL, but it queries a fixed set of "
                "tables derived from a Beancount ledger (`postings`, `entries`, `transactions`, `accounts`, `prices`, ...) and it understands accounting concepts "
                "that are painful in plain SQL: positions with cost basis, inventories that add up multi-currency lots (`sum(position)`), price conversion "
                "(`convert()`, `value()`), and period statements (`OPEN ON`, `CLOSE ON`, `CLEAR`). Aggregate queries group implicitly by the non-aggregate targets, "
                "and there are convenience statements `BALANCES`, `JOURNAL` and `PRINT`.")),
    dict(topic="postings_model", questions=[
        "What does one row of a BQL query represent?", "What table does a BQL SELECT query when there is no FROM clause?",
        "How does BQL flatten transactions?", "What is the postings table in BQL?"],
        answer=("Without a FROM clause a query runs over the `postings` table: every transaction is flattened into one row per posting (leg), and each row also carries "
                "the parent transaction's columns (`date`, `flag`, `payee`, `narration`, `tags`, `links`, `year`, ...). So `SELECT date, account, position` lists every "
                "leg of every transaction. Use `#transactions` for one row per transaction and `#entries` for one row per directive of any type."),
        examples=["SELECT date, account, position, narration LIMIT 10"]),
    dict(topic="from_vs_where", questions=[
        "What is the difference between FROM and WHERE in BQL?", "When do I use FROM and when WHERE in a BQL query?",
        "Does FROM filter whole transactions in bean-query v3?", "How do I select whole transactions that touch an account?"],
        answer=("In beanquery (v3) `FROM <expression>` and `WHERE` both filter rows of the table, so `FROM year = 2024` is equivalent to `WHERE year = 2024`. "
                "(In Beancount 2's bean-query, FROM selected whole transactions; that is no longer the case.) `FROM` is where the special period operators live: "
                "`OPEN ON`, `CLOSE ON` and `CLEAR`. To keep every posting of the transactions that touch some account, test the transaction-level set of accounts: "
                "`has_account('regex')` or `'Full:Account:Name' IN accounts`.\n\n"
                "```sql\nSELECT date, account, position\nWHERE has_account('Checking')\n```")),
    dict(topic="open_close_clear", questions=[
        "What do OPEN ON, CLOSE ON and CLEAR do in BQL?", "How do I produce an income statement or a balance sheet for a period in BQL?",
        "Explain the OPEN, CLOSE and CLEAR clauses.", "How do I get balances as of a date in BQL using CLOSE ON?"],
        answer=("They are period operators in the FROM clause. `OPEN ON <date>` replaces all entries before that date with summarizing entries that carry the opening "
                "balances (and rolls earlier income and expenses into equity), so the period starts from correct balances. `CLOSE ON <date>` drops entries on or after "
                "that date (so the date itself is excluded), and leaves income and expense balances alone, which is what an income statement needs. `CLEAR` moves the "
                "income and expense balances into equity at the end, which makes a balance sheet balance. Combine them for a full period:\n\n"
                "```sql\nSELECT account, sum(position)\nFROM OPEN ON 2024-01-01 CLOSE ON 2025-01-01 CLEAR\nWHERE account ~ '^(Assets|Liabilities|Equity)'\nGROUP BY account\n```"),
        examples=["SELECT account, sum(position) FROM OPEN ON 2024-01-01 CLOSE ON 2025-01-01 WHERE account ~ '^(Income|Expenses)' GROUP BY account ORDER BY account"]),
    dict(topic="positions_inventories", questions=[
        "What are positions and inventories in BQL?", "Why does sum(position) return several amounts?", "What is the difference between position, units, cost and value?",
        "What do units(), cost() and value() do to an inventory?"],
        answer=("A `position` is one holding: units of a commodity with an optional cost basis (like `10 AAPL {150 USD}`). Summing positions with `sum(position)` gives an "
                "`inventory`, which keeps different commodities and lots apart, so the result can contain several amounts. Convert it with: `units()` (strip the cost, "
                "keep only the commodity quantities), `cost()` (total at the original cost basis), `value()` (market value using the latest price directives) or "
                "`convert(x, 'CCY')` (convert to one currency at market prices). There is no `weight()` or `raw()` function; `weight` is a column of the postings table "
                "holding the amount used to balance the transaction."),
        examples=["SELECT account, units(sum(position)), cost(sum(position)) WHERE account ~ '^Assets' GROUP BY account"]),
    dict(topic="number_vs_position", questions=[
        "When should I use sum(number) instead of sum(position)?", "What is the difference between number and position in BQL?",
        "Is it safe to sum the number column?"],
        answer=("`number` is the plain decimal amount of a posting and `currency` its currency; `position` also carries the cost basis. `sum(number)` adds bare numbers, "
                "so it is only meaningful when everything you sum has the same currency (filter with `currency = 'USD'` or group by `currency`). `sum(position)` is always "
                "safe because the inventory keeps currencies apart. Arithmetic works on numbers (`number * 2`) but not on positions."),
        examples=["SELECT currency, sum(number) WHERE account ~ '^Expenses' GROUP BY currency"]),
    dict(topic="grouping", questions=[
        "How does GROUP BY work in BQL?", "Do I need GROUP BY when I use aggregate functions in BQL?", "What does the error about non-aggregates and GROUP-BY mean?",
        "Can I group by column position in BQL?"],
        answer=("Aggregate functions are `sum`, `count`, `min`, `max`, `first` and `last`. If a query mixes aggregates and plain columns, the plain columns become the "
                "grouping key: BQL groups implicitly, so `SELECT account, sum(position)` already groups by account (writing `GROUP BY account` is clearer). Every "
                "non-aggregate target must be covered by the grouping, otherwise you get \"all non-aggregates must be covered by GROUP-BY\". You can refer to targets by "
                "alias or by 1-based position (`GROUP BY 1`, `ORDER BY 2 DESC`). A group-by key must be hashable, so tags and links (sets) cannot be grouped directly; "
                "use `joinstr(tags)`.\n\n```sql\nSELECT account, sum(position) AS total\nWHERE account ~ '^Expenses'\nGROUP BY account\nORDER BY total DESC\n```")),
    dict(topic="sign_conventions", questions=[
        "Why are my income numbers negative in BQL?", "What are the sign conventions of Beancount accounts in query results?", "How do I show income as a positive number?"],
        answer=("Beancount is double-entry: money leaving an Income account is a negative posting, so Income totals come out negative (as do Liabilities and Equity), while "
                "Assets and Expenses are positive. Flip the sign when you want a positive income figure: `neg(sum(position))`, or `-sum(number)` for a single currency.\n\n"
                "```sql\nSELECT year, neg(sum(position)) AS income\nWHERE account ~ '^Income'\nGROUP BY year\n```")),
    dict(topic="no_avg", questions=[
        "How do I compute an average in BQL?", "Does BQL have an avg() function?", "How can I get the average monthly spending?"],
        answer=("BQL has no `avg()`. Sum and count instead; for an average per period, aggregate per period in a subquery and divide:\n\n"
                "```sql\nSELECT sum(total) / count(total) AS monthly_average\nFROM (SELECT year, month, sum(number) AS total WHERE account ~ '^Expenses' GROUP BY year, month)\n```"),
        examples=[]),
    dict(topic="no_count_distinct", questions=[
        "How do I count distinct values in BQL?", "Does BQL support count(DISTINCT x)?"],
        answer=("`count(DISTINCT x)` is not supported. `SELECT DISTINCT` is, so count the rows of a DISTINCT subquery:\n\n"
                "```sql\nSELECT count(*) AS payees\nFROM (SELECT DISTINCT payee)\n```")),
    dict(topic="regex_matching", questions=[
        "How do I do pattern matching like LIKE in BQL?", "Does BQL support LIKE?", "How does the ~ operator work in BQL?", "How do I match text case-insensitively in BQL?"],
        answer=("BQL has no `LIKE`. Use `~` for a regular-expression match and `!~` for a non-match. The pattern is searched anywhere in the string, so anchor it with `^` "
                "or `$` when you need a prefix or suffix match, and use `(?i)` for case-insensitive matching. `=` is exact equality, and `IN ('a', 'b')` tests a list. "
                "`pattern ?~ string` is the same as `string ~ pattern` with the operands swapped.\n\n"
                "```sql\nSELECT date, narration\nWHERE narration ~ '(?i)coffee' AND account !~ '^Assets'\n```")),
    dict(topic="tags_links", questions=[
        "How do I filter by tag in BQL?", "How do I check whether a transaction has a link in BQL?", "Why can't I GROUP BY tags?"],
        answer=("`tags` and `links` are sets of strings, so test membership with `IN`: `'trip' IN tags`. They cannot be compared with `=` or used as a GROUP BY key; "
                "for grouping use `joinstr(tags)`, and `length(tags) > 0` finds tagged entries.\n\n```sql\nSELECT date, narration, account, position\nWHERE 'trip' IN tags\n```")),
    dict(topic="dates", questions=[
        "How do I write dates in BQL?", "Why does comparing a date to a quoted string fail in BQL?", "How do I filter by a date range in BQL?"],
        answer=("Date literals are written without quotes, as `2024-03-31`. Comparing a date column with a quoted string fails with an operator error (`greater(date, str)` "
                "not supported). Combine comparisons with AND, or use `BETWEEN` (inclusive on both ends). The `year`, `month` and `day` columns exist on postings and entries; "
                "on other tables use `year(date)`, `month(date)`. `today()`, `date_add(d, days)` and `date_diff(d1, d2)` handle relative dates.\n\n"
                "```sql\nSELECT date, narration, position\nWHERE date >= 2024-03-01 AND date < 2024-04-01\n```")),
    dict(topic="having", questions=[
        "How does HAVING work in BQL?", "Why can't I use an alias in HAVING?", "How do I filter groups after aggregation in BQL?"],
        answer=("`HAVING` belongs to `GROUP BY` and filters groups after aggregation. Its expression must contain an aggregate and it cannot refer to a target alias, so "
                "repeat the aggregate: `HAVING sum(number) > 100`, not `HAVING total > 100`. Aggregates are not allowed in `WHERE`.\n\n"
                "```sql\nSELECT account, sum(number) AS total\nWHERE account ~ '^Expenses'\nGROUP BY account HAVING sum(number) > 100\n```")),
    dict(topic="pivot", questions=[
        "How does PIVOT BY work in BQL?", "How do I turn a grouped result into a matrix in BQL?", "What are the rules for PIVOT BY?"],
        answer=("`PIVOT BY a, b` turns a grouped result into a matrix: the values of the first column become rows and the values of the second become columns. Both "
                "columns must be in the SELECT targets (by name or 1-based index), they must be different, and the second must be a GROUP BY column.\n\n"
                "```sql\nSELECT account, year, sum(number)\nWHERE account ~ '^Expenses'\nGROUP BY account, year\nPIVOT BY account, year\n```")),
    dict(topic="order_limit", questions=[
        "How does ORDER BY work in BQL?", "Can I sort each column in a different direction in BQL?", "Where do NULLs sort in BQL?"],
        answer=("`ORDER BY` accepts several columns, each with its own `ASC` or `DESC` (ascending by default): `ORDER BY year DESC, month ASC`. Older versions applied one "
                "direction to all columns, which is no longer the case. NULL values sort as smaller than any other value. You can order by an alias, a 1-based column "
                "number or an expression, and `LIMIT n` keeps the first n rows.")),
    dict(topic="statements", questions=[
        "What are BALANCES, JOURNAL and PRINT in BQL?", "What do the BALANCES, JOURNAL and PRINT statements do?", "How do I dump entries as Beancount text with BQL?"],
        answer=("They are shortcuts. `BALANCES` prints the balance of each account (`sum(position)` grouped by account, in account order). `JOURNAL 'regex'` lists the "
                "postings of matching accounts with a running balance. `PRINT` outputs the selected directives as Beancount source text. `BALANCES` and `JOURNAL` accept "
                "`AT cost` or `AT units` to change the valuation, and all three accept a `FROM` clause. `PRINT` runs over directives, so its FROM cannot use posting "
                "columns like `account`; use `'Assets:Checking' IN accounts` or `has_account('Checking')`.\n\n"
                "```sql\nPRINT FROM has_account('Checking') AND year = 2024\n```"),
        examples=[]),
    dict(topic="tables_choice", questions=[
        "Which BQL table should I use: postings, transactions or entries?", "What is the difference between #postings, #transactions and #entries?"],
        answer=("Use `postings` (the default) when you need accounts, amounts or positions. Use `#transactions` for questions about whole transactions (one row each: counts, "
                "payees, narrations, dates, tags); it has no posting columns and no `year` column, so write `year(date) = 2024`. Use `#entries` for any directive type, "
                "filtering on `type` (`'transaction'`, `'open'`, `'price'`, ...). Other tables: `accounts`, `balances`, `prices`, `commodities`, `events`, `notes`, `documents`.\n\n"
                "```sql\nSELECT count(*)\nFROM #transactions\nWHERE year(date) = 2024\n```")),
    dict(topic="scalar_from_hash", questions=[
        "How do I evaluate an expression in BQL without querying the ledger?", "What does FROM # mean in BQL?"],
        answer=("`FROM #` selects the empty table, which has exactly one row and no columns, so it is handy for evaluating expressions and functions:\n\n"
                "```sql\nSELECT date_add(today(), 30) FROM #\n```")),
    dict(topic="subqueries", questions=[
        "Does BQL support subqueries?", "How do I use a subquery in BQL?"],
        answer=("Yes. A subquery can be the FROM source, and a one-column subquery can be used with `IN`:\n\n"
                "```sql\nSELECT max(total)\nFROM (SELECT year, month, sum(number) AS total WHERE account ~ '^Expenses' GROUP BY year, month)\n```\n\n"
                "```sql\nSELECT date, narration\nFROM #transactions\nWHERE payee IN (SELECT payee FROM #transactions WHERE narration ~ 'Rent')\n```")),
    dict(topic="metadata", questions=[
        "How do I read metadata in BQL?", "What is the difference between meta(), entry_meta() and any_meta()?"],
        answer=("`meta('key')` reads a metadata key of the posting, `entry_meta('key')` of the parent transaction, and `any_meta('key')` tries the posting and falls back to "
                "the transaction. Account and commodity metadata come from `open_meta(account, 'key')` and `commodity_meta(currency, 'key')`. Values have the generic "
                "`object` type and are NULL when the key is missing; convert with `int()`, `decimal()`, `str()` or `date()` when needed.\n\n"
                "```sql\nSELECT DISTINCT date, narration, entry_meta('invoice')\nWHERE entry_meta('invoice') IS NOT NULL\n```")),
    dict(topic="running_balance", questions=[
        "How do I get the balance of an account over time in BQL?", "How do I get month-end balances in BQL?", "What is the balance column?"],
        answer=("The `balance` column of a posting is the running balance (an inventory) of the postings so far. To read the balance at the end of each month, take the "
                "last value per month:\n\n```sql\nSELECT year, month, last(balance) AS end_balance\nWHERE account = 'Assets:Checking'\nGROUP BY year, month\nORDER BY year, month\n```"),
        examples=[]),
    dict(topic="balance_group_by_pitfall", questions=[
        "Why does GROUP BY account with last(balance) give the wrong total per account?",
        "I used last(balance) grouped by account for net worth and the numbers are wrong, why?",
        "Is last(balance) safe to use when a query covers more than one account?"],
        answer=("No: `balance` is a single running total computed over the rows the query sees, in date order, not one running "
                "total per account. Grouping by account and taking `last(balance)` still adds up postings from *every* account in "
                "the row set along the way, so an account that comes later in date order picks up the earlier accounts' amounts "
                "too. This is a real, reported beanquery gotcha, not a hypothetical: for two accounts with postings of 10 and 100, "
                "`last(balance)` reports 10 and 110, not 10 and 100. To get a correct total per account, group by account and use "
                "`sum(position)` instead:\n\n"
                "```sql\nSELECT account, sum(position)\nWHERE date <= 2024-12-31 AND account ~ '^(Assets|Liabilities)'\nGROUP BY account\n```\n\n"
                "`last(balance)` is only safe when the query is scoped to a single account (see the running-balance question above), "
                "not when it groups several accounts together.")),
    dict(topic="shell_commands", questions=[
        "What commands does the bean-query shell have?", "How do I list the tables or describe a table in bean-query?", "How do I see how a BQL query is parsed?"],
        answer=("Besides BQL statements the shell has dot-commands: `.tables` lists the tables, `.describe <table>` lists a table's columns and types, `.explain <query>` "
                "shows how a query is compiled, `.parse <query>` shows the parse tree, `.run` runs the named queries stored in the ledger, `.set` and `.format` change "
                "settings and the output format (text, csv, beancount), `.output` redirects output, `.reload` reloads the ledger, `.errors` shows ledger errors, and "
                ".history, .clear and .exit do what they say. `help select`, `help from` and `help where` list the columns and functions.")),
    dict(topic="v2_to_v3", questions=[
        "What changed in BQL between Beancount 2 and Beancount v3?", "What is different in beanquery compared to the old bean-query?"],
        answer=("BQL now lives in the separate `beanquery` package. Notable changes: `FROM <expr>` is a plain row filter (in Beancount 2 it selected whole transactions); "
                "`HAVING` is supported; each `ORDER BY` column has its own direction (previously one direction applied to all) and NULLs sort first; the functions `round()` "
                "and `empty()` and the casts `int()`, `decimal()`, `str()` and `date()` were added; `str()` is now a cast (the old behavior is `repr()`) and the old "
                "lenient `date(string)` parser is now `parse_date()`. beanquery also exposes the ledger as several tables (`#accounts`, `#prices`, `#balances`, ...) "
                "that can be queried directly.")),
    dict(topic="error_messages", questions=[
        "What does 'no function matches' mean in a BQL error?", "What does 'column not found in table' mean in BQL?", "Why do I get 'aggregates are not allowed in WHERE clause'?",
        "What does 'operator not supported' mean in BQL?"],
        answer=("These are compile errors. `no function matches \"f(type)\" name and argument types` means the function does not exist in BQL (for example `avg`) or the "
                "argument types are wrong. `column \"x\" not found in table \"t\"` means the column does not exist on the queried table (for example `year` on "
                "`#transactions`, or `account` on `#entries`). `aggregates are not allowed in WHERE clause` means an aggregate belongs in `HAVING` instead. "
                "`operator \"greater(date, str)\" not supported` means the operand types do not fit, for example a date compared with a quoted string: write the date "
                "literal without quotes. `syntax error` is a parse error, often from SQL-only syntax such as `LIKE`, `<>` (use `!=`) or `count(DISTINCT x)`.")),
    dict(topic="coalesce_cast", questions=[
        "How do I handle NULL values in BQL?", "Is there a coalesce function in BQL?"],
        answer=("Test with `IS NULL` / `IS NOT NULL`. `coalesce(a, b, ...)` returns the first non-NULL argument; all arguments must have the same type. NULLs sort before "
                "every other value.\n\n```sql\nSELECT date, coalesce(payee, 'unknown') AS payee\nFROM #transactions\n```")),
]


# ----------------------------------------------------------------------------- builders

def _example(rng: random.Random, question: str, answer: str, topic: str, kind: str) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer},
        ],
        "meta": {"task": "reference", "topic": kind + ":" + topic, "schema": "none"},
    }


def _validate(ex: Executor, answer: str, context: str) -> None:
    for block in FENCE.findall(answer):
        try:
            ex.run(block)
        except QueryError as e:
            raise AssertionError(f"reference example does not run ({context}): {block!r} -> {e.message}") from e


def table_examples(rng: random.Random, ex: Executor) -> list[dict]:
    out = []
    tables = ex.conn.tables
    listing = "\n".join(f"- `{n}`: {SHORT_DESC[n]}" for n in TABLE_ORDER)
    for q in ["Which tables can I query in BQL?", "What tables does bean-query provide?", "List the BQL tables and what each contains."]:
        out.append(_example(rng, q, f"The built-in tables (select with `FROM #name`; `postings` is the default) are:\n\n{listing}\n\nIn the shell, `.tables` lists them and `.describe <table>` shows the columns.", "tables", "table"))
    for name in TABLE_ORDER:
        cols = tables[name].columns
        cl = ", ".join(f"`{c}` ({types.name(col.dtype)})" for c, col in cols.items())
        answer = f"The `{name}` table is {TABLE_DESC[name]}.\n\nColumns: {cl}."
        for q in rng.sample([f"What columns does the {name} table have in BQL?", f"Describe the BQL table `{name}`.", f"What can I select from #{name}?", f"Show the schema of the {name} table in bean-query."], 2):
            out.append(_example(rng, q, answer, name, "table"))
    for name in ("postings", "entries"):
        for c, col in tables[name].columns.items():
            doc = COLUMN_DOC_OVERRIDES.get(c) or (col.__doc__ or "").strip()
            if not doc:
                continue
            q = rng.choice([f"What is the `{c}` column in the BQL {name} table?", f"What does `{c}` contain in {name}?", f"Explain the column {c} of #{name}."])
            out.append(_example(rng, q, f"`{c}` ({types.name(col.dtype)}): {doc}", f"{name}.{c}", "column"))
    return out


def function_examples(rng: random.Random, ex: Executor) -> list[dict]:
    out = []
    for name, funcs in sorted(query_compile.FUNCTIONS.items()):
        if name in EXCLUDED_FUNCTIONS:
            continue
        doc = FUNCTION_DOC_OVERRIDES.get(name) or next((f.__doc__.strip() for f in funcs if f.__doc__), None)
        if not doc:
            continue
        doc = " ".join(doc.split())
        sigs = "\n".join(f"- `{_sig(name, f)}`" for f in funcs)
        kind = "aggregate function" if issubclass(funcs[0], query_compile.EvalAggregator) else "function"
        note = f" {FUNCTION_NOTES[name]}" if name in FUNCTION_NOTES else ""
        answer = f"`{name}` is a BQL {kind}. {doc}{note}\n\nSignatures:\n{sigs}"
        example = FUNCTION_EXAMPLES.get(name)
        if example:
            answer += f"\n\nExample:\n\n```sql\n{example}\n```"
        _validate(ex, answer, f"function {name}")
        for q in rng.sample([f"What does the BQL function `{name}` do?", f"How do I use {name}() in bean-query?", f"What arguments does `{name}` take in BQL?", f"Explain {name}() in the Beancount query language."], 2):
            out.append(_example(rng, q, answer, name, "function"))
    return out


def concept_examples(rng: random.Random, ex: Executor) -> list[dict]:
    out = []
    for c in CONCEPTS:
        answer = c["answer"]
        _validate(ex, answer, f"concept {c['topic']}")
        for e in c.get("examples", []):
            ex.run(e)  # examples that are not shown but must stay valid
        for q in c["questions"]:
            out.append(_example(rng, q, answer, c["topic"], "concept"))
    return out


def build_reference_examples(rng: random.Random, ex: Executor) -> list[dict]:
    return table_examples(rng, ex) + function_examples(rng, ex) + concept_examples(rng, ex)
