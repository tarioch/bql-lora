#!/usr/bin/env python
"""Check a generated dataset (structure + ledger independence).

1. Structure: every line is ``{"messages": [[system,] user, assistant], "meta": {...}}`` (the system message is
   optional, see ``generate_dataset.py --no-system``), and every ```sql
   block in an assistant message parses as BQL.
2. Leak audit: in ``text2bql`` examples whose prompt has no ledger information (``schema == "none"``), every
   ledger-specific literal used in the BQL (full account names, currency codes, account keywords) must be
   traceable to the question. Otherwise the model would be trained to guess facts it cannot know.

Usage:
    python scripts/audit_dataset.py data/train.jsonl data/val.jsonl
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from beanquery import parser  # noqa: E402

ROOTS = {"Assets", "Liabilities", "Equity", "Income", "Expenses"}
GENERIC_LITERALS = {"*", "!", ":", "transaction", "month", "week", "dow", "quarter", "year"}
FENCE = re.compile(r"```sql\n(.*?)\n```", re.S)


def words(s: str) -> str:
    return re.sub(r"(?<=[a-z])(?=[A-Z])", " ", s).replace("-", " ").lower()


def leaks(question: str, bql: str) -> list[str]:
    problems = []
    ql = question.lower()
    for lit in re.findall(r"'([^']*)'|\"([^\"]*)\"", bql):
        lit = lit[0] or lit[1]
        core = lit.lstrip("^").rstrip("$")
        for part in (p.strip() for p in re.split(r"[|()]", core)):
            if not part or part in ROOTS or part in GENERIC_LITERALS:
                continue
            if re.fullmatch(r"[A-Z]{2,5}", part):  # a currency or commodity code
                if part not in question:
                    problems.append(f"currency {part}")
            elif ":" in part:  # a full account name or prefix
                if part not in question:
                    problems.append(f"account {part}")
            elif re.fullmatch(r"[A-Z][A-Za-z\-]+", part) and "account ~" in bql:  # keyword regexp
                if words(part).rstrip("s") not in ql and part not in question:
                    problems.append(f"keyword {part}")
    return problems


def main(paths: list[str]) -> int:
    n = n_none = 0
    bad_structure, bad_parse, bad_leaks = [], [], []
    for path in paths:
        for lineno, line in enumerate(open(path, encoding="utf-8"), 1):
            n += 1
            ex = json.loads(line)
            msgs = ex.get("messages", [])
            # A system message is optional (generate_dataset.py --no-system omits it).
            if [m["role"] for m in msgs] not in (["system", "user", "assistant"], ["user", "assistant"]) or "meta" not in ex:
                bad_structure.append(f"{path}:{lineno}")
                continue
            user, assistant = msgs[-2]["content"], msgs[-1]["content"]
            for block in FENCE.findall(assistant):
                try:
                    parser.parse(block)
                except Exception as e:  # noqa: BLE001
                    bad_parse.append((f"{path}:{lineno}", block, str(e).splitlines()[0]))
            meta = ex["meta"]
            if meta.get("task") == "text2bql" and meta.get("schema") == "none":
                n_none += 1
                found = leaks(user, meta["bql"])
                if found:
                    bad_leaks.append((f"{path}:{lineno}", found, user, meta["bql"]))

    print(f"{n} examples ({n_none} schema-free text2bql); structure errors: {len(bad_structure)}, "
          f"unparsable BQL blocks: {len(bad_parse)}, ledger leaks: {len(bad_leaks)}")
    for where in bad_structure[:5]:
        print("  structure:", where)
    for where, block, err in bad_parse[:5]:
        print("  parse:", where, block.replace("\n", " "), "->", err)
    for where, found, q, bql in bad_leaks[:10]:
        print("  leak:", where, found, "|", q, "|", bql.replace("\n", " "))
    return 1 if (bad_structure or bad_parse or bad_leaks) else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
