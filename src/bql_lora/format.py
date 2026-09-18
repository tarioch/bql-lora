"""Turn a validated Sample into a chat-format training example."""

from __future__ import annotations

import random

from .intents import Sample
from .ledger import Ledger
from .schema import SYSTEM_PROMPT, user_prompt

ANSWER_STYLES = ["code_then_explain", "code_then_explain", "code_only", "explain_then_code", "code_then_explain"]


def assistant_message(rng: random.Random, sample: Sample) -> str:
    style = rng.choice(ANSWER_STYLES) if sample.explanation else "code_only"
    code = f"```sql\n{sample.bql}\n```"
    if style == "code_only":
        return code
    if style == "explain_then_code":
        return f"{sample.explanation}\n\n{code}"
    return f"{code}\n\n{sample.explanation}"


def build_example(rng: random.Random, ledger: Ledger, sample: Sample) -> dict:
    return {
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt(ledger, sample.question)},
            {"role": "assistant", "content": assistant_message(rng, sample)},
        ],
        "meta": {"intent": sample.intent, "bql": sample.bql},
    }
