"""Turn a validated Sample into a chat-format training example."""

from __future__ import annotations

import random

from .intents import Sample
from .ledger import Ledger
from .schema import user_prompt


def assistant_message(rng: random.Random, sample: Sample) -> str:
    """Just the fenced query: the app runs it and has no use for an explanation, which is also less to learn."""
    return f"```sql\n{sample.bql}\n```"


def build_example(rng: random.Random, ledger: Ledger, sample: Sample, mode: str = "full") -> dict:
    # No system message (see README, "Why the data has no system prompt").
    return {
        "messages": [
            {"role": "user", "content": user_prompt(ledger, sample.question, mode)},
            {"role": "assistant", "content": assistant_message(rng, sample)},
        ],
        "meta": {"task": "text2bql", "intent": sample.intent, "schema": mode, "bql": sample.bql},
    }
