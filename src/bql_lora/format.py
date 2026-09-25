"""Turn a validated Sample into a chat-format training example."""

from __future__ import annotations

import random

from .intents import Sample
from .ledger import Ledger
from .schema import user_prompt

ANSWER_STYLES = ["code_then_explain", "code_then_explain", "code_only", "explain_then_code", "code_then_explain"]


def assistant_message(rng: random.Random, sample: Sample) -> str:
    style = rng.choice(ANSWER_STYLES) if sample.explanation else "code_only"
    code = f"```sql\n{sample.bql}\n```"
    if style == "code_only":
        return code
    if style == "explain_then_code":
        return f"{sample.explanation}\n\n{code}"
    return f"{code}\n\n{sample.explanation}"


def build_example(rng: random.Random, ledger: Ledger, sample: Sample, mode: str = "full") -> dict:
    # No system message: a constant one makes a fine-tune key on its presence rather than learning the behaviour
    # for any input (see README, "The system prompt: trigger or baked in?"). Qwen's chat template supplies its own
    # default system line for a conversation with none, which is what ollama/Modelfile bakes into the model.
    return {
        "messages": [
            {"role": "user", "content": user_prompt(ledger, sample.question, mode)},
            {"role": "assistant", "content": assistant_message(rng, sample)},
        ],
        "meta": {"task": "text2bql", "intent": sample.intent, "schema": mode, "bql": sample.bql},
    }
