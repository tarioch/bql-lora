"""The Ollama Modelfile, the training data and the code must all agree: no system message in training data, and
Qwen's own default system line baked into the Modelfile (see README, "The system prompt: trigger or baked in?")."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bql_lora.modelfile import QWEN_DEFAULT_SYSTEM, render_modelfile


def _system_value(text: str):
    m = re.search(r'(?s)^SYSTEM """(.*)"""\s*$', text, re.M)
    return m.group(1) if m else None


def test_committed_modelfile_is_what_the_generator_writes():
    committed = (ROOT / "ollama" / "Modelfile").read_bytes()
    assert committed == render_modelfile().encode("utf-8"), "run: python scripts/make_modelfile.py"


def test_modelfile_has_lf_line_endings_only():
    # Ollama keeps a carriage return found in a template or system prompt, which changes what the model sees.
    assert b"\r" not in (ROOT / "ollama" / "Modelfile").read_bytes()
    assert "\r" not in render_modelfile()


def test_system_block_is_qwens_default_line():
    # Verified with check_system_prompt_sensitivity.py: a model trained with no system message in its data answers
    # 8 of 8 test questions in BQL under this line (Qwen's chat template's own default for a conversation with
    # none), vs. only 1 of 8 for an earlier model that had a custom system prompt baked into every training example.
    assert _system_value((ROOT / "ollama" / "Modelfile").read_text(encoding="utf-8")) == QWEN_DEFAULT_SYSTEM
    assert _system_value(render_modelfile()) == QWEN_DEFAULT_SYSTEM


def test_template_reads_only_messages_and_has_no_tool_calling():
    text = render_modelfile()
    template = re.search(r'(?s)TEMPLATE """(.*?)"""', text).group(1)
    assert ".Messages" in template
    # ".System" would print the system prompt twice: Ollama also passes it as the first message.
    assert ".System" not in template
    assert not re.search(r"Tools|ToolCalls|tool_call|fim_", text)


def _first_example(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())


def test_training_data_has_no_system_message():
    for name in ("train.jsonl", "val.jsonl"):
        with open(ROOT / "data" / name, encoding="utf-8") as f:
            for line in f:
                assert [m["role"] for m in json.loads(line)["messages"]] == ["user", "assistant"], \
                    f"data/{name} has a system message; a constant one gives a fine-tune something to key on " \
                    "instead of learning the behaviour for any input"


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("ok")
