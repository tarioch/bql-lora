"""The Ollama Modelfile, the training data and the code must all use the same system prompt and format."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bql_lora.modelfile import render_modelfile
from bql_lora.schema import SYSTEM_PROMPT


def test_committed_modelfile_is_what_the_generator_writes():
    committed = (ROOT / "ollama" / "Modelfile").read_bytes()
    assert committed == render_modelfile().encode("utf-8"), "run: python scripts/make_modelfile.py"


def test_modelfile_has_lf_line_endings_only():
    # Ollama keeps a carriage return found in a template or system prompt, which changes what the model sees.
    assert b"\r" not in (ROOT / "ollama" / "Modelfile").read_bytes()


def test_system_block_is_exactly_the_system_prompt():
    text = (ROOT / "ollama" / "Modelfile").read_text(encoding="utf-8")
    assert re.search(r'(?s)^SYSTEM """(.*)"""\s*$', text, re.M).group(1) == SYSTEM_PROMPT


def test_template_reads_only_messages_and_has_no_tool_calling():
    text = render_modelfile()
    template = re.search(r'(?s)TEMPLATE """(.*?)"""', text).group(1)
    assert ".Messages" in template
    # ".System" would print the system prompt twice: Ollama also passes it as the first message.
    assert ".System" not in template
    assert not re.search(r"Tools|ToolCalls|tool_call|fim_", text)


def test_training_data_uses_the_same_system_prompt():
    for name in ("train.jsonl", "val.jsonl"):
        with open(ROOT / "data" / name, encoding="utf-8") as f:
            first = json.loads(f.readline())
        assert first["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}, f"{name} was built with a different system prompt; regenerate it"


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("ok")
