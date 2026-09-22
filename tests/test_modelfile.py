"""The Ollama Modelfile, the training data and the code must all use the same system prompt and format."""

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bql_lora.format import strip_system
from bql_lora.modelfile import QWEN_DEFAULT_SYSTEM, SYSTEM_CHOICES, render_modelfile
from bql_lora.schema import SYSTEM_PROMPT


def _system_value(text: str):
    m = re.search(r'(?s)^SYSTEM """(.*)"""\s*$', text, re.M)
    return m.group(1) if m else None


def test_committed_modelfile_is_what_the_generator_writes():
    committed = (ROOT / "ollama" / "Modelfile").read_bytes()
    assert committed == render_modelfile().encode("utf-8"), "run: python scripts/make_modelfile.py"


def test_modelfile_has_lf_line_endings_only():
    # Ollama keeps a carriage return found in a template or system prompt, which changes what the model sees.
    assert b"\r" not in (ROOT / "ollama" / "Modelfile").read_bytes()
    for system in SYSTEM_CHOICES:
        assert "\r" not in render_modelfile(system=system)


def test_system_block_matches_the_chosen_system_message():
    # The committed Modelfile matches a model trained on data/no-system/ (scripts/train.py's default), where
    # Qwen's chat template injects Qwen's own stock line for an example with no system message. Verified with
    # scripts/check_system_prompt_sensitivity.py: 8/8 test questions answered in BQL under this choice.
    assert _system_value((ROOT / "ollama" / "Modelfile").read_text(encoding="utf-8")) == QWEN_DEFAULT_SYSTEM
    assert _system_value(render_modelfile(system="qwen")) == QWEN_DEFAULT_SYSTEM
    assert _system_value(render_modelfile(system="bql")) == SYSTEM_PROMPT
    assert _system_value(render_modelfile(system="none")) is None
    assert "SYSTEM" not in render_modelfile(system="none")


def test_unknown_system_choice_is_rejected():
    try:
        render_modelfile(system="mine")
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_template_reads_only_messages_and_has_no_tool_calling():
    for system in SYSTEM_CHOICES:
        text = render_modelfile(system=system)
        template = re.search(r'(?s)TEMPLATE """(.*?)"""', text).group(1)
        assert ".Messages" in template
        # ".System" would print the system prompt twice: Ollama also passes it as the first message.
        assert ".System" not in template
        assert not re.search(r"Tools|ToolCalls|tool_call|fim_", text)


def test_strip_system_only_removes_the_system_message():
    ex = {"messages": [{"role": "system", "content": "s"}, {"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}],
          "meta": {"task": "text2bql"}}
    out = strip_system(ex)
    assert [m["role"] for m in out["messages"]] == ["user", "assistant"]
    assert out["meta"] == ex["meta"]
    assert len(ex["messages"]) == 3, "the input must not be modified"


def _first_example(path: Path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.loads(f.readline())


def test_data_with_system_prompt_uses_the_current_system_prompt():
    for name in ("train.jsonl", "val.jsonl"):
        first = _first_example(ROOT / "data" / name)
        assert first["messages"][0] == {"role": "system", "content": SYSTEM_PROMPT}, f"data/{name} was built with a different system prompt; regenerate it"


def test_no_system_data_has_no_system_message():
    folder = ROOT / "data" / "no-system"
    if not (folder / "train.jsonl").exists() or not (folder / "val.jsonl").exists():
        return
    for name in ("train.jsonl", "val.jsonl"):
        with open(folder / name, encoding="utf-8") as f:
            for line in f:
                assert [m["role"] for m in json.loads(line)["messages"]] == ["user", "assistant"]


def test_both_data_variants_hold_the_same_examples():
    folder = ROOT / "data" / "no-system"
    if not (folder / "train.jsonl").exists() or not (folder / "val.jsonl").exists():
        return
    for name in ("train.jsonl", "val.jsonl"):
        with open(ROOT / "data" / name, encoding="utf-8") as a, open(folder / name, encoding="utf-8") as b:
            lines_a, lines_b = a.read().splitlines(), b.read().splitlines()
        assert len(lines_a) == len(lines_b), f"{name} differs in length"
        for line_a, line_b in zip(lines_a, lines_b):
            assert strip_system(json.loads(line_a)) == json.loads(line_b)


if __name__ == "__main__":
    for fn in list(globals().values()):
        if callable(fn) and getattr(fn, "__name__", "").startswith("test_"):
            fn()
    print("ok")
