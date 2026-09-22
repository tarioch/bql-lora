"""Render the Ollama Modelfile for a model fine-tuned on this dataset.

The Modelfile is generated rather than hand-edited so the system prompt baked into the model cannot drift from
the one the training data was built with (``schema.SYSTEM_PROMPT``).

Design notes, each learned the hard way:

* The template reads only ``.Messages``. Ollama passes the Modelfile ``SYSTEM`` prompt as the first message *and*
  as ``.System``, so a template that prints both feeds the model the system prompt twice.
* The output must use LF line endings. Ollama keeps a ``\\r`` found inside a template or system prompt, so a
  CRLF file changes the tokens the model sees compared with training. ``.gitattributes`` pins the committed file.
* ``temperature 0``: this model writes queries, so sampling randomness only hurts.
* The stop string is a safety net; the GGUF normally already ends generation on ``<|im_end|>``.
* No tool-calling or fill-in-the-middle sections: the model is trained on question -> ``sql`` block only.
"""

from __future__ import annotations

from .schema import SYSTEM_PROMPT

DEFAULT_GGUF = "qwen2.5-coder-7b-instruct.Q4_K_M.gguf"

# Qwen's own default system message: what its chat template injects when a training example has none.
QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# Which system message the Modelfile bakes in. It has to match how the model was trained:
#   qwen  Qwen's stock line (the default): Qwen's chat template injects it when a training example has no system
#         message, which is what data/no-system/ (scripts/train.py's default) uses. Measured with
#         check_system_prompt_sensitivity.py: 8/8 test questions answered in BQL, vs. 1/8 for a model trained WITH
#         the system prompt and then run without one.
#   bql   the training system prompt, for a model trained on data/ (with the prompt in every example)
#   none  no system message at all
SYSTEM_CHOICES = ("qwen", "bql", "none")

# Plain ChatML as used by Qwen2.5: <|im_start|>role\ncontent<|im_end|>\n ... <|im_start|>assistant\n
TEMPLATE = (
    "{{- range .Messages }}<|im_start|>{{ .Role }}\n"
    "{{ .Content }}<|im_end|>\n"
    "{{ end }}<|im_start|>assistant\n"
)


def render_modelfile(gguf: str = DEFAULT_GGUF, system: str = "qwen") -> str:
    """The Modelfile text (LF line endings, trailing newline). ``system`` is one of ``SYSTEM_CHOICES``."""
    if system not in SYSTEM_CHOICES:
        raise ValueError(f"system must be one of {SYSTEM_CHOICES}, not {system!r}")
    text = {"bql": SYSTEM_PROMPT, "qwen": QWEN_DEFAULT_SYSTEM, "none": None}[system]
    if text is not None and '"""' in text:
        raise ValueError("the system prompt contains a triple quote, which cannot be embedded in a Modelfile")
    out = (
        f"FROM {gguf}\n"
        f'TEMPLATE """{TEMPLATE}"""\n'
        "PARAMETER temperature 0\n"
        'PARAMETER stop "<|im_end|>"\n'
    )
    if text is not None:
        out += f'SYSTEM """{text}"""\n'
    return out
