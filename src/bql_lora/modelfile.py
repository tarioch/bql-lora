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

# Plain ChatML as used by Qwen2.5: <|im_start|>role\ncontent<|im_end|>\n ... <|im_start|>assistant\n
TEMPLATE = (
    "{{- range .Messages }}<|im_start|>{{ .Role }}\n"
    "{{ .Content }}<|im_end|>\n"
    "{{ end }}<|im_start|>assistant\n"
)


def render_modelfile(gguf: str = DEFAULT_GGUF) -> str:
    """The Modelfile text (LF line endings, trailing newline)."""
    if '"""' in SYSTEM_PROMPT:
        raise ValueError("SYSTEM_PROMPT contains a triple quote, which cannot be embedded in a Modelfile")
    return (
        f"FROM {gguf}\n"
        f'TEMPLATE """{TEMPLATE}"""\n'
        "PARAMETER temperature 0\n"
        'PARAMETER stop "<|im_end|>"\n'
        f'SYSTEM """{SYSTEM_PROMPT}"""\n'
    )
