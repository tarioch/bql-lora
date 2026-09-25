"""Render the Ollama Modelfile for a model fine-tuned on this dataset.

The Modelfile is generated rather than hand-edited so it cannot silently drift from what training actually used.

Training examples carry no system message (see README, "The system prompt: trigger or baked in?"): a constant one
gives a fine-tune something to key on instead of learning the behaviour for any input. Qwen's chat template
supplies its own default system line for a conversation with none, which is what the model was trained with, so
that line has to be the Modelfile's ``SYSTEM`` too (measured directly against Ollama: without it, only 1 of 8 test
questions came back as BQL; with it, 8 of 8 did).

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

DEFAULT_GGUF = "qwen2.5-coder-7b-instruct.Q4_K_M.gguf"

# Qwen's own default system message: what its chat template injects for a conversation with no system message of
# its own, at training time and at inference. The Modelfile has to bake this in explicitly, since Ollama's runtime
# has no equivalent fallback of its own for a template (like ours) that does not reference `.System`.
QWEN_DEFAULT_SYSTEM = "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."

# Plain ChatML as used by Qwen2.5: <|im_start|>role\ncontent<|im_end|>\n ... <|im_start|>assistant\n
TEMPLATE = (
    "{{- range .Messages }}<|im_start|>{{ .Role }}\n"
    "{{ .Content }}<|im_end|>\n"
    "{{ end }}<|im_start|>assistant\n"
)


def render_modelfile(gguf: str = DEFAULT_GGUF) -> str:
    """The Modelfile text (LF line endings, trailing newline)."""
    return (
        f"FROM {gguf}\n"
        f'TEMPLATE """{TEMPLATE}"""\n'
        "PARAMETER temperature 0\n"
        'PARAMETER stop "<|im_end|>"\n'
        f'SYSTEM """{QWEN_DEFAULT_SYSTEM}"""\n'
    )
