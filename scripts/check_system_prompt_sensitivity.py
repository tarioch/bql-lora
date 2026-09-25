#!/usr/bin/env python
"""Does a newly retrained model still answer in BQL regardless of the system message it's given?

Training data carries no system message (see README); Qwen's chat template fills in its own default line for a
conversation with none, and that is what ``ollama/Modelfile`` bakes in as ``SYSTEM``. This is a regression check for
that: it builds two temporary Ollama models from one GGUF - the real Modelfile (``default``) and the same file with
no ``SYSTEM`` line at all (``none``, a robustness check for a caller that supplies its own) - asks each the same
questions, and reports how many answers are BQL (contain a ```sql block) and how many match the ``default``
variant's answer exactly. Your own models are not touched: the temporary ones are removed at the end.

A model trained the way this repo does it should answer in BQL under both variants; a large gap between them is a
sign something about the training or the Modelfile has drifted.

Usage:
    python scripts/check_system_prompt_sensitivity.py --gguf path/to/model.Q4_K_M.gguf
    python scripts/check_system_prompt_sensitivity.py --gguf model.gguf --ollama "C:/path/to/ollama.exe"
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from bql_lora.modelfile import render_modelfile  # noqa: E402

VARIANTS = ("default", "none")


def render_variant(gguf: str, variant: str) -> str:
    text = render_modelfile(gguf)
    if variant == "none":
        text = re.sub(r'(?m)^SYSTEM """.*"""\n?', "", text)
    return text

DEFAULT_QUESTIONS = [
    "What is my net worth in GBP?",
    "How much did I spend on groceries in 2024?",
    "Show the last 10 transactions with Rewe",
    "What is my average monthly spending on all expenses in USD?",
    "Which expense accounts have a total above 500 CAD?",
    "Show all transactions tagged trip-rome-2024",
    "What does date_trunc do in BQL?",
    "Hello, how are you today?",
]
SQL_BLOCK = re.compile(r"```sql\s*(.*?)\s*```", re.S)
PREFIX = "bql-sysprobe-"


def chat(host: str, model: str, question: str, num_predict: int) -> str:
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": question}], "stream": False,
                       "options": {"seed": 1, "num_predict": num_predict}}).encode("utf-8")
    req = urllib.request.Request(f"{host}/api/chat", data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=300) as resp:
        return json.loads(resp.read().decode("utf-8"))["message"]["content"]


def ollama(exe: str, *args: str, cwd: str | None = None) -> None:
    # ollama prints Unicode progress bars; decode explicitly so Windows' default code page cannot choke on them
    result = subprocess.run([exe, *args], cwd=cwd, capture_output=True, text=True, encoding="utf-8", errors="replace")
    if result.returncode != 0:
        raise RuntimeError(f"`ollama {' '.join(args)}` failed: {(result.stderr or result.stdout).strip()}")


def one_line(text: str, limit: int = 150) -> str:
    flat = " | ".join(text.split("\n"))
    flat = re.sub(r"\s+", " ", flat)
    return flat if len(flat) <= limit else flat[:limit] + "..."


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", required=True, help="the GGUF of your fine-tuned model")
    ap.add_argument("--ollama", default="ollama", help="the ollama executable (default: ollama on PATH)")
    ap.add_argument("--host", default="http://localhost:11434")
    ap.add_argument("--questions-file", type=Path, help="one question per line instead of the built-in set")
    ap.add_argument("--num-predict", type=int, default=220, help="max tokens per answer")
    args = ap.parse_args()

    gguf = Path(args.gguf).resolve()
    if not gguf.exists():
        print(f"GGUF not found: {gguf}", file=sys.stderr)
        return 2
    questions = ([q.strip() for q in args.questions_file.read_text(encoding="utf-8").splitlines() if q.strip()]
                 if args.questions_file else DEFAULT_QUESTIONS)

    created: list[str] = []
    answers: dict[str, list[str]] = {}
    try:
        with tempfile.TemporaryDirectory() as tmp:
            for variant in VARIANTS:
                name = PREFIX + variant
                path = Path(tmp) / f"Modelfile.{variant}"
                path.write_bytes(render_variant(str(gguf), variant).encode("utf-8"))
                ollama(args.ollama, "create", name, "-f", str(path), cwd=tmp)
                created.append(name)
        for variant in VARIANTS:
            name = PREFIX + variant
            print(f"asking {len(questions)} questions with the '{variant}' Modelfile ...", file=sys.stderr)
            answers[variant] = [chat(args.host, name, q, args.num_predict) for q in questions]
            ollama(args.ollama, "stop", name)
    except (RuntimeError, urllib.error.URLError, OSError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1
    finally:
        for name in created:
            try:
                ollama(args.ollama, "rm", name)
            except RuntimeError as e:
                print(f"warning: could not remove {name}: {e}", file=sys.stderr)

    ref = VARIANTS[0]
    for i, q in enumerate(questions):
        print(f"\n[{i + 1}] {q}")
        for variant in VARIANTS:
            answer = answers[variant][i]
            sql = SQL_BLOCK.search(answer)
            if variant == ref:
                tag = "BQL (sql block)" if sql else "no sql block"
            elif answer == answers[ref][i]:
                tag = f"identical to '{ref}'"
            else:
                tag = "different sql block" if sql else "no sql block"
            print(f"   {variant:8s} {tag:22s} {one_line(answer)}")

    print(f"\n{'variant':10s}{'answers with a sql block':28s}identical to '{ref}'")
    for variant in VARIANTS:
        n_sql = sum(1 for a in answers[variant] if SQL_BLOCK.search(a))
        n_same = sum(1 for a, b in zip(answers[variant], answers[ref]) if a == b)
        print(f"{variant:10s}{n_sql:>2d} of {len(questions):<22d}{n_same} of {len(questions)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
