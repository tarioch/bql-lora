#!/usr/bin/env python
"""Write (or check) the Ollama Modelfile generated from the training setup.

Usage:
    python scripts/make_modelfile.py                          # write ollama/Modelfile
    python scripts/make_modelfile.py --gguf my.Q4_K_M.gguf -o /path/to/Modelfile
    python scripts/make_modelfile.py --check                  # exit 1 if ollama/Modelfile is out of date
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from bql_lora.modelfile import DEFAULT_GGUF, render_modelfile  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gguf", default=DEFAULT_GGUF, help="GGUF file the Modelfile's FROM line points at (default: %(default)s)")
    ap.add_argument("-o", "--out", type=Path, default=ROOT / "ollama" / "Modelfile")
    ap.add_argument("--check", action="store_true", help="do not write; fail if the file on disk differs from what would be generated")
    args = ap.parse_args()

    text = render_modelfile(args.gguf)
    if args.check:
        current = args.out.read_bytes() if args.out.exists() else b""
        if current != text.encode("utf-8"):
            print(f"{args.out} is out of date; run scripts/make_modelfile.py", file=sys.stderr)
            return 1
        print(f"{args.out} is up to date")
        return 0

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_bytes(text.encode("utf-8"))  # bytes: never let the platform turn LF into CRLF
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
