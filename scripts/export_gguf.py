#!/usr/bin/env python
"""Merge a LoRA adapter saved by scripts/train.py into the base model and export it as a GGUF for Ollama.

Uses Unsloth's own ``save_pretrained_gguf``, which merges the adapter into 16-bit weights, then converts and
quantizes with llama.cpp (a prebuilt copy at ``~/.unsloth/llama.cpp`` by default, or ``UNSLOTH_LLAMA_CPP_PATH``; it
installs one if none is found, which needs network access and a working C/C++ toolchain).

Run it with the same Python environment used for training. Needs several GB of temporary disk space for the merged
16-bit model, in addition to the GGUF itself (~4.5 GB for Q4_K_M on a 7B model).

Usage:
    python scripts/export_gguf.py                                      # outputs/bql-lora/adapter -> outputs/bql-lora/gguf
    python scripts/export_gguf.py --adapter path/to/adapter --quant q5_k_m
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

# Unsloth patches transformers/trl when it is imported, so it has to come before them.
from unsloth import FastLanguageModel  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--adapter", type=Path, default=ROOT / "outputs" / "bql-lora" / "adapter", help="the LoRA adapter directory saved by train.py")
    ap.add_argument("--out", type=Path, default=None, help="default: <adapter's parent>/gguf")
    ap.add_argument("--quant", default="q4_k_m", help="llama.cpp quantization method (default: %(default)s)")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    args = ap.parse_args()

    if not args.adapter.exists():
        print(f"adapter not found: {args.adapter}", file=sys.stderr)
        return 2
    out = args.out or args.adapter.parent / "gguf"

    print(f"loading {args.adapter} ...")
    model, tokenizer = FastLanguageModel.from_pretrained(str(args.adapter), max_seq_length=args.max_seq_length, dtype=None, load_in_4bit=True)

    print(f"merging and exporting {args.quant} GGUF to {out} ...")
    result = model.save_pretrained_gguf(str(out), tokenizer, quantization_method=args.quant)

    gguf_files = [Path(p) for p in result.get("gguf_files", [])] if isinstance(result, dict) else []
    if not gguf_files:
        gguf_files = sorted(out.glob("*.gguf"))
    if not gguf_files:
        print("export finished but no .gguf file was found; check the output above", file=sys.stderr)
        return 1

    gguf = gguf_files[0]
    print(f"\nwrote {gguf} ({gguf.stat().st_size / 2**30:.2f} GiB)")
    print("\nNext:\n"
          f'  python scripts/make_modelfile.py --gguf "{gguf}" -o Modelfile\n'
          "  ollama create bql -f Modelfile\n"
          '  ollama run bql "How much did I spend on groceries in 2024?"')
    return 0


if __name__ == "__main__":
    sys.exit(main())
