#!/usr/bin/env python
"""Fine-tune a Qwen2.5-Coder Instruct model on this dataset with Unsloth (QLoRA). Everything that decides what the
model learns is on the command line or printed, instead of hidden in a UI:

* the fully rendered first training example (so you can see exactly which system message the model is trained with),
* token length statistics, with a warning if any example would be truncated,
* loss only on the assistant's answer (the question, and any system message, are masked out).

Run it with the Python environment that has Unsloth installed, from anywhere:

    python scripts/train.py                                 # data/, 2 epochs
    python scripts/train.py --max-steps 2                   # smoke test: two optimizer steps

Then bake the matching system message into the Modelfile and check the result (the commands are printed at the end).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Unsloth patches transformers/trl when it is imported, so it has to come before them.
from unsloth import FastLanguageModel, is_bfloat16_supported  # noqa: E402
from unsloth.chat_templates import train_on_responses_only  # noqa: E402

import torch  # noqa: E402
from datasets import Dataset  # noqa: E402
from trl import SFTConfig, SFTTrainer  # noqa: E402

DEFAULT_MODEL = "unsloth/qwen2.5-coder-7b-instruct-bnb-4bit"
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]
# ChatML markers used by Qwen: the loss is computed only on what follows the assistant marker.
INSTRUCTION_PART = "<|im_start|>user\n"
RESPONSE_PART = "<|im_start|>assistant\n"


def read_examples(path: Path) -> list[list[dict]]:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line)["messages"] for line in f if line.strip()]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", type=Path, default=ROOT / "data", help="folder with train.jsonl and val.jsonl (default: data)")
    ap.add_argument("--model", default=DEFAULT_MODEL)
    ap.add_argument("--out", type=Path, default=ROOT / "outputs" / "bql-lora")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=-1, help="stop after this many optimizer steps (overrides --epochs); for smoke tests")
    ap.add_argument("--max-seq-length", type=int, default=1024)
    ap.add_argument("--batch-size", type=int, default=4, help="per-device batch size")
    ap.add_argument("--grad-accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--rank", type=int, default=32, help="LoRA rank (alpha is set equal to it)")
    ap.add_argument("--eval-steps", type=int, default=100)
    ap.add_argument("--seed", type=int, default=3407)
    ap.add_argument("--no-response-only", action="store_true", help="also train on the question and system tokens (not recommended)")
    ap.add_argument("--merge", action="store_true", help="also save a merged 16-bit model (about 15 GB)")
    ap.add_argument("--gguf", action="store_true", help="also export a Q4_K_M GGUF via Unsloth (needs llama.cpp tooling; not covered by the smoke test)")
    args = ap.parse_args()

    for name in ("train.jsonl", "val.jsonl"):
        if not (args.data / name).exists():
            print(f"missing {args.data / name}", file=sys.stderr)
            return 2

    model, tokenizer = FastLanguageModel.from_pretrained(args.model, max_seq_length=args.max_seq_length, dtype=None, load_in_4bit=True)
    model = FastLanguageModel.get_peft_model(
        model, r=args.rank, lora_alpha=args.rank, lora_dropout=0, bias="none", target_modules=LORA_TARGETS,
        use_gradient_checkpointing="unsloth", random_state=args.seed)

    # Render every conversation with the model's own chat template. A conversation without a system message gets
    # the template's default system line, which is what the Modelfile must then also contain (see the end).
    train_msgs, val_msgs = read_examples(args.data / "train.jsonl"), read_examples(args.data / "val.jsonl")
    render = lambda msgs: tokenizer.apply_chat_template(msgs, tokenize=False)  # noqa: E731
    train_text, val_text = [render(m) for m in train_msgs], [render(m) for m in val_msgs]

    print("\n=== first training example, exactly as the model will see it ===")
    print(train_text[0])
    print("=== end of example ===\n")
    if train_msgs[0][0]["role"] == "system":
        print("WARNING: this data has a system message; ollama/Modelfile bakes in Qwen's default line and expects "
              "training data to have none (see scripts/generate_dataset.py)", file=sys.stderr)
    lengths = sorted(len(ids) for ids in tokenizer(train_text, add_special_tokens=False)["input_ids"])
    p99 = lengths[int(len(lengths) * 0.99)]
    print(f"{len(train_text)} train / {len(val_text)} validation examples; tokens per example: median {lengths[len(lengths) // 2]}, "
          f"p99 {p99}, max {lengths[-1]}")
    if lengths[-1] > args.max_seq_length:
        print(f"WARNING: {sum(n > args.max_seq_length for n in lengths)} examples are longer than --max-seq-length "
              f"{args.max_seq_length} and will be truncated", file=sys.stderr)

    bf16 = is_bfloat16_supported()
    config = SFTConfig(
        output_dir=str(args.out / "checkpoints"), dataset_text_field="text", max_length=args.max_seq_length, packing=False, dataset_num_proc=1,
        per_device_train_batch_size=args.batch_size, per_device_eval_batch_size=args.batch_size, gradient_accumulation_steps=args.grad_accum,
        num_train_epochs=args.epochs, max_steps=args.max_steps, learning_rate=args.lr, lr_scheduler_type="linear", warmup_steps=0.03,  # a float below 1 means 3% of all steps
        weight_decay=0.01, optim="adamw_8bit", bf16=bf16, fp16=not bf16, logging_steps=10, eval_strategy="steps", eval_steps=args.eval_steps,
        save_strategy="no", report_to="none", seed=args.seed)
    trainer = SFTTrainer(model=model, processing_class=tokenizer, args=config,
                         train_dataset=Dataset.from_dict({"text": train_text}), eval_dataset=Dataset.from_dict({"text": val_text}))
    if not args.no_response_only:
        trainer = train_on_responses_only(trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART)
    first = trainer.train_dataset[0]
    if "labels" in first:
        # Proof of what the loss is computed on: labels are -100 everywhere the model is not trained.
        supervised = [t for t, label in zip(first["input_ids"], first["labels"]) if label != -100]
        print(f"=== the {len(supervised)} of {len(first['input_ids'])} tokens of the first example that receive loss ===")
        print(tokenizer.decode(supervised))
        print("=== end ===\n")

    result = trainer.train()
    metrics = trainer.evaluate()
    print(f"\ntrain loss {result.training_loss:.4f}   validation loss {metrics['eval_loss']:.4f}")
    if torch.cuda.is_available():
        print(f"peak GPU memory reserved: {torch.cuda.max_memory_reserved() / 2**30:.2f} GiB of "
              f"{torch.cuda.get_device_properties(0).total_memory / 2**30:.2f} GiB")

    adapter = args.out / "adapter"
    model.save_pretrained(str(adapter))
    tokenizer.save_pretrained(str(adapter))
    (args.out / "training_config.json").write_text(json.dumps({
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "train_loss": result.training_loss, "eval_loss": metrics["eval_loss"]}, indent=2), encoding="utf-8")
    print(f"saved the LoRA adapter to {adapter}")
    if args.merge:
        model.save_pretrained_merged(str(args.out / "merged"), tokenizer, save_method="merged_16bit")
    if args.gguf:
        model.save_pretrained_gguf(str(args.out / "gguf"), tokenizer, quantization_method="q4_k_m")

    print("\nNext: export the merged model as a GGUF (Q4_K_M), then\n"
          "  python scripts/make_modelfile.py --gguf <your.gguf> -o Modelfile\n"
          "  ollama create bql -f Modelfile\n"
          "  python scripts/check_system_prompt_sensitivity.py --gguf <your.gguf>")
    return 0


if __name__ == "__main__":
    sys.exit(main())
