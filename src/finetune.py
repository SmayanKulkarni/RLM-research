"""
finetune.py — QLoRA Fine-Tuning with Unsloth for RLM-MCP Tool Selection.

Loads verified REPL trajectories and fine-tunes qwen3.5:4b (HuggingFace version)
with QLoRA via Unsloth. Trains ONLY on assistant turns (system/user turns are masked).

Architecture:
    Base model    : Qwen/Qwen3.5-4B-Instruct (4-bit NF4 via bitsandbytes)
    LoRA rank     : 32 (r=32, alpha=32)
    Target modules: q_proj, k_proj, v_proj, o_proj, gate_proj, up_proj, down_proj
    Trainer       : TRL SFTTrainer + Unsloth patches (2x speed, ~60% VRAM reduction)

Output:
    finetuned/qwen3.5-4b-rlm-lora/  — LoRA adapter (for deployment with Unsloth)
    finetuned/qwen3.5-4b-rlm-merged/ — Optional merged model compatible with Ollama

Usage:
    conda run -n astro python -m src.finetune \\
        --data data/trajectories_verified.jsonl \\
        --output finetuned/qwen3.5-4b-rlm-lora \\
        --epochs 3

    # Resume from checkpoint:
    conda run -n astro python -m src.finetune --resume-from finetuned/qwen3.5-4b-rlm-lora/checkpoint-500
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Model Configuration ───────────────────────────────────────────────────────

# HuggingFace model IDs for qwen3.5:4b only.
# IMPORTANT: Do not add non-qwen3.5 fallbacks here.
MODEL_CANDIDATES = [
    "Qwen/Qwen3.5-4B",                           # Canonical HF base model
    "Qwen/Qwen3.5-4B-Instruct",                  # Optional instruct variant (if available)
]

# LoRA hyperparameters
LORA_R = 32
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LORA_TARGET_MODULES = [
    "q_proj", "k_proj", "v_proj", "o_proj",
    "gate_proj", "up_proj", "down_proj"
]

# ── Dataset Loading ───────────────────────────────────────────────────────────

def load_verified_trajectories(path: Path) -> list[dict]:
    """Load verified trajectories from JSONL file."""
    if not path.exists():
        print(f"ERROR: Verified trajectories not found at {path}")
        print("Run generate_trajectories.py then verify_trajectories.py first.")
        sys.exit(1)

    trajectories = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))

    print(f"Loaded {len(trajectories)} verified trajectories from {path}")
    return trajectories


def load_query_split_lookup() -> dict[str, dict]:
    """Load query split/category metadata from the leakage-safe manifests."""
    query_dir = PROJECT_ROOT / "test_data" / "queries"
    lookup: dict[str, dict] = {}

    all_path = query_dir / "all_queries.json"
    if all_path.exists():
        with open(all_path) as file_handle:
            for row in json.load(file_handle):
                query = row.get("query")
                if query:
                    lookup[query] = {
                        "query_category": row.get("category", "unknown"),
                        "difficulty": row.get("difficulty", "unknown"),
                    }

    for split_name in ("train", "dev", "test"):
        split_path = query_dir / f"{split_name}_queries.json"
        if not split_path.exists():
            continue
        with open(split_path) as file_handle:
            for row in json.load(file_handle):
                query = row.get("query")
                if not query:
                    continue
                base = lookup.setdefault(query, {})
                base["split"] = split_name
                base.setdefault("query_category", row.get("category", "unknown"))
                base.setdefault("difficulty", row.get("difficulty", "unknown"))

    return lookup


def enrich_trajectory_metadata(trajectory: dict, query_split_lookup: dict[str, dict]) -> dict:
    """Ensure every trajectory carries split/category metadata for partitioning."""
    row = dict(trajectory)
    metadata = dict(row.get("metadata", {}))
    query_meta = query_split_lookup.get(row.get("query", ""), {})

    metadata["split"] = metadata.get("split") or query_meta.get("split") or "train"
    metadata["query_category"] = (
        metadata.get("query_category") or query_meta.get("query_category") or "synthetic_selection"
    )
    metadata["difficulty"] = metadata.get("difficulty") or query_meta.get("difficulty") or "synthetic"
    row["metadata"] = metadata
    return row


def format_for_training(trajectory: dict, tokenizer) -> dict | None:
    """
    Convert a trajectory's 'messages' list into a tokenized training example.

    Uses the model's chat template to format the conversation.
    'messages' follows OpenAI format: [{"role": ..., "content": ...}]

    Returns dict with 'text' key for SFTTrainer, or None on error.
    """
    messages = trajectory.get("messages", [])
    if not messages:
        return None

    try:
        # apply_chat_template converts messages → formatted string
        # add_generation_prompt=False because our last turn is already assistant
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}
    except Exception as e:
        print(f"  [formatting error]: {e}")
        return None


def _count_tokens(tokenizer, text: str) -> int:
    """Count tokens robustly across tokenizer/processor variants."""
    # Standard HF tokenizer path
    if hasattr(tokenizer, "encode"):
        return len(tokenizer.encode(text))

    # Processor-like path (returns dict/tensor with input_ids)
    if callable(tokenizer):
        encoded = tokenizer(text)
        if isinstance(encoded, dict) and "input_ids" in encoded:
            input_ids = encoded["input_ids"]
            # input_ids may be a flat list or batch list
            if isinstance(input_ids, list) and input_ids and isinstance(input_ids[0], list):
                return len(input_ids[0])
            if isinstance(input_ids, list):
                return len(input_ids)

    raise TypeError(
        f"Unsupported tokenizer type for token counting: {type(tokenizer).__name__}"
    )


def build_hf_dataset(trajectories: list[dict], tokenizer, max_length: int):
    """Build a HuggingFace Dataset from the verified trajectories."""
    from datasets import Dataset

    formatted = []
    skipped = 0

    for traj in trajectories:
        item = format_for_training(traj, tokenizer)
        if item is not None:
            # Pre-filter: skip sequences that are too long to avoid OOM
            token_count = _count_tokens(tokenizer, item["text"])
            if token_count <= max_length:
                formatted.append(item)
            else:
                skipped += 1

    print(f"  Formatted: {len(formatted)} / {len(trajectories)} trajectories")
    if skipped > 0:
        print(f"  Skipped {skipped} trajectories exceeding max_length={max_length}")

    return Dataset.from_list(formatted)


def partition_trajectories(
    trajectories: list[dict],
    train_split: str,
    eval_split: str | None,
    allow_random_split_fallback: bool,
) -> tuple[list[dict], list[dict]]:
    """Partition trajectories using explicit split metadata when available."""
    train_rows: list[dict] = []
    eval_rows: list[dict] = []

    for row in trajectories:
        split = row.get("metadata", {}).get("split", "train")
        if split == train_split:
            train_rows.append(row)
        elif eval_split and split == eval_split:
            eval_rows.append(row)

    if train_rows and eval_rows:
        return train_rows, eval_rows

    if not allow_random_split_fallback:
        missing = []
        if not train_rows:
            missing.append(f"train split '{train_split}'")
        if eval_split and not eval_rows:
            missing.append(f"eval split '{eval_split}'")
        print(
            "ERROR: Could not build explicit split-aware datasets for "
            + ", ".join(missing)
            + ". Use a corpus that includes split metadata or pass --allow-random-split-fallback."
        )
        sys.exit(1)

    print("⚠ Falling back to random 90/10 split because explicit split metadata is incomplete")
    shuffled = list(trajectories)
    if len(shuffled) < 2:
        return shuffled, []

    from random import Random
    rng = Random(42)
    rng.shuffle(shuffled)
    cut = max(1, int(round(len(shuffled) * 0.9)))
    cut = min(cut, len(shuffled) - 1)
    return shuffled[:cut], shuffled[cut:]


# ── Model Loading ─────────────────────────────────────────────────────────────

def load_model_and_tokenizer(
    model_id: str | None,
    max_seq_length: int,
    load_in_4bit: bool = True,
):
    """
    Load the model and tokenizer using Unsloth's FastLanguageModel.

    Tries qwen3.5:4b model candidates in order if no explicit model_id is given.
    """
    from unsloth import FastLanguageModel

    candidates = [model_id] if model_id else MODEL_CANDIDATES

    for candidate in candidates:
        candidate_lc = candidate.lower()
        if "qwen3.5" not in candidate_lc or "4b" not in candidate_lc:
            print(f"  Skipping non-qwen3.5-4b candidate: {candidate}")
            continue

        print(f"\nAttempting to load: {candidate}")
        try:
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=candidate,
                max_seq_length=max_seq_length,
                load_in_4bit=load_in_4bit,
                dtype=None,  # Auto-detect (bfloat16 on Ampere+)
            )

            # Some Qwen variants return a processor wrapper (e.g., Qwen3VLProcessor)
            # while SFT expects a tokenizer-like object.
            if hasattr(tokenizer, "tokenizer"):
                print(f"  Unwrapping processor -> tokenizer ({type(tokenizer).__name__})")
                tokenizer = tokenizer.tokenizer

            print(f"✓ Loaded: {candidate}")
            return model, tokenizer, candidate
        except Exception as e:
            print(f"  Failed ({type(e).__name__}): {e}")
            continue

    print("ERROR: Could not load Qwen3.5-4B model. Check internet connection, HuggingFace access, and model availability.")
    sys.exit(1)


def apply_lora(model, tokenizer):
    """Apply LoRA adapters using Unsloth's get_peft_model (optimised for speed)."""
    from unsloth import FastLanguageModel

    model = FastLanguageModel.get_peft_model(
        model,
        r=LORA_R,
        target_modules=LORA_TARGET_MODULES,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        bias="none",
        use_gradient_checkpointing="unsloth",  # Unsloth's long-context checkpointing
        random_state=42,
        use_rslora=False,
        loftq_config=None,
    )

    # Print trainable params
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nModel parameters:")
    print(f"  Total     : {total:,}")
    print(f"  Trainable : {trainable:,} ({100*trainable/total:.2f}%)")

    return model


def apply_assistant_only_loss_mask(trainer, args):
    """Apply assistant-only masking using Unsloth response-only utility."""
    try:
        from unsloth.chat_templates import train_on_responses_only
        trainer = train_on_responses_only(
            trainer,
            instruction_part=args.user_marker,
            response_part=args.assistant_marker,
        )
        print("✓ Assistant-only loss masking enabled")
        return trainer
    except Exception as e:
        if args.allow_unmasked_fallback:
            print(f"⚠ Could not enable assistant-only masking: {type(e).__name__}: {e}")
            print("  Continuing without masking due to --allow-unmasked-fallback")
            return trainer
        raise RuntimeError(
            "Assistant-only masking failed. Use --allow-unmasked-fallback to continue without masking. "
            f"Root cause: {type(e).__name__}: {e}"
        )


def print_label_mask_stats(trainer, max_batches: int = 1):
    """Print ratio of non-masked labels to verify assistant-only masking."""
    try:
        dataloader = trainer.get_train_dataloader()
        inspected = 0
        total = 0
        active = 0
        for batch in dataloader:
            labels = batch.get("labels")
            if labels is None:
                break
            total += labels.numel()
            active += int((labels != -100).sum().item())
            inspected += 1
            if inspected >= max_batches:
                break
        if total > 0:
            ratio = active / total
            print(f"  Label mask check: active={active} / total={total} ({ratio:.2%})")
    except Exception as e:
        print(f"  Label mask check skipped: {type(e).__name__}: {e}")


# ── Training ──────────────────────────────────────────────────────────────────

def train(
    model,
    tokenizer,
    train_dataset,
    eval_dataset,
    output_dir: Path,
    args,
):
    """Run SFT training using TRL's SFTTrainer with Unsloth patches."""
    from trl import SFTTrainer, SFTConfig

    output_dir.mkdir(parents=True, exist_ok=True)

    train_ds = train_dataset
    eval_ds = eval_dataset

    print(f"\nDataset split:")
    print(f"  Train : {len(train_ds)}")
    print(f"  Eval  : {len(eval_ds)}")

    if len(train_ds) == 0:
        print("ERROR: Empty train dataset after split partitioning")
        sys.exit(1)
    if len(eval_ds) == 0:
        print("ERROR: Empty eval dataset after split partitioning")
        sys.exit(1)

    # Compute steps
    effective_batch = args.per_device_batch_size * args.grad_accumulation
    steps_per_epoch = max(1, len(train_ds) // effective_batch)
    total_steps = steps_per_epoch * args.epochs
    # Eval & save once per epoch to minimise disk usage
    eval_steps = max(1, steps_per_epoch)
    save_steps = max(1, steps_per_epoch)
    log_steps = max(1, steps_per_epoch // 4)  # 4 log lines per epoch

    print(f"\nTraining plan:")
    print(f"  Epochs       : {args.epochs}")
    print(f"  Batch size   : {args.per_device_batch_size} × {args.grad_accumulation} accum = {effective_batch}")
    print(f"  Steps/epoch  : {steps_per_epoch}")
    print(f"  Total steps  : {total_steps}")
    print(f"  Eval every   : {eval_steps} steps (once/epoch)")
    print(f"  Save every   : {save_steps} steps (once/epoch)")

    training_args = SFTConfig(
        output_dir=str(output_dir),
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.per_device_batch_size,
        per_device_eval_batch_size=args.per_device_batch_size,
        gradient_accumulation_steps=args.grad_accumulation,
        warmup_ratio=0.05,
        learning_rate=args.learning_rate,
        lr_scheduler_type="cosine",
        weight_decay=0.01,
        fp16=False,
        bf16=True,            # RTX 4070 SUPER supports bfloat16
        optim="adamw_8bit",   # 8-bit AdamW via bitsandbytes (lower VRAM)
        logging_steps=log_steps,
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="steps",
        save_steps=save_steps,
        save_total_limit=1,           # Keep ONLY 1 checkpoint (storage-safe)
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none",    # No wandb/tensorboard by default
        seed=42,
        max_seq_length=args.max_seq_length,
        # SFT-specific: only compute loss on assistant tokens
        dataset_text_field="text",
        packing=False,       # Don't pack — preserve conversation boundaries
    )

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        args=training_args,
    )

    if not args.no_assistant_only_loss:
        trainer = apply_assistant_only_loss_mask(trainer, args)
        print_label_mask_stats(trainer, max_batches=1)

    # Resume from checkpoint if specified
    resume_from = args.resume_from if hasattr(args, "resume_from") and args.resume_from else None

    print(f"\n{'='*60}")
    print(f"Starting fine-tuning...")
    if resume_from:
        print(f"Resuming from: {resume_from}")
    print(f"{'='*60}\n")

    trainer_stats = trainer.train(resume_from_checkpoint=resume_from)

    # Print training summary
    print(f"\n{'='*60}")
    print(f"TRAINING COMPLETE")
    print(f"  Runtime  : {trainer_stats.metrics.get('train_runtime', 0):.1f}s")
    print(f"  Steps    : {trainer_stats.metrics.get('train_steps_per_second', 0):.2f} steps/s")
    print(f"  Loss     : {trainer_stats.metrics.get('train_loss', 0):.4f}")
    print(f"{'='*60}")

    # ── Cleanup intermediate checkpoints to free disk ──────────────────
    import shutil
    for ckpt in sorted(output_dir.glob("checkpoint-*")):
        print(f"  Removing checkpoint: {ckpt.name}")
        shutil.rmtree(ckpt, ignore_errors=True)
    print("  ✓ Intermediate checkpoints cleaned")

    return trainer


def save_model(model, tokenizer, output_dir: Path, merge: bool = False):
    """Save the LoRA adapter. Optionally merge with base and save full model."""
    # Always save LoRA adapter (small, fast, used with Unsloth for inference)
    lora_path = output_dir
    print(f"\nSaving LoRA adapter to: {lora_path}")
    model.save_pretrained(str(lora_path))
    tokenizer.save_pretrained(str(lora_path))
    print("✓ LoRA adapter saved")

    if merge:
        # Merge LoRA weights into base model → can then convert to GGUF for Ollama
        merged_path = output_dir.parent / (output_dir.name + "-merged")
        print(f"\nMerging LoRA into base model → {merged_path}")
        merged_path.mkdir(parents=True, exist_ok=True)
        from unsloth import FastLanguageModel
        model.save_pretrained_merged(
            str(merged_path),
            tokenizer,
            save_method="merged_16bit",
        )
        print("✓ Merged model saved (16-bit)")

        # Also export to GGUF for Ollama compatibility
        gguf_path = output_dir.parent / (output_dir.name + "-gguf")
        print(f"\nExporting to GGUF (Q4_K_M) → {gguf_path}")
        try:
            model.save_pretrained_gguf(
                str(gguf_path),
                tokenizer,
                quantization_method="q4_k_m",
            )
            print("✓ GGUF export complete — importable into Ollama via:")
            print(f"   ollama create rlm-mcp-qwen -f {gguf_path}/Modelfile")
        except Exception as e:
            print(f"  GGUF export failed (non-critical): {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="QLoRA fine-tune Qwen3.5-4B for RLM-MCP tool selection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic run
  conda run -n astro python -m src.finetune

  # Custom data and output
  conda run -n astro python -m src.finetune \\
      --data data/trajectories_verified.jsonl \\
      --output finetuned/qwen3.5-4b-rlm-lora

  # Quick test (1 epoch)
  conda run -n astro python -m src.finetune --epochs 1 --max-seq-length 1024

  # Merge + export GGUF for Ollama after training
  conda run -n astro python -m src.finetune --merge-and-export
        """,
    )
    parser.add_argument(
        "--data", type=str, default="data/trajectories_verified.jsonl",
        help="Path to verified trajectories JSONL",
    )
    parser.add_argument(
        "--output", type=str, default="finetuned/qwen3.5-4b-rlm-lora",
        help="Output directory for LoRA adapter",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help=f"HuggingFace model ID (default: tries {MODEL_CANDIDATES[0]} first)",
    )
    parser.add_argument(
        "--epochs", type=int, default=3,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--per-device-batch-size", type=int, default=2,
        help="Per-device batch size (keep at 2 for 12GB VRAM)",
    )
    parser.add_argument(
        "--grad-accumulation", type=int, default=8,
        help="Gradient accumulation steps (effective batch = batch × accum)",
    )
    parser.add_argument(
        "--learning-rate", type=float, default=2e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--max-seq-length", type=int, default=2048,
        help="Max sequence length (tokens). Longer = more VRAM.",
    )
    parser.add_argument(
        "--merge-and-export", action="store_true",
        help="After training, merge LoRA with base model and export GGUF for Ollama",
    )
    parser.add_argument(
        "--resume-from", type=str, default=None,
        help="Resume training from a checkpoint directory",
    )
    parser.add_argument(
        "--no-assistant-only-loss", action="store_true",
        help="Disable assistant-only loss masking (not recommended).",
    )
    parser.add_argument(
        "--assistant-marker", type=str, default="<|im_start|>assistant\n",
        help="Assistant marker used for response-only masking.",
    )
    parser.add_argument(
        "--user-marker", type=str, default="<|im_start|>user\n",
        help="User marker used for response-only masking.",
    )
    parser.add_argument(
        "--allow-unmasked-fallback", action="store_true",
        help="Continue training if assistant-only masking cannot be applied.",
    )
    parser.add_argument(
        "--train-split", type=str, default="train",
        choices=["train", "dev", "test"],
        help="Split label used for the training partition.",
    )
    parser.add_argument(
        "--eval-split", type=str, default="dev",
        choices=["train", "dev", "test"],
        help="Split label used for the evaluation partition.",
    )
    parser.add_argument(
        "--allow-random-split-fallback", action="store_true",
        help="Fall back to a random 90/10 split if the dataset lacks explicit split metadata.",
    )
    parser.add_argument(
        "--no-clean-output-first", action="store_true",
        help="Do not delete existing output model directories before training.",
    )
    args = parser.parse_args()

    data_path = PROJECT_ROOT / args.data
    output_dir = PROJECT_ROOT / args.output

    if not args.no_clean_output_first:
        cleanup_targets = [
            output_dir,
            output_dir.parent / (output_dir.name + "-merged"),
            output_dir.parent / (output_dir.name + "-gguf"),
        ]
        for target in cleanup_targets:
            if target.exists():
                print(f"Cleaning existing model artifact: {target}")
                shutil.rmtree(target, ignore_errors=True)

    print(f"\n{'='*60}")
    print(f"RLM-MCP QLoRA Fine-Tuning")
    print(f"{'='*60}")
    print(f"  Data     : {data_path}")
    print(f"  Output   : {output_dir}")
    print(f"  Epochs   : {args.epochs}")
    print(f"  LR       : {args.learning_rate}")
    print(f"  Max len  : {args.max_seq_length}")
    print(f"{'='*60}\n")

    # 1. Load model
    model, tokenizer, model_id = load_model_and_tokenizer(
        model_id=args.model,
        max_seq_length=args.max_seq_length,
    )
    print(f"  Base model loaded: {model_id}")

    # 2. Apply LoRA
    model = apply_lora(model, tokenizer)

    # 3. Load + format dataset
    print("\nPreparing dataset...")
    trajectories = load_verified_trajectories(data_path)
    query_split_lookup = load_query_split_lookup()
    trajectories = [enrich_trajectory_metadata(traj, query_split_lookup) for traj in trajectories]

    split_counts: dict[str, int] = {}
    for traj in trajectories:
        split = traj.get("metadata", {}).get("split", "train")
        split_counts[split] = split_counts.get(split, 0) + 1
    print(f"  Split composition: {split_counts}")

    train_rows, eval_rows = partition_trajectories(
        trajectories,
        train_split=args.train_split,
        eval_split=args.eval_split,
        allow_random_split_fallback=args.allow_random_split_fallback,
    )

    train_dataset = build_hf_dataset(train_rows, tokenizer, max_length=args.max_seq_length)
    eval_dataset = build_hf_dataset(eval_rows, tokenizer, max_length=args.max_seq_length)

    if len(train_dataset) < 10:
        print(f"ERROR: Only {len(train_dataset)} training examples — too few to train. Generate more trajectories.")
        sys.exit(1)
    if len(eval_dataset) == 0:
        print("ERROR: No evaluation examples available after formatting.")
        sys.exit(1)

    # 4. Train
    trainer = train(model, tokenizer, train_dataset, eval_dataset, output_dir, args)

    # 5. Save
    save_model(model, tokenizer, output_dir, merge=args.merge_and_export)

    print(f"\n✓ Done! LoRA adapter saved to: {output_dir}")
    print(f"\nTo evaluate the fine-tuned model, load the LoRA adapter via Unsloth:")
    print(f"  from unsloth import FastLanguageModel")
    print(f"  model, tokenizer = FastLanguageModel.from_pretrained('{output_dir}')")
    print(f"\nOr run the baseline evaluation with the merged Ollama model:")
    print(f"  ollama create rlm-mcp-qwen -f {output_dir}-gguf/Modelfile")
    print(f"  conda run -n astro python -m src.run_baseline --all --model rlm-mcp-qwen")


if __name__ == "__main__":
    main()
