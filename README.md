# RLM MCP Research: Fine-Tuning Small Language Models for Tool Selection

## Project Overview

This repository implements an end-to-end research workflow for fine-tuning small language models (SLMs) to excel at **MCP (Model Context Protocol) tool selection** within Recursive Language Model (RLM) environments. The primary focus is improving tool-selection accuracy and parameter extraction for the Qwen 3.5-4B model through supervised fine-tuning and preference optimization.

### What is RLM?

A Recursive Language Model augments a standard LLM with a Python REPL environment where:
- The input prompt is stored as a persistent variable
- The model can write Python code to interact with its context (search, filter, chunk, extract)
- The model can recursively call sub-models on relevant text slices
- This enables processing of massive contexts (10M+ tokens) while maintaining accuracy and cost-efficiency

### The Tool Selection Problem

In an RLM environment, the first critical task is **tool discovery and selection**: given a user query and a registry of available tools, identify the right tool(s) and construct correct parameters. Our research focuses on training SLMs to:

1. **Discover tools** by searching/filtering through large registries (10–100 tools)
2. **Extract parameters** by parsing requirements from natural language queries
3. **Generate valid code** to interact with a Python REPL for verification
4. **Handle iterative correction** when tool selection fails

This is challenging for 4B models because they require careful supervision to avoid hallucinating tool names and parameters.

## Project Layout

- `src/`: Core scripts for generation, corpus building, fine-tuning, and evaluation.
- `data/`: Training corpora and preference data (`*.jsonl`).
- `test_data/`: Registries, queries, and expected selections used for evaluation.
- `results/`: Benchmark summaries and comparison reports.
- `finetuned/`: LoRA adapters and merged model artifacts.
- `logs/`: Training and evaluation logs.
- `prompts/`: System prompts for different evaluation levels (level0–level2).
- `research/`: Deep-dive documentation on RLM theory, bottleneck analysis, and findings.

## Technical Approach & Methodology

### 1. Model Architecture

**Base Model:** Qwen 3.5-4B (4-bit quantized via BitsAndBytes)
- HuggingFace identifier: `Qwen/Qwen3.5-4B`
- Quantization: NF4 with 4-bit precision to fit on consumer GPUs (RTX 4070 SUPER)

**Fine-Tuning Strategy:** LoRA (Low-Rank Adaptation) via Unsloth
- **Why LoRA:** Reduces trainable parameters from 4B to ~20M while maintaining performance
- **Unsloth optimization:** 2–3× faster training vs. standard `peft` implementation
- **Adapter location:** `finetuned/qwen3.5-4b-rlm-lora-all-complex/`

**Training Configuration:**
- Epochs: 3
- Batch size: 1 (per-device) with 16× gradient accumulation = effective batch 16
- Max sequence length: 1536 tokens
- Learning rate: 5e-4 (via Unsloth default)
- Optimizer: AdamW with low-rank updates

### 2. Evaluation Framework

The project evaluates models across **7 core metrics** to measure tool-selection capability:

| Metric | Definition | Interpretation |
|--------|-----------|-----------------|
| **TSA** (Tool Selection Accuracy) | % of queries where the correct tool is identified | Primary measure of tool discovery |
| **PC** (Parameter Correctness) | % of required parameters correctly extracted from query | Measures parameter parsing accuracy |
| **RCV** (REPL Code Validity) | % of generated Python code that executes without syntax errors | Measures code generation quality |
| **E2E** (End-to-End) | % of complete, correct, and executable tool invocations | Product of TSA × PC (failure if any step fails) |
| **Failure Rate** | % of queries that produced invalid or hallucinated responses | Inverse of success count |
| **Pass@1** | Whether first attempt succeeded (single-trial evaluation) | Deterministic baseline |
| **Pass@K** | Whether any of K trials succeeded (multi-trial robustness) | Confidence with repetition |

### 3. Evaluation Contexts: Three Levels of Difficulty

**Level 0: Zero-Shot, No REPL (Baseline)**
- All 10–100 tool schemas listed directly in the prompt context
- Model must select tool via pure reading comprehension
- Typical performance: 100% TSA (tools fit in context), 57–75% PC (parameter errors)

**Level 1: Constrained REPL Discovery**
- Models interact with a Python REPL to search tool registries by name/description
- Must write code to filter/search, execute code, parse results
- Typical performance: 60–80% TSA, 40–50% PC (REPL setup aids discovery but parameter extraction remains hard)

**Level 2: Open-Ended Iterative Search**
- Models must discover tools, extract parameters, AND handle refinement/correction
- Allows multiple REPL interactions with recovery from failures
- Typical performance: 40–70% TSA, 30–50% PC (unconstrained interaction allows recovery but requires reasoning)

### 4. Data-to-Model Pipeline

The end-to-end workflow:

```
1. TRAJECTORY GENERATION
   └─ Generate synthetic (query, tool, params, REPL output) tuples
      └ Variants per task: 11+ (with parameter noise, paraphrasing)
   
2. CORPUS BUILDING & DEDUPLICATION
   └─ Merge trajectories from multiple sources (raw, augmented, complex_coding)
   └─ Filter duplicates and low-quality entries
   └─ Balance tool/parameter coverage (min 6 per tool)
   └─ Target: 2200 training examples
   
3. VERIFICATION & SUPERVISION HARDENING
   └─ Stage 1: Basic schema validation (JSON well-formedness)
   └─ Stage 2: Tool and parameter existence checks
   └─ Stage 3: Ground-truth parameter validation against known benchmarks
   └─ Output: Clean, verified corpus with correct parameter labels

4. PREFERENCE DATA GENERATION
   └─ Generate negative/contrastive examples (wrong tool, wrong params)
   └─ Create pairwise preference data for future DPO/ORPO training
   
5. FINE-TUNING
   └─ Train LoRA adapter on verified corpus
   └─ Mask non-assistant tokens (only learn final decision)
   └─ Eval on hold-out dev split (prevent leakage)
   
6. EVALUATION & COMPARISON
   └─ Run base model on test queries
   └─ Run fine-tuned model on same test queries
   └─ Compute 7-metric deltas across registry sizes & modes
   └─ Generate human-readable reports with confidence intervals
```

### 5. Test Dataset Design

To ensure evaluation integrity:
- **Train set:** 4 simple queries (discoverable by name/description)
- **Dev set:** 4 simple queries (used for early stopping during training)
- **Test set:** 28 queries (25 simple, 3 discovery-only)

Each test query is paired with:
- Expected tool from the registry
- Required parameter set (from ground truth)
- Multiple registry sizes (small_10, medium_25, large_50, xlarge_100)
- Query modes: selection (pick a tool), discovery (find tool by behavior), mixed

This ensures that fine-tuning improvements generalize beyond memorization.

## Common Workflows

### 1. Generate Complex Coding Trajectories

```bash
conda run -n astro python -m src.generate_complex_coding_trajectories \
  --registries medium_25 large_50 xlarge_100 \
  --variants-per-task 11 \
  --output data/trajectories_complex_coding.jsonl
```

### 2. Build a Robust Combined Corpus

```bash
conda run -n astro python -m src.build_robust_trajectories \
  --inputs \
    data/trajectories_verified.jsonl \
    data/trajectories_verified_new.jsonl \
    data/trajectories_augmented.jsonl \
    data/trajectories_final.jsonl \
    data/trajectories_final_qwen35.jsonl \
    data/trajectories_complex_coding.jsonl \
  --output data/trajectories_all_complex_full.jsonl \
  --output-preferences data/trajectory_preferences_all_complex_full.jsonl \
  --output-summary results/trajectory_corpus_summary_all_complex_full.json \
  --include-splits train dev \
  --augment-splits train \
  --min-per-registry-tool 6 \
  --target-total 2200 \
  --correction-ratio 0.35
```

### 3. Fine-Tune (Auto-Cleans Old Output)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
conda run -n astro python -u -m src.finetune \
  --data data/trajectories_all_complex_full.jsonl \
  --output finetuned/qwen3.5-4b-rlm-lora-all-complex \
  --epochs 3 \
  --train-split train \
  --eval-split dev \
  --max-seq-length 1536 \
  --per-device-batch-size 1 \
  --grad-accumulation 16
```

Note: `src.finetune` now removes existing output artifacts by default before starting a new run.
Use `--no-clean-output-first` to disable that behavior.

### 4. Compare Base vs Fine-Tuned

Example (test split, selection mode):

```bash
conda run -n astro python -m src.run_hf_comparison \
  --model-label base_qwen35_4b_all_complex \
  --model-name Qwen/Qwen3.5-4B \
  --query-split test \
  --trials 5 \
  --task-mode selection \
  --registries small_10 medium_25 large_50 xlarge_100

conda run -n astro python -m src.run_hf_comparison \
  --model-label finetuned_qwen35_4b_all_complex \
  --model-name finetuned/qwen3.5-4b-rlm-lora-all-complex \
  --query-split test \
  --trials 5 \
  --task-mode selection \
  --registries small_10 medium_25 large_50 xlarge_100
```

Then generate comparison report:

```bash
conda run -n astro python -m src.run_hf_comparison \
  --compare-only \
  --query-split test \
  --trials 5 \
  --task-mode selection \
  --registries small_10 medium_25 large_50 xlarge_100 \
  --base-summary results/hf_base_qwen35_4b_all_complex_summary_test.json \
  --finetuned-summary results/hf_finetuned_qwen35_4b_all_complex_summary_test.json \
  --report-path results/hf_base_vs_finetuned_all_complex_report_test.md \
  --manifest-path results/hf_comparison_manifest_all_complex_test.json
```

## Progress Monitoring

```bash
tail -f logs/finetune_all_complex.log
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader -l 2
pgrep -af src.finetune
```

## Key Outputs

- Combined corpus: `data/trajectories_all_complex_full.jsonl`
- Corpus summary: `results/trajectory_corpus_summary_all_complex_full.json`
- Fine-tuned adapter: `finetuned/qwen3.5-4b-rlm-lora-all-complex/`
- Comparison report: `results/hf_base_vs_finetuned_all_complex_report_test.md`

## Understanding Improvements & Results

### Metric Interpretation

When comparing base vs. fine-tuned models:

- **Positive delta → Improvement:** Fine-tuned model outperforms base
- **Negative delta → Regression:** Fine-tuned model underperforms (e.g., due to overfitting to training distribution)
- **Zero delta → No change:** Models perform identically on that registry/level

### Expected Patterns

**Level 0 (Zero-Shot):**
- Usually minimal improvement (models already see all tools in context)
- Fine-tuning may hurt if it causes "overthinking" on trivial tasks

**Level 1 (REPL Constrained):**
- Largest gains are here — fine-tuning teaches models to write better REPL queries
- Typical improvement: +15–35% on TSA, +10–20% on PC

**Level 2 (Open-Ended):**
- More variable — depends on whether model learns recovery strategies
- Fine-tuning on simpler trajectories may not transfer well to complex reasoning
- Possible regressions if training data is too narrow

**By Registry Size:**
- **Small (10 tools):** Usually higher baseline accuracy; less room for improvement
- **Large (50+ tools):** Higher potential gains (fine-tuning teaches searching, filtering)
- **Mixed results:** If fine-tuning overfits to training distribution, may underperform on diverse registries

### Interpreting Deltas Example

From test results on dev_selection split:
```
Level 1, medium_25 registry: +0.3333 delta_tsa, +0.3333 delta_pc, +0.3333 delta_e2e
  → 3 additional correct predictions out of 9 queries
  → Model learned better REPL interaction for medium-sized registries

Level 2, small_10 registry: -1.0 delta_pc, -1.0 delta_e2e
  → 10 queries all failed parameter extraction
  → Possible overfitting to Level 1 distribution (fewer Level 2 training examples)
```

## Research Questions & Known Limitations

### Why 4B Models Struggle with Tool Selection

1. **Context overload:** Listing 50+ tool schemas adds 5–10K tokens; leaves little room for reasoning
2. **Hallucination risk:** Smaller models confabulate tool names and parameters not in the context
3. **REPL gap:** Models tend to output code + fake outputs in a single pass instead of executing iteratively
4. **Parameter extraction:** Parsing required vs. optional parameters and their types is error-prone without strong supervision

### Why Fine-Tuning Helps (and Sometimes Hurts)

**Helps:**
- Aligns output format (cleaner code, better structured parameters)
- Improves REPL interaction patterns (write search code, wait for results, parse correctly)
- Boosts confidence on seen tool types/patterns (from training coverage)

**Hurts:**
- Overfitting to training tools (poor generalization to unseen tools)
- Distribution shift (training on simple queries, testing on discovery-only tasks)
- Token budget constraints (fine-tuned prompts may become more verbose)

### Open Issues

- **Preference tuning:** Current work uses supervised fine-tuning; DPO/ORPO phase is planned but pending
- **Negative examples:** Could improve parameter extraction with explicit contrastive pairs
- **Deeper RLM recursion:** Paper suggests multi-level recursion; current implementation uses single-level

## Citation & References

This research builds on:
- **RLM Paper:** Zhang et al., "Recursive Language Models" (2025)
- **LoRA:** Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models" (ICLR 2022)
- **CodeAct:** Wang et al., "CodeAct: Executable Code Action Agent" (ICML 2024)

## Troubleshooting

### GPU Out-of-Memory Errors

If you see `CUDA out of memory` during comparisons:

```bash
# Set memory expansion (helps reduce fragmentation)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Run sequentially (not parallel)
conda run -n astro python -m src.run_hf_comparison --model-label base ... 
conda run -n astro python -m src.run_hf_comparison --model-label finetuned ...
```

### Training Is Very Slow

If fine-tuning takes >1 hour per epoch:

```bash
# Verify Unsloth is loaded (check logs for "Unsloth patching" messages)
# If not, check that transformers version matches (5.2.0+)
conda list | grep -E "transformers|unsloth"

# Increase batch size if VRAM permits (max ~4–8 per RTX 4070)
--per-device-batch-size 2 --grad-accumulation 8
```

### Evaluation Queries Not Found

Verify test data exists:

```bash
ls test_data/queries/{train,dev,test}_queries.json
ls test_data/ground_truth/{train,dev,test}_expected_selections.json
```
