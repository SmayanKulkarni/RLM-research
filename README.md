# RLM MCP Research

This repository contains data generation, fine-tuning, and evaluation workflows for MCP tool-selection models (Qwen 3.5 + LoRA/Unsloth).

## Project Layout

- `src/`: Core scripts for generation, corpus building, fine-tuning, and evaluation.
- `data/`: Training corpora and preference data (`*.jsonl`).
- `test_data/`: Registries, queries, and expected selections used for evaluation.
- `results/`: Benchmark summaries and comparison reports.
- `finetuned/`: LoRA adapters and merged model artifacts.
- `logs/`: Training and evaluation logs.

## Environment

Use the `astro` conda environment (recommended):

```bash
conda run -n astro python -c "import sys; print(sys.executable)"
```

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
