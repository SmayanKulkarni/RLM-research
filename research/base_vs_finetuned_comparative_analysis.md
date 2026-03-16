# Comparative Analysis: Base vs Fine-Tuned Qwen3.5-4B (MCP/RLM Evaluation)

**Date:** 2026-03-16  
**Scope:** Comparative analysis of the base model (`Qwen/Qwen3.5-4B`) vs fine-tuned adapter (`finetuned/qwen3.55-4b-rlm-lora`) using the repository’s RLM/MCP evaluator.

## 1) Evaluation Setup

- **Runner:** `src/run_hf_comparison.py`
- **Evaluator:** `src/evaluator.py`
- **Test matrix:**
  - Levels: `0`, `1`, `2`
  - Registries: `small_10`, `medium_25`
  - Cells evaluated: 6 total combinations
- **Output artifacts:**
  - `results/hf_base_qwen35_4b_summary.json`
  - `results/hf_finetuned_qwen355_4b_summary.json`
  - `results/hf_base_vs_finetuned_comparison.json`

---

## 2) What Each Metric Means

All metrics are computed over test queries in a given matrix cell.

### 2.1 Tool Selection Accuracy (TSA)
- **Definition:** Fraction of queries where predicted tool name matches expected tool name.
- **Formula:**
  \[
  \text{TSA} = \frac{\#\{\text{tool\_correct}=\text{True}\}}{N}
  \]
- **Interpretation:** Higher is better. Measures tool-name selection quality.

### 2.2 Parameter Correctness (PC)
- **Definition:** Fraction of queries where required expected parameters are correctly matched.
- **Formula:**
  \[
  \text{PC} = \frac{\#\{\text{params\_correct}=\text{True}\}}{N}
  \]
- **Interpretation:** Higher is better. Captures argument fidelity.

### 2.3 REPL Code Validity (RCV)
- **Definition:** Average per-query ratio of successful REPL executions across attempted REPL steps.
- **Per-query code validity:**
  \[
  \text{code\_validity}_i = \frac{\#\text{successful REPL steps}}{\#\text{REPL steps}}
  \]
- **Aggregated metric:**
  \[
  \text{RCV} = \frac{1}{N}\sum_{i=1}^{N}\text{code\_validity}_i
  \]
- **Special case:** At Level 0 (no REPL), evaluator sets `code_validity = 1.0` (not applicable baseline).
- **Interpretation:** Higher is better. Indicates execution robustness of generated code/commands.

### 2.4 End-to-End Accuracy (E2E)
- **Definition:** Fraction of queries where both tool and parameters are correct.
- **Formula:**
  \[
  \text{E2E} = \frac{\#\{\text{tool\_correct} \land \text{params\_correct}\}}{N}
  \]
- **Interpretation:** Higher is better. Practical single-run success proxy.

### 2.5 Average Turns
- **Definition:** Average scaffold turns used per query.
- **Formula:**
  \[
  \text{Avg Turns} = \frac{1}{N}\sum_{i=1}^{N}\text{turns\_used}_i
  \]
- **Interpretation:** Lower is often better for efficiency, but must be read jointly with accuracy.

### 2.6 Failure Rate
- **Definition:** Fraction of queries where scaffold did **not** successfully finish (`success=False`).
- **Formula:**
  \[
  \text{Failure Rate} = \frac{\#\{\neg\text{success}\}}{N}
  \]
- **Interpretation:** Lower is better. Reliability indicator.

### 2.7 Context Savings %
- **Definition:** Token savings from REPL-style context management.
- **Formula (evaluator property):**
  \[
  \text{Context Savings} = 1 - \frac{\text{context\_tokens\_with\_repl}}{\text{context\_tokens\_without\_repl}}
  \]
- **Interpretation:** Higher is better. In this run, all cells report `1.0`, so this metric is not discriminative for base-vs-finetuned comparison here.

---

## 3) Cell-by-Cell Comparative Results

`Δ = finetuned - base`

### 3.1 Level 0 (no REPL loop)

| Cell | ΔTSA | ΔPC | ΔRCV | ΔE2E | ΔFailure |
|---|---:|---:|---:|---:|---:|
| level0_small_10 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 |
| level0_medium_25 | 0.0000 | -0.0833 | 0.0000 | -0.0833 | 0.0000 |

**Reading:** Nearly flat, with a slight PC/E2E drop on `medium_25`.

### 3.2 Level 1 (constrained REPL)

| Cell | ΔTSA | ΔPC | ΔRCV | ΔE2E | ΔFailure |
|---|---:|---:|---:|---:|---:|
| level1_small_10 | +0.2857 | +0.1429 | +0.2897 | +0.1429 | -0.2857 |
| level1_medium_25 | +0.0833 | 0.0000 | +0.1125 | 0.0000 | 0.0000 |

**Reading:** Strongest positive shift after fine-tuning, especially on `small_10`; reliability and execution quality improved materially.

### 3.3 Level 2 (few-shot/fuller REPL)

| Cell | ΔTSA | ΔPC | ΔRCV | ΔE2E | ΔFailure |
|---|---:|---:|---:|---:|---:|
| level2_small_10 | -0.1429 | -0.1429 | +0.0214 | -0.1429 | +0.1429 |
| level2_medium_25 | -0.0833 | -0.0833 | +0.0194 | -0.0833 | +0.0833 |

**Reading:** Mixed behavior: slight code-validity improvement but reduced selection/parameter accuracy and higher failure.

---

## 4) Level-Wise Macro Delta Summary (Average of two registries)

| Level | ΔTSA | ΔPC | ΔRCV | ΔE2E | ΔFailure |
|---|---:|---:|---:|---:|---:|
| Level 0 | 0.0000 | -0.0416 | 0.0000 | -0.0416 | 0.0000 |
| Level 1 | +0.1845 | +0.0714 | +0.2011 | +0.0714 | -0.1429 |
| Level 2 | -0.1131 | -0.1131 | +0.0204 | -0.1131 | +0.1131 |

**Key takeaway:** Fine-tuning helps most in **Level 1 constrained REPL** (the main target regime), while **Level 2** needs more targeted data and/or prompt strategy tuning.

---

## 5) Runtime/Operational Observations

- Finetuned runs were generally faster in multi-turn cells in this execution set (not a quality metric by itself).
- Due to GPU memory constraints, base and finetuned evaluations were run in separate processes; this does not affect metric definitions, but is important for reproducibility.

---

## 6) Practical Interpretation for MCP/RLM Work

1. **Where fine-tuning is clearly beneficial:** constrained REPL trajectories (Level 1), reflected by gains in TSA/RCV and reduced failures.
2. **What remains weak:** Level 2 generalization, where tool/parameter correctness regressed despite slightly better execution validity.
3. **What this suggests:** current data likely over-optimizes execution discipline in constrained loops and under-covers richer/few-shot decision patterns.

---

## 7) Limits of This Comparison

- Single-run metrics only (no repeated-trial `pass^k` style reliability yet).
- Registry scope in this run is `small_10` + `medium_25` (not `large_50`/`xlarge_100`).
- `context_savings_pct` is constant in this output and not useful for discriminating model quality here.

---

## 8) Recommended Next Step

Extend the runner to repeated trials per query and report pass-style reliability curves (`pass@1`, `pass@k` proxy, variance/CI) for each level/registry cell, then compare base vs fine-tuned on those reliability-centric metrics.

---

## 9) Post-Analysis Update (2026-03-16, Later Session)

The recommendation above has now been partially implemented.

### What changed after this analysis was written
- `src/evaluator.py` now supports repeated-trial evaluation with:
  - `pass_at_1`
  - `pass_at_k`
  - `pass_power_k_proxy`
  - bootstrap confidence intervals
- `src/run_hf_comparison.py` now supports:
  - repeated-trial runs via `--trials`
  - task mode selection via `--task-mode`
  - markdown report generation via `--report-path`
  - reproducibility manifest generation via `--manifest-path`
  - paired query-level significance outputs for repeated-trial summaries

### New evidence produced after this analysis
- Small repeated-trial dev-split artifacts:
  - `results/hf_base_qwen35_4b_summary_dev.json`
  - `results/hf_finetuned_qwen355_4b_summary_dev.json`
  - `results/hf_base_vs_finetuned_report_dev.md`
  - `results/hf_comparison_manifest_dev.json`

### Updated interpretation
- The project has moved from metric-definition stage into reliability-measurement stage.
- However, the dev repeated-trial slice is too small to revise the core narrative confidently.
- The broad all-registry repeated-trial run is the key next artifact for updating this comparative analysis rigorously.
