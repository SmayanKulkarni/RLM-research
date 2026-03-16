# RP-Focused Improvement Plan: Stronger RLM + Fine-Tuning Results

**Date:** 2026-03-16  
**Objective:** Convert current promising-but-mixed outcomes into robust, publication-ready evidence for an RP.

## 1) Evidence-Based Diagnosis from Current Results

## 1.1 What is already strong
- Fine-tuning clearly improves **Level 1 (constrained REPL)**:
  - Macro deltas: `ΔTSA +0.1845`, `ΔPC +0.0714`, `ΔRCV +0.2011`, `ΔFailure -0.1429`.
- This confirms the core hypothesis that trajectory fine-tuning helps in iterative REPL control loops.

## 1.2 What is currently weak
- **Level 2 regression** after fine-tuning:
  - Macro deltas: `ΔTSA -0.1131`, `ΔPC -0.1131`, `ΔE2E -0.1131`, `ΔFailure +0.1131`.
- Current comparisons are based on small query counts per cell, so confidence is limited.

## 1.3 Root-cause risks found in pipeline
1. **Train/eval contamination risk (high impact)**
   - `src/generate_trajectories.py` explicitly loads benchmark queries from `test_data/queries/all_queries.json` and injects them into training pair generation.
   - This can inflate apparent generalization and weakens research claims.

2. **Instruction/level mismatch in training data (high impact)**
   - Most generated trajectories are Level-2-style conversations; Level-specific behavior appears under-covered/imbalanced.
   - Observed pattern (Level 1 up, Level 2 down) suggests optimization is not stable across prompt regimes.

3. **Objective mismatch in finetuning (medium-high impact)**
   - `src/finetune.py` comments claim response-only learning, but there is no explicit response-mask setup; likely training over full sequence tokens.
   - This can dilute signal for decision-critical assistant spans (`<code>` + `FINAL(...)`).

4. **Data quality blind spots (medium impact)**
   - Verifier checks tool match and executable code, but does not deeply verify parameter semantic correctness beyond basic structure/non-empty constraints.
   - Weak parameter supervision aligns with some PC instability.

5. **Evaluation design is not yet paper-grade (high impact)**
   - Single-run metrics only; no repeated-trial reliability (`pass^k` style), no confidence intervals, no significance tests.
   - Current result set is too small to support strong claims.

---

## 2) Improvements to RLM System (Scaffold + Prompt + Control)

## 2.1 Make the RLM loop stateful and explicit
Implement a simple finite-state policy in scaffold execution:
- `DISCOVER` → require `search/list/filter`
- `VERIFY` → require `get_schema`
- `DECIDE` → output `FINAL(...)`

If model outputs `FINAL(...)` before `VERIFY`, auto-nudge once with strict instruction.  
Expected impact: reduce wrong-tool early commitments and stabilize Level 2.

## 2.2 Add failure-aware recovery policy
When failure pattern is detected (no FINAL by turn 6, repeated empty searches, or repeated same code):
- inject a compact corrective hint template,
- force one schema check on top candidate,
- terminate early if still non-progressing.

Expected impact: lower failure rate and turn bloat.

## 2.3 Tighten output constraints
- Require strict JSON schema for `FINAL` tool-selection tasks.
- Keep discovery tasks separate from selection tasks during scoring.

Expected impact: higher parser reliability and cleaner E2E accounting.

## 2.4 Prompt simplification for Level 2
Current Level 2 prompt may be overlong and style-variable. Use:
- 1 canonical example for strict selection,
- 1 canonical example for discovery,
- explicit “one code block at a time” rule,
- explicit “extract required params only from query text or default policy”.

Expected impact: better transfer across heterogeneous query forms.

---

## 3) Improvements to Fine-Tuning Pipeline

## 3.1 Remove benchmark leakage (must-do before new runs)
- Stop loading `test_data/queries/all_queries.json` into training generation.
- Create strict split files:
  - `train_queries.jsonl`
  - `dev_queries.jsonl`
  - `test_queries.jsonl` (never seen during generation/augmentation)

Expected impact: credible generalization claims.

## 3.2 Enforce response-only training correctly
In `src/finetune.py`, ensure loss is computed only on assistant outputs (especially `<code>` and `FINAL`).
- If using TRL/Unsloth response masking utilities, enable them explicitly.
- Validate by checking label mask ratio statistics.

Expected impact: stronger learning signal for decision behavior.

## 3.3 Rebalance dataset by level and difficulty
Create a curriculum-balanced dataset:
- 30% Level 1-style
- 50% Level 2-style
- 20% hard/adversarial cases (ambiguous wording, near-synonym tools)

Also balance by tool frequency to reduce overfitting to common tools.

Expected impact: fix Level 2 regressions and improve robustness.

## 3.4 Add hard negatives + preference-style data
For each positive trajectory, generate 1–2 negatives:
- wrong tool, right params
- right tool, wrong params
- malformed code but plausible reasoning

Use either:
- DPO/ORPO-style preference tuning, or
- SFT with explicit correction turns.

Expected impact: better discrimination and parameter precision.

## 3.5 Parameter-grounded supervision
Upgrade verifier semantic stage to check required parameter value correctness against expected extraction heuristics/labels (not just non-empty dict).

Expected impact: improved `parameter_correctness` and `E2E` stability.

---

## 4) Evaluation Upgrades Needed for RP-Quality Claims

## 4.1 Repeated-trial reliability
Add repeated runs per query/cell and report:
- `pass@1`
- `pass@k` proxy
- `pass^k`-style reliability proxy
- variance / 95% CI

Minimum: 5–10 trials per query for stochastic settings.

## 4.2 Expand matrix breadth
Run all levels across:
- `small_10`, `medium_25`, `large_50`, `xlarge_100`.

## 4.3 Add significance testing
For base vs fine-tuned deltas:
- bootstrap CIs for TSA/PC/E2E
- paired significance tests (query-level outcomes)

## 4.4 Add ablations that support a publishable narrative
Core ablations:
1. No-REPL vs constrained-REPL vs few-shot-REPL
2. No-finetune vs SFT-only vs SFT+preference tuning
3. Data size scaling (200 / 500 / 1000+ trajectories)
4. Leakage-controlled vs leakage-contaminated (to show evaluation rigor)

---

## 5) Concrete 3-Phase Execution Plan (Fastest Path to Strong Results)

## Phase A (2–3 days): Validity hardening
1. Remove train/test overlap in generation pipeline.
2. Add strict split manifests and rerun dataset build.
3. Add repeated-trial evaluation support and CI output.

**Deliverable:** clean baseline table with uncertainty bands.

## Phase B (3–5 days): Model quality uplift
1. Enable true response-only loss.
2. Rebalance dataset by level/tool/difficulty.
3. Add hard negatives and parameter-semantic checks.
4. Retrain LoRA with 2–3 hyperparameter sweeps.

**Deliverable:** improved Level 1 retained + Level 2 recovered.

## Phase C (3–4 days): RP-ready evidence
1. Run full registry matrix + repeated trials.
2. Produce ablation tables and error taxonomy.
3. Freeze best checkpoint and write reproducibility appendix.

**Deliverable:** paper-grade result section with clear claim boundaries.

---

## 6) Quantitative Targets for “Strong Results”

Use these as acceptance thresholds for RP write-up:
- Level 1: `E2E >= 0.85` with `failure_rate <= 0.08` across medium_25 and large_50.
- Level 2: recover to at least parity with base on TSA/PC, then exceed by `+0.05` absolute.
- Stability: bootstrap 95% CI excluding zero for main claimed deltas.
- Reliability: pass-style metric uplift visible across k (not just pass@1 single-shot).

---

## 7) Suggested RP Claim Framing (if targets are met)

"For Qwen3.5-4B-class SLMs, trajectory-grounded fine-tuning significantly improves constrained REPL tool-selection reliability under MCP-like registries, and with leakage-controlled evaluation plus repeated-trial reliability metrics, gains remain robust at medium-to-large tool scales."

This framing is strong but still realistic and defensible.

---

## 8) Implementation Status As Of 2026-03-16

This section converts the roadmap into a live execution ledger.

### 8.1 Completed roadmap items
- Leakage-safe split generation is implemented.
- Training-generation leakage control is implemented.
- Split-aware evaluation loading is implemented.
- Repeated-trial reliability metrics are implemented.
- Stateful scaffold recovery logic is implemented.
- Discovery-vs-selection evaluation separation is implemented.
- Level-2 prompt simplification is implemented.
- Assistant-only fine-tuning loss masking is implemented.
- Parameter-grounded Stage-3 semantic verification is implemented.
- Reproducibility manifest and markdown comparison report generation are implemented for HF comparison.

### 8.2 Partially completed roadmap items
- Significance testing is partially complete:
   - approximate CI-based significance exists,
   - paired bootstrap CIs exist,
   - paired sign-test p-values exist,
   - but broad-coverage statistical evidence is still pending because larger runs are not finished.
- Wave-3 evaluation breadth is partially complete:
   - small dev repeated-trial runs are done,
   - broad all-registry run is in progress.
- Fine-tuning supervision improvements are partially complete:
   - assistant-only loss masking is done,
   - split-manifest-aware train/eval partitioning is still pending.

### 8.3 Not-yet-implemented roadmap items
- Trajectory provenance metadata enrichment.
- Dataset rebalancing by level/difficulty/tool frequency.
- Hard-negative generation.
- Correction-turn or preference-pair dataset path.
- DPO/ORPO-style preference-tuning stage.
- Full ablation orchestration.
- End-to-end appendix-grade report generator across all pipelines.

### 8.4 Current evidence state
- The initial single-run comparison supported the main Level-1 benefit hypothesis.
- The small repeated-trial dev run showed that some fine-tuned regressions remain real on narrow slices, especially in Level 0 and Level 2.
- The active broad repeated-trial run is necessary before drawing stronger updated claims.

### 8.5 Immediate next required work
1. Finish the all-registry repeated-trial benchmark currently running.
2. Generate a consolidated RP-facing report from the broader benchmark.
3. Replace fine-tuning random split logic with split-manifest-aware partitioning.
4. Add provenance and balancing metadata to the trajectory pipeline.
5. Implement negative-data generation and a preference-tuning path.
