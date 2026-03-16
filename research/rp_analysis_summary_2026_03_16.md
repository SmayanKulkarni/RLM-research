# RP Analysis Summary — 2026-03-16

This file summarizes the diagnosis and improvement direction for achieving publication-grade MCP/RLM results.

## Primary diagnosis
- Fine-tuning is effective in constrained REPL mode (Level 1).
- Fine-tuning currently regresses quality in richer few-shot REPL mode (Level 2).
- Current evaluation lacks repeated-trial reliability and confidence intervals.

## Critical blocker
- Training/evaluation contamination risk exists because trajectory generation currently imports benchmark query sets into training generation (`src/generate_trajectories.py`).

## Action direction
- Enforce strict train/dev/test separation and rerun training.
- Switch evaluation to repeated-trial pass-style metrics with CIs.
- Rebalance and harden fine-tuning data (Level 2 coverage, hard negatives, parameter-semantic checks).
- Tighten scaffold control policy and recovery logic for multi-turn failures.

## Full roadmap
See: `research/rp_strong_results_improvement_plan.md`

## Current execution status
- Completed:
	- split manifests
	- split-safe trajectory seeding
	- repeated-trial metrics
	- discovery-vs-selection evaluation modes
	- Level-2 prompt simplification
	- assistant-only loss masking
	- parameter-grounded semantic verification
	- HF comparison manifests and markdown reports
- In progress:
	- broad repeated-trial all-registry HF comparison run
- Not completed yet:
	- split-aware fine-tune train/eval partitioning
	- provenance-rich trajectory metadata
	- hard-negative / correction-turn data generation
	- preference-tuning stage
	- formal ablation harness

## Interpretation update
- The repository is no longer at the planning-only stage.
- Core validity infrastructure is implemented.
- Core evaluation/reporting infrastructure is implemented.
- The remaining gap is now less about basic architecture and more about scaling evidence quality and enriching training data.
