## Plan: RP-Grade RLM Uplift
### Date : 15 March 
**DRAFT** — This plan turns the improvement document into an execution sequence that covers data validity, scaffold/RLM behavior, training quality, evaluation rigor, and RP-ready reporting. The highest-risk issue is leakage from generate_trajectories.py, so the plan starts by hardening splits and evaluation before any new training. After that, it upgrades the RLM loop in scaffold.py, fixes supervision in finetune.py, strengthens verification in verify_trajectories.py, and expands the experiment/reporting stack in run_hf_comparison.py and run_baseline.py. The intended outcome is a leakage-controlled, repeated-trial, statistically supported result package strong enough for an RP.

## Execution Status

### Status legend
- `COMPLETED` means implemented, validated, and logged.
- `IN PROGRESS` means code exists or runtime execution has started, but final consolidated evidence is not yet complete.
- `NOT STARTED` means the item is still pending implementation.

### Step-by-step status ledger

1. **Define experiment governance and split policy** — `COMPLETED`
- Completed in:
   - `src/generate_test_data.py`
   - `test_data/queries/train_queries.json`
   - `test_data/queries/dev_queries.json`
   - `test_data/queries/test_queries.json`
   - `test_data/ground_truth/train_expected_selections.json`
   - `test_data/ground_truth/dev_expected_selections.json`
   - `test_data/ground_truth/test_expected_selections.json`
- What was done:
   - Added deterministic split generation.
   - Separated train/dev/test query and GT manifests.
   - Established split-aware evaluation file resolution.
- Remaining under this step:
   - Add a more formal experiment-governance document or schema if needed.

2. **Remove benchmark leakage from trajectory generation** — `COMPLETED`
- Completed in:
   - `src/generate_trajectories.py`
- What was done:
   - Added `--seed-query-split`.
   - Defaulted to `none` so benchmark queries are not injected into training generation.
   - Added explicit warnings for risky settings (`test`, `all`).
- Remaining under this step:
   - Extend split provenance through augmentation outputs as explicit metadata.

3. **Make trajectory datasets provenance-aware and rebalanceable** — `NOT STARTED`
- Current gap:
   - Verified/final trajectory records still do not carry a full explicit metadata schema for split/source/difficulty/augmentation lineage.
- Remaining work:
   - Add metadata fields to generated and augmented trajectories.
   - Add dataset balancing/reweighting stage before fine-tuning.

4. **Strengthen trajectory verification for parameter-grounded supervision** — `COMPLETED`
- Completed in:
   - `src/verify_trajectories.py`
- What was done:
   - Added expected-parameter lookup from known GT manifests.
   - Stage 3 now validates required schema params.
   - Stage 3 now checks expected param values for benchmark-known queries.
   - Reject reasons are more specific.
- Remaining under this step:
   - Generalize param-grounded checks for fully synthetic queries beyond benchmark manifests.

5. **Add hard-negative and correction data generation** — `NOT STARTED`
- Current gap:
   - No negative-pair data schema has been added yet.
   - No synthetic wrong-tool/right-param or wrong-param variants are generated yet.
- Remaining work:
   - Extend `src/augment_trajectories.py`.
   - Define pairwise or correction-turn data format.

6. **Fix fine-tuning supervision and dataset splitting** — `PARTIALLY COMPLETED`
- Completed in:
   - `src/finetune.py`
- What was done:
   - Added assistant-only loss masking path.
   - Added label-mask diagnostics.
   - Added CLI controls for marker tuning and fallback behavior.
- Still pending:
   - Replace random row-level train/eval split with split-manifest-aware assignment.
   - Prevent paraphrase siblings or augmentation siblings from crossing train/eval boundaries.

7. **Add a second-stage preference-tuning path** — `NOT STARTED`
- Current gap:
   - No DPO/ORPO training entrypoint exists.
- Remaining work:
   - Add a second trainer pipeline.
   - Add pairwise dataset schema.

8. **Refactor the RLM scaffold into explicit stateful control** — `COMPLETED`
- Completed in:
   - `src/scaffold.py`
- What was done:
   - Added `DISCOVER` / `VERIFY` / `DECIDE`-style gating behavior.
   - Added verification-before-final nudges.
   - Added repeated-code detection.
   - Added low-signal-output recovery hints.
- Remaining under this step:
   - Quantitatively validate impact on larger runs.

9. **Add failure-aware recovery and termination policy** — `COMPLETED`
- Completed in:
   - `src/scaffold.py`
- What was done:
   - Added recovery hints for low-signal outputs.
   - Added repetition handling.
   - Added delayed forced transition from discovery to verification.
   - Added termination guard when recovery attempts are exhausted.
- Remaining under this step:
   - Tune thresholds after broader evaluation results are available.

10. **Tighten output contracts and separate task modes** — `COMPLETED`
- Completed in:
   - `src/evaluator.py`
   - `src/run_baseline.py`
   - `src/run_hf_comparison.py`
   - `prompts/level2.txt`
- What was done:
   - Added explicit discovery-vs-selection evaluation mode.
   - Added `--task-mode {selection,discovery,mixed}` to runners.
   - Simplified discovery output handling.

11. **Simplify and normalize prompts, especially Level 2** — `COMPLETED`
- Completed in:
   - `prompts/level2.txt`
- What was done:
   - Reduced prompt length.
   - Added one-code-block-per-turn rule.
   - Added explicit workflow contract.
   - Split selection vs discovery output contract.
- Remaining under this step:
   - Optional cleanup/versioning for `prompts/level1.txt`.

12. **Upgrade evaluator for repeated-trial reliability** — `COMPLETED`
- Completed in:
   - `src/evaluator.py`
   - `src/run_baseline.py`
   - `src/run_hf_comparison.py`
- What was done:
   - Added repeated-trial evaluation path.
   - Added pass-style metrics and bootstrap CIs.
   - Added repeated-trial runner integration.

13. **Expand experiment runners and comparison outputs** — `PARTIALLY COMPLETED`
- Completed in:
   - `src/run_baseline.py`
   - `src/run_hf_comparison.py`
- What was done:
   - Added `large_50` and `xlarge_100` support in runners.
   - Added report generation.
   - Added reproducibility manifest generation.
- Still pending:
   - Finish and analyze the currently running broad repeated-trial matrix.
   - Convert broad-run outputs into final consolidated RP-facing summaries.

14. **Add significance testing and RP-facing stats** — `PARTIALLY COMPLETED`
- Completed in:
   - `src/run_hf_comparison.py`
   - `results/hf_base_vs_finetuned_report_dev.md`
- What was done:
   - Added approximate CI-based delta significance.
   - Added query-level paired bootstrap CIs.
   - Added paired sign-test p-values.
   - Added markdown report generation.
- Still pending:
   - Run these methods on a broad enough benchmark matrix to support stronger claims.

15. **Formalize ablation framework** — `NOT STARTED`
- Current gap:
   - No ablation-specific orchestration or table generator exists yet.
- Remaining work:
   - Add experiment labels and ablation presets.
   - Generate ablation summary artifacts automatically.

16. **Build reproducibility and RP reporting outputs** — `PARTIALLY COMPLETED`
- Completed in:
   - `src/run_hf_comparison.py`
   - `results/hf_comparison_manifest_dev.json`
   - `results/hf_base_vs_finetuned_report_dev.md`
- What was done:
   - Added manifest generation.
   - Added markdown report generation.
- Still pending:
   - Add equivalent manifest/report generation to baseline and training pipelines.
   - Build appendix-ready consolidated tables.

17. **Sequence execution in three delivery waves** — `IN PROGRESS`
- Wave 1 is effectively complete.
- Wave 2 is mostly complete at the code level, but still needs larger-run validation.
- Wave 3 has begun experimentally because broad repeated-trial benchmarking is already running, but its core items remain unfinished.

### Current high-level state
- `Wave 1`: mostly complete and validated.
- `Wave 2`: major code changes implemented; empirical validation still ongoing.
- `Wave 3`: benchmark expansion in progress; hard negatives, preference tuning, full RP appendix generation still pending.

### Active execution right now
- A broad repeated-trial HF comparison run is currently running in the `astro` environment for:
   - `query_split=all`
   - `task_mode=selection`
   - `trials=5`
   - registries: `small_10`, `medium_25`, `large_50`, `xlarge_100`
- This run is intended to produce the first stronger all-coverage significance-bearing report.

**Steps**
1. **Define experiment governance and split policy**
   - Add explicit split artifacts for train/dev/test query sets and experiment metadata under test_data and research.
   - Refactor query-loading away from the benchmark-only path in generate_trajectories.py, run_baseline.py, and evaluator.py.
   - Introduce a split-aware loader boundary so `train` data can never consume `test` queries.
   - Record split-generation assumptions and seed policy in RESEARCH_LOG.md and a reproducibility manifest in research.

2. **Remove benchmark leakage from trajectory generation**
   - Replace `existing_qt` / `qt_pairs` coupling in generate_trajectories.py with split-scoped query sources.
   - Ensure generated training trajectories only use train queries or synthetic train-only queries, never all_queries.json.
   - Add guardrails in `run()` so accidental inclusion ofs held-out queries fails fast.
   - Update augmentation paths in augment_trajectories.py to preserve split provenance.

3. **Make trajectory datasets provenance-aware and rebalanceable**
   - Extend trajectory records produced by generate_trajectories.py and augment_trajectories.py with metadata fields for split, source, registry, augmentation type, inferred level, category, and difficulty.
   - Use labels already available in generate_test_data.py and category accessors in tool_registry.py to derive sampling strata.
   - Build a balancing stage before finetune.py so training can target level mix, tool-frequency flattening, and hard-case oversampling.

4. **Strengthen trajectory verification for parameter-grounded supervision**
   - Upgrade `stage3_semantic()` in verify_trajectories.py to validate required parameter keys and value correctness, not just tool identity and non-empty dicts.
   - Reuse evaluation-style matching logic from evaluator.py and formalize extraction heuristics currently embedded in augment_trajectories.py.
   - Emit structured reject reasons for wrong-tool, wrong-param, malformed-code, and ambiguous-label cases so those failures can later become negatives.

5. **Add hard-negative and correction data generation**
   - Extend augment_trajectories.py to synthesize negative examples: wrong tool/right params, right tool/wrong params, and valid-looking but failed REPL code.
   - Preserve links between each positive and its negatives in a paired-data schema for future preference tuning.
   - Add a correction-turn SFT representation as the first implementation target, since no pairwise trainer exists yet in finetune.py.

6. **Fix fine-tuning supervision and dataset splitting**
   - Replace row-level random split behavior in finetune.py with split-manifest-based train/dev assignment to prevent paraphrase siblings crossing splits.
   - Enforce actual assistant-only loss in finetune.py and finetune.py, with diagnostics that report masked-token ratios and assistant-token coverage.
   - Keep LoRA as the default path but parameterize curriculum/balancing options so multiple training slices can be compared cleanly.

7. **Add a second-stage preference-tuning path**
   - Introduce a separate trainer entrypoint or mode alongside finetune.py for DPO/ORPO-style runs using the new positive/negative pairs.
   - Keep this modular so SFT-only, SFT+correction, and SFT+preference become explicit ablation modes rather than ad hoc experiments.
   - Log training recipe, model ID, seed, and dataset manifest in the output directory and RESEARCH_LOG.md.

8. **Refactor the RLM scaffold into explicit stateful control**
   - Upgrade `MCPRLMScaffold.run()` in scaffold.py from a reactive loop to explicit `DISCOVER` → `VERIFY` → `DECIDE` phases.
   - Require evidence of schema verification before accepting a selection `FINAL(...)` for tool-selection tasks.
   - Track per-run state such as searched keywords, verified tools, repeated code, and stalled turns.
   - Leave repl_engine.py mostly unchanged; it already supports persistent namespace and turn history.

9. **Add failure-aware recovery and termination policy**
   - Replace the current generic fallback in scaffold.py with failure-pattern-specific nudges:
     - no `FINAL(...)` by threshold turn,
     - repeated empty or low-signal search,
     - repeated identical code block,
     - skipping `get_schema()` before selection.
   - Add one recovery attempt path and then terminate deterministically to avoid noisy long-tail failures and inflated turn counts.

10. **Tighten output contracts and separate task modes**
    - Split tool-selection and discovery behaviors now mixed in level2.txt.
    - Keep strict JSON `FINAL({...})` for selection tasks and a separate explicit contract for discovery tasks.
    - Update parsing in `parse_tool_from_answer()` and scoring behavior in evaluator.py so discovery is not forced through tool-selection logic.
    - Align runner filtering in run_baseline.py with the new task-mode distinction.

11. **Simplify and normalize prompts, especially Level 2**
    - Rewrite level2.txt to use fewer examples, one code block per turn, and a simpler decision contract.
    - Consider parallel cleanup of level1.txt so prompt family differences are intentional and minimal.
    - Version prompts and surface prompt IDs in result artifacts so prompt changes become ablations instead of hidden drift.

12. **Upgrade evaluator for repeated-trial reliability**
    - Extend `evaluate_single()` / `evaluate_all()` in evaluator.py to support repeated trials per query.
    - Compute per-query and per-cell pass-style metrics, trial variance, and confidence intervals.
    - Preserve both raw trial outputs and aggregated summaries so paired analysis remains possible later.

13. **Expand experiment runners and comparison outputs**
    - Expand matrices in run_hf_comparison.py to include `large_50` and `xlarge_100`.
    - Expand `run_all()` in run_baseline.py to cover the full registry set consistently.
    - Extend `write_comparison()` in run_hf_comparison.py to include confidence intervals, repeated-trial summaries, and significance results.

14. **Add significance testing and RP-facing stats**
    - Use per-query `EvalResult` outputs from evaluator.py for paired bootstrap confidence intervals and query-level significance tests.
    - Report significance for `tool_selection_accuracy`, `parameter_correctness`, and `end_to_end_accuracy`, and optionally `failure_rate`.
    - Store both machine-readable stats JSON and a human-readable markdown summary in results and research.

15. **Formalize ablation framework**
    - Add explicit experiment knobs for:
      - no-REPL vs constrained REPL vs few-shot REPL,
      - baseline vs SFT vs SFT+correction vs SFT+preference,
      - dataset size scaling,
      - prompt version,
      - leakage-controlled vs old contaminated setup for internal methodology comparison.
    - Use run_baseline.py, run_hf_comparison.py, scaffold.py, level2.txt, and finetune.py as the main ablation surfaces.

16. **Build reproducibility and RP reporting outputs**
    - Add experiment manifests that capture split IDs, prompt version, model/adaptor path, LoRA config, seed, registry set, and command-line args.
    - Extend current reporting anchors in baseline_summary.json, hf_base_vs_finetuned_comparison.json, base_vs_finetuned_comparative_analysis.md, and RESEARCH_LOG.md.
    - Produce RP-facing tables: main results, ablations, error taxonomy, and reproducibility appendix.

17. **Sequence execution in three delivery waves**
    - **Wave 1: validity hardening** — splits, leakage removal, repeated-trial evaluation, CI/statistics.
    - **Wave 2: model-quality uplift** — scaffold policy, prompt cleanup, verification hardening, rebalanced response-only SFT.
    - **Wave 3: strongest-results push** — hard negatives, preference tuning, full matrix reruns, ablations, RP package.

**Verification**
- Validate no test queries are reachable from training generation paths in generate_trajectories.py and augmentation outputs in augment_trajectories.py.
- Confirm split integrity by checking that train/dev/test query overlap is zero and sibling augmented samples do not cross partitions.
- Run baseline and HF comparison suites from run_baseline.py and run_hf_comparison.py on `small_10`, `medium_25`, `large_50`, and `xlarge_100`.
- Re-run with repeated trials and verify CI/statistics are emitted for all main metrics.
- Compare pre/post scaffold changes using identical prompts and registries to isolate control-policy gains.
- Retrain at least one leakage-controlled SFT model and verify Level 1 gains persist while Level 2 reaches parity or better.
- Generate final RP result tables and ensure every number is traceable to a manifest-backed artifact in results or research.

**Decisions**
- Leakage removal comes before any new training so later gains are publishable.
- Repeated-trial evaluation is treated as core infrastructure, not a later add-on.
- Correction-turn SFT should precede full DPO/ORPO because pairwise training infrastructure does not yet exist in the repo.
- Scaffold policy changes belong in `run()` within scaffold.py, while REPL persistence remains in repl_engine.py.
- Discovery and tool-selection should be separated at prompt, parser, and scorer levels to avoid mixed-contract noise.

## Remaining Work Summary

### Highest-priority remaining changes
1. Add trajectory provenance metadata and split-aware training/eval assignment.
2. Add hard-negative synthesis and correction-turn or pairwise preference data.
3. Add a preference-tuning stage.
4. Finish the broad repeated-trial matrix and write consolidated RP result tables.
5. Add ablation orchestration and appendix-ready reporting across baseline/training/comparison.

### Recommended immediate next sequence after the active benchmark finishes
1. Summarize the all-split repeated-trial outputs into a new RP-facing markdown report.
2. Replace fine-tuning random split logic with split-manifest-aware partitioning.
3. Add provenance fields to generated and augmented trajectory data.
4. Extend augmentation to generate negative/correction data.