# Research Log — Scaling MCP Access for SLMs via RLM

> **Project:** Scaling MCP Access for Small Language Models via the REPL Approach  
> **Team:** TY Engineering Students  
> **Primary Reference:** "Recursive Language Models" (Zhang et al., 2025)  
> **Started:** 2026-03-01

---

*This is a living document. Append new entries at the bottom. Never overwrite existing entries.*

---

## 2026-03-01 — Project Initialization

**Session goal:** Set up research agent scaffold and establish project structure.

### Actions Taken
- Created `.agents/skills/research/SKILL.md` — research agent skill definition
- Created `.agents/workflows/deep-research.md` — deep research workflow
- Created `.agents/workflows/paper-analysis.md` — paper analysis workflow
- Created `AGENTS.md` — project-level agent context
- Initialized this research log

### Initial Project Understanding
- **Core problem:** SLMs (1B–7B params) have context windows too small (2K–8K tokens) for effective MCP tool use
- **Proposed approach:** Adapt the RLM framework (REPL-based context externalization) to let SLMs access MCP servers
- **Key challenge:** SLMs lack the coding/reasoning ability to write Python code for REPL navigation
- **Candidate solutions under investigation:**
  1. Fine-tune SLM on synthetic RLM trajectories
  2. Use predefined Python scripts instead of open-ended code generation
  3. RLM "no sub-calls" variant (REPL without recursive LM invocations)

### Next Steps
- [x] Deep dive into RLM paper — extract all technical details relevant to SLMs
- [x] Research SLM capabilities landscape (coding, tool use, fine-tuning results)
- [x] Identify comprehensive bottleneck list
- [x] Research existing MCP + small model work
- [x] Propose candidate paper topics

---

## 2026-03-01 — Full Research Pass: RLM Deep Dive + Bottleneck Analysis + Topic Proposals

**Session goal:** Complete all future-steps research: deep RLM analysis, bottleneck identification, topic proposals, POA, feasibility.

**Method:** 9 web searches + full RLM paper extraction (33 pages via PyMuPDF) covering: SLM tool-calling benchmarks, MCP token overhead, SLM code generation capabilities, fine-tuning with synthetic trajectories, recursive LM frameworks (THREAD/DisCIPL/ReDel), context management systems (MemGPT/MemWalker/ReSum), Unsloth QLoRA feasibility, and MCP dynamic tool loading.

### Key Findings

**RLM Paper:**
- "No sub-calls" ablation gives 3.3× improvement even without sub-LM capability — critical for SLMs
- System prompt is fixed across all experiments — can be simplified for SLMs
- Paper explicitly states: "models without sufficient coding capabilities struggle as RLMs" — defines our research gap
- RLM paper suggests training models as RLMs as future work — validates our fine-tuning approach

**SLM Capabilities:**
- Fine-tuned OPT-350M achieved 77.55% on ToolBench — proves tiny models can learn tool use
- DisCIPL: Llama-3.2-1B went 4% → 87% with Planner-Follower pattern — massive gains possible
- THREAD: 10–50% absolute gains for Llama-3-8b and CodeLlama-7b — recursive patterns help small models
- Phi-3 Mini (3.8B): Strong coding, supports 128K context
- QLoRA fine-tuning of 3B models feasible on T4 with Unsloth (~2-4 hours)

**MCP Overhead:**
- 50 tool definitions consume 20,000–25,000 tokens (400–500 tokens/tool)
- "MCP tax" can consume >16% of context before any user interaction
- Dynamic tool loading can achieve 98.7% token reduction

### Deliverables Created
1. `research/rlm_deep_dive.md` — Full RLM paper analysis with numerical results and SLM mapping
2. `research/bottleneck_analysis.md` — 3 issues + 6 bottlenecks + 3 solutions + 6 additional issues
3. `research/topic_proposals_and_poa.md` — 3 topic proposals + 5-phase POA + feasibility assessment

### Recommended Topic
"REPL-Augmented Tool Access for Small Language Models: Enabling MCP Interaction Beyond Context Limits"

### Open Questions
- What is the minimum model size for effective REPL interaction?
- How much synthetic training data is needed for reliable REPL navigation?
- Does the Planner-Follower pattern work better than direct fine-tuning for this use case?
- Can predefined scripts achieve comparable results with less training?

---

## 2026-03-09 — Implementation-Focused Deep Research

**Session goal:** Answer 7 critical implementation questions about MCP structure exploitability, SLM capabilities, REPL environments, fine-tuning approaches, and verify all existing citations.

**Method:** 15 web searches covering: MCP JSON schema specification, SLM CoT/tool-calling benchmarks (BFCL, MCPMark, TAU-Bench), MCP scaling for SLMs, DisCIPL verification, THREAD verification, CodeAct analysis, REPL alternatives for agents, fine-tuning vs prompting vs scaffolding, SLM HumanEval benchmarks, xLAM function calling, GRPO/RL for tool calling, knowledge distillation approaches, MCP dynamic/lazy loading, and Qwen3.5 capabilities.

### Key Findings

1. **MCP has highly exploitable structure** — fixed JSON schema (name/description/inputSchema) makes tool selection closer to information retrieval than open-ended reasoning. Multi-level lazy loading achieves 98.7% token reduction.
2. **Nobody has done exactly what we're proposing** — xLAM approached function calling differently (purpose-built, no REPL); no RLM+MCP+SLM work exists.
3. **xLAM-1B surpasses GPT-3.5 on function calling** — proves 1B models can learn tool use with enough targeted data (60K examples via APIGen).
4. **GRPO (RL with verifiable rewards) is a strong alternative to SFT** — perfect for MCP tool selection where rewards are programmatically verifiable.
5. **Qwen3.5-4B is the optimal primary model** — strong coding performance, Unsloth-supported, GRPO-compatible, tool calling supported.
6. **All major citations verified** — DisCIPL (arXiv:2504.07081, COLM 2025), THREAD (NAACL 2025), CodeAct (arXiv:2402.01030).

### Deliverables Created
- `research/implementation_research_q_and_a.md` — Full answers to 7 questions with citations and implementation roadmap

### Open Questions
- What is the actual zero-shot performance of Qwen3.5-4B on MCP tool selection? (needs empirical testing)
- How much data does GRPO need vs. SFT for acceptable tool selection accuracy?
- Can the constrained REPL approach (predefined functions) alone match full REPL performance?

---

## 2026-03-10 — Implementation POA: REPL Design, System Prompts, Evaluation

**Session goal:** Answer 4 implementation clarification questions (MCP test server, REPL design, system prompt, evaluation) and create comprehensive implementation POA.

**Method:** Extracted RLM system prompts from Appendix D (pages 25-27), read APIGen PDF (3-stage verification pipeline, 3,673 APIs), 4 web searches on MCP test servers, Python SDK, and evaluation metrics.

### Key Findings

1. **MCP test server: Use synthetic tool registries** — we don't need a full MCP server at this stage. Generate JSON tool descriptions at scales of 10/25/50/100 tools. Collect real schemas from MCP reference servers. APIGen's 3-stage verification pipeline (format→execution→semantic) should be adopted for trajectory QA.
2. **Constrained REPL design finalized** — 3-level architecture: Level 1 (predefined functions only: `search()`, `get_schema()`, `list_names()`), Level 2 (simple Python + regex), Level 3 (full REPL). Code skeletons written for `MCPToolRegistry`, `REPLEngine`, `MCPRLMScaffold`.
3. **System prompt adapted from RLM Appendix D** — role, context, capabilities, examples, output format. Prompt version should evolve with each testing level.
4. **7 evaluation metrics defined** — TSA, Parameter Correctness, REPL Code Validity, E2E Task Completion, Turns Used, Context Tokens Saved, Timeout Rate. Decision framework: if RCV≥80% and TSA≥50% at Level 1, fine-tuning may not be critical.

### Deliverables Created
- `research/implementation_poa.md` — Full implementation POA with code skeletons, system prompt templates, evaluation framework, and 6-phase timeline

---

## 2026-03-14 — Post-Last-Push Exhaustive Change Audit (Uncommitted Work)

**Session goal:** Systematically append all work completed after the last git push, with an exhaustive inventory and outcome snapshot.

### Git Baseline (Source of Truth)
- Current branch: `main`
- Upstream tracking: `origin/main`
- Last pushed commit at audit time: `5479557` (`HEAD == origin/main`)
- Interpretation: there are **no additional commits** after the last push; all post-push work is currently in **working-tree modifications + untracked artifacts**.

### Tracked File Modifications (4 files)
1. `.gitignore`
  - Added `.env` ignore rule for local secret management (`GROQ_API_KEY`, etc.).

2. `configs/repl_config.yaml`
  - Default SLM model updated:
    - from `qwen3.5:4b`
    - to `qwen3.5:4b`

3. `src/slm_interface.py`
  - `SLMConfig.model_name` default updated to `qwen3.5:4b` for runtime consistency with config.

4. `requirements.txt`
  - Reworked into phase-structured dependency layout with install notes.
  - `ollama` minimum version increased (`>=0.6.0`).
  - Added trajectory generation dependency: `groq`.
  - Added fine-tuning stack dependencies: `transformers`, `trl`, `peft`, `datasets`, `accelerate`, `bitsandbytes`.
  - Added explicit install guidance for CUDA `torch` and `unsloth` in `astro` env.

### New Source Modules Added (5 files)
1. `src/generate_trajectories.py`
  - Synthetic trajectory generation pipeline using Groq (`llama-3.3-70b-versatile`).
  - Grounds model-produced code blocks by executing them via real `REPLEngine`.
  - Builds multi-turn SFT-ready conversations with real `[REPL OUTPUT]` turns.
  - Includes query generation per tool and registry-aware generation loops.

2. `src/verify_trajectories.py`
  - APIGen-style 3-stage verifier:
    - Stage 1: format
    - Stage 2: execution re-check against real REPL
    - Stage 3: semantic correctness (`predicted_tool == correct_tool`, params, registry membership)
  - Produces verified and rejected JSONL outputs with reject reasons.

3. `src/augment_trajectories.py`
  - Offline dataset scaling without external API calls:
    - Cross-registry re-grounding
    - Template-based synthesis
    - Query paraphrasing (templates + synonym substitutions)
  - Re-executes augmented code on real REPL before retaining examples.

4. `src/finetune.py`
  - QLoRA fine-tuning pipeline for `Qwen/Qwen3.5-4B*` using Unsloth + TRL SFT.
  - LoRA config: `r=32`, `alpha=32`, `dropout=0.05`, standard attention/MLP targets.
  - Includes dataset formatting, token-length filtering, train/eval split, checkpoint cleanup, optional merged and GGUF export.

5. `src/run_hf_comparison.py`
  - HuggingFace/Unsloth inference wrapper (`HFSLM`) and evaluation matrix runner.
  - Compares base vs fine-tuned performance across level/registry settings.
  - Writes summary and delta comparison JSON outputs.

### New Data Artifacts (Untracked, `data/`)
- `data/trajectories_raw.jsonl` — 117 lines
- `data/trajectories_verified.jsonl` — 117 lines
- `data/trajectories_raw_new.jsonl` — 40 lines
- `data/trajectories_verified_new.jsonl` — 40 lines
- `data/trajectories_augmented.jsonl` — 388 lines
- `data/trajectories_final.jsonl` — 388 lines
- `data/trajectories_final_qwen35.jsonl` — 418 lines

### New Fine-Tuning Artifacts (Untracked, `finetuned/`)
- Adapter package present at `finetuned/qwen3.55-4b-rlm-lora/` containing:
  - `adapter_config.json`
  - `adapter_model.safetensors`
  - `tokenizer.json`
  - `tokenizer_config.json`
  - `chat_template.jinja`
  - `README.md`
- Compiled kernel/cache outputs present: `unsloth_compiled_cache/` with 74 files.

### New Evaluation Result Files (Untracked, `results/` on 2026-03-12)
- `results/level0_small_10_20260312_172730.json`
- `results/level1_small_10_20260312_172804.json`
- `results/level2_small_10_20260312_172842.json`
- `results/level0_medium_25_20260312_172915.json`
- `results/level1_medium_25_20260312_173024.json`
- `results/level2_medium_25_20260312_173147.json`
- `results/level0_large_50_20260312_173308.json`

### Metrics Snapshot from Newly Added Result Files
1. **Small-10 registry (7 queries)**
  - Level 0: TSA 1.0000, PC 0.7143, RCV 1.0000, E2E 0.7143, avg turns 1.00, failure 0.0000
  - Level 1: TSA 0.8571, PC 0.8571, RCV 1.0000, E2E 0.8571, avg turns 3.43, failure 0.0000
  - Level 2: TSA 0.8571, PC 0.8571, RCV 1.0000, E2E 0.8571, avg turns 3.71, failure 0.0000

2. **Medium-25 registry (12 queries)**
  - Level 0: TSA 0.9167, PC 0.6667, RCV 1.0000, E2E 0.6667, avg turns 1.00, failure 0.0000
  - Level 1: TSA 0.9167, PC 0.9167, RCV 1.0000, E2E 0.9167, avg turns 4.00, failure 0.0833
  - Level 2: TSA 0.9167, PC 0.9167, RCV 1.0000, E2E 0.9167, avg turns 4.42, failure 0.0833

3. **Large-50 registry (25 queries, Level 0 run available in untracked set)**
  - Level 0: TSA 0.9600, PC 0.6400, RCV 1.0000, E2E 0.6400, avg turns 1.00, failure 0.0000

### Additional New Workspace Additions
- `.github/agents/AI Reseach RLM.agent.md` added (autonomous agent profile/config, 500+ lines).

### Post-Push Working Tree Summary (at audit time)
- Modified tracked files: 4
- Untracked high-level roots: `.github/`, `data/`, `finetuned/`, `results/`, `src/`, `unsloth_compiled_cache/`
- Untracked new source files: 5
- Untracked new result files: 7
- Untracked fine-tuned adapter files: 6

### Immediate Next Actions
- Stage curated subsets intentionally (`src/*`, config/requirements, selected `results/`) based on repo size policy.
- Decide whether `data/`, `finetuned/`, and `unsloth_compiled_cache/` should be committed, moved to release assets, or ignored.
- If committing datasets/artifacts, add versioning notes (generation date, model, and pipeline stage) in a companion manifest.

---

## 2026-03-15 — Benchmark-Aligned Base vs Fine-Tuned Qwen3.5 Evaluation (MCP/RLM Context)
**Session goal:** Use web-validated benchmark definitions (MCPMark, BFCL, τ²-bench) and run comparable base-vs-finetuned Qwen3.5 evaluations in this repository.

**Method:**
- Web verification of benchmark protocols and metrics:
  - MCPMark docs + intro (task scope, pass@1 / pass@K / pass^K / avg@K, 127-task benchmark)
  - BFCL leaderboard + BFCL evaluation methodology (multi-turn, multi-step, state + response checks)
  - τ²-bench repo + OpenReview paper page (pass^k reliability metric, dual-control agent-user-tool setup)
- Local execution in `astro` env:
  - Ran base model benchmark matrix via `src/run_hf_comparison.py` (single-model mode)
  - Ran fine-tuned LoRA benchmark matrix via separate process to avoid GPU memory carry-over
  - Generated `results/hf_base_vs_finetuned_comparison.json`

### Findings

1. **Benchmark protocol alignment (web-validated)**
- **MCPMark** focuses on MCP-native, verifiable task completion across service environments and reports pass-based aggregate metrics (pass@1 / pass@K / pass^K / avg@K). 🟢 EMPIRICAL
- **BFCL** emphasizes function/tool correctness under single-turn + multi-turn settings; newer variants include state-aware correctness in multi-step trajectories. 🟢 EMPIRICAL
- **τ²-bench** evaluates tool-agent-user interaction reliability and explicitly uses **pass^k** to capture consistency across repeated trials. 🟢 EMPIRICAL

2. **Metric mapping from our repo to external benchmarks**
- `tool_selection_accuracy` ≈ tool-call selection correctness (closest to core FC accuracy dimensions in BFCL/MCPMark). 🟡 ANALYTICAL
- `parameter_correctness` ≈ argument/schema correctness (BFCL-style argument fidelity). 🟡 ANALYTICAL
- `end_to_end_accuracy` ≈ pass@1-like success proxy for one run per task (not identical, but closest local equivalent). 🟡 ANALYTICAL
- `repl_code_validity` captures executable trajectory quality (closest to RLM/CodeAct-style execution robustness, complementary to pass metrics). 🟡 ANALYTICAL
- `failure_rate` is a practical reliability failure proxy (inverse signal to pass-style metrics). 🟡 ANALYTICAL

3. **Base vs Fine-tuned (Qwen3.5) — observed deltas (small_10 + medium_25 matrix)**
- **Level 1 (constrained REPL)** improved most after fine-tuning:
  - `small_10`: TSA +0.2857, RCV +0.2897, failure -0.2857
  - `medium_25`: TSA +0.0833, RCV +0.1125
- **Level 0 (no REPL)** remained flat or slightly worse on parameter correctness (as expected: this level does not exploit REPL trajectory learning).
- **Level 2 (few-shot REPL)** showed mixed behavior: RCV slightly improved, but TSA/E2E dropped on this sample.

4. **Interpretation for MCP+RLM context management**
- Fine-tuning on grounded trajectories appears to primarily strengthen **execution discipline and stability in constrained REPL loops** (Level 1), which is the key RLM bottleneck addressed by this project.
- Gains are not uniformly distributed across prompt regimes; additional data curation should prioritize Level-2-style trajectory diversity and hard cases.

### Artifacts Produced
- `results/hf_base_qwen35_4b_summary.json`
- `results/hf_finetuned_qwen355_4b_summary.json`
- `results/hf_base_vs_finetuned_comparison.json`

### Open Questions
- Convert current single-run `end_to_end_accuracy` into true pass^k-style reliability by running repeated trials per query/config.
- Add state-based end-state checks (BFCL-style) for richer multi-turn evaluation beyond FINAL extraction.
- Add MCPMark-compatible task wrappers for direct external benchmark comparability.

### Sources
- [MCPMark Docs, 2026] "MCPMark Introduction" — https://mcpmark.ai/docs/introduction
- [MCPMark, 2026] "MCPMark Homepage" — https://mcpmark.ai
- [Patil et al., 2025] "BFCL Leaderboard" — https://gorilla.cs.berkeley.edu/leaderboard.html
- [Mao et al., 2024] "BFCL V3 Multi-Turn & Multi-Step Evaluation" — https://gorilla.cs.berkeley.edu/blogs/13_bfcl_v3_multi_turn.html
- [Barres et al., 2025] "τ²-bench GitHub" — https://github.com/sierra-research/tau2-bench
- [Yao et al., 2025] "τ-bench (ICLR 2025)" — https://openreview.net/forum?id=roNSXZpUDN

---

## 2026-03-16 — Repo Update Log (Qwen3.5-Only + Comparison Pipeline Stabilization)

**Session goal:** Record all completed implementation updates before next experimentation cycle.

### Updates Logged
1. **Qwen version normalization (repo-wide):**
- Removed/updated all `Qwen2.5` references in active code/docs to keep the project strictly on `Qwen3.5` scope.
- Runtime defaults remain aligned to `qwen3.5:4b` in execution entrypoints.

2. **HF comparison pipeline stabilization (`src/run_hf_comparison.py`):**
- Refactored execution to support **single-model-per-process** benchmarking to avoid CUDA OOM from sequential dual-load.
- Added CLI options for explicit model run + post-hoc merge flow:
  - `--model-label`
  - `--model-name`
  - `--compare-only`
- Added explicit cleanup flow (object release + GC + CUDA cache clear) between phases.

3. **Benchmark execution completion (base vs fine-tuned):**
- Executed base model matrix run and persisted summary.
- Executed fine-tuned LoRA matrix run in separate process and persisted summary.
- Executed comparison merge step and generated delta report JSON.

### New/Updated Artifacts Confirmed
- `results/hf_base_qwen35_4b_summary.json`
- `results/hf_finetuned_qwen355_4b_summary.json`
- `results/hf_base_vs_finetuned_comparison.json`

### Recorded Outcome Snapshot
- Level 1 (constrained REPL) shows strongest post-finetune gains (tool selection + execution reliability).
- Level 0 remains mostly flat/slightly lower on parameter correctness.
- Level 2 remains mixed, indicating further trajectory diversity and repeated-trial evaluation are needed.

### Next Logged Priority
- Add repeated-trial pass-style reliability (`pass^k`-like) aggregation on top of current single-run metrics.

---

## 2026-03-16 — Phase A Implementation Start (Leakage Control + Reliability Metrics)

**Session goal:** Begin execution of the RP improvement roadmap with concrete code changes focused on evaluation validity and reproducibility.

### Implemented Changes
1. **Deterministic split manifests added (`train/dev/test`)**
- Updated `src/generate_test_data.py` to create:
  - `test_data/queries/train_queries.json`
  - `test_data/queries/dev_queries.json`
  - `test_data/queries/test_queries.json`
  - `test_data/ground_truth/train_expected_selections.json`
  - `test_data/ground_truth/dev_expected_selections.json`
  - `test_data/ground_truth/test_expected_selections.json`
- Split generation is deterministic (`seed=42`) and keeps discovery/simple mix explicit.

2. **Training leakage path reduced in trajectory generation**
- Updated `src/generate_trajectories.py` to support `--seed-query-split` with options:
  - `none` (default, leakage-safe)
  - `train`, `dev`, `test`, `all`
- Default behavior now avoids auto-injecting benchmark queries.
- Added warnings for risky settings (`test` / `all`).

3. **Split-aware evaluation runners**
- Updated `src/run_baseline.py`:
  - Added `--query-split {all,train,dev,test}`
  - Added `resolve_eval_paths(...)` for split-specific files.
- Updated `src/run_hf_comparison.py`:
  - Added split-aware case loading via `--query-split`.
  - Added configurable matrix registries via `--registries`.

4. **Repeated-trial reliability metrics + confidence intervals**
- Updated `src/evaluator.py` with `evaluate_repeated(...)`:
  - pass-style metrics: `pass_at_1`, `pass_at_k`, `pass_power_k_proxy`
  - mean-over-trials reliability for TSA/PC/failure
  - bootstrap confidence intervals (95% default)
- Updated `src/run_baseline.py` to expose `--trials` and execute repeated-trial mode.
- Updated `src/run_hf_comparison.py` to expose `--trials` and include reliability deltas in comparison output when available.

### Validation Completed (astro environment)
- Regenerated test data successfully with new split files.
- CLI sanity checks passed for:
  - `python -m src.run_baseline --help` (shows `--query-split`, `--trials`)
  - `python -m src.run_hf_comparison --help` (shows `--query-split`, `--trials`, `--registries`)

### Next Step
- Implement scaffold policy upgrades (`DISCOVER → VERIFY → DECIDE`) and failure-aware recovery logic to address Level-2 instability.

---

## 2026-03-16 — Phase B Implementation Progress (Supervision + Semantic Verification)

**Session goal:** Continue roadmap execution by improving training supervision quality and semantic verification strictness.

### Implemented Changes
1. **Assistant-only loss masking in fine-tuning** (`src/finetune.py`)
- Added explicit response-only masking hook via Unsloth chat-template utility.
- Added CLI controls:
  - `--no-assistant-only-loss`
  - `--assistant-marker`
  - `--user-marker`
  - `--allow-unmasked-fallback`
- Added runtime label-mask diagnostics to report active label ratio in train batches.

2. **Parameter-grounded semantic checks** (`src/verify_trajectories.py`)
- Added expected-parameter lookup loading from known GT manifests (`all/train/dev/test`).
- Stage 3 now validates:
  - required schema params exist and are non-empty
  - expected param values match when query is known in manifests
- Added more specific semantic reject reasons (missing required param, mismatch keys/values).

### Validation Completed (astro environment)
- `python -m src.verify_trajectories --help` executes cleanly.
- `python -m src.finetune --help` executes cleanly and shows new masking controls.
- Static error checks report no errors in modified files.

### Next Step
- Implement task-mode separation for evaluation/prompting (tool-selection vs discovery) and finalize Level-2 prompt simplification for cleaner E2E accounting.

---

## 2026-03-16 — Phase B Implementation Progress (Task Modes + Prompt Simplification + Significance)

**Session goal:** Implement the remaining roadmap items for cleaner task accounting and stronger statistical reporting.

### Implemented Changes
1. **Discovery vs selection task-mode separation**
- Updated `src/evaluator.py` to support discovery query scoring:
  - Added `parse_discovery_answer(...)`.
  - In discovery mode (`expected_tool == DISCOVERY` or `category == discovery`), success is evaluated as successful non-empty `FINAL(...)` answer.
- Updated `src/run_baseline.py`:
  - Added `--task-mode {selection,discovery,mixed}`.
  - Filtering now supports explicit task subsets.
- Updated `src/run_hf_comparison.py`:
  - Added `--task-mode {selection,discovery,mixed}`.
  - Registry filtering now respects selected task mode.

2. **Level-2 prompt simplification**
- Rewrote `prompts/level2.txt` with a more compact contract:
  - Explicit `DISCOVER → VERIFY → DECIDE` workflow.
  - One-code-block-per-turn rule.
  - Separate output contracts for selection vs discovery tasks.
  - Reduced examples to one clear selection and one discovery pattern.

3. **Significance-style statistics in comparison outputs**
- Extended repeated-trial evaluator outputs in `src/evaluator.py`:
  - Added `mean_e2e_over_trials` + bootstrap CI.
- Extended `src/run_hf_comparison.py` comparison writer:
  - Adds approximate delta CI for key deltas when CIs are available.
  - Adds `excludes_zero` significance flag for each supported delta.

### Validation Completed (astro environment)
- `python -m src.run_baseline --help` shows `--task-mode` and runs clean.
- `python -m src.run_hf_comparison --help` shows `--task-mode` and runs clean.
- Static error checks: no errors in updated Python files.

### Next Step
- Implement query-level paired significance testing (beyond CI overlap approximation) and add explicit markdown stats report generation for RP appendices.

---

## 2026-03-16 — First Repeated-Trial Significance Run (Dev Split, Selection Mode)

**Session goal:** Execute the first end-to-end repeated-trial benchmark comparison with significance-bearing outputs.

### Execution (astro environment)
- Base model run:
  - `python -m src.run_hf_comparison --model-label base_qwen35_4b --model-name Qwen/Qwen3.5-4B --query-split dev --task-mode selection --trials 5 --registries small_10 medium_25`
- Fine-tuned model run:
  - `python -m src.run_hf_comparison --model-label finetuned_qwen355_4b --model-name finetuned/qwen3.55-4b-rlm-lora --query-split dev --task-mode selection --trials 5 --registries small_10 medium_25`
- Merge comparison:
  - `python -m src.run_hf_comparison --compare-only --base-summary results/hf_base_qwen35_4b_summary_dev.json --finetuned-summary results/hf_finetuned_qwen355_4b_summary_dev.json`

### Artifacts Produced
- `results/hf_base_qwen35_4b_summary_dev.json`
- `results/hf_finetuned_qwen355_4b_summary_dev.json`
- `results/hf_base_vs_finetuned_comparison.json`
- `results/hf_dev_selection_trials5_summary.json`

### Observed Pattern
- On this small dev split (1 query in small_10, 3 queries in medium_25), fine-tuned model underperformed base in several Level 0 and Level 2 cells.
- Several delta CIs marked `excludes_zero` for negative deltas where outcomes were deterministic across repeated trials.
- This run is useful as a reliability sanity check but not sufficient alone for broad claims due to very small dev sample size.

### Next Step
- Run repeated-trial comparison on larger coverage (`query_split=all`, `registries=small_10 medium_25 large_50 xlarge_100`) and then add query-level paired significance tests.

---

## 2026-03-16 — Systematic Execution Ledger Consolidation

**Session goal:** Consolidate all implementation work completed so far into the main project documentation and explicitly separate completed work, in-progress work, and remaining work.

### Documentation updates completed
1. `research/plan.md`
- Expanded from a roadmap into a status-bearing execution ledger.
- Added per-step implementation status:
  - completed
  - partially completed
  - in progress
  - not started
- Added a detailed remaining-work summary.

2. `research/rp_strong_results_improvement_plan.md`
- Added an implementation-status section summarizing roadmap execution state.

3. `research/rp_analysis_summary_2026_03_16.md`
- Updated from diagnosis-only summary to diagnosis + current execution state.

4. `research/base_vs_finetuned_comparative_analysis.md`
- Added a post-analysis update section documenting what changed after the original write-up.

### Current implementation state at consolidation time

**Completed core infrastructure:**
- Split-safe query manifests
- Leakage-safe trajectory seeding
- Split-aware runners
- Repeated-trial evaluator
- Bootstrap confidence intervals
- Discovery-vs-selection task modes
- Simplified Level-2 prompt
- Stateful scaffold recovery logic
- Assistant-only loss masking
- Parameter-grounded Stage-3 verification
- HF comparison markdown report generation
- HF comparison reproducibility manifest generation
- Query-level paired significance outputs

**In progress:**
- Broad repeated-trial HF comparison across all registries in selection mode

**Still pending:**
- Trajectory provenance metadata enrichment
- Split-manifest-aware fine-tuning train/eval partitioning
- Hard-negative / correction-turn data generation
- Preference-tuning stage
- Ablation harness and appendix-grade automated reporting across all pipelines

### Why this consolidation matters
- The project is now beyond research design and partial prototyping.
- A substantial portion of the measurement and control-stack implementation is complete.
- The main unresolved risk is no longer missing infrastructure; it is whether broader empirical evidence will support the intended RP narrative after stricter evaluation.

### Next Step
- Wait for the active all-registry repeated-trial run to finish, then convert its outputs into a consolidated RP-facing result report and use that report to prioritize the next training/data interventions.


