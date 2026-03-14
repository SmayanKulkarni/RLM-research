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

**Method:** 15 web searches covering: MCP JSON schema specification, SLM CoT/tool-calling benchmarks (BFCL, MCPMark, TAU-Bench), MCP scaling for SLMs, DisCIPL verification, THREAD verification, CodeAct analysis, REPL alternatives for agents, fine-tuning vs prompting vs scaffolding, SLM HumanEval benchmarks, xLAM function calling, GRPO/RL for tool calling, knowledge distillation approaches, MCP dynamic/lazy loading, and Qwen2.5-Coder capabilities.

### Key Findings

1. **MCP has highly exploitable structure** — fixed JSON schema (name/description/inputSchema) makes tool selection closer to information retrieval than open-ended reasoning. Multi-level lazy loading achieves 98.7% token reduction.
2. **Nobody has done exactly what we're proposing** — xLAM approached function calling differently (purpose-built, no REPL); no RLM+MCP+SLM work exists.
3. **xLAM-1B surpasses GPT-3.5 on function calling** — proves 1B models can learn tool use with enough targeted data (60K examples via APIGen).
4. **GRPO (RL with verifiable rewards) is a strong alternative to SFT** — perfect for MCP tool selection where rewards are programmatically verifiable.
5. **Qwen2.5-Coder-3B is the optimal primary model** — ~80% HumanEval, Unsloth-supported, GRPO-compatible, tool calling supported.
6. **All major citations verified** — DisCIPL (arXiv:2504.07081, COLM 2025), THREAD (NAACL 2025), CodeAct (arXiv:2402.01030).

### Deliverables Created
- `research/implementation_research_q_and_a.md` — Full answers to 7 questions with citations and implementation roadmap

### Open Questions
- What is the actual zero-shot performance of Qwen2.5-Coder-3B on MCP tool selection? (needs empirical testing)
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
    - from `qwen2.5-coder:3b`
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

