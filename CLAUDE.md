# RLM × MCP — Research Workspace

## Thesis (3 sentences)
SLMs (1B–7B params) have context windows (2K–8K tokens) too small to load 50 MCP tool schemas (~20K tokens). We adapt the RLM framework — which externalizes context into a persistent Python REPL — so SLMs navigate a `tools_registry` API instead of loading all schemas at once. We generate synthetic REPL trajectories and fine-tune Qwen2.5-Coder-3B (QLoRA) to improve REPL interaction capability.

## Current Phase
- Phase 0 DONE: L0=100% TSA (zero-shot baseline, all schemas in context)
- Phase 1 DONE: L1=28–33% TSA, 71% fail rate. Root cause: SLM hallucinates REPL output in one turn
- Phase 1.5 IN PROGRESS: stop_sequences=["</code>"], XML delimiters, check reorder in scaffold.py
- Phase A (Week 1): prepare_toucan → generate_trajectories (10k) → generate_negatives (3k) → build_splits
- Phase B (Week 2): SFT Qwen2.5-3B + optional DPO on Colab T4 or RTX 4070Ti
- Phase C (Week 3): Full eval matrix (all levels × all registries), ablations
- Phase D (Week 4): Paper writing

## Source File Map
```
src/scaffold.py              MCPRLMScaffold: SLM↔REPL loop  ← THIS IS THE AGENT HARNESS
src/repl_engine.py           REPLEngine + REPLConfig (3 levels of allowed imports/builtins)
src/slm_interface.py         SLMInterface: Ollama client, stop_sequences=["</code>"]
src/tool_registry.py         MCPToolRegistry: constrained API (list_names, search, get_schema)
src/evaluator.py             MCPToolEvaluator: 7 metrics (TSA, PC, RCV, E2E, turns, savings, fail%)
src/run_baseline.py          CLI: python -m src.run_baseline --level 1 --registry small_10 [--all]
src/generate_trajectories.py Groq API (llama-3.3-70b-versatile) → real REPL execution → JSONL
src/verify_trajectories.py   3-stage APIGen: format → execution → semantic
src/augment_trajectories.py  Cross-registry re-grounding + paraphrase augmentation
src/build_splits.py          Stratified train/dev/test split builder (80/10/10)
src/prepare_toucan.py        HF: Agent-Ark/Toucan-1.5M → REPL format adapter
src/generate_negatives.py    Hard negative generation for DPO training
```

## Key Data Paths
```
test_data/tool_registries/        small_10.json, medium_25.json, large_50.json, xlarge_100.json
test_data/queries/                all_queries.json (simple/multi_select/discovery categories)
test_data/ground_truth/           expected_selections.json  [SEALED — never modify]
data/trajectories_raw.jsonl       raw Groq output (pre-verification)
data/trajectories_verified.jsonl  post-verification (3-stage APIGen filter)
data/splits/{train,dev,test}.jsonl
results/                          {level}_{registry}_{timestamp}.json per run
research/RESEARCH_LOG.md          append-only living research log
```

## Common CLI Commands
```bash
python -m src.run_baseline --level 0 --registry small_10
python -m src.run_baseline --all                            # WARNING: skip xlarge_100 on M1
python -m src.generate_trajectories --registries small_10 medium_25 --queries-per-tool 8
python -m src.verify_trajectories --input data/trajectories_raw.jsonl
python -m src.augment_trajectories --input data/trajectories_verified.jsonl
python -m src.build_splits
python -m src.prepare_toucan --output data/toucan_repl.jsonl --max-examples 20000
```

## Hardware & API Constraints (HARD LIMITS — never violate)
- M1 MacBook Air 8GB: Ollama only. No GPU training. No xlarge_100 baseline (OOM).
- Colab T4 16GB (free): QLoRA Qwen2.5-3B at 4-bit ~3.5GB VRAM. Checkpoint every 500 steps.
- RTX 4070Ti 12GB: Available via teammate for Qwen3.5-4B or larger runs.
- GROQ_API_KEY in .env: free tier ~30 req/min. ALWAYS sleep(2.0) between calls.
- Ollama: local at localhost:11434. Run `ollama list` to verify model is loaded.

---

## ═══ MODEL SELECTION — AUTONOMOUS RULES ═══

**Follow these rules automatically, without being asked:**

| Task Type | Model | Rationale |
|-----------|-------|-----------|
| Reading files, searching code, directory listing | **Haiku** | Delegation via Task tool |
| Writing Python functions, fixing bugs, CLI | **Sonnet** | Default for 90% of work |
| Analyzing eval results, writing summaries | **Sonnet** | Standard analysis |
| Designing new pipeline stages / REPL levels | **Opus** | Architecture decisions |
| Debugging root cause spanning 3+ source files | **Opus** | Deep multi-file tracing |
| Writing paper sections (intro, methodology, related work) | **Opus** | Academic writing quality |
| Interpreting unexpected metric patterns (why L2 > L1?) | **Opus** | Cross-domain reasoning |
| Subagent file-reading tasks | **Haiku** | CLAUDE_CODE_SUBAGENT_MODEL is set |

**Proactive escalation:** If a task escalates mid-session from simple to architectural, say:
> "This spans 3+ source files and requires cross-domain reasoning — recommend `/model claude-opus-4-6`."

**Never use Opus for:** File reads, CLI runs, result formatting, single-file edits.

---

## ═══ SUBAGENT DELEGATION RULES ═══

**Delegate to subagents when:**
1. Task involves reading 5+ files without writing (use Haiku subagent via Task tool)
2. Task is clearly separable (trajectory generation ≠ eval analysis)
3. Parallel workstreams needed (Phase A: TOUCAN + trajectories + negatives)

**Available project agents** (in `.claude/agents/`):
- `research-orchestrator` — Coordinates multi-phase research, assembles RESEARCH_LOG entries
- `eval-agent` — Runs baselines, diagnoses failure modes, recommends next experiments
- `trajectory-agent` — Manages Groq API trajectory generation with rate-limit compliance
- `python-ml-reviewer` — Reviews Python/ML code (research-specific checklist)
- `data-pipeline-agent` — TOUCAN processing, augmentation, split building

**Context passing:** Before delegating, write task context to `SHARED_TASK_NOTES.md`. Each agent reads this on start.

---

## ═══ AGENT HARNESS FRAMEWORK ═══

`MCPRLMScaffold` IS an agent harness. When improving it, apply these 4 constraints:
1. **Action space quality** — `MCPToolRegistry` API: narrow schemas, stable names, predictable outputs
2. **Observation quality** — `REPLEngine` output: clear status, truncation limits, actionable errors
3. **Recovery quality** — XML `<code>` fallback, nudge messages, max_turns cap
4. **Context budget** — `output_chars` limits (2000/4000), system prompt size, conversation growth

See: `.claude/skills/agent-harness-construction.md`

---

## ═══ EXA PAPER SEARCH (Phase D) ═══

Use the `exa` MCP for all academic paper searches. Prefer over web search for research tasks.

**Standard paper search** (finding related work, checking citations):
```
exa.search: query="MCP tool selection small language models", category="research paper", type="deep", includeDomains=["arxiv.org", "semanticscholar.org"]
```

**Deep researcher** (synthesize a topic from scratch, Phase D related work section):
```
exa.deep_researcher_start: topic="reinforcement learning from language feedback tool use SLMs"
→ returns job_id
exa.deep_researcher_check: job_id=<id>   ← poll until status="done"
```

**When to use each tool:**
| Task | Tool | Notes |
|------|------|-------|
| Find a specific paper by title/author | `exa.search` | type="auto" |
| Find all papers on a topic | `exa.search` | type="deep", category="research paper" |
| Background reading for a section | `exa.search` | includeDomains=["arxiv.org"] |
| Synthesize entire related work section | `exa.deep_researcher_start` | Async, takes 30–120s |
| Check if your approach is novel | `exa.search` | Use thesis keywords |

**Key papers to have exa find when writing the paper:**
- MCP Bridge (arXiv:2504.08999) — our primary baseline
- RLM original paper — cite for REPL framework
- APIGen [Liu et al., 2024] — cite for verification pipeline
- Qwen2.5-Coder paper — cite for model choice
- Unsloth/QLoRA papers — cite for fine-tuning approach

See: `.claude/skills/exa-research-search.md` for full patterns.

---

## ═══ RESEARCH STANDARDS ═══
- Cite every factual claim: [Author et al., Year]
- Evidence labels: EMPIRICAL / ANALYTICAL / INTUITION / UNVERIFIED
- Append to `research/RESEARCH_LOG.md` — never overwrite existing entries
- Every experiment: save results to `results/{level}_{registry}_{timestamp}.json`

---

## ═══ SESSION PROTOCOL ═══
**Session start:** Read last 3 entries of `research/RESEARCH_LOG.md`, check `SHARED_TASK_NOTES.md`, report current phase + immediate next action in 1–2 sentences.

**Session end:** Use `/save-research-session` command to write structured session state. Update `SHARED_TASK_NOTES.md`.

**Long experiments (>30 min):** Prepare and verify the command, then have the user run it with `nohup` in a separate terminal. Do not babysit long-running scripts inside Claude.

---

## DO NOT
- Commit `.env` (contains GROQ_API_KEY and OPENAI_API_KEY)
- Call Groq API without `sleep(2.0)` between requests
- Modify `test_data/ground_truth/` — it is the sealed eval set
- Run `--all` baselines including `xlarge_100` on M1 (OOM)
- Auto-commit `data/` or `results/` files without human review
- Use Opus for file reads, simple code edits, or CLI execution
