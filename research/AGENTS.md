# AGENTS.md — RLM × MCP Research Project

## Project Overview
This workspace is a **research project** investigating how to **scale MCP (Model Context Protocol) access for Small Language Models (SLMs) via the REPL-based approach** introduced in the Recursive Language Models (RLM) paper (Zhang et al., 2025).

**Core Thesis:** SLMs (~1B–7B parameters) cannot effectively use MCP servers because the tool descriptions alone saturate their small context windows. By adapting the RLM framework — which offloads context into a persistent Python REPL environment — we aim to enable SLMs to interact with MCP tools without context window overflow.

## Directory Layout
```
rlm/
├── AGENTS.md                          # ← You are here. Project-level agent context.
├── .agents/
│   ├── skills/
│   │   └── research/
│   │       └── SKILL.md               # Research agent skill definition
│   └── workflows/
│       ├── deep-research.md           # Workflow: deep web + paper research
│       └── paper-analysis.md          # Workflow: systematic paper analysis
├── references/
│   ├── RLM.pdf                        # Primary reference paper
│   ├── rlm_ref.txt                    # User's initial notes & intuitions
│   └── name CVResearchEngineer.md     # Reference agent scaffold example
└── research/
    └── RESEARCH_LOG.md                # Living research log (append-only)
```

## Key References
| Reference | Path | Purpose |
|-----------|------|---------|
| RLM Paper | `references/RLM.pdf` | Primary methodology source — "Recursive Language Models" by Alex L. Zhang et al. |
| User Notes | `references/rlm_ref.txt` | User's initial understanding and intuitions from the RLM paper |
| Scaffold Example | `references/name CVResearchEngineer.md` | Reference for agent scaffold design (different project) |

## Research Context

### The RLM Paper — Key Technical Details
- **Core idea:** Treat LLM input as an *external environment variable* in a Python REPL rather than feeding it directly into the context window.
- **Mechanism:** The LLM writes/executes Python code to interact with its context — regex filtering, chunking, keyword search, sub-LM calls.
- **Recursive sub-calls:** The root LLM can spawn sub-LLM queries on relevant context slices, enabling arbitrarily deep reasoning chains.
- **Scale:** Demonstrated on inputs >10M tokens, outperforming base models and existing scaffolds by up to 2× while keeping costs comparable.
- **Limitation (critical for us):** "Models without sufficient coding capabilities struggle as RLMs" — this is the central challenge for SLMs.

### Our Research Questions
1. **Can the RLM REPL approach extend effective context for SLMs?** (beyond their 2K–8K native windows)
2. **Can this enable SLMs to access MCP servers?** (where tool descriptions alone may consume the entire context)
3. **How do we address the coding/reasoning gap in SLMs?** (they can't write the code needed to navigate the REPL)

### Known Bottlenecks
- SLMs lack the coding/reasoning capability to generate proper REPL interaction code
- SLMs can't create proper Chain-of-Thought within the RLM scaffold
- Potential solutions under investigation: fine-tuning on synthetic RLM trajectories, predefined Python scripts, RLM-no-sub-calls mode

## Agent Conventions
1. **Always cite sources.** Every claim must reference a paper, article, or empirical result. No hallucinated metrics.
2. **Use `research/RESEARCH_LOG.md` as the living document.** Append findings, never overwrite.
3. **Web search is mandatory** for every research step. Don't rely solely on the model's training data.
4. **Distinguish clearly** between: (a) established facts from papers, (b) user's intuitions, (c) agent's analysis/suggestions.
5. **Feasibility lens:** All proposals must consider that the team consists of TY engineering students with minimal-to-zero funding (consumer GPUs at best, free-tier APIs, open-source models only).
