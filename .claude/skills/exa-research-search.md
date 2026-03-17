# Exa Research Paper Search

## What This Skill Is
How to use the Exa MCP server for academic paper search in the RLM×MCP project. Covers standard search, deep research synthesis, structured output extraction, and Phase D paper writing workflows.

## When to Use
- Finding related work for any claim in the paper (Phase D)
- Checking if a result or approach is novel before claiming it
- Finding the correct arXiv ID for a paper you want to cite
- Synthesizing a related work section from scratch
- Quick lookup: "what did X paper actually say about Y?"

---

## Quick Reference: Tool Selection

| Task | Tool | Key Parameters |
|------|------|---------------|
| Find specific paper | `exa.search` | `type: "auto"`, `query: "title or author"` |
| Find papers on a topic | `exa.search` | `type: "deep"`, `category: "research paper"` |
| arXiv-only search | `exa.search` | `includeDomains: ["arxiv.org"]` |
| Synthesize a topic | `exa.deep_researcher_start` | `topic: "..."` then poll with `deep_researcher_check` |
| Extract structured citations | `exa.search` + `outputSchema` | See structured output section |

---

## Standard Research Paper Search

Use for: finding 5–20 papers on a topic, checking related work exists.

```
Tool: exa.search
Parameters:
  query: "tool selection small language models context compression"
  type: "deep"
  category: "research paper"
  includeDomains: ["arxiv.org", "semanticscholar.org", "aclanthology.org"]
  numResults: 10
```

**When NOT to use `category: "research paper"`:** If you want blog posts, GitHub repos, or documentation alongside papers, omit the category filter.

---

## Deep Researcher (Phase D — Related Work Section)

Use for: synthesizing an entire sub-topic from scratch. Returns a structured report with citations.

**Step 1 — Start the job:**
```
Tool: exa.deep_researcher_start
Parameters:
  topic: "reinforcement learning language model tool use context efficiency"
```
Returns: `{ "job_id": "xxx", "status": "running" }`

**Step 2 — Poll until done:**
```
Tool: exa.deep_researcher_check
Parameters:
  job_id: "xxx"
```
Poll every 30 seconds. Status changes: `running → done`. Takes 30–120 seconds.

**Output:** Structured report with summary, key findings, and cited URLs. Use citations directly in RESEARCH_LOG.md entries.

**When to use deep researcher vs. standard search:**
- Standard search: you know roughly what you're looking for (1 specific topic)
- Deep researcher: you need a comprehensive survey of a sub-field (Phase D related work)

---

## Structured Output Extraction

Use when you want to extract specific fields (title, year, abstract, arXiv ID) from search results:

```
Tool: exa.search
Parameters:
  query: "MCP tool selection benchmark evaluation"
  type: "deep"
  category: "research paper"
  outputSchema: {
    "type": "object",
    "properties": {
      "title": {"type": "string"},
      "year": {"type": "integer"},
      "arxiv_id": {"type": "string"},
      "abstract_summary": {"type": "string"}
    }
  }
```

Returns structured objects instead of raw text — easier to paste into RESEARCH_LOG.md citation lists.

---

## Key Papers to Search For (This Project)

Run these searches at the start of Phase D:

| Paper | Search Query | Why |
|-------|-------------|-----|
| MCP Bridge | `"MCP Bridge tool selection arXiv 2504.08999"` | Primary baseline — 73% F1 on MCPToolBench |
| RLM original | `"RLM reinforcement learning memory Python REPL context"` | Framework we adapt |
| APIGen | `"APIGen API generation training data Liu 2024"` | Verification pipeline citation |
| Qwen2.5-Coder | `"Qwen2.5-Coder code language model"` | Model choice citation |
| MCPToolBench | `"MCPToolBench benchmark MCP evaluation"` | Eval benchmark citation |
| SLM context windows | `"small language model context window limitation 1B 3B"` | Motivation section |

---

## Exa Search Patterns by Phase

**Phase 1.5 (current):** Not needed — debugging scaffold, no literature needed.

**Phase A:** Not needed — trajectory generation is engineering, not literature.

**Phase B (fine-tuning):**
- `"QLoRA fine-tuning instruction following tool use"` — training methodology
- `"Unsloth Qwen fine-tuning memory efficient"` — implementation reference

**Phase C (eval matrix):**
- `"few-shot prompting tool selection improvement"` — explaining L2 > L1 finding
- `"registry size scaling tool retrieval accuracy"` — explaining degradation curve

**Phase D (paper):** Use deep researcher for each section:
- Introduction: `"MCP adoption Claude API tool calling growth 2024 2025"`
- Related work: `"context compression SLM tool selection retrieval"`
- Methodology: `"REPL Python execution trajectory generation training data"`
- Conclusion: `"fine-tuning small models tool use alignment"`

---

## Free Tier Limits

- Exa free tier: 1,000 searches/month
- Deep researcher: counts as ~5–10 standard searches per job
- Budget: 1,000 searches is more than enough for this project (Phase D needs ~50–100 total)
- API key is in `.claude/settings.json` under the exa MCP HTTP transport URL

---

## Integration Notes

The Exa MCP is configured as HTTP transport in `.claude/settings.json`:
```json
"exa": {
  "type": "http",
  "url": "https://mcp.exa.ai/mcp?exaApiKey=<key>"
}
```

This is Exa's recommended approach — more reliable than the npx package, no local install needed, always uses latest API version.
