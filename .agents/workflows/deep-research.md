---
description: Systematic deep research workflow using web search, paper reading, and evidence synthesis. Use when researching any topic related to the SLM × MCP × RLM project.
---

# Deep Research Workflow

Use this workflow whenever conducting research on a topic. Every step must produce traceable, cited evidence.

## Prerequisites
- Read `AGENTS.md` for project context
- Read `.agents/skills/research/SKILL.md` for research conventions
- Have `research/RESEARCH_LOG.md` open for appending findings

---

## Step 1: Define the Research Question
Before searching, write down:
- **Exact question** we're trying to answer
- **Why it matters** for our SLM × MCP × RLM project
- **What would a good answer look like** (what evidence would be compelling?)

Append a new entry header to `research/RESEARCH_LOG.md`.

## Step 2: Broad Search Sweep (3–5 queries)
// turbo-all
Run at least 3 diverse `search_web` queries:
1. **Direct query**: The exact research question
2. **Adjacent query**: Related but different angle (e.g., if researching "SLM tool use", also search "small language model function calling")
3. **Negation query**: Search for failures/limitations (e.g., "small language models limitations tool calling")
4. **(Optional) Competition query**: What solutions already exist?
5. **(Optional) Benchmark query**: How is performance measured in this area?

Record all queries and their result summaries.

## Step 3: Read Primary Sources
For the top 3–5 most relevant results:
1. Use `read_url_content` to extract the full text
2. For arXiv papers, read the abstract, introduction, methodology, and results sections
3. Extract: key claims, experimental setup, metrics, limitations
4. Note the citation count and publication venue (if available)

## Step 4: Build Evidence Table
Create a structured comparison table:

| Claim/Finding | Source | Evidence Type | Relevance to Our Project |
|---------------|--------|---------------|--------------------------|
| [Finding] | [Author, Year] | 🟢/🟡/🔵/🔴 | [How it connects] |

## Step 5: Identify Gaps
Answer:
- What has **NOT** been studied in this area?
- What assumptions in existing work **don't hold** for SLMs?
- Where is there **conflicting evidence**?

## Step 6: Synthesize & Document
Write findings to the appropriate research document:
- Append to `research/RESEARCH_LOG.md` with full entry format
- If this is a major topic, create a dedicated `research/[topic].md` file
- Always include: Sources section, Open Questions, and Implications for our project

## Step 7: Cross-Reference
Check if findings connect to or contradict:
- Previous entries in `research/RESEARCH_LOG.md`
- The RLM paper findings (see `references/RLM.pdf` analysis)
- User's intuitions from `references/rlm_ref.txt`

Flag any contradictions or confirmations explicitly.
