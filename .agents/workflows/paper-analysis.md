---
description: Systematic paper analysis workflow for extracting deep, implementation-level insights from academic papers. Use when analyzing any research paper relevant to the project.
---

# Paper Analysis Workflow

Use this workflow when analyzing any research paper in detail. The goal is NOT a surface-level summary — it's extracting specific technical details that inform our project.

## Prerequisites
- Read `AGENTS.md` for project context
- Read `.agents/skills/research/SKILL.md` for research conventions
- Ensure the paper PDF is in `references/` directory

---

## Step 1: Extract Paper Text
```python
# Use PyMuPDF to extract text from the PDF
import fitz
doc = fitz.open('references/<paper>.pdf')
for page in doc:
    text = page.get_text()
    # Process page by page
```

Record: Title, Authors, Year, Venue/ArXiv ID, Citation count (via web search)

## Step 2: Structural Mapping
Create a section-by-section outline:
- What is each section about?
- How many pages for each section?
- Where are the key figures and tables?
- What is in the appendix?

## Step 3: Deep Extraction (Section by Section)

### 3a. Problem Statement & Motivation
- What exact problem are they solving?
- What are the baseline methods they compare against?
- What metrics do they use?
- **Our lens:** Does this problem statement overlap with ours?

### 3b. Methodology
- Exact system architecture (draw it out if needed)
- Algorithm pseudocode (extract or reconstruct)
- Implementation details: languages, libraries, API calls, hardware
- System prompts (extract verbatim if provided)
- **Our lens:** What can we reuse? What must we modify for SLMs?

### 3c. Experiments & Results
- Extract ALL numerical results into a table
- Note which results are statistically significant vs. not
- Identify the ablation studies — these reveal what matters most
- **Our lens:** Which ablation is closest to our SLM scenario?

### 3d. Limitations & Future Work
- Author-stated limitations
- Unstated limitations you can identify
- Future work suggestions — are any aligned with our direction?
- **Our lens:** Can our paper address any of these gaps?

### 3e. Related Work Section
- List all cited papers that are relevant to our project
- Group them by theme (long context, tool use, SLMs, fine-tuning, etc.)
- Flag papers we should read next

## Step 4: Cross-Reference via Web Search
For the paper's key claims:
1. Search for subsequent papers that cite this work
2. Search for implementations or reproductions
3. Search for critiques or alternative approaches
4. Check if the paper's benchmarks are still considered standard

## Step 5: Produce Analysis Document
Write to `research/<paper_name>_deep_dive.md` with this structure:

```markdown
# Deep Dive: [Paper Title]
**Authors:** [...]  **Year:** [...]  **Venue:** [...]

## TL;DR (3 sentences max)

## Technical Architecture
[Detailed description with diagrams if needed]

## Key Results Table
| Benchmark | Method | Score | Cost | Notes |
|-----------|--------|-------|------|-------|

## Ablation Insights
[What ablations reveal about what actually matters]

## Relevance to Our Project
[Specific connections to SLM × MCP × RLM]

## What We Can Reuse
[Concrete techniques, prompts, architectures]

## What We Must Change
[What doesn't work for SLMs and why]

## Papers to Read Next
[From references section, prioritized]

## Sources
[Full reference list]
```

## Step 6: Update Research Log
Append a summary entry to `research/RESEARCH_LOG.md` noting:
- Paper analyzed
- Key takeaways for our project
- New questions raised
