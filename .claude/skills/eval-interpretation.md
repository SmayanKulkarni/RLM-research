# Evaluation Interpretation

## What This Skill Is
How to read, interpret, and act on RLM×MCP evaluation results. Covers the 7-metric framework, diagnostic decision trees, registry size effects, and how to write results into RESEARCH_LOG.md.

## When to Use
- After running `python -m src.run_baseline` and seeing results
- When comparing multiple runs to find the best config
- When writing a RESEARCH_LOG entry about experimental results
- Use with eval-harness.md for the full evaluation framework

---

## Quick Reference: What Good Numbers Look Like

| Metric | Broken | Needs Work | Good | Great |
|--------|--------|-----------|------|-------|
| TSA | <30% | 30–50% | 50–70% | >70% ← beats MCP Bridge |
| RCV | 0% ← bug | <50% | 50–80% | >80% |
| Fail% | >60% | 30–60% | 10–30% | <10% |
| Avg Turns | 1 ← guessing | 2–4 | 4–7 | — |
| Context Savings | <50% | 50–80% | 80–95% | >95% |

**Research target:** TSA > 73% (beats MCP Bridge F1 on MCPToolBench)

---

## Diagnostic Decision Tree

```
STEP 1: Is RCV = 0%?
  YES → CRITICAL BUG: REPL not executing
      → Check: scaffold.py — is FINAL() check BEFORE code extraction?
      → Fix: reorder to extract + execute code FIRST, then check for FINAL()
      → Check: slm_interface.py — are stop_sequences=["</code>"] active?
  NO → Continue

STEP 2: Is fail_rate > 50%?
  YES → Scaffold instability
      → Check: raw_slm_outputs for FINAL() presence
      → If FINAL() missing: prompt not clear enough about format
        → Check: prompts/level{n}.txt — are XML instructions explicit?
      → If FINAL() present but not parsed: check evaluator.py parsing regex
  NO → Continue

STEP 3: Is TSA < 30%?
  YES → SLM not finding the right tool
      → Check: avg_turns — if avg_turns ≈ 1, SLM is guessing not navigating
        → avg_turns=1 means code blocks aren't executing (back to RCV)
        → avg_turns=8+ means SLM is looping without convergence
      → Check: level — Level 0 (no REPL) should always be higher than Level 1 initially
  NO → Continue

STEP 4: Is there a regression vs. previous run?
  YES → Check what changed
      → git diff to see modified files
      → Check: same registry? Same model? Same seed?
      → Verify test_data/ground_truth wasn't modified
  NO → Improvement confirmed — document in RESEARCH_LOG
```

---

## Registry Size Effect

Expected TSA degradation as registry grows:

| Registry Size | Expected TSA drop vs. small_10 |
|-------------|-------------------------------|
| small_10 (10 tools) | Baseline |
| medium_25 (25 tools) | ~10–15% drop |
| large_50 (50 tools) | ~20–25% drop |
| xlarge_100 (100 tools) | ~30–40% drop |

**Why:** More tools = more navigation steps needed = more turns = more context pressure. This is EXPECTED behavior, not a bug. After fine-tuning, the gap should narrow significantly (the paper's claim).

---

## Context Savings Interpretation

`context_savings_pct = 1 - (tokens_with_repl / tokens_without_repl)`

- **Without REPL (Level 0):** All 50 tool schemas loaded upfront (~20K tokens)
- **With REPL (Level 1):** Only system prompt + conversation (~2K–5K tokens)

Expected savings: ~75–95% depending on registry size. If savings < 50%, something is wrong with the REPL approach (maybe too many get_schema() calls).

---

## Writing Results to RESEARCH_LOG.md

**Always use this format:**

```markdown
## [DATE] — [What Was Tested]

**Config:** Level={}, Registry={}, Model={}, n={}
**TSA:** X% (prev: Y%, delta: +/-Z%)
**RCV:** X%  **E2E:** X%  **fail:** X%  **avg_turns:** X

**Evidence label:** EMPIRICAL  ← always this for actual baseline runs

**Interpretation:** [One sentence per metric that changed significantly]
TSA improved from 28% to 45% because [specific mechanism — e.g., stop sequences now halt generation at </code>, forcing real REPL execution].

**Decision:** [What this means for next step]
**References:** [arXiv:XXXX if applicable]
```

---

## Comparing Multiple Runs

When you have 6+ result files, create a comparison table:

```markdown
| Date | Level | Registry | TSA | RCV | E2E | fail% | Change |
|------|-------|----------|-----|-----|-----|-------|--------|
| 2026-03-10 | L1 | small_10 | 28% | 0% | 28% | 71% | Pre-fix baseline |
| 2026-03-17 | L1 | small_10 | ??% | ??% | ??% | ??% | Post-fix |
```

Use this table in the memory file `~/.claude/projects/.../memory/model_results.md` so it persists across sessions.

---

## Escalation Triggers

Escalate to Opus (`/model claude-opus-4-6`) when:
- TSA improves >15% after a single fix (need architectural explanation for paper)
- TSA regresses unexpectedly after a change
- Metric pattern is contradictory (e.g., RCV high but TSA low)
- Results differ significantly between registry sizes (>25% TSA gap)
