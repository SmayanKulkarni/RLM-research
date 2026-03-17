# Evaluation Harness

## What This Skill Is
A framework for eval-driven development — defining success metrics before implementation, running structured evaluations, and interpreting results for research decisions. Adapted from ECC's eval-harness skill for the RLM×MCP 7-metric evaluation framework.

## When to Use
- Before implementing any scaffold change (define what success looks like first)
- When running `python -m src.run_baseline` or analyzing `results/` JSON files
- When comparing model variants or registry sizes
- Use the `/run-eval-baseline` command for structured baseline runs

---

## Core Principle: Evals First

**Before any implementation change:**
1. State the hypothesis: "Adding stop sequences will increase RCV from 0% to >50%"
2. Define the metric: RCV (REPL Code Validity) in `results/{run}.json`
3. Define success threshold: RCV > 50% at Level 1 is meaningful improvement
4. Run the baseline FIRST to establish the pre-change measurement
5. Make the change
6. Run the eval again
7. Report the delta

This prevents HiPPO-driven development (Highest Paid Person's Opinion).

---

## Our 7 Metrics Explained

| Metric | Key in JSON | What It Measures | Research Significance |
|--------|------------|-----------------|----------------------|
| **TSA** | `tool_selection_accuracy` | % queries where SLM predicted correct tool | Primary metric — beats MCP Bridge 73% F1 = success |
| **PC** | `parameter_correctness` | % queries with correct parameters given correct tool | Quality beyond just naming the tool |
| **RCV** | `repl_code_validity` | % of executed code blocks that succeeded | Health indicator — 0% means REPL not working at all |
| **E2E** | `end_to_end_accuracy` | TSA × PC (both correct) | True end-to-end success |
| **Avg Turns** | `avg_repl_turns` | Mean REPL interaction rounds per query | Efficiency — fewer turns = less context consumption |
| **Context Savings** | context_savings | (1 - with_repl_tokens / without_repl_tokens) | Main thesis metric — does REPL save tokens? |
| **Fail Rate** | `failure_rate` | % queries producing no FINAL() answer | Stability metric — high fail = broken scaffold |

### Metric Relationships

```
High fail_rate → scaffold stability problem (check stop sequences, code extraction)
RCV = 0% → REPL not executing (check scaffold.py check order)
TSA high + RCV low → SLM guessing without REPL (not the behavior we want)
TSA high + RCV high + fail low → ideal: REPL-driven correct tool selection
avg_turns high → context budget pressure (check truncation limits)
```

---

## Evaluation Decision Tree

```
Run baseline
    ↓
TSA > 70%? → YES → MILESTONE: Beat MCP Bridge. Append to RESEARCH_LOG (EMPIRICAL)
    ↓ NO
fail_rate > 50%?
    → YES: Scaffold broken. Check FINAL() parsing, check stop sequences
    → NO: Continue diagnosis
    ↓
RCV = 0%?
    → YES: REPL not executing. FINAL() check runs before code extraction
    → NO: Continue diagnosis
    ↓
TSA < baseline?
    → YES: Regression. Check prompt changes, registry file integrity
    → NO: Flat performance = data or model capacity problem
```

---

## pass@k vs pass^k (for paper)

When reporting results:
- **pass@k**: probability that AT LEAST ONE of k attempts succeeds (optimistic bound)
- **pass^k**: probability that ALL k attempts succeed (reliability bound)

We use **pass^1** (single attempt, must succeed) because in production, an SLM gets one shot at tool selection.

For ablations, consider:
- pass@3: run 3 times, count once if any succeeds (temperature=0.7)
- Report both pass@1 and pass@3 in the paper

---

## Evaluation Directory Structure

```
results/
  baseline_summary.json          — aggregated across all configs
  level0_small10_{ts}.json       — per-run detailed results
  level1_medium25_{ts}.json
  ...

.claude/evals/                   — eval definitions (create this)
  eval_l1_scaffold_fix.md        — "does stop_sequence fix increase RCV?"
  eval_l1_vs_l2.md               — "does few-shot help?"
  eval_registry_scaling.md       — "how does TSA degrade with registry size?"
```

---

## Result JSON Structure

```json
{
  "config": {
    "level": 1,
    "registry": "medium_25",
    "model": "qwen2.5-coder:3b",
    "timestamp": "2026-03-17T10:00:00"
  },
  "metrics": {
    "tool_selection_accuracy": 0.33,
    "parameter_correctness": 0.33,
    "repl_code_validity": 0.0,
    "end_to_end_accuracy": 0.33,
    "avg_repl_turns": 7.6,
    "context_savings_pct": 0.87,
    "failure_rate": 0.58,
    "total_queries": 12
  },
  "per_query_results": [...]
}
```

---

## Writing Up Results (RESEARCH_LOG Format)

```markdown
## [DATE] — Eval: [Description]

**Hypothesis:** [What we expected to happen]
**Change made:** [Specific modification]
**Before:** TSA=X%  RCV=Y%  fail=Z%  (n=N)
**After:**  TSA=X%  RCV=Y%  fail=Z%  (n=N)
**Delta:**  TSA +X%  RCV +Y%  fail -Z%

**Evidence label:** EMPIRICAL
**Interpretation:** [What the numbers mean mechanistically]
**Decision:** [What to do next]
**References:** [arXiv:XXXX if applicable]
```
