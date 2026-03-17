Review the MCPRLMScaffold as an agent harness using the 4-constraint framework from agent-harness-construction skill.

## Context to Read First

```
src/scaffold.py
src/repl_engine.py
src/slm_interface.py
src/tool_registry.py
.claude/skills/agent-harness-construction.md
```

## Review Framework — 4 Constraints

### 1. Action Space Quality
*Are the tools the SLM can call narrow, stable, and predictable?*

Evaluate `MCPToolRegistry` API:
- Method signatures: `list_names()`, `search(query)`, `get_schema(name)`, `filter_by_category()`, `count()`
- Are return types consistent and documented?
- What happens on invalid inputs? Are errors informative?
- Is the API minimal enough that an SLM can learn it from a short prompt?

### 2. Observation Quality
*Does REPLEngine output give the SLM actionable feedback after each code block?*

Evaluate `REPLEngine.execute()` output:
- Truncation: are the `max_output_chars` limits appropriate per level?
- Error format: do exceptions include enough context to fix the code?
- Success format: is the output structured to help the SLM decide on the next action?
- Missing indicators: does the SLM know if a code block ran successfully vs. partially?

### 3. Recovery Quality
*What happens when the SLM fails to produce valid code?*

Evaluate `scaffold.py` failure handling:
- Nudge messages: what does the SLM see when it fails to produce `<code>` tags?
- Fallback: does it try markdown fences if XML extraction fails?
- Max turns: does the scaffold terminate gracefully or hang?
- Loop detection: can it detect if the SLM is repeating the same failed code?

### 4. Context Budget
*Is the conversation growing within a manageable context window?*

Evaluate context growth per turn:
- System prompt size: how many tokens does `prompts/level{n}.txt` consume?
- Per-turn overhead: REPL turn format adds how many tokens?
- At 10 turns (max): what's the estimated total context consumption?
- Does context_savings% metric actually reflect real compression?

## Output Format

For each constraint:
```
[Constraint Name]
Current State: [what exists now, specific line references]
Gap: [what's missing or suboptimal]
Specific Fix: [exact code change or config change]
Expected Impact: [what metric this should improve and by roughly how much]
```

## This is NOT a Code Review

Do not flag style issues, typing, or formatting. Focus exclusively on the harness architecture and its impact on TSA and RCV metrics.
