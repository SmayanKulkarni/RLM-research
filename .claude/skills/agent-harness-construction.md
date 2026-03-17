# Agent Harness Construction

## What This Skill Is
A framework for designing and evaluating the quality of an agent harness — the scaffolding that mediates between an LLM and its available tools. Based on ECC's agent-harness-construction skill, adapted for the RLM×MCP REPL scaffold.

## When to Use
- When reviewing or improving `src/scaffold.py`, `src/repl_engine.py`, `src/tool_registry.py`
- When TSA or RCV metrics are lower than expected
- When adding a new REPL level or registry size
- Use the `/harness-review` command to run a structured audit

---

## The 4 Harness Constraints

### 1. Action Space Quality
The tools available to the agent must be: **narrow, stable, and predictable**.

**Principles:**
- Purposeful tools: each method solves exactly one problem
- Stable names: never rename a tool the model has trained on
- Narrow schemas: minimal required parameters, sensible defaults
- Predictable outputs: same input → same output type always

**For MCPToolRegistry:**
```python
# Good: narrow, predictable
tools_registry.search("file operations")          # → list[str]
tools_registry.get_schema("read_file")            # → dict

# Bad: too broad, unpredictable
tools_registry.get_everything()                   # → ???
```

**Evaluation question:** Can an SLM learn the full API from 5 examples? If not, it's too complex.

### 2. Observation Quality
The agent must receive feedback that is **actionable, clear, and appropriately sized**.

**Principles:**
- Status indicators: success/failure signal on every execution
- Summaries: for large outputs, provide a summary + pointer ("first 10 results shown...")
- Actionable errors: "KeyError: 'read_file' not in registry" not just "KeyError"
- Artifact references: point to where results were stored, not just dump them

**For REPLEngine output:**
```python
# Good: actionable
"Error: 'write_file' not found. Available: ['read_file', 'list_dir']. Try tools_registry.search('write')"

# Bad: useless
"KeyError"
```

**Evaluation question:** Given only the REPL output, can the SLM identify exactly what to fix?

### 3. Recovery Quality
The harness must handle failures **gracefully and deterministically**.

**Principles:**
- Root cause hints: tell the agent WHY something failed
- Safe retry instructions: show what a correct retry would look like
- Explicit stopping conditions: "if you've failed 3 times, call FINAL() with your best guess"
- No silent failures: every failed execution produces visible feedback

**For MCPRLMScaffold:**
```python
# Good recovery flow
if not code_blocks and turn > 2:
    nudge = "Use <code> tags to call tools_registry methods. Example:\n<code>\ntools_registry.search('your query')\n</code>"

# Bad: no guidance
if not code_blocks:
    pass  # SLM gets no feedback, repeats mistake
```

**Evaluation question:** Can the SLM self-correct after a single failure without help from the system prompt?

### 4. Context Budget
Every turn, the conversation grows. The harness must **stay within the SLM's context window**.

**Calculation for 10-turn max:**
```
System prompt:       ~1,500 tokens (Level 1)
Per user message:    ~200 tokens avg
Per REPL output:     ~500 tokens (truncated to 2000 chars ≈ 500 tokens)
Per SLM response:    ~300 tokens avg

Total at 10 turns:   1,500 + 10 × (200 + 500 + 300) = ~11,500 tokens
Qwen2.5-3B context:  ~8,192 tokens native

→ PROBLEM: At 10 turns, context exceeds native window.
→ FIX: Truncate output more aggressively, compress earlier turns, reduce max_turns to 6.
```

**Evaluation question:** What is the token count at turn 5? At turn 10? Does it exceed the SLM's context window?

---

## RLM×MCP Harness Current State (as of Phase 1.5)

| Constraint | Current State | Gap | Priority Fix |
|-----------|--------------|-----|-------------|
| Action space | 5 registry methods, narrow | Good | None needed |
| Observation | Truncated to 2000/4000 chars, errors pass through | Errors not enhanced with hints | Add "Available tools: ..." to KeyError messages |
| Recovery | Nudge message if no code blocks | Nudge fires too late (after 3 attempts) | Move nudge to turn 1 if no code blocks |
| Context budget | 10 turns max, 2000 char truncation | At 10 turns likely exceeds 8K tokens | Reduce max_turns to 6, compress earlier turns |

---

## Hybrid Architecture (Recommended)

From ECC research: the hybrid approach (ReAct planning + typed tool execution) outperforms pure function-calling or pure exploration.

For our scaffold:
- **ReAct layer:** SLM reasons about which registry method to call
- **Typed execution layer:** `MCPToolRegistry` enforces schemas, catches errors
- **Feedback layer:** `REPLEngine` returns structured, truncated, error-enhanced output

This matches exactly what Level 1 + Level 2 attempts to do. The missing piece is the feedback layer quality.
