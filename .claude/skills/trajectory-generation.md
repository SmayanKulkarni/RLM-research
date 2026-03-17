# Trajectory Generation

## What This Skill Is
Protocol for generating high-quality REPL interaction trajectories using Groq API (llama-3.3-70b-versatile) as the expert model, with real REPL execution via REPLEngine to produce ground-truth training data.

## When to Use
- Before running `python -m src.generate_trajectories`
- When planning Phase A Week 1 generation batch
- When yield rate is below expected and debugging is needed
- Use `/phase-a-kickoff` for the full Phase A orchestration

---

## Pre-flight Checklist (ALWAYS complete before running)

```bash
# 1. GROQ_API_KEY present
grep "GROQ_API_KEY" .env || { echo "ABORT: Set GROQ_API_KEY in .env"; exit 1; }

# 2. Ollama running (needed for REPL execution during generation)
pgrep -x "ollama" && echo "OK" || echo "START: ollama serve"

# 3. Disk space (plan for 500MB+ for full pipeline)
df -h data/

# 4. Registry files exist
ls test_data/tool_registries/

# 5. Ground truth exists (needed for semantic verification)
ls test_data/ground_truth/expected_selections.json
```

---

## Groq Free-Tier Budget

| Parameter | Value |
|-----------|-------|
| Rate limit | ~30 req/min, ~14,400 req/day |
| With sleep(2): | ~28 effective req/min |
| Expected yield | ~70% pass rate through verify |
| To get 10k verified | Need ~14,300 raw calls |
| Time estimate | ~510 min = 8.5h continuous |
| Days needed | 1 full day at free tier |

**Budget calculation formula:**
```
total_calls = num_registries × num_tools × queries_per_tool × levels
```

**Thresholds:**
- < 200 calls: run directly in Claude
- 200–500 calls: run in terminal with progress monitoring
- > 500 calls: run with `nohup` overnight

---

## APIGen 3-Stage Verification Pipeline

After generation, ALWAYS run `verify_trajectories.py` (implements APIGen from [Liu et al., 2024]):

**Stage 1 — Format check:**
- Valid JSONL: every line parseable
- Required fields: `messages`, `tool`, `params`
- FINAL() call parseable as JSON: `{"tool": "name", "params": {...}}`

**Stage 2 — Execution check:**
- Every `<code>` block in the trajectory re-runs against REPLEngine
- REPLEngine uses same config as the level being trained on
- Any execution failure → discard trajectory

**Stage 3 — Semantic check:**
- Predicted tool in FINAL() matches `expected_selections.json` ground truth
- Predicted params are a subset of required params (at minimum)

**Expected yield:** ~70% pass all 3 stages. If < 50%, something is systematically wrong — investigate Stage 2 failures first.

---

## Failure Modes and Recovery

| Error Type | Symptom | Cause | Recovery |
|-----------|---------|-------|---------|
| GROQ_FAIL 429 | HTTP 429 in logs | Rate limit exceeded | Increase `sleep` to 3s, reduce batch |
| PARSE_FAIL | No FINAL() in response | Groq LLM output malformed | Skip trajectory, count as loss |
| REPL_FAIL | Stage 2 execution error | Code block can't execute | Check REPLConfig.allowed_imports for that level |
| KEY_MISMATCH | Stage 3 semantic fail | Tool name doesn't match ground truth | Flag for manual review, exclude from train set |
| TIMEOUT | Request hangs | Groq API slow | Set request timeout, retry once |

---

## Quality Expectations by Registry Size

| Registry | Tools | Expected per-query difficulty | Expected yield |
|---------|-------|------------------------------|----------------|
| small_10 | 10 | Low — few tools to navigate | ~80% |
| medium_25 | 25 | Medium — requires search | ~70% |
| large_50 | 50 | High — multiple navigation steps | ~60% |
| xlarge_100 | 100 | Very high — complex search chains | ~50% |

---

## Trajectory Quality Signals

**Good trajectory:**
- 2–5 REPL turns before FINAL()
- Uses `search()` or `filter_by_category()` before `get_schema()`
- Code outputs look like real registry API responses (not hallucinated)
- FINAL() params are non-empty and match the schema

**Bad trajectory (discard):**
- FINAL() on turn 1 (guessing without REPL)
- Same code repeated 3+ times (stuck in loop)
- Fake REPL output in code comments (hallucinated execution)
- Empty params: `{"tool": "name", "params": {}}`

---

## Post-Generation (Mandatory)

```bash
# Verify all trajectories (do before augmentation or split building)
python -m src.verify_trajectories --input data/trajectories_raw.jsonl

# Report expected:
# Format OK: ~100% (should all pass)
# Execution OK: ~85–90%
# Semantic OK: ~70% of execution-passing
# Overall yield: ~70%

# Only proceed to augmentation if yield > 50%
# If yield < 50%: investigate Stage 2 (execution) failures first
```
