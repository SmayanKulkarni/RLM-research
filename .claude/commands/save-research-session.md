Save the current research session state for future resumption. Creates a structured session file AND updates SHARED_TASK_NOTES.md.

## Session File Location
`~/.claude/sessions/YYYY-MM-DD-rlm-[goal-slug]-session.tmp`

## Required Sections (do not skip any)

### 1. What We Were Working On
One sentence: the specific goal of this session.

### 2. What Worked ✓
Specific approaches, commands, or configurations that produced useful results. Be precise — "tried X" is insufficient; write "ran X with Y flag, produced Z result."

### 3. What Did NOT Work ✗ (MOST IMPORTANT)
**This section prevents retrying dead ends in future sessions.**
Format: "Tried [specific action] → [exact failure] → [why it failed]"

Examples:
- "Tried `--level 3` flag → doesn't exist in run_baseline.py yet"
- "Tried running xlarge_100 locally → OOM at 6.2GB, confirmed M1 limit"
- "Tried Groq batch size 50 → hit 429 rate limit after 47 calls, need ≤30/min"

### 4. Untried Approaches
Things worth trying next session that weren't attempted yet.

### 5. Files Modified
List every file changed and what was changed.

### 6. Key Decisions Made
Architecture or research decisions made this session, with brief rationale.

### 7. Blockers
Anything preventing next steps. Be specific.

### 8. Exact Next Action
ONE specific command or task to do first next session — no ambiguity.

---

## Also update SHARED_TASK_NOTES.md

Write a 3-sentence summary:
1. What was accomplished
2. Current status of ongoing work
3. Next immediate action

The SHARED_TASK_NOTES.md is read by all subagents at session start — keep it current and actionable.
