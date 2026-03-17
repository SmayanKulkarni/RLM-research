#!/usr/bin/env bash
# SessionStart hook — fires when Claude Code opens this project.
# Prints a context briefing to stderr (visible in Claude Code terminal).

PROJ=/Users/rahul/Desktop/rlm
LOG="$PROJ/research/RESEARCH_LOG.md"
NOTES="$PROJ/SHARED_TASK_NOTES.md"

echo "╔══════════════════════════════════════╗" >&2
echo "║   RLM×MCP Session Start              ║" >&2
echo "╚══════════════════════════════════════╝" >&2

# Last 20 lines of RESEARCH_LOG
if [ -f "$LOG" ]; then
  echo "▸ Last Research Log Entry:" >&2
  tail -20 "$LOG" >&2
  echo "" >&2
fi

# SHARED_TASK_NOTES
if [ -f "$NOTES" ]; then
  echo "▸ Shared Task Notes:" >&2
  cat "$NOTES" >&2
  echo "" >&2
fi

# Environment checks
if ! pgrep -x "ollama" > /dev/null 2>&1; then
  echo "⚠️  WARNING: Ollama not running. Start with: ollama serve" >&2
fi

if [ ! -f "$PROJ/.env" ]; then
  echo "⚠️  WARNING: .env file missing (GROQ_API_KEY needed for trajectory generation)" >&2
elif ! grep -q "GROQ_API_KEY" "$PROJ/.env" 2>/dev/null; then
  echo "⚠️  WARNING: GROQ_API_KEY not found in .env" >&2
fi

# Check for unverified trajectory data
if [ -f "$PROJ/data/trajectories_raw.jsonl" ] && [ ! -f "$PROJ/data/trajectories_verified.jsonl" ]; then
  echo "📋 REMINDER: trajectories_raw.jsonl exists but is unverified." >&2
  echo "   Run: python -m src.verify_trajectories --input data/trajectories_raw.jsonl" >&2
fi

echo "" >&2
exit 0
