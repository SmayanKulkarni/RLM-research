#!/usr/bin/env bash
# PreCompact hook — fires before Claude compresses the context window.
# Logs a compaction event to RESEARCH_LOG.md so we know context was lost mid-session.

PROJ=/Users/rahul/Desktop/rlm
LOG="$PROJ/research/RESEARCH_LOG.md"
SESSIONS_DIR=~/.claude/sessions
DATE=$(date "+%Y-%m-%d %H:%M")

mkdir -p "$SESSIONS_DIR"

echo "⚠ Context compaction triggered at $DATE" >&2
echo "  Check SHARED_TASK_NOTES.md for continuation context." >&2

if [ -f "$LOG" ]; then
  {
    echo ""
    echo "<!-- compaction:$DATE -->"
    echo "> ⚠ **Context compacted at $DATE** — Session hit 50% context threshold."
    echo "> If mid-experiment, check SHARED_TASK_NOTES.md for continuation instructions."
    echo ""
  } >> "$LOG"
fi

exit 0
