#!/usr/bin/env bash
# Stop hook — fires at end of each Claude response turn.
# Scans for new result JSON files and auto-appends metrics to RESEARCH_LOG.md.

PROJ=/Users/rahul/Desktop/rlm
LOG="$PROJ/research/RESEARCH_LOG.md"
RESULTS="$PROJ/results"

[ ! -d "$RESULTS" ] && exit 0
[ ! -f "$LOG" ] && exit 0

# Find result JSONs modified more recently than the log's last modification
RECENT=$(find "$RESULTS" -name "*.json" -not -name "baseline_summary.json" -newer "$LOG" 2>/dev/null)

[ -z "$RECENT" ] && exit 0

DATE=$(date "+%Y-%m-%d %H:%M")

{
  echo ""
  echo "---"
  echo ""
  echo "## $DATE — Auto-logged Results"
  echo ""
  for f in $RECENT; do
    FNAME=$(basename "$f")
    python3 -c "
import json, sys
try:
    d = json.load(open('$f'))
    m = d.get('metrics', {})
    def fmt(v):
        return f'{v:.1%}' if isinstance(v, float) else str(v)
    tsa  = fmt(m.get('tool_selection_accuracy', '?'))
    rcv  = fmt(m.get('repl_code_validity', '?'))
    e2e  = fmt(m.get('end_to_end_accuracy', '?'))
    fail = fmt(m.get('failure_rate', '?'))
    turns = m.get('avg_repl_turns', '?')
    n    = m.get('total_queries', '?')
    print(f'**$FNAME**  TSA={tsa}  RCV={rcv}  E2E={e2e}  fail={fail}  avg_turns={turns}  n={n}')
except Exception as e:
    print(f'**$FNAME**  [parse error: {e}]')
" 2>/dev/null
    echo ""
  done
  echo "*← Auto-logged by session_stop hook. Add manual notes above.*"
  echo ""
} >> "$LOG"

echo "✓ session_stop: logged $(echo "$RECENT" | wc -l | tr -d ' ') result(s) to RESEARCH_LOG.md" >&2
exit 0
