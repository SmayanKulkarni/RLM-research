#!/usr/bin/env bash
# PostToolUse hook (matcher: Write|Edit) — fires after Claude writes or edits a file.
# Auto-runs ruff check + format on Python files.

INPUT=$(cat)

FILE=$(python3 -c "
import sys, json
try:
    d = json.loads(sys.stdin.read())
    path = d.get('file_path', d.get('path', ''))
    print(path)
except:
    print('')
" <<< "$INPUT" 2>/dev/null)

if [[ "$FILE" == *.py ]] && [ -n "$FILE" ]; then
  cd /Users/rahul/Desktop/rlm
  if command -v ruff &> /dev/null; then
    ruff check "$FILE" --fix --quiet 2>/dev/null
    ruff format "$FILE" --quiet 2>/dev/null
    echo "✓ ruff: auto-fixed $(basename $FILE)" >&2
  else
    echo "⚠ ruff not found — install with: pip install ruff" >&2
  fi
fi

exit 0
