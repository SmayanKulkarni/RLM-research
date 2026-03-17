Coordinate Phase A (Week 1) multi-workstream data preparation: TOUCAN + trajectory generation + negative generation.

## Overview

Phase A has 3 workstreams that can run in parallel (different data sources, no dependencies until build_splits):
- **Stream 1:** TOUCAN processing (`data-pipeline-agent`)
- **Stream 2:** Groq trajectory generation (`trajectory-agent`) — runs with nohup overnight
- **Stream 3:** Hard negative generation (`data-pipeline-agent`)

## Pre-flight (run before anything)

```bash
df -h data/           # Need 500MB+ free
grep GROQ_API_KEY .env # Must exist
pgrep -x ollama       # Must be running
```

## Stream Execution Order

**Step 1 — Start TOUCAN (fastest, ~2h):**
```bash
python -m src.prepare_toucan --output data/toucan_repl.jsonl --max-examples 20000
```

**Step 2 — Start trajectory generation in background (~8.5h at free tier):**
```bash
nohup python -m src.generate_trajectories \
  --registries small_10 medium_25 large_50 \
  --queries-per-tool 8 \
  --levels 1 2 \
  --output data/trajectories_raw.jsonl \
  --seed 42 \
  2>&1 | tee data/generation_log.txt &
echo "Trajectory gen PID: $! — Monitor: tail -f data/generation_log.txt"
```

**Step 3 — Start negatives (while trajectories run, ~1h):**
```bash
python -m src.generate_negatives --count 3000
```

**Step 4 — After all streams complete — verify and build:**
```bash
python -m src.verify_trajectories --input data/trajectories_raw.jsonl
python -m src.augment_trajectories --input data/trajectories_verified.jsonl
python -m src.build_splits
```

## Timeline Estimate
- TOUCAN: ~2h
- Trajectories: ~8.5h (overnight)
- Negatives: ~1h
- Verify + splits: ~1h
- Total wall-clock: ~10h (mostly waiting for trajectories overnight)

## Completion Report

After all steps, report:
- Final corpus size (train/dev/test counts)
- Tool coverage across registry sizes
- Yield rate from trajectory generation
- Any quality issues flagged by verify_trajectories
