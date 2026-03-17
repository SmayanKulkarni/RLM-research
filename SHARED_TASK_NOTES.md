# Shared Task Notes — RLM×MCP
*Read by all agents at session start. Keep current and actionable.*

## Current Sprint Goal
Complete Phase 1.5 scaffold fixes (stop_sequences + XML delimiters) and verify RCV > 0% on L1 re-run.

## Status
- [x] Phase 0 baselines: L0=100% TSA ✓
- [x] Phase 1 baselines: L1=28–33% TSA, root cause identified ✓
- [ ] Phase 1.5: scaffold fixes (stop_sequences + XML delimiter + check reorder) — IN PROGRESS
- [ ] Phase 1.5 verification: re-run L1 baselines to confirm RCV > 0%
- [ ] Phase A: TOUCAN + 10k trajectories + 3k negatives + build_splits
- [ ] Phase B: QLoRA fine-tuning Qwen2.5-3B

## Last Action
2026-03-17 — Claude Code scaffold set up (CLAUDE.md, settings.json, hooks, agents, commands, skills, memory files).

## Next Action
Re-run `python -m src.run_baseline --level 1 --registry small_10` after confirming scaffold fixes in `src/slm_interface.py` and `src/scaffold.py` are active. Expected: RCV > 0% (previously 0% due to FINAL() checked before code extraction).

## Active Blockers
None currently.

## Do NOT Retry
- Do NOT run `--all` with `xlarge_100` on M1 (confirmed OOM at 6.2GB)
- Do NOT call Groq API without `sleep(2.0)` between requests (429 rate limit)
