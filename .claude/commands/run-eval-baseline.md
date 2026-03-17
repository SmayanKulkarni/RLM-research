Run an evaluation baseline with full pre-flight checks, execution, and diagnostic interpretation.

Arguments: $ARGUMENTS (e.g., "level=1 registry=medium_25" or "level=0 registry=small_10 model=qwen2.5-coder:3b")

## Delegation

Delegate entirely to `@eval-agent` with the parsed arguments.

The eval-agent will:
1. Run pre-flight checks (Ollama, registry file, xlarge_100 hardware guard)
2. Execute: `python -m src.run_baseline --level {level} --registry {registry}`
3. Report all 7 metrics with delta vs. stored baseline
4. Run failure mode diagnosis if TSA < 50%
5. Recommend the next experiment

## Post-run Rule

If TSA deviates from baseline by **more than 10% in either direction**:
→ Escalate to Opus for root cause analysis. Say:
> "This deviation is significant. Recommend Opus for root cause: `/model claude-opus-4-6`"

If TSA > 70% (research target beaten):
→ Immediately flag as milestone. Append to RESEARCH_LOG.md with evidence label EMPIRICAL.
