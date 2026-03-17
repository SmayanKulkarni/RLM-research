Run a complete research sprint on a goal, orchestrating: context loading → success criteria → phased execution → RESEARCH_LOG entry.

Arguments: $ARGUMENTS (e.g., "analyze scaffold fix impact on L1 TSA" or "investigate why L2 outperforms L1")

## Execution Steps

1. **Load context:**
   - Read last 5 entries of `research/RESEARCH_LOG.md`
   - Read `SHARED_TASK_NOTES.md`
   - Report current phase and what has been done

2. **Define success criteria upfront:**
   - What specific question will be answered?
   - What metric threshold constitutes success?
   - What will be written to RESEARCH_LOG at the end?

3. **Phase the work** (Research → Experiment → Analyze → Conclude):
   - Research: what do we know, what does existing data show?
   - Experiment: what needs to be run to answer the question?
   - Analyze: interpret results against success criteria
   - Conclude: what is the decision, what is the next step?

4. **Delegate to specialists:**
   - Baseline runs → `@eval-agent`
   - Trajectory data → `@trajectory-agent`
   - Data pipeline → `@data-pipeline-agent`
   - Code review → `@python-ml-reviewer`

5. **Update SHARED_TASK_NOTES.md** after each phase with findings.

6. **Write RESEARCH_LOG entry** (mandatory final step):
   ```markdown
   ## [DATE] — [Sprint Goal]
   **Findings:** [2–3 sentences]
   **Metrics:** [key numbers]
   **Evidence label:** EMPIRICAL / ANALYTICAL / INTUITION / UNVERIFIED
   **Decision:** [what to do next]
   **Open questions:** [unresolved items]
   ```

7. **Model escalation:** If findings reveal unexpected architecture-level issues, say:
   > "These findings require architectural reasoning. Recommend switching to Opus: `/model claude-opus-4-6`"
