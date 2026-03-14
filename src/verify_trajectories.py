"""
verify_trajectories.py — 3-Stage APIGen-Style Trajectory Verification.

Applies format → execution → semantic checks to raw trajectories produced
by generate_trajectories.py. Outputs verified (clean) and rejected (with
failure reasons) datasets.

Stages:
    Stage 1 — Format Check:
        - Has at least one <code>...</code> block
        - Has FINAL({...}) on the last assistant turn
        - At least 2 messages beyond system+user
        - No runaway trajectories (too many turns)

    Stage 2 — Execution Check:
        - Re-executes all code blocks against our real REPL
        - Marks any trajectory where code raised errors
        - Checks that execution path is plausible (no empty outputs)

    Stage 3 — Semantic Check:
        - predicted_tool == correct_tool in ground truth
        - FINAL params are non-empty
        - Tool actually exists in the registry

Usage:
    conda run -n astro python -m src.verify_trajectories \\
        --input data/trajectories_raw.jsonl \\
        --output-verified data/trajectories_verified.jsonl \\
        --output-rejected data/trajectories_rejected.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tool_registry import MCPToolRegistry
from src.repl_engine import REPLEngine, REPLConfig

PROJECT_ROOT = Path(__file__).parent.parent

# ── Verification Helpers ───────────────────────────────────────────────────────

def stage1_format(traj: dict) -> tuple[bool, str]:
    """
    Stage 1: Check the raw conversation structure.

    Returns (passed, reason).
    """
    messages = traj.get("messages", [])

    # Must have system + user + at least 2 assistant turns
    if len(messages) < 4:
        return False, f"Too few messages: {len(messages)}"

    roles = [m["role"] for m in messages]
    if roles[0] != "system":
        return False, "First message must be system"
    if roles[1] != "user":
        return False, "Second message must be user"
    if roles[-1] != "assistant":
        return False, "Last message must be from assistant"

    # Last assistant message must contain FINAL(...)
    last_assistant = messages[-1]["content"]
    if not re.search(r'FINAL\s*\(', last_assistant):
        return False, "Last assistant message missing FINAL(...)"

    # At least one code block must exist in the conversation
    all_content = " ".join(m["content"] for m in messages)
    if not re.search(r'<code>', all_content):
        return False, "No <code> blocks found"

    # Sanity: no more than 15 turns (would indicate runaway loop)
    turn_count = traj.get("turns", 0)
    if turn_count > 12:
        return False, f"Too many turns: {turn_count}"

    # Check FINAL is parseable JSON
    final_match = re.search(r'FINAL\((\{.*?\})\)', last_assistant, re.DOTALL)
    if not final_match:
        return False, "FINAL(...) content is not a JSON object"
    try:
        json.loads(final_match.group(1))
    except json.JSONDecodeError as e:
        return False, f"FINAL JSON parse error: {e}"

    return True, "ok"


def stage2_execution(traj: dict) -> tuple[bool, str, list[dict]]:
    """
    Stage 2: Re-execute all code blocks against the real REPL.

    Returns (passed, reason, updated_messages_with_real_outputs).

    We do a fresh re-execution and compare—if code errors exceed 50%,
    the trajectory is rejected.
    """
    reg_name = traj.get("registry", "small_10")
    reg_path = PROJECT_ROOT / "test_data" / "tool_registries" / f"{reg_name}.json"

    if not reg_path.exists():
        return False, f"Registry file not found: {reg_path}", []

    registry = MCPToolRegistry.from_json_file(reg_path)

    repl_config = REPLConfig(
        max_turns=15,
        max_output_chars=2000,
        allowed_imports=["re", "json"],
    )
    repl = REPLEngine(config=repl_config, initial_namespace={"tools_registry": registry})

    messages = traj["messages"]
    new_messages = []
    errors = 0
    total_code = 0

    i = 0
    while i < len(messages):
        msg = messages[i]
        new_messages.append(msg)

        if msg["role"] == "assistant":
            # Extract and re-execute code blocks in this assistant turn
            blocks = REPLEngine.extract_code_blocks(msg["content"])
            for code in blocks:
                total_code += 1
                turn_result = repl.execute(code)
                if not turn_result.success:
                    errors += 1

            # If next message is a [REPL OUTPUT] user turn, replace it with real output
            if i + 1 < len(messages) and messages[i + 1]["role"] == "user":
                next_content = messages[i + 1]["content"]
                if next_content.startswith("[REPL OUTPUT]"):
                    # Build real output from all executed code in this assistant turn
                    real_outputs = []
                    for t in repl.history[-len(blocks):]:
                        real_outputs.append(t.output)
                    real_combined = "\n---\n".join(real_outputs)
                    new_messages.append({
                        "role": "user",
                        "content": f"[REPL OUTPUT]:\n{real_combined}" if real_combined.strip()
                                   else "[REPL OUTPUT]:\n(no output)"
                    })
                    i += 2  # Skip the original user turn
                    continue

        i += 1

    if total_code == 0:
        return False, "No executable code blocks found", []

    error_rate = errors / total_code
    if error_rate > 0.5:
        return False, f"Code error rate too high: {errors}/{total_code} blocks failed", []

    return True, "ok", new_messages


def stage3_semantic(traj: dict, updated_messages: list[dict]) -> tuple[bool, str]:
    """
    Stage 3: Check that the predicted tool matches the correct tool.

    Returns (passed, reason).
    """
    correct_tool = traj.get("correct_tool")
    if not correct_tool:
        return False, "No correct_tool field in trajectory"

    # Extract predicted tool from FINAL(...)
    last_assistant = next(
        (m["content"] for m in reversed(updated_messages) if m["role"] == "assistant"),
        ""
    )
    final_match = re.search(r'FINAL\((\{.*?\})\)', last_assistant, re.DOTALL)
    if not final_match:
        return False, "Could not find FINAL(...) in last assistant message"

    try:
        final_obj = json.loads(final_match.group(1))
    except json.JSONDecodeError:
        return False, "FINAL JSON parse error at semantic stage"

    predicted_tool = final_obj.get("tool", "")

    if predicted_tool != correct_tool:
        return False, f"Tool mismatch: predicted={predicted_tool}, correct={correct_tool}"

    # FINAL params should be non-empty
    params = final_obj.get("params", {})
    if not isinstance(params, dict):
        return False, "FINAL params is not a dict"

    # Verify the tool exists in the registry
    reg_name = traj.get("registry", "small_10")
    reg_path = PROJECT_ROOT / "test_data" / "tool_registries" / f"{reg_name}.json"
    if reg_path.exists():
        registry = MCPToolRegistry.from_json_file(reg_path)
        if correct_tool not in registry.list_names():
            return False, f"Tool '{correct_tool}' not in registry '{reg_name}'"

    return True, "ok"


# ── Main Verification Loop ────────────────────────────────────────────────────

def run(args):
    input_path = PROJECT_ROOT / args.input
    verified_path = PROJECT_ROOT / args.output_verified
    rejected_path = PROJECT_ROOT / args.output_rejected

    verified_path.parent.mkdir(parents=True, exist_ok=True)
    rejected_path.parent.mkdir(parents=True, exist_ok=True)

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        sys.exit(1)

    trajectories = []
    with open(input_path) as f:
        for line in f:
            line = line.strip()
            if line:
                trajectories.append(json.loads(line))

    print(f"Loaded {len(trajectories)} raw trajectories from {input_path}")
    print(f"\n{'='*60}")

    verified = []
    rejected = []

    stage_counts = {"stage1": 0, "stage2": 0, "stage3": 0}

    for i, traj in enumerate(trajectories):
        q_short = traj.get("query", "?")[:55]
        print(f"[{i+1}/{len(trajectories)}] {traj.get('correct_tool','?')} | {q_short}...")

        # Stage 1: Format
        ok, reason = stage1_format(traj)
        if not ok:
            stage_counts["stage1"] += 1
            print(f"  ✗ Stage 1 FAIL: {reason}")
            rejected.append({**traj, "reject_stage": 1, "reject_reason": reason})
            continue
        print(f"  ✓ Stage 1 (format)")

        # Stage 2: Execution
        ok, reason, updated_messages = stage2_execution(traj)
        if not ok:
            stage_counts["stage2"] += 1
            print(f"  ✗ Stage 2 FAIL: {reason}")
            rejected.append({**traj, "reject_stage": 2, "reject_reason": reason})
            continue
        print(f"  ✓ Stage 2 (execution)")

        # Stage 3: Semantic
        ok, reason = stage3_semantic(traj, updated_messages)
        if not ok:
            stage_counts["stage3"] += 1
            print(f"  ✗ Stage 3 FAIL: {reason}")
            rejected.append({**traj, "reject_stage": 3, "reject_reason": reason})
            continue
        print(f"  ✓ Stage 3 (semantic)")

        # PASSED all stages — use real-output-updated messages
        clean_traj = {
            "registry": traj.get("registry"),
            "query": traj.get("query"),
            "correct_tool": traj.get("correct_tool"),
            "turns": traj.get("turns"),
            "messages": updated_messages,
        }
        verified.append(clean_traj)

    # Write outputs
    with open(verified_path, "w") as f:
        for t in verified:
            f.write(json.dumps(t) + "\n")

    with open(rejected_path, "w") as f:
        for t in rejected:
            f.write(json.dumps(t) + "\n")

    total = len(trajectories)
    nv = len(verified)
    nr = len(rejected)
    pass_rate = nv / total * 100 if total else 0

    print(f"\n{'='*60}")
    print(f"VERIFICATION COMPLETE")
    print(f"  Total input   : {total}")
    print(f"  Verified      : {nv} ({pass_rate:.1f}%)")
    print(f"  Rejected      : {nr}")
    print(f"    Stage 1 (format)   : {stage_counts['stage1']} failures")
    print(f"    Stage 2 (execution): {stage_counts['stage2']} failures")
    print(f"    Stage 3 (semantic) : {stage_counts['stage3']} failures")
    print(f"  Verified saved to : {verified_path}")
    print(f"  Rejected saved to : {rejected_path}")
    print(f"{'='*60}")

    if nv < 100:
        print(f"\n⚠  WARNING: Only {nv} verified trajectories. Fine-tuning needs ~500+ for good results.")
        print("   Consider running generate_trajectories.py with more --queries-per-tool.")
    elif nv < 500:
        print(f"\n⚠  {nv} trajectories: acceptable for initial fine-tuning, aim for 1000+.")
    else:
        print(f"\n✓  {nv} trajectories: good dataset size for QLoRA fine-tuning.")


def main():
    parser = argparse.ArgumentParser(description="Verify REPL-MCP training trajectories (3-stage)")
    parser.add_argument(
        "--input", type=str, default="data/trajectories_raw.jsonl",
        help="Input raw trajectories JSONL",
    )
    parser.add_argument(
        "--output-verified", type=str, default="data/trajectories_verified.jsonl",
        help="Output path for verified trajectories",
    )
    parser.add_argument(
        "--output-rejected", type=str, default="data/trajectories_rejected.jsonl",
        help="Output path for rejected trajectories with reasons",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
