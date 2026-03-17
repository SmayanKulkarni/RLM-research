"""
generate_negatives.py — Hard Negative Generation for DPO Training.

For each verified positive trajectory, generates hard negatives of two types:

  1. wrong_tool   : Agent selects a plausible but incorrect tool.
                   The FINAL({...}) points to a semantically similar but wrong tool.

  2. wrong_param  : Agent selects the correct tool but supplies hallucinated/wrong
                   parameter values that don't match the user's query.

Output is a DPO-pairs JSONL where each line is:
    {
        "query":            str,
        "correct_tool":     str,
        "negative_type":    "wrong_tool" | "wrong_param",
        "chosen_messages":  list[dict],   # the positive (correct) trajectory
        "rejected_messages":list[dict],   # the negative (incorrect) trajectory
    }

This format is consumed by finetune_dpo.py (TRL DPOTrainer).

Usage:
    conda run -n astro python -m src.generate_negatives \\
        --positives data/trajectories_verified.jsonl \\
        --positives data/toucan_repl.jsonl \\
        --output data/negatives_dpo.jsonl \\
        --target 3000

References:
    - TP-LLaMA (NeurIPS 2024): SFT + DPO cascade beats SFT-only
    - APIGen (Salesforce, arXiv:2406.18518): hard negatives improve tool selection precision
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tool_registry import MCPToolRegistry

PROJECT_ROOT = Path(__file__).parent.parent


# ── Wrong-tool Selection ───────────────────────────────────────────────────────

def _tool_keywords(tool: dict) -> set[str]:
    """Extract meaningful keywords from a tool's name and description."""
    text = f"{tool.get('name', '')} {tool.get('description', '')}".lower()
    words = re.findall(r"[a-z]{3,}", text)
    stop = {
        "the", "and", "for", "with", "from", "that", "this", "tool", "will",
        "can", "use", "get", "set", "create", "update", "delete", "returns",
        "list", "all", "any", "one", "two", "three", "given", "used",
    }
    return {w for w in words if w not in stop}


def find_plausible_wrong_tool(
    registry: MCPToolRegistry,
    correct_tool: str,
    query: str,
    rng: random.Random,
) -> str | None:
    """
    Pick a plausible wrong tool — one that shares keywords with the query but is
    NOT the correct tool.

    Strategy (in priority order):
      1. Same category as the correct tool, different tool
      2. Any tool whose keywords overlap with the query keywords
      3. Random fallback (excluding correct tool)

    Returns None if the registry has only one tool.
    """
    all_tools = registry._tools  # type: ignore[attr-defined]
    if len(all_tools) < 2:
        return None

    correct_kws = _tool_keywords(next((t for t in all_tools if t["name"] == correct_tool), {}))
    query_kws = set(re.findall(r"[a-z]{3,}", query.lower()))

    # Compute keyword overlap score for every other tool
    candidates: list[tuple[int, str]] = []
    for t in all_tools:
        if t["name"] == correct_tool:
            continue
        kws = _tool_keywords(t)
        # Prefer tools that share query keywords (and some correct-tool keywords)
        # but avoid tools that are identical in name structure
        overlap_query = len(kws & query_kws)
        overlap_correct = len(kws & correct_kws)
        # Score: some query overlap is good, but don't pick something too similar
        score = overlap_query * 2 - overlap_correct
        candidates.append((score, t["name"]))

    if not candidates:
        return None

    # Sort by score descending, then pick randomly from top 3
    candidates.sort(key=lambda x: -x[0])
    top_n = min(3, len(candidates))
    return rng.choice(candidates[:top_n])[1]


# ── Wrong-param Generation ─────────────────────────────────────────────────────

_STRING_SUBSTITUTIONS = [
    "example_value",
    "test_string",
    "placeholder_data",
    "sample_input",
    "dummy_value",
]

_NUMERIC_MULTIPLIERS = [0, 2, 10, -1, 999]


def generate_wrong_params(
    correct_params: dict,
    schema: dict | None,
    rng: random.Random,
) -> dict:
    """
    Generate wrong parameter values that look plausible but are incorrect.

    Strategy:
      - String values: substitute with a fixed wrong string
      - Numeric values: multiply by a wrong factor (including 0 or negative)
      - Boolean values: flip
      - Missing required params: return params with one required field removed
      - If no params to corrupt, add a spurious extra field

    Always ensures the result is DIFFERENT from correct_params.
    """
    if not correct_params:
        # No params to corrupt — add a spurious field
        return {"_hallucinated": "this_param_does_not_exist"}

    wrong = dict(correct_params)  # shallow copy

    # Find required params from schema
    required = []
    if schema:
        required = schema.get("inputSchema", {}).get("required", [])

    # Try to corrupt one param (prioritise required ones)
    targets = [k for k in required if k in wrong] or list(wrong.keys())
    if not targets:
        return {"_hallucinated": "this_param_does_not_exist"}

    key = rng.choice(targets)
    val = wrong[key]

    if isinstance(val, bool):
        wrong[key] = not val
    elif isinstance(val, (int, float)):
        multiplier = rng.choice(_NUMERIC_MULTIPLIERS)
        wrong[key] = val * multiplier if multiplier != 0 else 0
    elif isinstance(val, str):
        sub = rng.choice(_STRING_SUBSTITUTIONS)
        wrong[key] = sub if sub != val else sub + "_wrong"
    elif isinstance(val, list):
        wrong[key] = []
    elif isinstance(val, dict):
        wrong[key] = {}
    else:
        wrong[key] = None

    # Verify we actually changed something
    if wrong == correct_params:
        wrong["_extra_wrong_param"] = "injected_by_negative_generator"

    return wrong


# ── Trajectory Mutation ────────────────────────────────────────────────────────

def _swap_final_in_messages(
    messages: list[dict],
    new_tool: str,
    new_params: dict,
) -> list[dict]:
    """
    Return a copy of messages where the last FINAL({...}) is replaced
    with new_tool and new_params.

    All intermediate messages (DISCOVER, VERIFY, REPL OUTPUTs) are kept
    identical — only the final decision changes. This isolates the error
    to the DECIDE step, which is the hardest step to learn.
    """
    import copy
    new_messages = copy.deepcopy(messages)

    final_json = json.dumps({"tool": new_tool, "params": new_params})
    final_str = f"FINAL({final_json})"

    # Find last assistant message and swap FINAL(...)
    for i in range(len(new_messages) - 1, -1, -1):
        msg = new_messages[i]
        if msg["role"] != "assistant":
            continue
        content = msg["content"]
        # Replace existing FINAL(...) with new one
        if re.search(r"FINAL\s*\(", content):
            # Use lambda to avoid re.sub treating backslashes in replacement as escapes
            new_content = re.sub(
                r"FINAL\s*\(\{.*?\}\)",
                lambda _: final_str,
                content,
                flags=re.DOTALL,
            )
            if new_content == content:
                # Regex didn't match (greedy issue) — append replacement
                new_content = re.sub(
                    r"FINAL\s*\(.*$",
                    lambda _: final_str,
                    content,
                    flags=re.DOTALL,
                )
            new_messages[i]["content"] = new_content
            break

    return new_messages


# ── Per-trajectory Negative Generation ────────────────────────────────────────

def generate_negatives_from_trajectory(
    traj: dict,
    rng: random.Random,
    n_wrong_tool: int = 1,
    n_wrong_param: int = 1,
) -> list[dict]:
    """
    Generate hard negatives from a single positive trajectory.

    Returns a list of DPO-pair dicts:
        {query, correct_tool, negative_type, chosen_messages, rejected_messages}

    May return fewer than n_wrong_tool + n_wrong_param records if the registry
    is too small or params are uncorruptible.
    """
    messages = traj.get("messages", [])
    correct_tool = traj.get("correct_tool") or traj.get("predicted_tool", "")
    query = traj.get("query", "")

    if not messages or not correct_tool or not query:
        return []

    # Build registry from the trajectory's tools list
    tools_list = traj.get("tools")
    if tools_list:
        registry = MCPToolRegistry(tools_list)
    else:
        # Try to load from registry field
        reg_name = traj.get("registry", "small_10")
        reg_path = PROJECT_ROOT / "test_data" / "tool_registries" / f"{reg_name}.json"
        if not reg_path.exists():
            return []
        registry = MCPToolRegistry.from_json_file(reg_path)

    schema = registry.get_schema(correct_tool)
    correct_params = traj.get("final_answer", {}).get("params", {})
    if not correct_params:
        # Try to extract from FINAL(...) in messages
        for msg in reversed(messages):
            if msg["role"] == "assistant":
                m = re.search(r"FINAL\((\{.*?\})\)", msg["content"], re.DOTALL)
                if m:
                    try:
                        obj = json.loads(m.group(1))
                        correct_params = obj.get("params", {})
                    except json.JSONDecodeError:
                        pass
                break

    results = []

    # ── Type 1: Wrong tool ────────────────────────────────────────────
    for _ in range(n_wrong_tool):
        wrong_tool = find_plausible_wrong_tool(registry, correct_tool, query, rng)
        if wrong_tool is None:
            break
        # Keep the correct params but point to wrong tool
        rejected_msgs = _swap_final_in_messages(messages, wrong_tool, correct_params)
        results.append({
            "query": query,
            "correct_tool": correct_tool,
            "negative_type": "wrong_tool",
            "wrong_tool": wrong_tool,
            "chosen_messages": messages,
            "rejected_messages": rejected_msgs,
        })

    # ── Type 2: Wrong params ──────────────────────────────────────────
    for _ in range(n_wrong_param):
        wrong_params = generate_wrong_params(correct_params, schema, rng)
        if wrong_params == correct_params:
            break
        rejected_msgs = _swap_final_in_messages(messages, correct_tool, wrong_params)
        results.append({
            "query": query,
            "correct_tool": correct_tool,
            "negative_type": "wrong_param",
            "wrong_params": wrong_params,
            "chosen_messages": messages,
            "rejected_messages": rejected_msgs,
        })

    return results


# ── Main Loop ─────────────────────────────────────────────────────────────────

def run(args):
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Load positives from all specified input files
    positives: list[dict] = []
    for input_path_str in args.positives:
        input_path = PROJECT_ROOT / input_path_str
        if not input_path.exists():
            print(f"WARNING: Input file not found, skipping: {input_path}")
            continue
        count_before = len(positives)
        with open(input_path) as f:
            for line in f:
                line = line.strip()
                if line:
                    positives.append(json.loads(line))
        added = len(positives) - count_before
        print(f"Loaded {added:,} positives from {input_path}")

    if not positives:
        print("ERROR: No positive trajectories loaded.")
        sys.exit(1)

    print(f"Total positives: {len(positives):,}")
    print(f"Target negatives: {args.target:,}")
    print(f"\n{'='*60}")

    rng = random.Random(args.seed)
    rng.shuffle(positives)

    stats = {
        "processed": 0,
        "wrong_tool": 0,
        "wrong_param": 0,
        "skipped": 0,
    }

    out_file = open(output_path, "w")
    total_written = 0

    try:
        for traj in positives:
            if total_written >= args.target:
                break

            remaining = args.target - total_written
            # Generate 1 of each type per positive, stop when we hit target
            n_wt = min(1, remaining)
            n_wp = min(1, remaining - n_wt)

            negatives = generate_negatives_from_trajectory(
                traj, rng,
                n_wrong_tool=n_wt,
                n_wrong_param=n_wp,
            )

            if not negatives:
                stats["skipped"] += 1
                continue

            for neg in negatives:
                if total_written >= args.target:
                    break
                out_file.write(json.dumps(neg) + "\n")
                total_written += 1
                if neg["negative_type"] == "wrong_tool":
                    stats["wrong_tool"] += 1
                else:
                    stats["wrong_param"] += 1

            stats["processed"] += 1

            if stats["processed"] % 500 == 0:
                print(f"  [{stats['processed']:,} positives processed | "
                      f"{total_written:,} negatives written]")
    finally:
        out_file.close()

    print(f"\n{'='*60}")
    print(f"HARD NEGATIVE GENERATION COMPLETE")
    print(f"  Positives processed : {stats['processed']:,}")
    print(f"  Skipped (no registry): {stats['skipped']:,}")
    print(f"  Negatives written   : {total_written:,}")
    print(f"    wrong_tool        : {stats['wrong_tool']:,}")
    print(f"    wrong_param       : {stats['wrong_param']:,}")
    print(f"  Output              : {output_path}")
    print(f"{'='*60}")

    if total_written < args.target:
        print(f"\nWARNING: Only generated {total_written:,} negatives (target: {args.target:,}).")
        print("  Try adding more --positives files or lowering --target.")
    else:
        print(f"\n✓  {total_written:,} DPO training pairs ready.")


def main():
    parser = argparse.ArgumentParser(
        description="Generate hard negative trajectories for DPO fine-tuning"
    )
    parser.add_argument(
        "--positives", type=str, action="append", required=True,
        metavar="PATH",
        help="Input JSONL file of positive trajectories (repeat for multiple files)",
    )
    parser.add_argument(
        "--output", type=str, default="data/negatives_dpo.jsonl",
        help="Output JSONL file path (relative to project root)",
    )
    parser.add_argument(
        "--target", type=int, default=3000,
        help="Target number of negative DPO pairs to generate (default: 3000)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
