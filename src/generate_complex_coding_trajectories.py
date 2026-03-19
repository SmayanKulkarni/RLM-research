"""
generate_complex_coding_trajectories.py — Build hard coding-focused trajectories.

This generator creates deeper, multi-step tool-selection trajectories for complex
software-engineering tasks (debugging, refactors, CI fixes, data checks), grounded
by executing every <code> block against the real REPL.

Output is split-aware and metadata-rich so it can be merged into robust fine-tune
corpora.

Usage:
    conda run -n astro python -m src.generate_complex_coding_trajectories \
        --registries medium_25 large_50 xlarge_100 \
        --variants-per-task 3 \
        --output data/trajectories_complex_coding.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import random
from pathlib import Path
from typing import Any

from src.repl_engine import REPLEngine, REPLConfig
from src.tool_registry import MCPToolRegistry

PROJECT_ROOT = Path(__file__).parent.parent


COMPLEX_TASKS = [
    {
        "tool": "search_code",
        "category": "development",
        "keywords": ["search code TODO"],
        "query": (
            "Investigate our auth timeout regression by scanning /srv/app/src for all TODO or FIXME "
            "notes related to session refresh in Python files only, and prioritize findings from handlers."
        ),
        "params": {
            "path": "/srv/app/src",
            "pattern": "TODO|FIXME|session refresh",
            "file_extension": ".py",
        },
    },
    {
        "tool": "git_clone",
        "category": "development",
        "keywords": ["clone repository"],
        "query": (
            "Set up a repro workspace by cloning https://github.com/acme/payment-service into "
            "/home/dev/repros/payment-service before we test the failing webhook branch."
        ),
        "params": {
            "repo_url": "https://github.com/acme/payment-service",
            "destination_path": "/home/dev/repros/payment-service",
        },
    },
    {
        "tool": "git_commit",
        "category": "development",
        "keywords": ["git commit changes"],
        "query": (
            "After fixing flaky test retries, commit the staged changes in /workspace/rlm with message "
            "'stabilize retry policy for integration tests'."
        ),
        "params": {
            "repo_path": "/workspace/rlm",
            "message": "stabilize retry policy for integration tests",
        },
    },
    {
        "tool": "create_github_issue",
        "category": "development",
        "keywords": ["github issue bug report"],
        "query": (
            "Open a GitHub issue in acme/traffic-forecast documenting production memory spikes after "
            "deploy, include title 'memory spike after rollout' and mention p95 latency impact."
        ),
        "params": {
            "repo": "acme/traffic-forecast",
            "title": "memory spike after rollout",
            "body": "Observed memory spikes after rollout with p95 latency degradation.",
        },
    },
    {
        "tool": "validate_json",
        "category": "development",
        "keywords": ["validate json schema"],
        "query": (
            "Before release, validate the JSON payload at /tmp/release/config.json against our API schema "
            "so malformed feature flags do not break startup."
        ),
        "params": {
            "json_string": "{\"feature_flag\": true}",
        },
    },
    {
        "tool": "parse_csv",
        "category": "data_storage",
        "keywords": ["parse csv file"],
        "query": (
            "Parse /data/experiments/run_204_metrics.csv so we can inspect failed test rows and compare "
            "loss drift across nightly runs."
        ),
        "params": {
            "file_path": "/data/experiments/run_204_metrics.csv",
            "delimiter": ",",
        },
    },
    {
        "tool": "analyze_csv",
        "category": "data_storage",
        "keywords": ["analyze csv statistics"],
        "query": (
            "Run a statistical analysis on /data/traffic/validation_window.csv to inspect outliers and "
            "feature distribution drift before retraining."
        ),
        "params": {
            "file_path": "/data/traffic/validation_window.csv",
        },
    },
    {
        "tool": "query_database",
        "category": "data_storage",
        "keywords": ["query database"],
        "query": (
            "Execute a diagnostic query against analytics DB to list top 50 failed inference jobs in the "
            "last 24 hours including model_version and error_code for incident triage."
        ),
        "params": {
            "query": "SELECT model_version, error_code, count(*) FROM inference_failures WHERE ts > NOW() - INTERVAL '24 hours' GROUP BY model_version, error_code ORDER BY count(*) DESC LIMIT 50",
        },
    },
    {
        "tool": "run_shell_command",
        "category": "system",
        "keywords": ["run shell command logs"],
        "query": (
            "Run a shell command to inspect the latest 200 lines from /var/log/pipeline.log and filter for "
            "timeout or OOM markers so we can debug CI instability quickly."
        ),
        "params": {
            "command": "tail -n 200 /var/log/pipeline.log | grep -E 'timeout|OOM|out of memory'",
        },
    },
    {
        "tool": "read_file",
        "category": "filesystem",
        "keywords": ["read file config"],
        "query": (
            "Read /etc/rlm/repl_config.yaml to verify whether max_turns and timeout settings were changed "
            "during yesterday's hotfix rollout."
        ),
        "params": {
            "path": "/etc/rlm/repl_config.yaml",
        },
    },
    {
        "tool": "write_file",
        "category": "filesystem",
        "keywords": ["write file patch"],
        "query": (
            "Write a temporary patch note to /tmp/hotfix_notes.txt summarizing rollback steps and owners "
            "for tonight's emergency release review."
        ),
        "params": {
            "path": "/tmp/hotfix_notes.txt",
            "content": "Rollback playbook: stop service, restore previous checkpoint, validate health checks.",
        },
    },
    {
        "tool": "compress_files",
        "category": "filesystem",
        "keywords": ["compress logs archive"],
        "query": (
            "Compress /var/tmp/debug_artifacts into a zip archive before sharing with platform engineering "
            "for deep postmortem analysis."
        ),
        "params": {
            "input_path": "/var/tmp/debug_artifacts",
            "output_path": "/var/tmp/debug_artifacts.zip",
        },
    },
]


QUERY_PREFIXES = [
    "For a high-priority incident, ",
    "In a production debugging scenario, ",
    "For an in-depth root-cause analysis, ",
]

QUERY_SUFFIXES = [
    "Focus on correctness over speed.",
    "Assume strict rollback requirements.",
    "Treat this as a blocker for tonight's release.",
    "Prioritize reproducibility and clear diagnostics.",
    "Account for flaky CI behavior in your approach.",
    "Assume on-call handoff notes must be explicit.",
    "Keep operational risk minimal.",
    "Bias toward deterministic investigation steps.",
    "Include safeguards suitable for production workflows.",
    "Assume this affects multiple services in staging.",
    "Treat latency regressions as a primary concern.",
    "Minimize ambiguity in tool and parameter selection.",
]


def load_system_prompt(level: int, num_tools: int) -> str:
    path = PROJECT_ROOT / "prompts" / f"level{level}.txt"
    return path.read_text().replace("{num_tools}", str(num_tools))


def get_registries(registry_names: list[str]) -> dict[str, MCPToolRegistry]:
    registries: dict[str, MCPToolRegistry] = {}
    for name in registry_names:
        path = PROJECT_ROOT / "test_data" / "tool_registries" / f"{name}.json"
        if not path.exists():
            continue
        registries[name] = MCPToolRegistry.from_json_file(path)
    return registries


def extract_code_blocks(text: str) -> list[str]:
    return REPLEngine.extract_code_blocks(text)


def params_for_schema(task_params: dict[str, Any], schema: dict[str, Any] | None) -> dict[str, Any]:
    if not schema:
        return dict(task_params)
    props = schema.get("inputSchema", {}).get("properties", {})
    required = schema.get("inputSchema", {}).get("required", [])

    final_params: dict[str, Any] = {}
    for key in props:
        if key in task_params:
            final_params[key] = task_params[key]

    for key in required:
        if key in final_params:
            continue
        ptype = props.get(key, {}).get("type", "string")
        if ptype in {"integer", "number"}:
            final_params[key] = 1
        elif ptype == "boolean":
            final_params[key] = True
        elif ptype == "array":
            final_params[key] = []
        else:
            final_params[key] = f"{key}_value"
    return final_params


def build_assistant_steps(task: dict[str, Any], tool_schema: dict[str, Any], params: dict[str, Any]) -> list[str]:
    tool_name = task["tool"]
    keyword = task["keywords"][0]
    category = task["category"]

    return [
        (
            "THOUGHT: This is a complex engineering request, so I should first discover candidate tools "
            "using targeted coding keywords.\n"
            f"<code>\nprint(tools_registry.search(\"{keyword}\"))\n</code>"
        ),
        (
            "THOUGHT: I should narrow by category to avoid selecting a similarly named but incorrect tool.\n"
            f"<code>\nprint(tools_registry.filter_by_category(\"{category}\"))\n</code>"
        ),
        (
            "THOUGHT: Now I will verify the exact schema of the best candidate before finalizing parameters.\n"
            f"<code>\nprint(tools_registry.get_schema(\"{tool_name}\"))\n</code>"
        ),
        (
            "THOUGHT: The schema confirms the right tool; I can now return a grounded FINAL selection.\n"
            f"FINAL({json.dumps({'tool': tool_name, 'params': params}, sort_keys=True)})"
        ),
    ]


def execute_grounded_messages(
    query: str,
    task: dict[str, Any],
    registry_name: str,
    registry: MCPToolRegistry,
    prompt_level: int,
) -> dict[str, Any] | None:
    tool_schema = registry.get_schema(task["tool"])
    if not tool_schema:
        return None

    params = params_for_schema(task.get("params", {}), tool_schema)

    repl = REPLEngine(
        config=REPLConfig(max_turns=10, max_output_chars=2500, allowed_imports=["re", "json"]),
        initial_namespace={"tools_registry": registry},
    )

    system_prompt = load_system_prompt(prompt_level, registry.count())
    messages: list[dict[str, Any]] = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    steps = build_assistant_steps(task, tool_schema, params)
    turns = 0

    for step in steps:
        messages.append({"role": "assistant", "content": step})
        code_blocks = extract_code_blocks(step)
        for code in code_blocks:
            result = repl.execute(code)
            turns += 1
            messages.append({"role": "user", "content": f"[REPL OUTPUT]:\n{result.output}"})
            if not result.success:
                return None

    return {
        "registry": registry_name,
        "query": query,
        "correct_tool": task["tool"],
        "turns": turns,
        "messages": messages,
        "metadata": {
            "split": "train",
            "query_category": "complex_coding",
            "difficulty": "hard",
            "trajectory_kind": "positive",
            "prompt_level": prompt_level,
            "registry_size": registry.count(),
        },
        "provenance": {
            "source_dataset": "complex_coding_generator",
            "grounded_with_real_repl": True,
            "lineage": ["complex_coding_generator"],
        },
    }


def query_variants(base_query: str, variants_per_task: int) -> list[str]:
    variants: list[str] = [base_query]
    if variants_per_task <= 1:
        return variants

    base_tail = base_query[0].lower() + base_query[1:] if base_query else base_query
    i = 0
    while len(variants) < variants_per_task:
        prefix = QUERY_PREFIXES[i % len(QUERY_PREFIXES)]
        suffix = QUERY_SUFFIXES[(i // len(QUERY_PREFIXES)) % len(QUERY_SUFFIXES)]
        candidate = f"{prefix}{base_tail} {suffix}".strip()
        if candidate not in variants:
            variants.append(candidate)
        i += 1

    return variants


def dedupe(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    final_rows: list[dict[str, Any]] = []
    for row in rows:
        key = json.dumps(
            {
                "registry": row.get("registry"),
                "tool": row.get("correct_tool"),
                "query": row.get("query"),
            },
            sort_keys=True,
        )
        if key in seen:
            continue
        seen.add(key)
        final_rows.append(row)
    return final_rows


def run(args: argparse.Namespace) -> None:
    random.seed(args.seed)
    registries = get_registries(args.registries)
    if not registries:
        raise SystemExit("No valid registries found.")

    all_rows: list[dict[str, Any]] = []
    skipped = 0

    for registry_name, registry in registries.items():
        available = set(registry.list_names())
        for task in COMPLEX_TASKS:
            if task["tool"] not in available:
                continue
            for query in query_variants(task["query"], args.variants_per_task):
                row = execute_grounded_messages(
                    query=query,
                    task=task,
                    registry_name=registry_name,
                    registry=registry,
                    prompt_level=args.prompt_level,
                )
                if row is None:
                    skipped += 1
                    continue
                all_rows.append(row)

    all_rows = dedupe(all_rows)
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as file_handle:
        for row in all_rows:
            file_handle.write(json.dumps(row) + "\n")

    by_registry: dict[str, int] = {}
    for row in all_rows:
        by_registry[row["registry"]] = by_registry.get(row["registry"], 0) + 1

    print("=" * 60)
    print("COMPLEX CODING TRAJECTORY GENERATION COMPLETE")
    print(f"  Total trajectories: {len(all_rows)}")
    print(f"  Skipped          : {skipped}")
    print(f"  Per registry     : {by_registry}")
    print(f"  Output           : {output_path}")
    print("=" * 60)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate complex coding trajectories")
    parser.add_argument(
        "--registries",
        nargs="+",
        default=["medium_25", "large_50", "xlarge_100"],
        help="Registry names under test_data/tool_registries",
    )
    parser.add_argument(
        "--variants-per-task",
        type=int,
        default=3,
        help="How many query variants to generate per complex task",
    )
    parser.add_argument(
        "--prompt-level",
        type=int,
        default=2,
        choices=[0, 1, 2],
        help="Prompt template level to embed in system message",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data/trajectories_complex_coding.jsonl",
        help="Output JSONL path relative to project root",
    )
    parser.add_argument("--seed", type=int, default=42)
    run(parser.parse_args())


if __name__ == "__main__":
    main()