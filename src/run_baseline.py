"""
run_baseline.py — CLI entry point for Phase 1 baseline testing.

Runs Level 0 (zero-shot), Level 1 (constrained REPL), and Level 2
(few-shot REPL) evaluations, saving results to results/ directory.

Usage:
    python -m src.run_baseline --level 0 --registry small_10
    python -m src.run_baseline --level 1 --registry medium_25
    python -m src.run_baseline --level 1 --level 2 --registry large_50
    python -m src.run_baseline --all
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tool_registry import MCPToolRegistry
from src.repl_engine import REPLEngine, REPLConfig
from src.slm_interface import SLMInterface, SLMConfig
from src.scaffold import MCPRLMScaffold
from src.evaluator import MCPToolEvaluator, QueryTestCase


def load_system_prompt(level: int) -> str:
    """Load the system prompt for the given level."""
    prompts_dir = PROJECT_ROOT / "prompts"
    prompt_file = prompts_dir / f"level{level}.txt"
    if not prompt_file.exists():
        raise FileNotFoundError(f"System prompt not found: {prompt_file}")
    return prompt_file.read_text()


def load_test_cases() -> list[QueryTestCase]:
    """Load test queries and ground truth."""
    queries_path = PROJECT_ROOT / "test_data" / "queries" / "all_queries.json"
    gt_path = PROJECT_ROOT / "test_data" / "ground_truth" / "expected_selections.json"

    if not queries_path.exists() or not gt_path.exists():
        print("Test data not found. Generating...")
        from src.generate_test_data import generate_test_data
        generate_test_data(str(PROJECT_ROOT))

    return MCPToolEvaluator.load_test_cases(queries_path, gt_path)


def resolve_eval_paths(query_split: str) -> tuple[Path, Path]:
    """Resolve query and ground-truth file paths for a split."""
    if query_split == "all":
        queries_path = PROJECT_ROOT / "test_data" / "queries" / "all_queries.json"
        gt_path = PROJECT_ROOT / "test_data" / "ground_truth" / "expected_selections.json"
    else:
        queries_path = PROJECT_ROOT / "test_data" / "queries" / f"{query_split}_queries.json"
        gt_path = PROJECT_ROOT / "test_data" / "ground_truth" / f"{query_split}_expected_selections.json"

    return queries_path, gt_path


def filter_test_cases(
    cases: list[QueryTestCase],
    registry: MCPToolRegistry,
    task_mode: str = "selection",
) -> list[QueryTestCase]:
    """Filter test cases to only include those with tools in the registry."""
    available = set(registry.list_names())
    filtered = []
    for case in cases:
        is_discovery = case.expected_tool == "DISCOVERY" or case.category == "discovery"

        if task_mode == "selection" and is_discovery:
            continue
        if task_mode == "discovery" and not is_discovery:
            continue

        if is_discovery:
            filtered.append(case)
            continue

        if case.expected_tool in available:
            filtered.append(case)

    return filtered


def _task_mode_suffix(task_mode: str) -> str:
    return "" if task_mode == "selection" else f"_{task_mode}"


def run_evaluation(
    level: int,
    registry_name: str,
    model_name: str = "qwen3.5:4b",
    query_split: str = "all",
    trials: int = 1,
    task_mode: str = "selection",
    verbose: bool = True,
):
    """
    Run a single evaluation at the given level and registry size.

    Args:
        level: Testing level (0, 1, or 2).
        registry_name: Name of the registry file (e.g., 'small_10').
        model_name: Ollama model name.
        verbose: Print detailed results.

    Returns:
        Tuple of (metrics dict, results list).
    """
    # Load registry
    registry_path = PROJECT_ROOT / "test_data" / "tool_registries" / f"{registry_name}.json"
    if not registry_path.exists():
        print(f"Registry not found: {registry_path}")
        print("Generating test data first...")
        from src.generate_test_data import generate_test_data
        generate_test_data(str(PROJECT_ROOT))

    registry = MCPToolRegistry.from_json_file(registry_path)

    # Load and filter test cases
    queries_path, gt_path = resolve_eval_paths(query_split)
    if not queries_path.exists() or not gt_path.exists():
        print(f"Query split files not found for split='{query_split}'. Generating test data...")
        from src.generate_test_data import generate_test_data
        generate_test_data(str(PROJECT_ROOT))

    all_cases = MCPToolEvaluator.load_test_cases(queries_path, gt_path)
    cases = filter_test_cases(all_cases, registry, task_mode=task_mode)

    if not cases:
        print(f"No test cases match the {registry_name} registry. Skipping.")
        return None, None

    # Load system prompt
    prompt = load_system_prompt(level)

    # Create SLM interface
    slm_config = SLMConfig(model_name=model_name)

    # Create scaffold
    repl_config = REPLConfig.for_level(max(level, 1))  # Level 0 uses Level 1 config
    repl = REPLEngine(
        config=repl_config,
        initial_namespace={"tools_registry": registry},
    )
    formatted_prompt = prompt.replace("{num_tools}", str(registry.count()))
    scaffold = MCPRLMScaffold(
        slm=SLMInterface(slm_config),
        repl_engine=repl,
        system_prompt=formatted_prompt,
    )

    # Create evaluator and run
    evaluator = MCPToolEvaluator(scaffold=scaffold, tools_registry=registry)
    if trials > 1:
        metrics, results = evaluator.evaluate_repeated(
            cases,
            level=level,
            trials=trials,
            verbose=verbose,
        )
    else:
        metrics, results = evaluator.evaluate_all(cases, level=level, verbose=verbose)

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_suffix = _task_mode_suffix(task_mode)
    output_path = PROJECT_ROOT / "results" / f"level{level}_{registry_name}{mode_suffix}_{timestamp}.json"
    evaluator.save_results(metrics, results, output_path)

    return metrics, results


def run_all(model_name: str = "qwen3.5:4b", task_mode: str = "selection"):
    """Run all baseline evaluations (Levels 0, 1, 2 × registry sizes)."""
    registries = ["small_10", "medium_25", "large_50", "xlarge_100"]
    levels = [0, 1, 2]

    all_metrics = {}

    for registry_name in registries:
        for level in levels:
            print(f"\n{'#'*60}")
            print(f"  Level {level} × {registry_name}")
            print(f"{'#'*60}")

            try:
                metrics, _ = run_evaluation(
                    level=level,
                    registry_name=registry_name,
                    model_name=model_name,
                    query_split="all",
                    trials=1,
                    task_mode=task_mode,
                )
                if metrics:
                    key = f"level{level}_{registry_name}"
                    if isinstance(metrics, dict):
                        all_metrics[key] = metrics
                    else:
                        all_metrics[key] = metrics.to_dict()
            except Exception as e:
                print(f"  ❌ Error: {e}")
                filtered.append(case)
                continue

    # Save summary
    summary_path = PROJECT_ROOT / "results" / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\n✅ All baselines complete. Summary saved to {summary_path}")

    return all_metrics


def main():
    parser = argparse.ArgumentParser(
        description="Run MCP-RLM baseline evaluations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.run_baseline --level 1 --registry small_10
    python -m src.run_baseline --level 0 --level 1 --registry medium_25 --model qwen3.5:4b
  python -m src.run_baseline --all
        """,
    )
    parser.add_argument(
        "--level", type=int, action="append",
        help="Testing level(s): 0=zero-shot, 1=constrained REPL, 2=few-shot REPL",
    )
    parser.add_argument(
        "--registry", type=str, default="small_10",
        choices=["small_10", "medium_25", "large_50", "xlarge_100"],
        help="Tool registry size to use",
    )
    parser.add_argument(
        "--model", type=str, default="qwen3.5:4b",
        help="Ollama model name (default: qwen3.5:4b)",
    )
    parser.add_argument(
        "--query-split", type=str, default="all",
        choices=["all", "train", "dev", "test"],
        help="Which query split to evaluate.",
    )
    parser.add_argument(
        "--trials", type=int, default=1,
        help="Repeated trials per query for reliability metrics (>=1).",
    )
    parser.add_argument(
        "--task-mode", type=str, default="selection",
        choices=["selection", "discovery", "mixed"],
        help="Evaluate selection-only, discovery-only, or mixed query sets.",
    )
    parser.add_argument(
        "--all", action="store_true",
        help="Run all levels × all registries",
    )
    parser.add_argument(
        "--generate-data", action="store_true",
        help="Only generate test data, don't run evaluation",
    )
    parser.add_argument(
        "-q", "--quiet", action="store_true",
        help="Less verbose output",
    )

    args = parser.parse_args()

    if args.generate_data:
        from src.generate_test_data import generate_test_data
        generate_test_data(str(PROJECT_ROOT))
        return

    if args.all:
        run_all(model_name=args.model, task_mode=args.task_mode)
        return

    levels = args.level or [1]
    for level in levels:
        run_evaluation(
            level=level,
            registry_name=args.registry,
            model_name=args.model,
            query_split=args.query_split,
            trials=args.trials,
            task_mode=args.task_mode,
            verbose=not args.quiet,
        )


if __name__ == "__main__":
    main()
