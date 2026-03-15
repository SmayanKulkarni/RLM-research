"""
Phase 1 Baseline Evaluation Runner.

Runs Level 0/1/2 evaluations on small_10 and medium_25 registries.
Saves results per-run and a combined summary.
"""

import sys
import json
import time
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tool_registry import MCPToolRegistry
from src.repl_engine import REPLEngine, REPLConfig
from src.slm_interface import SLMInterface, SLMConfig
from src.scaffold import MCPRLMScaffold
from src.evaluator import MCPToolEvaluator, QueryTestCase

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS_DIR = PROJECT_ROOT / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def load_test_cases():
    """Load all test cases."""
    return MCPToolEvaluator.load_test_cases(
        PROJECT_ROOT / "test_data" / "queries" / "all_queries.json",
        PROJECT_ROOT / "test_data" / "ground_truth" / "expected_selections.json",
    )


def filter_cases(cases, registry, include_discovery=False):
    """Filter test cases to only those whose expected_tool is in the registry."""
    available = set(registry.list_names())
    filtered = []
    for c in cases:
        if c.expected_tool == "DISCOVERY":
            if include_discovery:
                filtered.append(c)
            continue
        if c.expected_tool in available:
            filtered.append(c)
    return filtered


def run_single_eval(level, registry_name, slm, all_cases):
    """Run one evaluation: level × registry."""
    registry_path = PROJECT_ROOT / "test_data" / "tool_registries" / f"{registry_name}.json"
    registry = MCPToolRegistry.from_json_file(registry_path)

    cases = filter_cases(all_cases, registry)
    if not cases:
        print(f"  [SKIP] No matching test cases for {registry_name}")
        return None, None

    # Load prompt
    prompt = (PROJECT_ROOT / "prompts" / f"level{level}.txt").read_text()
    formatted_prompt = prompt.replace("{num_tools}", str(registry.count()))

    # Build scaffold
    repl_config = REPLConfig.for_level(max(level, 1))
    repl = REPLEngine(config=repl_config, initial_namespace={"tools_registry": registry})
    scaffold = MCPRLMScaffold(slm=slm, repl_engine=repl, system_prompt=formatted_prompt)

    evaluator = MCPToolEvaluator(scaffold=scaffold, tools_registry=registry)
    metrics, results = evaluator.evaluate_all(cases, level=level, verbose=True)

    # Save per-run results
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = RESULTS_DIR / f"level{level}_{registry_name}_{ts}.json"
    evaluator.save_results(metrics, results, out_path)

    return metrics, results


def main():
    print("=" * 60)
    print("  PHASE 1: BASELINE EVALUATION")
    print(f"  Model: qwen3.5:4b")
    print(f"  Time: {datetime.now().isoformat()}")
    print("=" * 60)

    # Initialize SLM configs
    # Level 0 (zero-shot): no stop sequences needed (no REPL)
    # Level 1/2: stop sequences enforce turn-taking at </code>
    slm_level0 = SLMInterface(SLMConfig(
        model_name="qwen3.5:4b",
        temperature=0.1,
        max_tokens=2048,
        stop_sequences=[],  # No REPL → no stop sequences
    ))
    slm_repl = SLMInterface(SLMConfig(
        model_name="qwen3.5:4b",
        temperature=0.1,
        max_tokens=2048,
        stop_sequences=["</code>"],  # Enforce turn-taking
    ))

    all_cases = load_test_cases()
    print(f"\nLoaded {len(all_cases)} total test cases")

    # Test matrix: levels × registries
    test_matrix = [
        (0, "small_10"),
        (0, "medium_25"),
        (1, "small_10"),
        (1, "medium_25"),
        (2, "small_10"),
        (2, "medium_25"),
    ]

    summary = {}
    for level, reg_name in test_matrix:
        print(f"\n{'#' * 60}")
        print(f"  Level {level} × {reg_name}")
        print(f"{'#' * 60}")

        start = time.time()
        current_slm = slm_level0 if level == 0 else slm_repl
        metrics, results = run_single_eval(level, reg_name, current_slm, all_cases)
        elapsed = time.time() - start

        if metrics:
            key = f"level{level}_{reg_name}"
            summary[key] = metrics.to_dict()
            summary[key]["total_time_seconds"] = round(elapsed, 1)
            print(f"\n  ⏱  Completed in {elapsed:.0f}s")

    # Save combined summary
    summary_path = RESULTS_DIR / "baseline_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Print final summary table
    print("\n" + "=" * 80)
    print("  BASELINE SUMMARY")
    print("=" * 80)
    print(f"{'Config':<25} {'TSA':>6} {'PC':>6} {'RCV':>6} {'E2E':>6} {'Turns':>6} {'Fail':>6} {'Time':>6}")
    print("-" * 80)
    for key, m in summary.items():
        print(
            f"{key:<25} "
            f"{m['tool_selection_accuracy']:>5.1%} "
            f"{m['parameter_correctness']:>5.1%} "
            f"{m['repl_code_validity']:>5.1%} "
            f"{m['end_to_end_accuracy']:>5.1%} "
            f"{m['avg_turns']:>5.1f} "
            f"{m['failure_rate']:>5.1%} "
            f"{m.get('total_time_seconds', 0):>5.0f}s"
        )
    print("=" * 80)
    print(f"\nResults saved to: {summary_path}")


if __name__ == "__main__":
    main()
