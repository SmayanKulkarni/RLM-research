"""
MCPToolEvaluator — Evaluates SLM performance on MCP tool selection tasks.

Computes 7 metrics as defined in implementation_poa.md § Q4:
  1. Tool Selection Accuracy (TSA)
  2. Parameter Correctness (PC)
  3. REPL Code Validity (RCV)
  4. End-to-End Task Completion (E2E)
  5. Average REPL Turns
  6. Context Tokens Saved
  7. Failure Rate

Design Reference: implementation_poa.md § Q4 (Evaluation)
"""

from __future__ import annotations

import json
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .scaffold import MCPRLMScaffold, ScaffoldResult
from .tool_registry import MCPToolRegistry


@dataclass
class QueryTestCase:
    """A single test case for evaluation."""
    query: str
    expected_tool: str
    expected_params: dict = field(default_factory=dict)
    category: str = "simple"       # simple, multi_select, discovery
    difficulty: str = "easy"       # easy, medium, hard


@dataclass
class EvalResult:
    """Result of evaluating a single test case."""
    query: str
    expected_tool: str
    predicted_answer: str | None
    predicted_tool: str | None
    predicted_params: dict | None
    tool_correct: bool
    params_correct: bool
    code_validity: float
    turns_used: int
    success: bool                 # Did scaffold produce FINAL()?
    elapsed_seconds: float

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "expected_tool": self.expected_tool,
            "predicted_tool": self.predicted_tool,
            "tool_correct": self.tool_correct,
            "params_correct": self.params_correct,
            "code_validity": self.code_validity,
            "turns_used": self.turns_used,
            "success": self.success,
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


@dataclass
class EvalMetrics:
    """Aggregated evaluation metrics."""
    tool_selection_accuracy: float
    parameter_correctness: float
    repl_code_validity: float
    end_to_end_accuracy: float
    avg_turns: float
    failure_rate: float
    total_queries: int
    context_tokens_with_repl: int = 0
    context_tokens_without_repl: int = 0

    @property
    def context_savings_pct(self) -> float:
        if self.context_tokens_without_repl == 0:
            return 0.0
        return 1 - (self.context_tokens_with_repl / self.context_tokens_without_repl)

    def to_dict(self) -> dict:
        return {
            "tool_selection_accuracy": round(self.tool_selection_accuracy, 4),
            "parameter_correctness": round(self.parameter_correctness, 4),
            "repl_code_validity": round(self.repl_code_validity, 4),
            "end_to_end_accuracy": round(self.end_to_end_accuracy, 4),
            "avg_turns": round(self.avg_turns, 2),
            "failure_rate": round(self.failure_rate, 4),
            "context_savings_pct": round(self.context_savings_pct, 4),
            "total_queries": self.total_queries,
        }

    def summary(self) -> str:
        lines = [
            "=" * 50,
            "  EVALUATION RESULTS",
            "=" * 50,
            f"  Tool Selection Accuracy : {self.tool_selection_accuracy:.1%}",
            f"  Parameter Correctness   : {self.parameter_correctness:.1%}",
            f"  REPL Code Validity      : {self.repl_code_validity:.1%}",
            f"  End-to-End Accuracy     : {self.end_to_end_accuracy:.1%}",
            f"  Avg REPL Turns          : {self.avg_turns:.1f}",
            f"  Failure Rate            : {self.failure_rate:.1%}",
            f"  Context Savings         : {self.context_savings_pct:.1%}",
            f"  Total Queries           : {self.total_queries}",
            "=" * 50,
        ]
        return "\n".join(lines)


class MCPToolEvaluator:
    """Evaluate SLM performance on MCP tool selection tasks."""

    def __init__(
        self,
        scaffold: MCPRLMScaffold,
        tools_registry: MCPToolRegistry,
    ):
        self.scaffold = scaffold
        self.tools_registry = tools_registry

    # ── Data Loading ─────────────────────────────────────────────────

    @staticmethod
    def load_test_cases(queries_path: str | Path, ground_truth_path: str | Path) -> list[QueryTestCase]:
        """
        Load test cases from JSON files.

        Expected format for queries:
            [{"query": "...", "category": "...", "difficulty": "..."}, ...]

        Expected format for ground truth:
            [{"query": "...", "expected_tool": "...", "expected_params": {...}}, ...]
        """
        with open(queries_path) as f:
            queries = json.load(f)
        with open(ground_truth_path) as f:
            ground_truth = json.load(f)

        # Build lookup by query text
        gt_lookup = {gt["query"]: gt for gt in ground_truth}

        cases = []
        for q in queries:
            gt = gt_lookup.get(q["query"], {})
            cases.append(QueryTestCase(
                query=q["query"],
                expected_tool=gt.get("expected_tool", ""),
                expected_params=gt.get("expected_params", {}),
                category=q.get("category", "simple"),
                difficulty=q.get("difficulty", "easy"),
            ))
        return cases

    # ── Answer Parsing ───────────────────────────────────────────────

    @staticmethod
    def parse_tool_from_answer(answer: str | None) -> tuple[str | None, dict | None]:
        """
        Extract tool name and params from a FINAL() answer.

        Tries multiple formats:
          1. JSON: {"tool": "name", "params": {...}}
          2. Plain text: tool_name or "tool_name"
        """
        if answer is None:
            return None, None

        # Try JSON parsing
        try:
            data = json.loads(answer)
            if isinstance(data, dict):
                tool = data.get("tool", data.get("name"))
                params = data.get("params", data.get("parameters", {}))
                return tool, params
        except (json.JSONDecodeError, TypeError):
            pass

        # Try extracting JSON from within the answer (handles nested braces)
        # Find the outermost { ... } pair
        brace_depth = 0
        start_idx = None
        for i, c in enumerate(answer):
            if c == '{':
                if brace_depth == 0:
                    start_idx = i
                brace_depth += 1
            elif c == '}':
                brace_depth -= 1
                if brace_depth == 0 and start_idx is not None:
                    try:
                        data = json.loads(answer[start_idx:i+1])
                        if isinstance(data, dict):
                            tool = data.get("tool", data.get("name"))
                            params = data.get("params", data.get("parameters", {}))
                            return tool, params
                    except (json.JSONDecodeError, TypeError):
                        pass
                    start_idx = None


        # Plain text — try to extract a tool name
        clean = answer.strip().strip('"').strip("'")
        if clean and " " not in clean:
            return clean, None

        return None, None

    @staticmethod
    def parse_discovery_answer(answer: str | None) -> str | None:
        """Extract free-text discovery answer from FINAL(...) output."""
        if answer is None:
            return None
        clean = str(answer).strip()
        if not clean:
            return None
        return clean

    # ── Evaluation ───────────────────────────────────────────────────

    def evaluate_single(
        self,
        test_case: QueryTestCase,
        level: int = 1,
        verbose: bool = False,
    ) -> EvalResult:
        """Run and evaluate a single test case."""
        # Reset REPL state for each query
        self.scaffold.reset(self.tools_registry)

        # Run the scaffold
        if level == 0:
            result = self.scaffold.run_zero_shot(test_case.query, self.tools_registry)
        else:
            result = self.scaffold.run(test_case.query)

        # Parse the answer
        predicted_tool, predicted_params = self.parse_tool_from_answer(result.answer)

        is_discovery = (
            test_case.expected_tool == "DISCOVERY"
            or test_case.category.lower() == "discovery"
        )

        if is_discovery:
            discovery_answer = self.parse_discovery_answer(result.answer)
            tool_correct = bool(result.success and discovery_answer)
            params_correct = tool_correct
            predicted_tool = "DISCOVERY" if tool_correct else predicted_tool
            predicted_params = None
        else:
            # Check tool correctness
            tool_correct = (
                predicted_tool is not None
                and predicted_tool.lower() == test_case.expected_tool.lower()
            )

            # Check params correctness
            params_correct = False
            if tool_correct and test_case.expected_params and predicted_params:
                # Check required params are present and match
                params_correct = all(
                    k in predicted_params and str(predicted_params[k]).lower() == str(v).lower()
                    for k, v in test_case.expected_params.items()
                )
            elif tool_correct and not test_case.expected_params:
                params_correct = True  # No params expected

        # Code validity
        code_validity = 0.0
        if result.repl_history:
            successful = sum(1 for h in result.repl_history if h.success)
            code_validity = successful / len(result.repl_history)
        elif level == 0:
            code_validity = 1.0  # No REPL used, not applicable

        if verbose:
            status = "✅" if tool_correct else "❌"
            print(f"  {status} Query: {test_case.query[:60]}...")
            print(f"     Expected: {test_case.expected_tool} | Got: {predicted_tool}")
            if not tool_correct and result.answer:
                print(f"     Raw answer: {result.answer[:100]}...")

        return EvalResult(
            query=test_case.query,
            expected_tool=test_case.expected_tool,
            predicted_answer=result.answer,
            predicted_tool=predicted_tool,
            predicted_params=predicted_params,
            tool_correct=tool_correct,
            params_correct=params_correct,
            code_validity=code_validity,
            turns_used=result.turns,
            success=result.success,
            elapsed_seconds=result.elapsed_seconds,
        )

    def evaluate_all(
        self,
        test_cases: list[QueryTestCase],
        level: int = 1,
        verbose: bool = True,
    ) -> tuple[EvalMetrics, list[EvalResult]]:
        """
        Run evaluation across all test cases.

        Args:
            test_cases: List of QueryTestCase objects.
            level: REPL level (0, 1, or 2).
            verbose: Print per-query results.

        Returns:
            Tuple of (aggregated metrics, individual results).
        """
        if verbose:
            print(f"\n{'='*50}")
            print(f"  Running Level {level} evaluation ({len(test_cases)} queries)")
            print(f"  Model: {self.scaffold.slm}")
            print(f"  Tools: {self.tools_registry}")
            print(f"{'='*50}\n")

        results = []
        for i, tc in enumerate(test_cases):
            if verbose:
                print(f"[{i+1}/{len(test_cases)}]", end="")
            result = self.evaluate_single(tc, level=level, verbose=verbose)
            results.append(result)

        metrics = self._aggregate_metrics(results)

        if verbose:
            print(metrics.summary())

        return metrics, results

    def evaluate_repeated(
        self,
        test_cases: list[QueryTestCase],
        level: int = 1,
        trials: int = 5,
        verbose: bool = True,
        bootstrap_samples: int = 1000,
        bootstrap_confidence: float = 0.95,
    ) -> tuple[dict, list[list[EvalResult]]]:
        """Run repeated-trial evaluation for pass-style reliability metrics."""
        if trials < 1:
            raise ValueError("trials must be >= 1")

        trial_results: list[list[EvalResult]] = []
        per_trial_metrics: list[dict] = []

        for trial in range(trials):
            if verbose:
                print(f"\n--- Trial {trial + 1}/{trials} ---")
            metrics, results = self.evaluate_all(test_cases, level=level, verbose=verbose)
            trial_results.append(results)
            per_trial_metrics.append(metrics.to_dict())

        n = len(test_cases)
        if n == 0:
            raise ValueError("No test cases provided")

        # Query-level pass-style stats over repeated trials
        e2e_success_matrix = [
            [bool(trial_results[t][i].tool_correct and trial_results[t][i].params_correct) for t in range(trials)]
            for i in range(n)
        ]
        tsa_matrix = [
            [bool(trial_results[t][i].tool_correct) for t in range(trials)]
            for i in range(n)
        ]
        pc_matrix = [
            [bool(trial_results[t][i].params_correct) for t in range(trials)]
            for i in range(n)
        ]
        failure_matrix = [
            [not bool(trial_results[t][i].success) for t in range(trials)]
            for i in range(n)
        ]

        pass_at_1_values = [float(row[0]) for row in e2e_success_matrix]
        pass_at_k_values = [float(any(row)) for row in e2e_success_matrix]
        pass_power_k_values = [float(all(row)) for row in e2e_success_matrix]

        # Mean per-query rates across trials (stability signal)
        tsa_mean_query = [sum(row) / trials for row in tsa_matrix]
        pc_mean_query = [sum(row) / trials for row in pc_matrix]
        fail_mean_query = [sum(row) / trials for row in failure_matrix]
        e2e_mean_query = [sum(row) / trials for row in e2e_success_matrix]

        summary = {
            "trials": trials,
            "aggregated": per_trial_metrics[0],
            "pass_at_1": round(sum(pass_at_1_values) / n, 4),
            "pass_at_k": round(sum(pass_at_k_values) / n, 4),
            "pass_power_k_proxy": round(sum(pass_power_k_values) / n, 4),
            "mean_tsa_over_trials": round(sum(tsa_mean_query) / n, 4),
            "mean_pc_over_trials": round(sum(pc_mean_query) / n, 4),
            "mean_failure_over_trials": round(sum(fail_mean_query) / n, 4),
            "mean_e2e_over_trials": round(sum(e2e_mean_query) / n, 4),
            "confidence_intervals": {
                "pass_at_1": self._bootstrap_ci(pass_at_1_values, bootstrap_samples, bootstrap_confidence),
                "pass_at_k": self._bootstrap_ci(pass_at_k_values, bootstrap_samples, bootstrap_confidence),
                "pass_power_k_proxy": self._bootstrap_ci(pass_power_k_values, bootstrap_samples, bootstrap_confidence),
                "mean_tsa_over_trials": self._bootstrap_ci(tsa_mean_query, bootstrap_samples, bootstrap_confidence),
                "mean_pc_over_trials": self._bootstrap_ci(pc_mean_query, bootstrap_samples, bootstrap_confidence),
                "mean_failure_over_trials": self._bootstrap_ci(fail_mean_query, bootstrap_samples, bootstrap_confidence),
                "mean_e2e_over_trials": self._bootstrap_ci(e2e_mean_query, bootstrap_samples, bootstrap_confidence),
            },
            "per_trial_metrics": per_trial_metrics,
        }

        return summary, trial_results

    def _aggregate_metrics(self, results: list[EvalResult]) -> EvalMetrics:
        """Aggregate per-query results into EvalMetrics."""
        n = len(results)
        return EvalMetrics(
            tool_selection_accuracy=sum(r.tool_correct for r in results) / n,
            parameter_correctness=sum(r.params_correct for r in results) / n,
            repl_code_validity=sum(r.code_validity for r in results) / n if any(r.code_validity > 0 for r in results) else 0.0,
            end_to_end_accuracy=sum(r.tool_correct and r.params_correct for r in results) / n,
            avg_turns=sum(r.turns_used for r in results) / n,
            failure_rate=sum(not r.success for r in results) / n,
            total_queries=n,
            context_tokens_without_repl=self.tools_registry.get_total_tokens_estimate(),
        )

    @staticmethod
    def _bootstrap_ci(values: list[float], samples: int = 1000, confidence: float = 0.95) -> dict:
        """Bootstrap confidence interval for mean(values)."""
        if not values:
            return {"low": 0.0, "high": 0.0}

        rng = random.Random(42)
        n = len(values)
        means = []
        for _ in range(samples):
            sample = [values[rng.randrange(n)] for _ in range(n)]
            means.append(sum(sample) / n)

        means.sort()
        lower_idx = int(((1 - confidence) / 2) * (samples - 1))
        upper_idx = int((1 - (1 - confidence) / 2) * (samples - 1))

        return {
            "low": round(means[lower_idx], 4),
            "high": round(means[upper_idx], 4),
        }

    # ── Results I/O ──────────────────────────────────────────────────

    @staticmethod
    def save_results(
        metrics: EvalMetrics | dict,
        results: list[EvalResult] | list[list[EvalResult]],
        output_path: str | Path,
    ):
        """Save evaluation results to a JSON file."""
        if isinstance(metrics, EvalMetrics):
            metrics_payload = metrics.to_dict()
        else:
            metrics_payload = metrics

        if results and isinstance(results[0], list):
            results_payload = [[r.to_dict() for r in trial] for trial in results]
        else:
            results_payload = [r.to_dict() for r in results]

        output = {
            "metrics": metrics_payload,
            "results": results_payload,
        }
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w") as f:
            json.dump(output, f, indent=2)
        print(f"Results saved to {output_path}")
