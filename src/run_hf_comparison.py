from __future__ import annotations

import argparse
import json
import gc
import sys
import time
import random
import math
from datetime import datetime
from pathlib import Path

import torch
from unsloth import FastLanguageModel

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluator import MCPToolEvaluator
from src.repl_engine import REPLConfig, REPLEngine
from src.scaffold import MCPRLMScaffold
from src.tool_registry import MCPToolRegistry


class HFSLM:
    def __init__(self, model_name: str, stop_sequences: list[str] | None = None, max_tokens: int = 512):
        self.model_name = model_name
        self.stop_sequences = stop_sequences or []
        self.max_tokens = max_tokens

        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=model_name,
            max_seq_length=2048,
            load_in_4bit=True,
            dtype=None,
        )

        if hasattr(tokenizer, "tokenizer"):
            tokenizer = tokenizer.tokenizer

        FastLanguageModel.for_inference(model)
        self.model = model
        self.tokenizer = tokenizer

    def _truncate_stop(self, text: str) -> str:
        cut = None
        for stop in self.stop_sequences:
            idx = text.find(stop)
            if idx != -1:
                end = idx + len(stop)
                cut = end if cut is None else min(cut, end)
        return text if cut is None else text[:cut]

    def generate(self, messages: list[dict[str, str]]) -> str:
        prompt = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {key: value.to(self.model.device) for key, value in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(
                **inputs,
                max_new_tokens=self.max_tokens,
                do_sample=False,
                temperature=0.1,
                top_p=0.9,
                repetition_penalty=1.1,
                pad_token_id=self.tokenizer.eos_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
                use_cache=True,
            )

        new_tokens = output[0][inputs["input_ids"].shape[1]:]
        text = self.tokenizer.decode(new_tokens, skip_special_tokens=True)
        return self._truncate_stop(text)

    def __repr__(self) -> str:
        return f"HFSLM(model={self.model_name})"

    def set_stop_sequences(self, stop_sequences: list[str]):
        self.stop_sequences = stop_sequences


def load_cases(query_split: str = "all"):
    if query_split == "all":
        queries_path = PROJECT_ROOT / "test_data" / "queries" / "all_queries.json"
        ground_truth_path = PROJECT_ROOT / "test_data" / "ground_truth" / "expected_selections.json"
    else:
        queries_path = PROJECT_ROOT / "test_data" / "queries" / f"{query_split}_queries.json"
        ground_truth_path = PROJECT_ROOT / "test_data" / "ground_truth" / f"{query_split}_expected_selections.json"

    return MCPToolEvaluator.load_test_cases(queries_path, ground_truth_path)


def filter_cases(cases, registry, task_mode: str = "selection"):
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


def run_matrix(
    model_label: str,
    model_name: str,
    query_split: str = "all",
    include_registries: list[str] | None = None,
    trials: int = 1,
    task_mode: str = "selection",
) -> dict:
    all_cases = load_cases(query_split=query_split)
    selected_registries = include_registries or ["small_10", "medium_25"]
    matrix = [
        (0, reg_name) for reg_name in selected_registries
    ] + [
        (1, reg_name) for reg_name in selected_registries
    ] + [
        (2, reg_name) for reg_name in selected_registries
    ]

    summary = {}
    slm = HFSLM(model_name=model_name, stop_sequences=[])

    for level, registry_name in matrix:
        registry_path = PROJECT_ROOT / "test_data" / "tool_registries" / f"{registry_name}.json"
        registry = MCPToolRegistry.from_json_file(registry_path)
        cases = filter_cases(all_cases, registry, task_mode=task_mode)

        prompt = (PROJECT_ROOT / "prompts" / f"level{level}.txt").read_text()
        prompt = prompt.replace("{num_tools}", str(registry.count()))

        repl_config = REPLConfig.for_level(max(level, 1))
        repl = REPLEngine(config=repl_config, initial_namespace={"tools_registry": registry})
        if level == 0:
            slm.set_stop_sequences([])
        else:
            slm.set_stop_sequences(["</code>"])
        scaffold = MCPRLMScaffold(slm=slm, repl_engine=repl, system_prompt=prompt)
        evaluator = MCPToolEvaluator(scaffold=scaffold, tools_registry=registry)

        start = time.time()
        if trials > 1:
            repeated_metrics, trial_results = evaluator.evaluate_repeated(
                cases,
                level=level,
                trials=trials,
                verbose=False,
            )
            result = dict(repeated_metrics["aggregated"])
            result["pass_at_1"] = repeated_metrics["pass_at_1"]
            result["pass_at_k"] = repeated_metrics["pass_at_k"]
            result["pass_power_k_proxy"] = repeated_metrics["pass_power_k_proxy"]
            result["confidence_intervals"] = repeated_metrics["confidence_intervals"]

            per_query_e2e_rates = []
            per_query_tsa_rates = []
            per_query_pc_rates = []
            for i, case in enumerate(cases):
                tsa_hits = []
                pc_hits = []
                e2e_hits = []
                for trial_idx in range(trials):
                    r = trial_results[trial_idx][i]
                    tsa = 1.0 if r.tool_correct else 0.0
                    pc = 1.0 if r.params_correct else 0.0
                    e2e = 1.0 if (r.tool_correct and r.params_correct) else 0.0
                    tsa_hits.append(tsa)
                    pc_hits.append(pc)
                    e2e_hits.append(e2e)

                per_query_tsa_rates.append({"query": case.query, "rate": round(sum(tsa_hits) / trials, 4)})
                per_query_pc_rates.append({"query": case.query, "rate": round(sum(pc_hits) / trials, 4)})
                per_query_e2e_rates.append({"query": case.query, "rate": round(sum(e2e_hits) / trials, 4)})

            result["per_query_rates"] = {
                "tsa": per_query_tsa_rates,
                "pc": per_query_pc_rates,
                "e2e": per_query_e2e_rates,
            }
        else:
            metrics, _ = evaluator.evaluate_all(cases, level=level, verbose=False)
            result = metrics.to_dict()

        result["elapsed_seconds"] = round(time.time() - start, 1)
        result["trials"] = trials
        summary[f"level{level}_{registry_name}"] = result
        print(model_label, f"level{level}_{registry_name}", result)

    output_suffix = "" if query_split == "all" else f"_{query_split}"
    output_path = PROJECT_ROOT / "results" / f"hf_{model_label}_summary{output_suffix}.json"
    with open(output_path, "w") as file_handle:
        json.dump(summary, file_handle, indent=2)

    del slm
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    print("saved", output_path)
    return summary


def write_comparison(base: dict, finetuned: dict):
    def paired_bootstrap_diff_ci(base_values: list[float], fin_values: list[float], samples: int = 2000, confidence: float = 0.95) -> dict:
        if not base_values or not fin_values or len(base_values) != len(fin_values):
            return {"low": 0.0, "high": 0.0}
        rng = random.Random(42)
        n = len(base_values)
        diffs = []
        for _ in range(samples):
            idxs = [rng.randrange(n) for _ in range(n)]
            mean_base = sum(base_values[i] for i in idxs) / n
            mean_fin = sum(fin_values[i] for i in idxs) / n
            diffs.append(mean_fin - mean_base)
        diffs.sort()
        lower_idx = int(((1 - confidence) / 2) * (samples - 1))
        upper_idx = int((1 - (1 - confidence) / 2) * (samples - 1))
        return {"low": round(diffs[lower_idx], 4), "high": round(diffs[upper_idx], 4)}

    def paired_sign_test_pvalue(base_values: list[float], fin_values: list[float]) -> float:
        if not base_values or not fin_values or len(base_values) != len(fin_values):
            return 1.0

        diffs = [f - b for b, f in zip(base_values, fin_values)]
        nonzero = [d for d in diffs if d != 0]
        n = len(nonzero)
        if n == 0:
            return 1.0

        pos = sum(1 for d in nonzero if d > 0)
        neg = n - pos
        k = min(pos, neg)

        tail = 0.0
        for i in range(k + 1):
            tail += math.comb(n, i) * (0.5 ** n)

        pvalue = min(1.0, 2 * tail)
        return round(pvalue, 4)

    def extract_query_rates(section: dict, metric_key: str) -> tuple[list[str], list[float]]:
        qrates = section.get("per_query_rates", {}).get(metric_key, []) if isinstance(section, dict) else []
        queries = [row.get("query", "") for row in qrates]
        rates = [float(row.get("rate", 0.0)) for row in qrates]
        return queries, rates

    def approx_delta_ci(base_ci: dict, fin_ci: dict) -> tuple[float, float]:
        low = round(float(fin_ci["low"]) - float(base_ci["high"]), 4)
        high = round(float(fin_ci["high"]) - float(base_ci["low"]), 4)
        return low, high

    comparison = {}
    for key in base:
        if key not in finetuned:
            continue
        comparison[key] = {
            "base": base[key],
            "finetuned": finetuned[key],
            "delta_tsa": round(finetuned[key]["tool_selection_accuracy"] - base[key]["tool_selection_accuracy"], 4),
            "delta_pc": round(finetuned[key]["parameter_correctness"] - base[key]["parameter_correctness"], 4),
            "delta_rcv": round(finetuned[key]["repl_code_validity"] - base[key]["repl_code_validity"], 4),
            "delta_e2e": round(finetuned[key]["end_to_end_accuracy"] - base[key]["end_to_end_accuracy"], 4),
            "delta_fail": round(finetuned[key]["failure_rate"] - base[key]["failure_rate"], 4),
        }
        if "pass_at_1" in base[key] and "pass_at_1" in finetuned[key]:
            comparison[key]["delta_pass_at_1"] = round(finetuned[key]["pass_at_1"] - base[key]["pass_at_1"], 4)
        if "pass_at_k" in base[key] and "pass_at_k" in finetuned[key]:
            comparison[key]["delta_pass_at_k"] = round(finetuned[key]["pass_at_k"] - base[key]["pass_at_k"], 4)
        if "pass_power_k_proxy" in base[key] and "pass_power_k_proxy" in finetuned[key]:
            comparison[key]["delta_pass_power_k_proxy"] = round(
                finetuned[key]["pass_power_k_proxy"] - base[key]["pass_power_k_proxy"],
                4,
            )

        base_ci = base[key].get("confidence_intervals", {}) if isinstance(base[key], dict) else {}
        fin_ci = finetuned[key].get("confidence_intervals", {}) if isinstance(finetuned[key], dict) else {}
        ci_map = {
            "delta_tsa": "mean_tsa_over_trials",
            "delta_pc": "mean_pc_over_trials",
            "delta_e2e": "mean_e2e_over_trials",
            "delta_fail": "mean_failure_over_trials",
            "delta_pass_at_1": "pass_at_1",
            "delta_pass_at_k": "pass_at_k",
            "delta_pass_power_k_proxy": "pass_power_k_proxy",
        }
        significance = {}
        for delta_key, ci_key in ci_map.items():
            if delta_key not in comparison[key]:
                continue
            if ci_key not in base_ci or ci_key not in fin_ci:
                continue
            low, high = approx_delta_ci(base_ci[ci_key], fin_ci[ci_key])
            significance[delta_key] = {
                "approx_ci": {"low": low, "high": high},
                "excludes_zero": (low > 0.0 or high < 0.0),
            }
        if significance:
            comparison[key]["significance"] = significance

        # Query-level paired significance when repeated-trial query rates are present
        paired = {}
        metric_alias = {
            "tsa": "delta_tsa",
            "pc": "delta_pc",
            "e2e": "delta_e2e",
        }
        for metric, delta_key in metric_alias.items():
            base_queries, base_rates = extract_query_rates(base[key], metric)
            fin_queries, fin_rates = extract_query_rates(finetuned[key], metric)
            if not base_rates or not fin_rates:
                continue
            if base_queries != fin_queries:
                continue
            ci = paired_bootstrap_diff_ci(base_rates, fin_rates)
            pvalue = paired_sign_test_pvalue(base_rates, fin_rates)
            paired[delta_key] = {
                "paired_bootstrap_ci": ci,
                "paired_sign_test_pvalue": pvalue,
                "excludes_zero": (ci["low"] > 0.0 or ci["high"] < 0.0),
            }
        if paired:
            comparison[key]["paired_significance"] = paired

    cmp_path = PROJECT_ROOT / "results" / "hf_base_vs_finetuned_comparison.json"
    with open(cmp_path, "w") as file_handle:
        json.dump(comparison, file_handle, indent=2)
    print("saved", cmp_path)


def write_markdown_report(comparison: dict, report_path: Path):
    lines = [
        "# HF Base vs Fine-Tuned Comparison Report",
        "",
        "Generated by `src/run_hf_comparison.py`.",
        "",
    ]

    for cell, payload in comparison.items():
        lines.append(f"## {cell}")
        for metric in ["delta_tsa", "delta_pc", "delta_e2e", "delta_fail", "delta_pass_at_1", "delta_pass_at_k", "delta_pass_power_k_proxy"]:
            if metric in payload:
                lines.append(f"- {metric}: {payload[metric]:+.4f}")

        paired = payload.get("paired_significance", {})
        if paired:
            lines.append("- paired_significance:")
            for metric, stats in paired.items():
                ci = stats.get("paired_bootstrap_ci", {"low": 0.0, "high": 0.0})
                pvalue = stats.get("paired_sign_test_pvalue", 1.0)
                excl = stats.get("excludes_zero", False)
                lines.append(
                    f"  - {metric}: CI[{ci['low']:+.4f}, {ci['high']:+.4f}], p={pvalue:.4f}, excludes_zero={excl}"
                )

        lines.append("")

    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines))
    print("saved", report_path)


def write_experiment_manifest(manifest_path: Path, payload: dict):
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(payload, indent=2))
    print("saved", manifest_path)


def main():
    parser = argparse.ArgumentParser(description="Run HF base vs finetuned benchmark comparisons")
    parser.add_argument("--model-label", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--compare-only", action="store_true")
    parser.add_argument("--query-split", type=str, default="all", choices=["all", "train", "dev", "test"])
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--task-mode", type=str, default="selection", choices=["selection", "discovery", "mixed"])
    parser.add_argument(
        "--registries",
        nargs="+",
        default=["small_10", "medium_25"],
        choices=["small_10", "medium_25", "large_50", "xlarge_100"],
    )
    parser.add_argument("--base-summary", type=str, default="results/hf_base_qwen35_4b_summary.json")
    parser.add_argument("--finetuned-summary", type=str, default="results/hf_finetuned_qwen355_4b_summary.json")
    parser.add_argument("--report-path", type=str, default="results/hf_base_vs_finetuned_report.md")
    parser.add_argument("--manifest-path", type=str, default="results/hf_comparison_manifest.json")
    args = parser.parse_args()

    if args.compare_only:
        with open(PROJECT_ROOT / args.base_summary) as file_handle:
            base = json.load(file_handle)
        with open(PROJECT_ROOT / args.finetuned_summary) as file_handle:
            finetuned = json.load(file_handle)
        write_comparison(base, finetuned)
        with open(PROJECT_ROOT / "results" / "hf_base_vs_finetuned_comparison.json") as file_handle:
            comparison = json.load(file_handle)
        write_markdown_report(comparison, PROJECT_ROOT / args.report_path)

        manifest = {
            "timestamp_utc": datetime.utcnow().isoformat() + "Z",
            "mode": "compare_only",
            "inputs": {
                "base_summary": args.base_summary,
                "finetuned_summary": args.finetuned_summary,
            },
            "outputs": {
                "comparison_json": "results/hf_base_vs_finetuned_comparison.json",
                "report_markdown": args.report_path,
            },
            "settings": {
                "query_split": args.query_split,
                "task_mode": args.task_mode,
                "trials": args.trials,
                "registries": args.registries,
            },
        }
        write_experiment_manifest(PROJECT_ROOT / args.manifest_path, manifest)
        return

    if args.model_label and args.model_name:
        run_matrix(
            args.model_label,
            args.model_name,
            query_split=args.query_split,
            include_registries=args.registries,
            trials=args.trials,
            task_mode=args.task_mode,
        )
        return

    base = run_matrix(
        "base_qwen35_4b",
        "Qwen/Qwen3.5-4B",
        query_split=args.query_split,
        include_registries=args.registries,
        trials=args.trials,
        task_mode=args.task_mode,
    )
    finetuned = run_matrix(
        "finetuned_qwen355_4b",
        "finetuned/qwen3.55-4b-rlm-lora",
        query_split=args.query_split,
        include_registries=args.registries,
        trials=args.trials,
        task_mode=args.task_mode,
    )
    write_comparison(base, finetuned)
    with open(PROJECT_ROOT / "results" / "hf_base_vs_finetuned_comparison.json") as file_handle:
        comparison = json.load(file_handle)
    write_markdown_report(comparison, PROJECT_ROOT / args.report_path)

    manifest = {
        "timestamp_utc": datetime.utcnow().isoformat() + "Z",
        "mode": "full",
        "inputs": {
            "base_model": "Qwen/Qwen3.5-4B",
            "finetuned_model": "finetuned/qwen3.55-4b-rlm-lora",
        },
        "outputs": {
            "base_summary": f"results/hf_base_qwen35_4b_summary{'' if args.query_split == 'all' else f'_{args.query_split}'}.json",
            "finetuned_summary": f"results/hf_finetuned_qwen355_4b_summary{'' if args.query_split == 'all' else f'_{args.query_split}'}.json",
            "comparison_json": "results/hf_base_vs_finetuned_comparison.json",
            "report_markdown": args.report_path,
        },
        "settings": {
            "query_split": args.query_split,
            "task_mode": args.task_mode,
            "trials": args.trials,
            "registries": args.registries,
        },
    }
    write_experiment_manifest(PROJECT_ROOT / args.manifest_path, manifest)


if __name__ == "__main__":
    main()
