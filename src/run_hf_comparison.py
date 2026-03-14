from __future__ import annotations

import json
import sys
import time
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


def load_cases():
    return MCPToolEvaluator.load_test_cases(
        PROJECT_ROOT / "test_data" / "queries" / "all_queries.json",
        PROJECT_ROOT / "test_data" / "ground_truth" / "expected_selections.json",
    )


def filter_cases(cases, registry):
    available = set(registry.list_names())
    filtered = []
    for case in cases:
        if case.expected_tool == "DISCOVERY":
            continue
        if case.expected_tool in available:
            filtered.append(case)
    return filtered


def run_matrix(model_label: str, model_name: str) -> dict:
    all_cases = load_cases()
    matrix = [
        (0, "small_10"),
        (0, "medium_25"),
        (1, "small_10"),
        (1, "medium_25"),
        (2, "small_10"),
        (2, "medium_25"),
    ]

    summary = {}
    slm = HFSLM(model_name=model_name, stop_sequences=[])

    for level, registry_name in matrix:
        registry_path = PROJECT_ROOT / "test_data" / "tool_registries" / f"{registry_name}.json"
        registry = MCPToolRegistry.from_json_file(registry_path)
        cases = filter_cases(all_cases, registry)

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
        metrics, _ = evaluator.evaluate_all(cases, level=level, verbose=False)
        result = metrics.to_dict()
        result["elapsed_seconds"] = round(time.time() - start, 1)
        summary[f"level{level}_{registry_name}"] = result
        print(model_label, f"level{level}_{registry_name}", result)

    output_path = PROJECT_ROOT / "results" / f"hf_{model_label}_summary.json"
    with open(output_path, "w") as file_handle:
        json.dump(summary, file_handle, indent=2)

    print("saved", output_path)
    return summary


def main():
    base = run_matrix("base_qwen35_4b", "Qwen/Qwen3.5-4B")
    finetuned = run_matrix("finetuned_qwen355_4b", "finetuned/qwen3.55-4b-rlm-lora")

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

    cmp_path = PROJECT_ROOT / "results" / "hf_base_vs_finetuned_comparison.json"
    with open(cmp_path, "w") as file_handle:
        json.dump(comparison, file_handle, indent=2)
    print("saved", cmp_path)


if __name__ == "__main__":
    main()
