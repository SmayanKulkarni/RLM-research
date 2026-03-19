"""
build_robust_trajectories.py — Construct a larger, split-aware training corpus.

This script consolidates verified / augmented trajectory JSONL files into a
single robust corpus with:
  - leakage-safe split inference from train/dev/test query manifests,
  - provenance metadata on every trajectory,
  - balancing by (registry, tool),
  - paraphrase-based positive augmentation,
  - correction-turn trajectories for hard-negative supervision,
  - optional preference-pair export for future DPO/ORPO experiments.

Usage:
    conda run -n astro python -m src.build_robust_trajectories \
        --inputs data/trajectories_verified.jsonl data/trajectories_augmented.jsonl \
        --output data/trajectories_robust.jsonl \
        --output-preferences data/trajectory_preferences.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
QUERY_DIR = PROJECT_ROOT / "test_data" / "queries"
GROUND_TRUTH_DIR = PROJECT_ROOT / "test_data" / "ground_truth"
REGISTRY_DIR = PROJECT_ROOT / "test_data" / "tool_registries"

SYNONYMS = {
    "send": ["deliver", "dispatch", "transmit"],
    "get": ["fetch", "retrieve", "obtain"],
    "create": ["make", "set up", "generate"],
    "search": ["look up", "find", "search for"],
    "read": ["open", "view", "display"],
    "write": ["save", "store", "put"],
    "list": ["show", "display", "enumerate"],
    "convert": ["transform", "change"],
    "check": ["verify", "validate", "inspect"],
    "weather": ["forecast", "weather conditions"],
    "email": ["e-mail", "mail"],
    "message": ["text", "notification"],
    "file": ["document", "file"],
    "directory": ["folder", "directory"],
}

CORRECTION_PROMPT = (
    "That selection is incorrect. Re-check the registry evidence and return the corrected "
    "FINAL JSON only."
)


def load_json(path: Path) -> Any:
    with open(path) as file_handle:
        return json.load(file_handle)


def load_jsonl(path: Path) -> list[dict]:
    rows: list[dict] = []
    with open(path) as file_handle:
        for line in file_handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def save_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as file_handle:
        for row in rows:
            file_handle.write(json.dumps(row) + "\n")


def registry_size_from_name(registry_name: str) -> int | None:
    match = re.search(r"_(\d+)$", registry_name)
    return int(match.group(1)) if match else None


def load_query_metadata_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    all_path = QUERY_DIR / "all_queries.json"
    if all_path.exists():
        for row in load_json(all_path):
            query = row.get("query")
            if query:
                lookup[query] = {
                    "query_category": row.get("category", "unknown"),
                    "difficulty": row.get("difficulty", "unknown"),
                }

    for split_name in ("train", "dev", "test"):
        split_path = QUERY_DIR / f"{split_name}_queries.json"
        if not split_path.exists():
            continue
        for row in load_json(split_path):
            query = row.get("query")
            if not query:
                continue
            base = lookup.setdefault(query, {})
            base["split"] = split_name
            base.setdefault("query_category", row.get("category", "unknown"))
            base.setdefault("difficulty", row.get("difficulty", "unknown"))
    return lookup


def load_ground_truth_lookup() -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for file_path in sorted(GROUND_TRUTH_DIR.glob("*_expected_selections.json")):
        split_name = file_path.name.replace("_expected_selections.json", "")
        for row in load_json(file_path):
            query = row.get("query")
            if not query:
                continue
            lookup[query] = {
                "split": split_name,
                "expected_tool": row.get("expected_tool"),
                "expected_params": row.get("expected_params", {}),
            }

    all_path = GROUND_TRUTH_DIR / "expected_selections.json"
    if all_path.exists():
        for row in load_json(all_path):
            query = row.get("query")
            if not query:
                continue
            base = lookup.setdefault(query, {})
            base.setdefault("expected_tool", row.get("expected_tool"))
            base.setdefault("expected_params", row.get("expected_params", {}))
    return lookup


def load_registry_catalog() -> dict[str, dict[str, dict[str, Any]]]:
    catalog: dict[str, dict[str, dict[str, Any]]] = {}
    for file_path in sorted(REGISTRY_DIR.glob("*.json")):
        payload = load_json(file_path)
        tools = payload.get("tools", payload)
        if isinstance(tools, dict):
            tools = tools.get("tools", [])
        catalog[file_path.stem] = {tool["name"]: tool for tool in tools}
    return catalog


def extract_final_payload(message_content: str) -> dict[str, Any] | None:
    match = re.search(r"FINAL\((\{.*?\})\)", message_content, re.DOTALL)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
        return payload if isinstance(payload, dict) else None
    except json.JSONDecodeError:
        return None


def replace_final_payload(message_content: str, payload: dict[str, Any]) -> str:
    replacement = f"FINAL({json.dumps(payload, sort_keys=True)})"
    if "FINAL(" not in message_content:
        return replacement
    return re.sub(
        r"FINAL\(\{.*?\}\)",
        lambda _match: replacement,
        message_content,
        flags=re.DOTALL,
    )


def paraphrase_query(query: str) -> list[str]:
    variants: list[str] = []

    words = query.split()
    for _ in range(3):
        new_words = list(words)
        swapped = False
        for index, word in enumerate(new_words):
            normalized = word.lower().strip(".,!?")
            if normalized in SYNONYMS and random.random() < 0.45:
                replacement = random.choice(SYNONYMS[normalized])
                if word[:1].isupper():
                    replacement = replacement.capitalize()
                new_words[index] = replacement
                swapped = True
        if swapped:
            variant = " ".join(new_words)
            if variant != query and variant not in variants:
                variants.append(variant)

    prefixes = [
        "Can you ", "Please ", "I need to ", "I'd like to ",
        "Help me ", "I want to ", "Could you ",
    ]
    stripped = query
    for prefix in prefixes:
        if query.lower().startswith(prefix.lower()):
            stripped = query[len(prefix):]
            break

    if stripped != query and stripped:
        prefix = random.choice(prefixes)
        candidate = prefix + stripped[0].lower() + stripped[1:]
        if candidate != query and candidate not in variants:
            variants.append(candidate)
    elif query:
        prefix = random.choice(prefixes)
        candidate = prefix + query[0].lower() + query[1:]
        if candidate != query and candidate not in variants:
            variants.append(candidate)

    return variants[:3]


def annotate_trajectory(
    trajectory: dict[str, Any],
    source_name: str,
    query_lookup: dict[str, dict[str, Any]],
    ground_truth_lookup: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    row = copy.deepcopy(trajectory)
    query = row.get("query", "")
    metadata = dict(row.get("metadata", {}))
    provenance = dict(row.get("provenance", {}))

    query_meta = query_lookup.get(query, {})
    ground_truth = ground_truth_lookup.get(query, {})

    split = metadata.get("split") or query_meta.get("split") or ground_truth.get("split") or "train"
    metadata["split"] = split
    metadata["query_category"] = metadata.get("query_category") or query_meta.get("query_category", "synthetic_selection")
    metadata["difficulty"] = metadata.get("difficulty") or query_meta.get("difficulty", "synthetic")
    metadata["trajectory_kind"] = metadata.get("trajectory_kind", "positive")
    metadata["registry_size"] = metadata.get("registry_size") or registry_size_from_name(row.get("registry", ""))
    metadata["prompt_level"] = metadata.get("prompt_level", 2)
    if ground_truth.get("expected_params") and "expected_params" not in metadata:
        metadata["expected_params"] = ground_truth["expected_params"]

    provenance["source_dataset"] = provenance.get("source_dataset", source_name)
    provenance["grounded_with_real_repl"] = provenance.get("grounded_with_real_repl", True)
    provenance.setdefault("lineage", [source_name])

    row["metadata"] = metadata
    row["provenance"] = provenance
    return row


def update_query_text(messages: list[dict[str, Any]], old_query: str, new_query: str) -> list[dict[str, Any]]:
    updated: list[dict[str, Any]] = []
    swapped = False
    for message in messages:
        next_message = copy.deepcopy(message)
        if not swapped and next_message.get("role") == "user" and next_message.get("content") == old_query:
            next_message["content"] = new_query
            swapped = True
        updated.append(next_message)
    return updated


def make_paraphrase_trajectory(base: dict[str, Any], new_query: str) -> dict[str, Any]:
    row = copy.deepcopy(base)
    old_query = row.get("query", "")
    row["query"] = new_query
    row["messages"] = update_query_text(row.get("messages", []), old_query, new_query)
    metadata = dict(row.get("metadata", {}))
    provenance = dict(row.get("provenance", {}))
    metadata["trajectory_kind"] = "positive"
    metadata["augmentation"] = "paraphrase"
    provenance["lineage"] = [*provenance.get("lineage", []), "paraphrase"]
    row["metadata"] = metadata
    row["provenance"] = provenance
    return row


def choose_distractor_tool(
    trajectory: dict[str, Any],
    registry_catalog: dict[str, dict[str, dict[str, Any]]],
) -> str | None:
    registry_name = trajectory.get("registry", "")
    correct_tool = trajectory.get("correct_tool", "")
    registry_tools = registry_catalog.get(registry_name, {})
    if correct_tool not in registry_tools:
        candidates = [name for name in registry_tools if name != correct_tool]
        return random.choice(candidates) if candidates else None

    correct_schema = registry_tools[correct_tool]
    correct_category = correct_schema.get("annotations", {}).get("category")
    same_category = [
        name for name, schema in registry_tools.items()
        if name != correct_tool and schema.get("annotations", {}).get("category") == correct_category
    ]
    if same_category:
        return random.choice(same_category)

    candidates = [name for name in registry_tools if name != correct_tool]
    return random.choice(candidates) if candidates else None


def corrupt_params(
    params: dict[str, Any],
    trajectory: dict[str, Any],
    registry_catalog: dict[str, dict[str, dict[str, Any]]],
) -> dict[str, Any]:
    corrupted = copy.deepcopy(params)
    registry_name = trajectory.get("registry", "")
    tool_name = trajectory.get("correct_tool", "")
    schema = registry_catalog.get(registry_name, {}).get(tool_name, {})
    required = schema.get("inputSchema", {}).get("required", []) if schema else []

    if required:
        drop_key = required[0]
        corrupted.pop(drop_key, None)
        if corrupted != params:
            return corrupted

    if corrupted:
        first_key = next(iter(corrupted))
        value = corrupted[first_key]
        if isinstance(value, (int, float)):
            corrupted[first_key] = value + 1
        elif isinstance(value, bool):
            corrupted[first_key] = not value
        else:
            corrupted[first_key] = f"incorrect_{first_key}"
        return corrupted

    return {"invalid": True}


def build_bad_final_message(
    trajectory: dict[str, Any],
    bad_type: str,
    registry_catalog: dict[str, dict[str, dict[str, Any]]],
) -> str | None:
    messages = trajectory.get("messages", [])
    if not messages:
        return None
    last_assistant = next((msg for msg in reversed(messages) if msg.get("role") == "assistant"), None)
    if last_assistant is None:
        return None

    payload = extract_final_payload(last_assistant.get("content", ""))
    if payload is None:
        return None

    if bad_type == "wrong_tool":
        distractor = choose_distractor_tool(trajectory, registry_catalog)
        if not distractor:
            return None
        payload["tool"] = distractor
    elif bad_type == "wrong_params":
        payload["params"] = corrupt_params(payload.get("params", {}), trajectory, registry_catalog)
    else:
        return None

    return replace_final_payload(last_assistant.get("content", ""), payload)


def build_correction_trajectory(
    trajectory: dict[str, Any],
    bad_type: str,
    registry_catalog: dict[str, dict[str, dict[str, Any]]],
) -> tuple[dict[str, Any] | None, dict[str, Any] | None]:
    messages = copy.deepcopy(trajectory.get("messages", []))
    if not messages:
        return None, None

    last_assistant_idx = None
    for index in range(len(messages) - 1, -1, -1):
        if messages[index].get("role") == "assistant":
            last_assistant_idx = index
            break
    if last_assistant_idx is None:
        return None, None

    bad_message = build_bad_final_message(trajectory, bad_type, registry_catalog)
    if bad_message is None:
        return None, None

    correct_message = copy.deepcopy(messages[last_assistant_idx])
    prefix = copy.deepcopy(messages[:last_assistant_idx])
    new_messages = prefix + [
        {"role": "assistant", "content": bad_message},
        {"role": "user", "content": CORRECTION_PROMPT},
        correct_message,
    ]

    row = copy.deepcopy(trajectory)
    metadata = dict(row.get("metadata", {}))
    provenance = dict(row.get("provenance", {}))
    metadata["trajectory_kind"] = "correction"
    metadata["negative_type"] = bad_type
    provenance["lineage"] = [*provenance.get("lineage", []), f"correction:{bad_type}"]
    row["metadata"] = metadata
    row["provenance"] = provenance
    row["messages"] = new_messages

    preference_pair = {
        "query": trajectory.get("query"),
        "registry": trajectory.get("registry"),
        "split": metadata.get("split", "train"),
        "negative_type": bad_type,
        "context_messages": prefix,
        "chosen": correct_message.get("content", ""),
        "rejected": bad_message,
    }
    return row, preference_pair


def dedupe_trajectories(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    seen: set[str] = set()
    deduped: list[dict[str, Any]] = []
    for row in rows:
        signature = json.dumps(
            {
                "registry": row.get("registry"),
                "query": row.get("query"),
                "correct_tool": row.get("correct_tool"),
                "messages": row.get("messages"),
                "kind": row.get("metadata", {}).get("trajectory_kind", "positive"),
                "negative_type": row.get("metadata", {}).get("negative_type"),
            },
            sort_keys=True,
        )
        if signature in seen:
            continue
        seen.add(signature)
        deduped.append(row)
    return deduped


def trajectory_signature(row: dict[str, Any]) -> str:
    return json.dumps(
        {
            "registry": row.get("registry"),
            "query": row.get("query"),
            "correct_tool": row.get("correct_tool"),
            "messages": row.get("messages"),
            "kind": row.get("metadata", {}).get("trajectory_kind", "positive"),
            "negative_type": row.get("metadata", {}).get("negative_type"),
        },
        sort_keys=True,
    )


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    split_counts = Counter(row.get("metadata", {}).get("split", "unknown") for row in rows)
    registry_counts = Counter(row.get("registry", "unknown") for row in rows)
    kind_counts = Counter(row.get("metadata", {}).get("trajectory_kind", "positive") for row in rows)
    positive_tool_counts = Counter(
        (row.get("registry", "unknown"), row.get("correct_tool", "unknown"))
        for row in rows
        if row.get("metadata", {}).get("trajectory_kind", "positive") == "positive"
    )

    return {
        "total": len(rows),
        "splits": dict(split_counts),
        "registries": dict(registry_counts),
        "trajectory_kinds": dict(kind_counts),
        "min_positive_per_registry_tool": min(positive_tool_counts.values()) if positive_tool_counts else 0,
        "max_positive_per_registry_tool": max(positive_tool_counts.values()) if positive_tool_counts else 0,
        "unique_registry_tool_pairs": len(positive_tool_counts),
    }


def build_dataset(args: argparse.Namespace) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    query_lookup = load_query_metadata_lookup()
    ground_truth_lookup = load_ground_truth_lookup()
    registry_catalog = load_registry_catalog()

    include_splits = set(args.include_splits)
    augment_splits = set(args.augment_splits)

    source_rows: list[dict[str, Any]] = []
    source_stats: dict[str, int] = {}
    for raw_path in args.inputs:
        path = PROJECT_ROOT / raw_path
        if not path.exists():
            print(f"[WARN] Skipping missing input: {path}")
            continue
        rows = load_jsonl(path)
        source_stats[raw_path] = len(rows)
        for row in rows:
            annotated = annotate_trajectory(row, raw_path, query_lookup, ground_truth_lookup)
            if annotated.get("metadata", {}).get("split") in include_splits:
                source_rows.append(annotated)

    positives = dedupe_trajectories(source_rows)
    existing_signatures = {trajectory_signature(row) for row in positives}
    counts_by_key = Counter(
        (row.get("registry", "unknown"), row.get("correct_tool", "unknown"))
        for row in positives
        if row.get("metadata", {}).get("trajectory_kind", "positive") == "positive"
    )

    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in positives:
        if row.get("metadata", {}).get("trajectory_kind") != "positive":
            continue
        key = (row.get("registry", "unknown"), row.get("correct_tool", "unknown"))
        grouped[key].append(row)

    generated_paraphrases: list[dict[str, Any]] = []
    for key, examples in grouped.items():
        registry_name, tool_name = key
        eligible_examples = [
            row for row in examples if row.get("metadata", {}).get("split") in augment_splits
        ]
        if not eligible_examples:
            continue
        while counts_by_key[key] < args.min_per_registry_tool:
            base = eligible_examples[counts_by_key[key] % len(eligible_examples)]
            variants = paraphrase_query(base.get("query", ""))
            created = False
            for variant in variants:
                candidate = make_paraphrase_trajectory(base, variant)
                signature = trajectory_signature(candidate)
                if signature in existing_signatures:
                    continue
                generated_paraphrases.append(candidate)
                existing_signatures.add(signature)
                counts_by_key[key] += 1
                created = True
                if counts_by_key[key] >= args.min_per_registry_tool:
                    break
            if not created:
                break

    positives = dedupe_trajectories(positives + generated_paraphrases)

    if args.target_total > len(positives):
        grouped = defaultdict(list)
        counts_by_key = Counter()
        for row in positives:
            if row.get("metadata", {}).get("trajectory_kind") != "positive":
                continue
            key = (row.get("registry", "unknown"), row.get("correct_tool", "unknown"))
            grouped[key].append(row)
            counts_by_key[key] += 1

        ordered_keys = sorted(grouped, key=lambda key: counts_by_key[key])
        cursor = 0
        while len(positives) < args.target_total and ordered_keys:
            key = ordered_keys[cursor % len(ordered_keys)]
            eligible_examples = [
                row for row in grouped[key] if row.get("metadata", {}).get("split") in augment_splits
            ]
            if not eligible_examples:
                cursor += 1
                if cursor > len(ordered_keys) * 2:
                    break
                continue

            base = random.choice(eligible_examples)
            variants = paraphrase_query(base.get("query", ""))
            created = False
            for variant in variants:
                candidate = make_paraphrase_trajectory(base, variant)
                signature = trajectory_signature(candidate)
                if signature in existing_signatures:
                    continue
                positives.append(candidate)
                existing_signatures.add(signature)
                grouped[key].append(candidate)
                counts_by_key[key] += 1
                created = True
                if len(positives) >= args.target_total:
                    break
            if not created:
                cursor += 1
            else:
                ordered_keys = sorted(grouped, key=lambda item: counts_by_key[item])

    positives = dedupe_trajectories(positives)

    correction_candidates = [
        row for row in positives
        if row.get("metadata", {}).get("split") in augment_splits
        and row.get("metadata", {}).get("trajectory_kind") == "positive"
    ]
    random.shuffle(correction_candidates)

    target_corrections = int(round(len(correction_candidates) * args.correction_ratio))
    correction_rows: list[dict[str, Any]] = []
    preference_pairs: list[dict[str, Any]] = []

    for row in correction_candidates:
        if len(correction_rows) >= target_corrections:
            break
        for bad_type in ("wrong_tool", "wrong_params"):
            correction_row, preference_pair = build_correction_trajectory(row, bad_type, registry_catalog)
            if correction_row is None:
                continue
            signature = trajectory_signature(correction_row)
            if signature in existing_signatures:
                continue
            correction_rows.append(correction_row)
            existing_signatures.add(signature)
            preference_pairs.append(preference_pair)
            if len(correction_rows) >= target_corrections:
                break

    all_rows = dedupe_trajectories(positives + correction_rows)
    summary = summarize(all_rows)
    summary["source_rows"] = source_stats
    summary["generated_paraphrases"] = len(generated_paraphrases)
    summary["generated_corrections"] = len(correction_rows)
    summary["preference_pairs"] = len(preference_pairs)
    return all_rows, preference_pairs, summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a robust, balanced trajectory corpus")
    parser.add_argument(
        "--inputs",
        nargs="+",
        default=["data/trajectories_verified.jsonl", "data/trajectories_augmented.jsonl"],
        help="Input JSONL trajectory files relative to project root",
    )
    parser.add_argument(
        "--output",
        default="data/trajectories_robust.jsonl",
        help="Output JSONL path relative to project root",
    )
    parser.add_argument(
        "--output-preferences",
        default="data/trajectory_preferences.jsonl",
        help="Optional preference-pair JSONL path relative to project root",
    )
    parser.add_argument(
        "--output-summary",
        default="results/trajectory_corpus_summary.json",
        help="Summary JSON path relative to project root",
    )
    parser.add_argument(
        "--include-splits",
        nargs="+",
        default=["train", "dev"],
        choices=["train", "dev", "test"],
        help="Splits to include in the final corpus",
    )
    parser.add_argument(
        "--augment-splits",
        nargs="+",
        default=["train"],
        choices=["train", "dev", "test"],
        help="Splits eligible for paraphrase and correction augmentation",
    )
    parser.add_argument(
        "--min-per-registry-tool",
        type=int,
        default=6,
        help="Minimum positive trajectories per (registry, tool)",
    )
    parser.add_argument(
        "--target-total",
        type=int,
        default=1200,
        help="Target total positives before correction-turn augmentation",
    )
    parser.add_argument(
        "--correction-ratio",
        type=float,
        default=0.35,
        help="Fraction of positive train examples to convert into correction-turn trajectories",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)

    rows, preferences, summary = build_dataset(args)

    output_path = PROJECT_ROOT / args.output
    save_jsonl(output_path, rows)

    if args.output_preferences:
        save_jsonl(PROJECT_ROOT / args.output_preferences, preferences)

    summary_path = PROJECT_ROOT / args.output_summary
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as file_handle:
        json.dump(summary, file_handle, indent=2)

    print("=" * 60)
    print("ROBUST TRAJECTORY CORPUS COMPLETE")
    print(f"  Total rows            : {summary['total']}")
    print(f"  Positive min per pair : {summary['min_positive_per_registry_tool']}")
    print(f"  Positive max per pair : {summary['max_positive_per_registry_tool']}")
    print(f"  Splits                : {summary['splits']}")
    print(f"  Kinds                 : {summary['trajectory_kinds']}")
    print(f"  Preference pairs      : {summary['preference_pairs']}")
    print(f"  Saved corpus          : {output_path}")
    print(f"  Saved summary         : {summary_path}")
    if args.output_preferences:
        print(f"  Saved preferences     : {PROJECT_ROOT / args.output_preferences}")
    print("=" * 60)


if __name__ == "__main__":
    main()