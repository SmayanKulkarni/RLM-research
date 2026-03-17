"""
build_splits.py — Stratified Train/Dev/Test Split Builder.

Combines all trajectory sources (TOUCAN, in-house REPL, augmented) into
clean, stratified splits for fine-tuning and evaluation.

Split design:
    Train : 80%  (used only for fine-tuning)
    Dev   : 10%  (used for hyperparameter selection — never for final reporting)
    Test  : 10%  (sealed — used only for final paper results)

Stratification:
    - Registry size bucket: small (≤15), medium (16–30), large (31–100)
    - Source: toucan / inhouse / augmented
    - Query type: selection / discovery (if present in metadata)

Leakage protection:
    - Queries appearing in test_data/ground_truth/ are always placed in test
    - Query-level deduplication across all splits (same query → same split)

Usage:
    conda run -n astro python -m src.build_splits \\
        --inputs data/toucan_repl.jsonl \\
        --inputs data/trajectories_verified.jsonl \\
        --output-dir data/splits \\
        --dev-ratio 0.10 \\
        --test-ratio 0.10 \\
        --seed 42
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterator

PROJECT_ROOT = Path(__file__).parent.parent


# ── Helpers ────────────────────────────────────────────────────────────────────

def _registry_bucket(traj: dict) -> str:
    """Bin registry size into small/medium/large for stratification."""
    size = traj.get("registry_size", 0)
    if size == 0:
        # Infer from registry name
        reg = traj.get("registry", "")
        if "small" in reg or reg.endswith("_10"):
            size = 10
        elif "medium" in reg or reg.endswith("_25"):
            size = 25
        elif "large" in reg or reg.endswith("_50"):
            size = 50
        else:
            size = 100
    if size <= 15:
        return "small"
    elif size <= 30:
        return "medium"
    else:
        return "large"


def _source_label(traj: dict) -> str:
    """Identify trajectory source."""
    src = traj.get("source", "")
    if src == "toucan":
        return "toucan"
    registry = traj.get("registry", "")
    if registry.startswith("toucan"):
        return "toucan"
    if traj.get("query_type") == "discovery":
        return "inhouse_discovery"
    return "inhouse"


def _stratum_key(traj: dict) -> str:
    return f"{_registry_bucket(traj)}_{_source_label(traj)}"


def load_sealed_test_queries() -> set[str]:
    """
    Load queries that must be placed in the test split (our benchmark queries).
    These are in test_data/ground_truth/ — they must never appear in train or dev.
    """
    sealed: set[str] = set()
    candidates = [
        PROJECT_ROOT / "test_data" / "ground_truth" / "expected_selections.json",
        PROJECT_ROOT / "test_data" / "ground_truth" / "test_expected_selections.json",
    ]
    for path in candidates:
        if not path.exists():
            continue
        with open(path) as f:
            data = json.load(f)
        for row in data:
            q = row.get("query", "").strip()
            if q:
                sealed.add(q)
    return sealed


def stream_jsonl(path: Path) -> Iterator[dict]:
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


# ── Main Split Logic ───────────────────────────────────────────────────────────

def run(args):
    output_dir = PROJECT_ROOT / args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load sealed test queries (leakage guard)
    sealed_test_queries = load_sealed_test_queries()
    print(f"Sealed test queries (benchmark): {len(sealed_test_queries)}")

    # Load all trajectories from all inputs
    all_trajs: list[dict] = []
    for input_path_str in args.inputs:
        input_path = PROJECT_ROOT / input_path_str
        if not input_path.exists():
            print(f"WARNING: Input not found, skipping: {input_path}")
            continue
        count_before = len(all_trajs)
        for traj in stream_jsonl(input_path):
            all_trajs.append(traj)
        added = len(all_trajs) - count_before
        print(f"Loaded {added:,} trajectories from {input_path.name}")

    if not all_trajs:
        print("ERROR: No trajectories loaded.")
        sys.exit(1)

    print(f"\nTotal loaded: {len(all_trajs):,}")

    # ── Step 1: Deduplicate by query (keep last seen per query) ──────────
    # This prevents the same query appearing in multiple splits.
    # We build a query→index map; later splits use this for assignment.
    query_to_indices: dict[str, list[int]] = defaultdict(list)
    for i, t in enumerate(all_trajs):
        q = t.get("query", "").strip()
        query_to_indices[q].append(i)

    # For queries with multiple trajectories, keep all (they may differ in level/registry)
    # but assign ALL trajectories with the same query to the same split.

    # ── Step 2: Group trajectories by stratum ────────────────────────────
    # Stratify by stratum_key, then assign queries to splits proportionally.
    stratum_queries: dict[str, list[str]] = defaultdict(list)
    seen_queries: set[str] = set()

    for traj in all_trajs:
        q = traj.get("query", "").strip()
        if q in seen_queries:
            continue
        seen_queries.add(q)
        key = _stratum_key(traj)
        stratum_queries[key].append(q)

    print(f"\nUnique queries: {len(seen_queries):,}")
    print(f"Strata: {dict((k, len(v)) for k, v in sorted(stratum_queries.items()))}")

    # ── Step 3: Assign queries to splits ─────────────────────────────────
    rng = random.Random(args.seed)
    query_split: dict[str, str] = {}

    # Force sealed queries into test
    for q in sealed_test_queries:
        if q in seen_queries:
            query_split[q] = "test"

    # Distribute remaining queries proportionally within each stratum
    for key, queries in stratum_queries.items():
        remaining = [q for q in queries if q not in query_split]
        rng.shuffle(remaining)

        n = len(remaining)
        n_test = max(1, int(n * args.test_ratio)) if n >= 3 else 0
        n_dev = max(1, int(n * args.dev_ratio)) if n >= 3 else 0
        n_train = n - n_test - n_dev

        for q in remaining[:n_test]:
            query_split[q] = "test"
        for q in remaining[n_test:n_test + n_dev]:
            query_split[q] = "dev"
        for q in remaining[n_test + n_dev:]:
            query_split[q] = "train"

    # ── Step 4: Assign trajectories based on their query's split ─────────
    train, dev, test = [], [], []

    for traj in all_trajs:
        q = traj.get("query", "").strip()
        split = query_split.get(q, "train")  # default to train if unseen query
        if split == "test":
            test.append(traj)
        elif split == "dev":
            dev.append(traj)
        else:
            train.append(traj)

    # Shuffle each split
    rng.shuffle(train)
    rng.shuffle(dev)
    rng.shuffle(test)

    # ── Step 5: Write outputs ─────────────────────────────────────────────
    splits = {"train": train, "dev": dev, "test": test}
    for split_name, trajs in splits.items():
        out_path = output_dir / f"{split_name}.jsonl"
        with open(out_path, "w") as f:
            for t in trajs:
                f.write(json.dumps(t) + "\n")
        print(f"  {split_name:6s}: {len(trajs):6,} trajectories → {out_path}")

    # ── Step 6: Write split manifest (query → split assignment) ──────────
    manifest_path = output_dir / "query_split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(query_split, f, indent=2)

    # ── Step 7: Diagnostic report ─────────────────────────────────────────
    total = len(all_trajs)
    print(f"\n{'='*60}")
    print(f"SPLIT SUMMARY")
    print(f"  Total trajectories  : {total:,}")
    print(f"  Train               : {len(train):,} ({len(train)/total*100:.1f}%)")
    print(f"  Dev                 : {len(dev):,} ({len(dev)/total*100:.1f}%)")
    print(f"  Test                : {len(test):,} ({len(test)/total*100:.1f}%)")
    print(f"  Sealed test queries : {sum(1 for q, s in query_split.items() if s == 'test' and q in sealed_test_queries)}")

    # Stratum breakdown per split
    print(f"\n  Stratum breakdown (train/dev/test):")
    all_strata = sorted(set(_stratum_key(t) for t in all_trajs))
    for stratum in all_strata:
        n_tr = sum(1 for t in train if _stratum_key(t) == stratum)
        n_dv = sum(1 for t in dev if _stratum_key(t) == stratum)
        n_te = sum(1 for t in test if _stratum_key(t) == stratum)
        print(f"    {stratum:30s}: {n_tr:5d} / {n_dv:4d} / {n_te:4d}")

    print(f"\n  Query manifest saved to: {manifest_path}")
    print(f"{'='*60}")

    # Warnings
    if len(dev) < 500:
        print(f"\nWARNING: Dev set has only {len(dev):,} examples. Need 500+ for reliable CIs.")
        print("  Add more --inputs (run generate_trajectories.py for more in-house data).")
    if len(test) < 500:
        print(f"WARNING: Test set has only {len(test):,} examples. Need 500+ for reliable CIs.")
    else:
        print(f"\n✓  Splits are ready. Use data/splits/train.jsonl for fine-tuning.")
        print(f"   NEVER use data/splits/test.jsonl during training or dev selection.")


def main():
    parser = argparse.ArgumentParser(
        description="Build stratified train/dev/test splits from trajectory files"
    )
    parser.add_argument(
        "--inputs", type=str, action="append", required=True,
        metavar="PATH",
        help="Input JSONL file (repeat for multiple files)",
    )
    parser.add_argument(
        "--output-dir", type=str, default="data/splits",
        help="Output directory for split files (default: data/splits)",
    )
    parser.add_argument(
        "--dev-ratio", type=float, default=0.10,
        help="Fraction to allocate to dev (default: 0.10)",
    )
    parser.add_argument(
        "--test-ratio", type=float, default=0.10,
        help="Fraction to allocate to test (default: 0.10)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
