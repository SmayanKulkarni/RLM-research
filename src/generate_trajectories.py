"""
generate_trajectories.py — Synthetic REPL-MCP Training Trajectory Generator.

Uses Groq API (llama-3.3-70b-versatile, free tier) to generate expert REPL
interaction trajectories for MCP tool selection. Code blocks are executed
against the REAL REPLEngine so all outputs are ground-truth (not hallucinated).

Pipeline per trajectory:
  1. Pick a (query, correct_tool, registry) triple
  2. Ask frontier model to write code steps leading to correct_tool
  3. Execute each <code> block against our real REPL → replace fake outputs
  4. Assemble multi-turn conversation for SFT

Usage:
    conda run -n astro python -m src.generate_trajectories \\
        --registries small_10 medium_25 large_50 \\
        --queries-per-tool 8 \\
        --output data/trajectories_raw.jsonl

Requires:
    GROQ_API_KEY environment variable (free at console.groq.com)
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
import time
from pathlib import Path
from typing import Any

sys.path.insert(0, str(Path(__file__).parent.parent))

# Load .env from project root (GROQ_API_KEY etc.)
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from groq import Groq

from src.tool_registry import MCPToolRegistry
from src.repl_engine import REPLEngine, REPLConfig

PROJECT_ROOT = Path(__file__).parent.parent

# ── Constants ─────────────────────────────────────────────────────────────────

GROQ_MODEL = "llama-3.3-70b-versatile"
QUERY_GEN_MODEL = "llama-3.3-70b-versatile"
RATE_LIMIT_DELAY = 2.0   # seconds between Groq calls (free tier safety)

# ── Prompts ───────────────────────────────────────────────────────────────────

QUERY_GENERATION_PROMPT = """You are generating realistic user queries for testing an AI assistant that selects MCP tools.

Generate {n} diverse, realistic user queries that would require using the tool: **{tool_name}**

Tool description: {tool_description}
Tool parameters: {params_summary}

Requirements:
- Each query should feel like something a real user would type
- Vary the phrasing, formality, and context
- Include the parameter values naturally in the query
- Do NOT mention the tool name directly — query from the user's perspective
- Output ONLY a JSON array of strings, no other text

Example output format:
["Query 1 here", "Query 2 here", "Query 3 here"]"""


TRAJECTORY_GENERATION_PROMPT = """You are an expert AI demonstrating REPL-based MCP tool selection. Generate a realistic step-by-step trajectory.

== REPL ENVIRONMENT ==
Variable `tools_registry` is pre-loaded. Available methods:
- tools_registry.search("keyword") → returns list of {{name, description}} dicts
- tools_registry.get_schema("tool_name") → returns full JSON schema with params
- tools_registry.list_names() → returns list of all tool name strings
- tools_registry.filter_by_category("cat") → returns list of {{name, description}} dicts
- tools_registry.count() → returns integer

== TASK ==
User query: {query}

The correct tool to select is: **{correct_tool}**
Its schema: {correct_schema}

== GENERATE THE TRAJECTORY ==
Write 2-4 steps of THOUGHT + CODE that naturally discover and select this tool.
End with FINAL({{"tool": "{correct_tool}", "params": {{...from query...}}}})

FORMAT (use <code> XML tags, not markdown backticks):

THOUGHT: <reasoning>
<code>
# python code here
print(...)
</code>

THOUGHT: <reasoning>
<code>
# more code
print(...)
</code>

THOUGHT: <final reasoning>
FINAL({{"tool": "<tool_name>", "params": {{<params extracted from query>}}}})

Rules:
- Use ONLY the registry API shown above
- Extract parameter values FROM THE USER QUERY
- Always end with FINAL(...) on its own line
- Keep thoughts concise (1-2 sentences)
- Make the search keywords natural (what would you search for given the query?)
- Do NOT generate [REPL OUTPUT] sections — those will be filled in automatically"""


# ── Query Generation ──────────────────────────────────────────────────────────

def generate_queries_for_tool(
    client: Groq,
    tool: dict,
    n: int = 8,
) -> list[str]:
    """Ask Groq to generate n diverse user queries for a given tool."""
    params = tool.get("inputSchema", {}).get("properties", {})
    params_summary = ", ".join(f"{k} ({v.get('type','?')})" for k, v in params.items())

    prompt = QUERY_GENERATION_PROMPT.format(
        n=n,
        tool_name=tool["name"],
        tool_description=tool.get("description", ""),
        params_summary=params_summary or "no required params",
    )

    try:
        resp = client.chat.completions.create(
            model=QUERY_GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=600,
        )
        text = resp.choices[0].message.content.strip()
        # Extract JSON array
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            queries = json.loads(match.group(0))
            return [q for q in queries if isinstance(q, str) and q.strip()]
    except Exception as e:
        print(f"    [query gen error for {tool['name']}]: {e}")

    return []


# ── Trajectory Generation ─────────────────────────────────────────────────────

def generate_raw_trajectory(
    client: Groq,
    query: str,
    correct_tool: str,
    registry: MCPToolRegistry,
) -> str | None:
    """Call Groq to get the raw trajectory text."""
    schema = registry.get_schema(correct_tool)

    prompt = TRAJECTORY_GENERATION_PROMPT.format(
        query=query,
        correct_tool=correct_tool,
        correct_schema=json.dumps(schema, indent=2) if schema else "{}",
    )

    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=1200,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [Groq error]: {e}")
        return None


def parse_and_ground(
    raw: str,
    query: str,
    registry: MCPToolRegistry,
    repl: REPLEngine,
    system_prompt: str,
) -> dict | None:
    """
    Parse Groq's trajectory, execute code blocks against our REAL REPL,
    and assemble a grounded multi-turn conversation.

    Returns a dict with 'messages', 'final_answer', 'turns' or None on failure.
    """
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    # Split on <code>...</code> blocks
    parts = re.split(r'(<code>.*?</code>)', raw, flags=re.DOTALL)

    current_assistant = ""
    turn_count = 0
    found_final = False
    final_answer = None

    for part in parts:
        if part.startswith('<code>') and part.endswith('</code>'):
            code = part[6:-7].strip()

            # Build assistant message: accumulated thought + this code block
            assistant_msg = current_assistant.strip() + "\n<code>\n" + code + "\n</code>"
            messages.append({"role": "assistant", "content": assistant_msg.strip()})
            current_assistant = ""

            # Execute against the REAL REPL with the registry loaded
            real_output = repl.execute(code).output
            turn_count += 1

            messages.append({
                "role": "user",
                "content": f"[REPL OUTPUT]:\n{real_output}"
            })
        else:
            # Check if this segment contains FINAL(...)
            final_match = re.search(
                r'FINAL\((\{.*?\})\)',
                part,
                re.DOTALL,
            )
            if final_match:
                final_idx = part.rfind('FINAL(')
                pre = part[:final_idx].strip()
                final_str = part[final_idx:final_match.end()].strip()

                full_msg = "\n".join(filter(None, [current_assistant.strip(), pre, final_str]))
                messages.append({"role": "assistant", "content": full_msg.strip()})

                try:
                    final_answer = json.loads(final_match.group(1))
                except json.JSONDecodeError:
                    # Fall back: extract just the tool name
                    tool_m = re.search(r'"tool"\s*:\s*"([^"]+)"', final_match.group(1))
                    if tool_m:
                        final_answer = {"tool": tool_m.group(1)}

                found_final = True
                current_assistant = ""
            else:
                current_assistant += part

    # Catch FINAL in trailing text
    if not found_final and current_assistant.strip():
        final_match = re.search(r'FINAL\((\{.*?\})\)', current_assistant, re.DOTALL)
        if final_match:
            messages.append({"role": "assistant", "content": current_assistant.strip()})
            try:
                final_answer = json.loads(final_match.group(1))
            except json.JSONDecodeError:
                tool_m = re.search(r'"tool"\s*:\s*"([^"]+)"', final_match.group(1))
                if tool_m:
                    final_answer = {"tool": tool_m.group(1)}
            found_final = True

    if not found_final or final_answer is None:
        return None

    return {
        "messages": messages,
        "final_answer": final_answer,
        "turns": turn_count,
        "raw": raw,
    }


def load_system_prompt(level: int = 2, num_tools: int = 25) -> str:
    path = PROJECT_ROOT / "prompts" / f"level{level}.txt"
    return path.read_text().replace("{num_tools}", str(num_tools))


# ── Ground Truth Loading ───────────────────────────────────────────────────────

def load_ground_truth() -> dict[str, str]:
    """Load expected tool selections from ground truth file."""
    gt_path = PROJECT_ROOT / "test_data" / "ground_truth" / "expected_selections.json"
    if gt_path.exists():
        with open(gt_path) as f:
            data = json.load(f)
        # Flatten: {query_text -> tool_name}
        result = {}
        for item in data if isinstance(data, list) else data.get("selections", []):
            result[item.get("query", "")] = item.get("expected_tool", "")
        return result
    return {}


def load_existing_queries(split: str = "all") -> list[dict]:
    """Load existing queries for a specific split."""
    if split == "all":
        q_path = PROJECT_ROOT / "test_data" / "queries" / "all_queries.json"
    else:
        q_path = PROJECT_ROOT / "test_data" / "queries" / f"{split}_queries.json"

    if q_path.exists():
        with open(q_path) as f:
            data = json.load(f)
        return data if isinstance(data, list) else data.get("queries", [])

    if split != "all":
        print(f"[WARN] Query split file not found: {q_path}")
    return []


# ── Main Generation Loop ───────────────────────────────────────────────────────

def run(args):
    api_key = args.api_key or os.environ.get("GROQ_API_KEY")
    if not api_key:
        print("ERROR: Set GROQ_API_KEY env var or pass --api-key")
        sys.exit(1)

    client = Groq(api_key=api_key)

    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    ground_truth = load_ground_truth()
    existing_queries = []
    if args.seed_query_split != "none":
        existing_queries = load_existing_queries(args.seed_query_split)
        print(
            f"Loaded {len(existing_queries)} seed queries from split='{args.seed_query_split}'."
        )

    # Build query→tool mapping from existing test data
    existing_qt = {}
    for item in existing_queries:
        q = item.get("query", "")
        t = item.get("expected_tool") or ground_truth.get(q)
        if q and t:
            existing_qt[q] = t

    all_trajectories = []
    total_attempted = 0
    total_ok = 0
    total_failed = 0

    for reg_name in args.registries:
        reg_path = PROJECT_ROOT / "test_data" / "tool_registries" / f"{reg_name}.json"
        if not reg_path.exists():
            print(f"[SKIP] Registry not found: {reg_path}")
            continue

        registry = MCPToolRegistry.from_json_file(reg_path)
        num_tools = registry.count()
        tools = registry._tools

        repl_config = REPLConfig(
            max_turns=10,
            max_output_chars=2000,
            allowed_imports=["re", "json"],
        )

        print(f"\n{'='*60}")
        print(f"Registry: {reg_name} ({num_tools} tools)")
        print(f"{'='*60}")

        # ── Collect (query, tool) pairs ────────────────────────────────────

        qt_pairs: list[tuple[str, str]] = []

        # 1. Optional seed queries that belong to this registry
        all_tool_names = set(registry.list_names())
        for q, t in existing_qt.items():
            if t in all_tool_names:
                qt_pairs.append((q, t))

        # 2. Generate new queries for each tool via Groq
        if args.queries_per_tool > 0:
            print(f"\nGenerating {args.queries_per_tool} queries per tool ({len(tools)} tools)...")
            for i, tool in enumerate(tools):
                print(f"  [{i+1}/{len(tools)}] {tool['name']}", end="", flush=True)
                new_queries = generate_queries_for_tool(client, tool, n=args.queries_per_tool)
                for q in new_queries:
                    qt_pairs.append((q, tool["name"]))
                print(f" → {len(new_queries)} queries")
                time.sleep(RATE_LIMIT_DELAY)

        # Shuffle for diversity
        random.shuffle(qt_pairs)

        print(f"\nTotal (query, tool) pairs for {reg_name}: {len(qt_pairs)}")
        print(f"Generating trajectories...\n")

        system_prompt = load_system_prompt(level=2, num_tools=num_tools)

        # Single REPL instance per registry — reset per trajectory
        repl = REPLEngine(
            config=repl_config,
            initial_namespace={"tools_registry": registry},
        )

        for idx, (query, correct_tool) in enumerate(qt_pairs):
            print(f"  [{idx+1}/{len(qt_pairs)}] tool={correct_tool} | q={query[:60]}...", end="")
            sys.stdout.flush()
            total_attempted += 1

            # Reset REPL state but keep registry loaded
            repl.reset(keep_initial={"tools_registry": registry})

            # Generate raw trajectory from Groq
            raw = generate_raw_trajectory(client, query, correct_tool, registry)
            if raw is None:
                print(" [GROQ FAIL]")
                total_failed += 1
                time.sleep(RATE_LIMIT_DELAY)
                continue

            # Parse and ground against our real REPL
            result = parse_and_ground(raw, query, registry, repl, system_prompt)

            if result is None:
                print(" [PARSE FAIL - no FINAL]")
                total_failed += 1
                time.sleep(RATE_LIMIT_DELAY)
                continue

            # Tag with metadata for verification
            traj = {
                "registry": reg_name,
                "query": query,
                "correct_tool": correct_tool,
                "predicted_tool": result["final_answer"].get("tool"),
                "messages": result["messages"],
                "final_answer": result["final_answer"],
                "turns": result["turns"],
            }

            all_trajectories.append(traj)
            total_ok += 1
            print(f" [OK, pred={traj['predicted_tool']}, turns={result['turns']}]")

            # Save incrementally (every 10 trajectories)
            if total_ok % 10 == 0:
                _save(all_trajectories, output_path)
                print(f"  >> Saved {total_ok} trajectories so far")

            time.sleep(RATE_LIMIT_DELAY)

    # Final save
    _save(all_trajectories, output_path)

    print(f"\n{'='*60}")
    print(f"GENERATION COMPLETE")
    print(f"  Attempted : {total_attempted}")
    print(f"  Succeeded : {total_ok}")
    print(f"  Failed    : {total_failed}")
    print(f"  Saved to  : {output_path}")
    print(f"{'='*60}")


def _save(trajectories: list[dict], path: Path):
    with open(path, "w") as f:
        for t in trajectories:
            f.write(json.dumps(t) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Generate REPL-MCP training trajectories via Groq")
    parser.add_argument(
        "--registries", nargs="+",
        default=["small_10", "medium_25", "large_50"],
        choices=["small_10", "medium_25", "large_50", "xlarge_100"],
        help="Tool registries to use",
    )
    parser.add_argument(
        "--queries-per-tool", type=int, default=8,
        help="Number of new queries to generate per tool via Groq (0=use existing only)",
    )
    parser.add_argument(
        "--output", type=str, default="data/trajectories_raw.jsonl",
        help="Output path for raw trajectories (relative to project root)",
    )
    parser.add_argument(
        "--api-key", type=str, default=None,
        help="Groq API key (or set GROQ_API_KEY env var)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--seed-query-split", type=str, default="none",
        choices=["none", "train", "dev", "test", "all"],
        help=(
            "Optional query split used to seed generation with existing queries. "
            "Default is 'none' to avoid train/eval leakage."
        ),
    )
    args = parser.parse_args()

    if args.seed_query_split in {"test", "all"}:
        print(
            f"[WARN] seed_query_split={args.seed_query_split} can cause train/eval leakage. "
            "Prefer --seed-query-split train for controlled experiments."
        )

    random.seed(args.seed)
    run(args)


if __name__ == "__main__":
    main()
