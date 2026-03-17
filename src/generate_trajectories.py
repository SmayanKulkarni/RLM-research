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

# ── REPL Level Definitions ─────────────────────────────────────────────────────
#
# Level 1 — constrained: only registry API methods, no Python imports
# Level 2 — simple Python: registry API + re, json modules allowed
#
# We generate trajectories for BOTH levels so the fine-tuned model learns
# to operate at both levels. The system prompt and allowed code differs.

LEVEL_CONFIGS = {
    1: {
        "allowed_imports": [],
        "prompt_note": (
            "LEVEL: Constrained REPL. Use ONLY the registry API methods — "
            "no Python builtins beyond print(). No imports."
        ),
        "code_style": "Only call tools_registry.* methods. No loops or Python logic.",
    },
    2: {
        "allowed_imports": ["re", "json"],
        "prompt_note": (
            "LEVEL: Simple Python REPL. You may use basic Python (loops, "
            "list comprehensions, re, json) in addition to registry API methods."
        ),
        "code_style": (
            "You may use re, json, and Python control flow alongside "
            "tools_registry.* calls."
        ),
    },
}

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


DISCOVERY_QUERY_GENERATION_PROMPT = """You are generating "discovery" queries for testing an AI that navigates an MCP tool registry.

A discovery query asks WHAT TOOLS ARE AVAILABLE for a category or use-case — it does NOT ask to actually execute a tool.

Category: {category}
Example tools in this category: {tool_names}

Generate {n} diverse discovery queries a user might ask. Examples:
- "What tools do you have for file management?"
- "Show me all the communication-related tools available."
- "Can I search the web? What web tools are there?"

Requirements:
- Each query asks about tool availability, not about executing a specific action
- Vary phrasing (questions, commands, casual language)
- Do NOT mention specific tool names — user doesn't know them
- Output ONLY a JSON array of strings

Example output format:
["Query 1 here", "Query 2 here"]"""


TRAJECTORY_GENERATION_PROMPT = """You are an expert AI demonstrating REPL-based MCP tool selection. Generate a realistic step-by-step trajectory.

== REPL ENVIRONMENT ==
Variable `tools_registry` is pre-loaded. Available methods:
- tools_registry.search("keyword") → returns list of {{name, description}} dicts
- tools_registry.get_schema("tool_name") → returns full JSON schema with params
- tools_registry.list_names() → returns list of all tool name strings
- tools_registry.filter_by_category("cat") → returns list of {{name, description}} dicts
- tools_registry.count() → returns integer

{level_note}

== TASK ==
User query: {query}

The correct tool to select is: **{correct_tool}**
Its schema: {correct_schema}

== GENERATE THE TRAJECTORY ==
Write 2-4 steps of THOUGHT + CODE that naturally discover and select this tool.
End with FINAL({{"tool": "{correct_tool}", "params": {{...from query...}}}})

{code_style_note}

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
- Use ONLY the allowed code described above
- Extract parameter values FROM THE USER QUERY
- Always end with FINAL(...) on its own line
- Keep thoughts concise (1-2 sentences)
- Make the search keywords natural (what would you search for given the query?)
- Do NOT generate [REPL OUTPUT] sections — those will be filled in automatically"""


DISCOVERY_TRAJECTORY_PROMPT = """You are demonstrating REPL-based MCP tool discovery. Generate a trajectory for a user asking WHAT TOOLS ARE AVAILABLE.

== REPL ENVIRONMENT ==
Variable `tools_registry` is pre-loaded. Available methods:
- tools_registry.search("keyword") → returns list of {{name, description}} dicts
- tools_registry.list_names() → returns list of all tool name strings
- tools_registry.filter_by_category("cat") → returns list of {{name, description}} dicts
- tools_registry.count() → returns integer

== TASK ==
User query: {query}

Category being explored: {category}
Tools in this category: {tool_names_and_descs}

== GENERATE THE TRAJECTORY ==
Write 1-2 steps of THOUGHT + CODE to list or search for tools in this category.
End with FINAL(plain text summary of the available tools).

FORMAT (use <code> XML tags):

THOUGHT: <reasoning about what to search/list>
<code>
# search or list tools
print(...)
</code>

THOUGHT: <summarize what was found>
FINAL(Here are the available tools for {category}: ...)

Rules:
- Use search() or filter_by_category() to find tools
- FINAL should be a plain-text summary, NOT a JSON object
- Keep it concise — just list the tool names and one-line descriptions
- Do NOT generate [REPL OUTPUT] sections"""


# ── Query Generation ──────────────────────────────────────────────────────────

def generate_discovery_queries_for_category(
    client: Groq,
    category: str,
    tool_names: list[str],
    n: int = 3,
) -> list[str]:
    """Generate discovery queries for a tool category."""
    prompt = DISCOVERY_QUERY_GENERATION_PROMPT.format(
        category=category,
        tool_names=", ".join(tool_names[:6]),
        n=n,
    )
    try:
        resp = client.chat.completions.create(
            model=QUERY_GEN_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.9,
            max_tokens=400,
        )
        text = resp.choices[0].message.content.strip()
        match = re.search(r'\[.*\]', text, re.DOTALL)
        if match:
            queries = json.loads(match.group(0))
            return [q for q in queries if isinstance(q, str) and q.strip()]
    except Exception as e:
        print(f"    [discovery query gen error for {category}]: {e}")
    return []


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
    level: int = 2,
) -> str | None:
    """Call Groq to get the raw trajectory text for the given REPL level."""
    schema = registry.get_schema(correct_tool)
    level_cfg = LEVEL_CONFIGS.get(level, LEVEL_CONFIGS[2])

    prompt = TRAJECTORY_GENERATION_PROMPT.format(
        query=query,
        correct_tool=correct_tool,
        correct_schema=json.dumps(schema, indent=2) if schema else "{}",
        level_note=level_cfg["prompt_note"],
        code_style_note=level_cfg["code_style"],
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


def generate_discovery_trajectory(
    client: Groq,
    query: str,
    category: str,
    tools_in_category: list[dict],
) -> str | None:
    """Call Groq to generate a discovery-mode trajectory."""
    names_and_descs = "\n".join(
        f"- {t['name']}: {t.get('description', '')}"
        for t in tools_in_category[:8]
    )
    prompt = DISCOVERY_TRAJECTORY_PROMPT.format(
        query=query,
        category=category,
        tool_names_and_descs=names_and_descs,
    )
    try:
        resp = client.chat.completions.create(
            model=GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.6,
            max_tokens=800,
        )
        return resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"    [Groq discovery error]: {e}")
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

    # Normalise levels list
    levels: list[int] = args.levels
    discovery_fraction: float = args.discovery_fraction

    for reg_name in args.registries:
        reg_path = PROJECT_ROOT / "test_data" / "tool_registries" / f"{reg_name}.json"
        if not reg_path.exists():
            print(f"[SKIP] Registry not found: {reg_path}")
            continue

        registry = MCPToolRegistry.from_json_file(reg_path)
        num_tools = registry.count()
        tools = registry._tools

        print(f"\n{'='*60}")
        print(f"Registry: {reg_name} ({num_tools} tools)")
        print(f"{'='*60}")

        # ── Collect (query, tool, query_type) triples ──────────────────────
        # query_type: "selection" | "discovery"

        qt_triples: list[tuple[str, str, str]] = []

        # 1. Optional seed queries
        all_tool_names = set(registry.list_names())
        for q, t in existing_qt.items():
            if t in all_tool_names:
                qt_triples.append((q, t, "selection"))

        # 2. Generate new selection queries per tool
        if args.queries_per_tool > 0:
            print(f"\nGenerating {args.queries_per_tool} queries per tool ({len(tools)} tools)...")
            for i, tool in enumerate(tools):
                print(f"  [{i+1}/{len(tools)}] {tool['name']}", end="", flush=True)
                new_queries = generate_queries_for_tool(client, tool, n=args.queries_per_tool)
                for q in new_queries:
                    qt_triples.append((q, tool["name"], "selection"))
                print(f" → {len(new_queries)} queries")
                time.sleep(RATE_LIMIT_DELAY)

        # 3. Generate discovery queries per category
        if discovery_fraction > 0:
            n_selection = len(qt_triples)
            n_discovery_target = max(1, int(n_selection * discovery_fraction / (1 - discovery_fraction)))
            categories = _get_categories_from_registry(registry)
            queries_per_cat = max(2, n_discovery_target // max(len(categories), 1))

            print(f"\nGenerating discovery queries (~{n_discovery_target} total, "
                  f"{len(categories)} categories)...")
            for cat, cat_tools in categories.items():
                tool_names = [t["name"] for t in cat_tools]
                disc_queries = generate_discovery_queries_for_category(
                    client, cat, tool_names, n=queries_per_cat
                )
                for q in disc_queries:
                    # Store category name as the "tool" field for discovery queries
                    qt_triples.append((q, cat, "discovery"))
                print(f"  {cat}: {len(disc_queries)} discovery queries")
                time.sleep(RATE_LIMIT_DELAY)

        # Shuffle for diversity
        random.shuffle(qt_triples)

        print(f"\nTotal (query, tool, type) triples for {reg_name}: {len(qt_triples)}")
        sel_count = sum(1 for _, _, qt in qt_triples if qt == "selection")
        disc_count = sum(1 for _, _, qt in qt_triples if qt == "discovery")
        print(f"  Selection: {sel_count} | Discovery: {disc_count}")
        print(f"  Levels to generate: {levels}")
        print(f"Generating trajectories...\n")

        for idx, (query, tool_or_cat, query_type) in enumerate(qt_triples):
            # ── Pick REPL level for this trajectory ───────────────────────
            level = random.choice(levels)

            repl_config = REPLConfig.for_level(level)
            system_prompt = load_system_prompt(level=level, num_tools=num_tools)
            repl = REPLEngine(
                config=repl_config,
                initial_namespace={"tools_registry": registry},
            )

            print(
                f"  [{idx+1}/{len(qt_triples)}] L{level} {query_type} "
                f"tool={tool_or_cat} | q={query[:50]}...",
                end="",
            )
            sys.stdout.flush()
            total_attempted += 1

            # ── Discovery trajectory ───────────────────────────────────────
            if query_type == "discovery":
                cat_tools = [t for t in tools
                             if tool_or_cat.lower() in str(t.get("annotations", {})).lower()
                             or tool_or_cat.lower() in t.get("description", "").lower()]
                if not cat_tools:
                    print(" [NO TOOLS IN CAT]")
                    total_failed += 1
                    continue

                raw = generate_discovery_trajectory(client, query, tool_or_cat, cat_tools)
                if raw is None:
                    print(" [GROQ FAIL]")
                    total_failed += 1
                    time.sleep(RATE_LIMIT_DELAY)
                    continue

                result = parse_and_ground(raw, query, registry, repl, system_prompt)
                if result is None:
                    print(" [PARSE FAIL]")
                    total_failed += 1
                    time.sleep(RATE_LIMIT_DELAY)
                    continue

                traj = {
                    "registry": reg_name,
                    "level": level,
                    "query_type": "discovery",
                    "query": query,
                    "correct_tool": None,
                    "predicted_tool": None,
                    "messages": result["messages"],
                    "final_answer": result["final_answer"],
                    "turns": result["turns"],
                }

            # ── Selection trajectory ───────────────────────────────────────
            else:
                correct_tool = tool_or_cat

                raw = generate_raw_trajectory(client, query, correct_tool, registry, level=level)
                if raw is None:
                    print(" [GROQ FAIL]")
                    total_failed += 1
                    time.sleep(RATE_LIMIT_DELAY)
                    continue

                result = parse_and_ground(raw, query, registry, repl, system_prompt)
                if result is None:
                    print(" [PARSE FAIL - no FINAL]")
                    total_failed += 1
                    time.sleep(RATE_LIMIT_DELAY)
                    continue

                traj = {
                    "registry": reg_name,
                    "level": level,
                    "query_type": "selection",
                    "query": query,
                    "correct_tool": correct_tool,
                    "predicted_tool": result["final_answer"].get("tool"),
                    "messages": result["messages"],
                    "final_answer": result["final_answer"],
                    "turns": result["turns"],
                }

            all_trajectories.append(traj)
            total_ok += 1
            print(f" [OK turns={result['turns']}]")

            # Incremental save every 10 trajectories
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


def _get_categories_from_registry(registry: MCPToolRegistry) -> dict[str, list[dict]]:
    """Group tools by their category annotation."""
    categories: dict[str, list[dict]] = {}
    for tool in registry._tools:
        cat = tool.get("annotations", {}).get("category", "general")
        categories.setdefault(cat, []).append(tool)
    return categories


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
    parser.add_argument(
        "--levels", type=int, nargs="+", default=[1, 2],
        choices=[1, 2],
        help=(
            "REPL levels to generate trajectories for. Each trajectory is randomly "
            "assigned one level. Default: [1, 2] (50%% L1, 50%% L2)."
        ),
    )
    parser.add_argument(
        "--discovery-fraction", type=float, default=0.15,
        help=(
            "Fraction of trajectories that should be discovery queries "
            "(0.0 = none, 0.15 = 15%%). Default: 0.15"
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
