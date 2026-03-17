"""
prepare_toucan.py — TOUCAN Dataset Adapter for REPL-MCP Fine-Tuning.

Downloads the TOUCAN-1.5M dataset (Agent-Ark/Toucan-1.5M on HuggingFace),
filters to single-tool selection tasks, and converts each entry into our
REPL-based multi-turn conversation format.

Conversion pipeline per TOUCAN entry:
  1. Extract (query, correct_tool, params, available_tools)
  2. Convert tools from OpenAI/MCP JSON schema → our MCPToolRegistry format
  3. Build a deterministic 2-step REPL trajectory:
       DISCOVER: tools_registry.search(keyword)
       VERIFY:   tools_registry.get_schema(tool_name)
       DECIDE:   FINAL({"tool": ..., "params": {...}})
  4. Execute each code block against real REPLEngine (ground-truth outputs)
  5. Save in our standard JSONL format (compatible with verify_trajectories.py)

Why this works:
  TOUCAN's trajectories use direct tool calls; ours teach REPL-based
  registry navigation. By re-grounding TOUCAN's (query, tool, params) triples
  in our REPL scaffold, we get large-scale training data for the DISCOVER→
  VERIFY→DECIDE workflow without another LLM API call.

Usage:
    python -m src.prepare_toucan \\
        --output data/toucan_repl.jsonl \\
        --max-examples 20000 \\
        --min-registry-size 5 \\
        --max-registry-size 100

Requires:
    pip install datasets huggingface_hub

Reference:
    TOUCAN: arXiv:2510.01179 — 1.5M MCP trajectories from ~500 real MCPs
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Iterator

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tool_registry import MCPToolRegistry
from src.repl_engine import REPLEngine, REPLConfig

PROJECT_ROOT = Path(__file__).parent.parent

# ── System Prompt (Level 1 — constrained REPL, no Python imports needed) ──────

SYSTEM_PROMPT_TEMPLATE = """You are an AI assistant that selects MCP tools to answer user requests.

== YOUR ENVIRONMENT ==
You have access to a REPL environment with a `tools_registry` object pre-loaded with {num_tools} available MCP tools. You cannot see all tools at once — use the registry functions to find the right one.

Available functions in the REPL:
- tools_registry.list_names() → list of all tool names
- tools_registry.search(query) → find tools matching a keyword (returns name + description)
- tools_registry.get_schema(tool_name) → get full tool definition with parameters
- tools_registry.filter_by_category(category) → filter by category
- tools_registry.count() → total number of tools

== HOW TO INTERACT ==
1. Think about what the user needs
2. Write Python code inside <code></code> tags to explore available tools
3. STOP after closing </code> — you will receive the real output
4. Use the output to reason and write more code if needed
5. When you've found the right tool, provide FINAL({{"tool": "name", "params": {{...}}}})

== IMPORTANT RULES ==
- ALWAYS write code inside <code></code> tags
- STOP after </code> and wait for the real output — DO NOT generate fake output
- Never guess tool names — always search or list first
- Use FINAL({{"tool": "name", "params": {{...}}}}) when you have your answer
"""

# ── Tool Format Conversion ─────────────────────────────────────────────────────

def openai_tool_to_mcp(tool: dict) -> dict | None:
    """
    Convert an OpenAI-style tool definition to our MCP format.

    Handles both:
      - OpenAI: {"type": "function", "function": {"name": ..., "parameters": ...}}
      - Already-MCP: {"name": ..., "inputSchema": ...}
      - Raw function: {"name": ..., "description": ..., "parameters": ...}
    """
    if not isinstance(tool, dict):
        return None

    # Already in our MCP format
    if "inputSchema" in tool and "name" in tool:
        return tool

    # OpenAI function-calling format
    if tool.get("type") == "function" and "function" in tool:
        fn = tool["function"]
        return {
            "name": fn.get("name", ""),
            "description": fn.get("description", ""),
            "inputSchema": fn.get("parameters", {"type": "object", "properties": {}}),
            "annotations": {"category": _infer_category(fn.get("name", ""))},
        }

    # Raw function dict (name + parameters at top level)
    if "name" in tool and ("parameters" in tool or "inputSchema" in tool):
        schema = tool.get("inputSchema") or tool.get("parameters") or {}
        return {
            "name": tool["name"],
            "description": tool.get("description", ""),
            "inputSchema": schema,
            "annotations": {"category": _infer_category(tool.get("name", ""))},
        }

    return None


def _infer_category(tool_name: str) -> str:
    """Infer a category from tool name using simple keyword matching."""
    name = tool_name.lower()
    rules = [
        (["email", "mail", "smtp"], "communication"),
        (["slack", "discord", "chat", "message", "notify"], "communication"),
        (["calendar", "event", "schedule", "meeting"], "communication"),
        (["sms", "text", "twilio"], "communication"),
        (["file", "read", "write", "fs", "directory", "folder", "path"], "filesystem"),
        (["database", "sql", "query", "db", "postgres", "mysql", "mongo"], "data"),
        (["csv", "excel", "spreadsheet", "table"], "data"),
        (["http", "request", "api", "fetch", "get", "post", "web", "url"], "web"),
        (["search", "scrape", "crawl", "browse"], "web"),
        (["weather", "forecast", "temperature"], "weather"),
        (["git", "github", "commit", "repo", "branch", "pr"], "code"),
        (["code", "python", "js", "run", "exec", "shell", "bash"], "code"),
        (["image", "photo", "picture", "screenshot", "resize"], "media"),
        (["pdf", "doc", "convert", "compress", "zip"], "utility"),
        (["time", "date", "clock", "timezone"], "utility"),
        (["translate", "language", "locale"], "utility"),
        (["stock", "price", "finance", "currency", "exchange"], "finance"),
        (["ml", "model", "inference", "embed", "classify", "predict"], "ml"),
        (["task", "todo", "list", "create", "manage"], "productivity"),
        (["auth", "login", "token", "key", "password", "hash"], "security"),
    ]
    for keywords, category in rules:
        if any(kw in name for kw in keywords):
            return category
    return "general"


# ── Search Keyword Extraction ──────────────────────────────────────────────────

def extract_search_keyword(query: str, tool_name: str) -> str:
    """
    Extract a natural search keyword from the query and tool name.

    Prefers words that appear in both the query and the tool name/description,
    falling back to splitting the tool name into readable words.
    """
    # Convert tool_name snake_case to words: "send_slack_message" → ["slack", "message"]
    tool_words = re.split(r'[_\-\s]+', tool_name.lower())
    # Remove generic verbs that appear in almost every tool
    stop_words = {"get", "set", "create", "read", "write", "list", "make",
                  "run", "use", "do", "a", "the", "to", "from", "with"}
    meaningful_words = [w for w in tool_words if w and w not in stop_words and len(w) > 2]

    if not meaningful_words:
        meaningful_words = tool_words[:2]

    # Check which meaningful words appear in the query (case-insensitive)
    query_lower = query.lower()
    query_hits = [w for w in meaningful_words if w in query_lower]

    if query_hits:
        # Use the most distinctive match from the query
        return query_hits[0]

    # Fall back to the most distinctive word from the tool name
    return meaningful_words[0] if meaningful_words else tool_name.split("_")[0]


# ── REPL Trajectory Builder ────────────────────────────────────────────────────

# Template thoughts for each step — varied to avoid training on identical text
_SEARCH_THOUGHTS = [
    "I need to find the right tool for this request. Let me search the registry.",
    "Let me search for a relevant tool to handle this task.",
    "I'll start by searching the tool registry for something that matches this request.",
    "To find the right tool, I'll search for relevant keywords.",
    "Let me look for tools related to this request.",
]

_SCHEMA_THOUGHTS = [
    "Found a candidate. Let me check its full schema to confirm it fits and get the parameter names.",
    "This looks relevant. I'll verify the schema before selecting it.",
    "Let me get the full tool definition to confirm the required parameters.",
    "I need to check the schema to extract the correct parameter names.",
    "This tool looks right. Let me verify its parameters.",
]

_FINAL_THOUGHTS = [
    "The schema confirms this is the right tool. I can now extract the parameters from the request.",
    "This matches the request. I have all the parameter information I need.",
    "The tool and parameters are clear. I'll finalize my selection.",
]


def build_repl_trajectory(
    query: str,
    correct_tool: str,
    params: dict,
    registry: MCPToolRegistry,
    repl: REPLEngine,
    num_tools: int,
    seed: int = 0,
) -> dict | None:
    """
    Build a deterministic 2-step REPL trajectory for a single (query, tool, params) triple.

    Steps:
      1. search(keyword)     → shows tool in results (DISCOVER)
      2. get_schema(tool)    → shows full schema  (VERIFY)
      3. FINAL(...)          →                    (DECIDE)

    All code blocks are executed against the real REPLEngine so outputs
    are ground-truth, not hallucinated.

    Returns None if the tool isn't found in the registry or REPL execution fails.
    """
    rng = random.Random(seed)

    # Sanity: tool must exist in this registry
    if correct_tool not in registry.list_names():
        return None

    system_prompt = SYSTEM_PROMPT_TEMPLATE.format(num_tools=num_tools)

    # ── Step 1: DISCOVER ─────────────────────────────────────────────────
    search_kw = extract_search_keyword(query, correct_tool)
    search_code = f'print(tools_registry.search("{search_kw}"))'
    search_result = repl.execute(search_code)

    # If search returned empty, fall back to list_names
    if not search_result.output.strip() or search_result.output.strip() in ("[]", "(no output)"):
        # Fall back: list all names and look for the tool
        search_kw = correct_tool.replace("_", " ").split()[0]
        search_code = f'print(tools_registry.search("{search_kw}"))'
        repl.reset(keep_initial={"tools_registry": registry})
        search_result = repl.execute(search_code)

    # Last resort: list all names
    if not search_result.output.strip() or search_result.output.strip() in ("[]", "(no output)"):
        search_code = f'print(tools_registry.list_names())'
        repl.reset(keep_initial={"tools_registry": registry})
        search_result = repl.execute(search_code)

    search_thought = rng.choice(_SEARCH_THOUGHTS)
    assistant_step1 = f"THOUGHT: {search_thought}\n<code>\n{search_code}\n</code>"
    repl_output1 = search_result.output.strip() or "(no output)"

    # ── Step 2: VERIFY ───────────────────────────────────────────────────
    schema_code = f'print(tools_registry.get_schema("{correct_tool}"))'
    schema_result = repl.execute(schema_code)

    if not schema_result.success or not schema_result.output.strip():
        return None  # Registry doesn't have this tool's schema

    schema_thought = rng.choice(_SCHEMA_THOUGHTS)
    assistant_step2 = f"THOUGHT: {schema_thought}\n<code>\n{schema_code}\n</code>"
    repl_output2 = schema_result.output.strip()

    # ── Step 3: DECIDE ───────────────────────────────────────────────────
    final_thought = rng.choice(_FINAL_THOUGHTS)
    final_json = json.dumps({"tool": correct_tool, "params": params})
    assistant_final = f"THOUGHT: {final_thought}\nFINAL({final_json})"

    # ── Assemble conversation ────────────────────────────────────────────
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
        {"role": "assistant", "content": assistant_step1},
        {"role": "user", "content": f"[REPL OUTPUT]:\n{repl_output1}"},
        {"role": "assistant", "content": assistant_step2},
        {"role": "user", "content": f"[REPL OUTPUT]:\n{repl_output2}"},
        {"role": "assistant", "content": assistant_final},
    ]

    return {
        "messages": messages,
        "turns": 2,
        "repl_output_search": repl_output1,
        "repl_output_schema": repl_output2,
    }


# ── TOUCAN Entry Parsing ───────────────────────────────────────────────────────
#
# Real TOUCAN-1.5M schema (observed from SFT config):
#   uuid         : str
#   subset_name  : str   — "irrelevant" | "multi-turn" | "single-turn" | etc.
#   question     : str   — plain string, or JSON-encoded list for multi-turn
#   target_tools : str   — "ToolName" or "Server::fn" or "S1::fn1, S2::fn2" (comma = multi)
#   tools        : str   — JSON-encoded list of OpenAI-format tool objects
#   messages     : str   — JSON-encoded list with roles: user/assistant/tool_call/tool_response


def _parse_json_field(raw: Any) -> Any:
    """Parse a field that may already be the target type or a JSON string."""
    if isinstance(raw, str):
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    return raw


def _extract_params_from_tool_call_messages(messages: list, tool_name: str) -> dict:
    """
    Extract call arguments from TOUCAN's tool_call messages.

    TOUCAN uses {"role": "tool_call", "content": "{'name': ..., 'arguments': '...'}"}.
    The content is a Python dict repr (single-quoted), so we use ast.literal_eval.
    """
    import ast

    for msg in messages:
        if msg.get("role") != "tool_call":
            continue
        content = msg.get("content", "")
        if not isinstance(content, str):
            continue
        try:
            call_obj = ast.literal_eval(content)
            if not isinstance(call_obj, dict):
                continue
            # Match by tool name (exact or suffix)
            call_name = call_obj.get("name", "")
            if call_name != tool_name and not call_name.endswith(f"-{tool_name}"):
                continue
            raw_args = call_obj.get("arguments", {})
            if isinstance(raw_args, str):
                try:
                    raw_args = json.loads(raw_args)
                except (json.JSONDecodeError, ValueError):
                    raw_args = {}
            return raw_args if isinstance(raw_args, dict) else {}
        except (ValueError, SyntaxError):
            continue
    return {}


def _resolve_tool_name(target_raw: str, tools_mcp: list[dict]) -> str | None:
    """
    Match a TOUCAN target_tools string to the actual tool name in the registry.

    TOUCAN uses naming like "Server Name::function_name" in target_tools,
    but tool objects use slugs like "server-name-function_name".

    Strategy (in priority order):
      1. Exact match against tool name
      2. The part after "::" (function name) matches tool name exactly
      3. The function-name part appears as a suffix in the tool slug
    """
    tool_names = [t["name"] for t in tools_mcp]

    # 1. Exact match
    if target_raw in tool_names:
        return target_raw

    # 2. Extract function name from "Server::function_name" notation
    fn_part = target_raw.split("::")[-1].strip() if "::" in target_raw else target_raw.strip()

    # 3. Exact match on function part
    if fn_part in tool_names:
        return fn_part

    # 4. Slug suffix match: tool slug ends with "-fn_part" or "_fn_part"
    fn_slug = fn_part.lower().replace(" ", "_").replace("-", "_")
    for name in tool_names:
        name_slug = name.lower().replace("-", "_")
        if name_slug.endswith(f"_{fn_slug}") or name_slug.endswith(f"-{fn_slug}"):
            return name
        if name_slug == fn_slug:
            return name

    return None


def parse_toucan_entry(entry: dict) -> list[dict]:
    """
    Decompose a TOUCAN-1.5M entry into a list of single-tool (query, tool, params) triples.

    TOUCAN entries are almost entirely multi-tool (comma-separated target_tools with
    a JSON-array question field, one question per tool call). We decompose each entry
    into N individual single-tool examples — one per turn.

    Filters:
      - Skips "irrelevant" subset (no matching tool by design)
      - Skips any turn whose tool cannot be matched in the registry

    Returns a list of dicts, each with keys: query, correct_tool, params, tools.
    The same tools list is shared across all turns from the same entry (full registry).
    """
    # ── Filter: skip irrelevant subset ──────────────────────────────────
    if entry.get("subset_name") == "irrelevant":
        return []

    # ── Parse tools (JSON string → list) ────────────────────────────────
    raw_tools = _parse_json_field(entry.get("tools", "[]"))
    if not isinstance(raw_tools, list) or not raw_tools:
        return []

    tools_mcp = []
    for t in raw_tools:
        mcp = openai_tool_to_mcp(t)
        if mcp and mcp.get("name"):
            tools_mcp.append(mcp)
    if not tools_mcp:
        return []

    # ── Parse questions (may be single string or JSON-encoded list) ──────
    question_raw = entry.get("question", "")
    if isinstance(question_raw, str) and question_raw.startswith("["):
        parsed_q = _parse_json_field(question_raw)
        questions = parsed_q if isinstance(parsed_q, list) else [question_raw]
    elif isinstance(question_raw, list):
        questions = question_raw
    else:
        questions = [question_raw] if question_raw.strip() else []

    if not questions:
        return []

    # ── Parse target tools (comma-separated list, one per turn) ──────────
    target_tools_str = entry.get("target_tools", "").strip()
    if not target_tools_str:
        return []

    target_tools_raw = [t.strip() for t in target_tools_str.split(",") if t.strip()]

    # ── Parse messages to extract call params ────────────────────────────
    messages_raw = _parse_json_field(entry.get("messages", "[]"))
    messages = messages_raw if isinstance(messages_raw, list) else []

    # ── Zip questions + target_tools into single-tool turns ──────────────
    # If counts don't match exactly, pair what we can (zip stops at shorter).
    results = []
    for question, target_raw in zip(questions, target_tools_raw):
        query = question.strip() if isinstance(question, str) else ""
        if not query:
            continue

        correct_tool = _resolve_tool_name(target_raw, tools_mcp)
        if correct_tool is None:
            continue

        params = _extract_params_from_tool_call_messages(messages, correct_tool)

        results.append({
            "query": query,
            "correct_tool": correct_tool,
            "params": params,
            "tools": tools_mcp,
        })

    return results


# ── Dataset Streaming ──────────────────────────────────────────────────────────

def stream_toucan(
    dataset_name: str = "Agent-Ark/Toucan-1.5M",
    config: str = "SFT",
    split: str = "train",
    shuffle_seed: int = 42,
) -> Iterator[dict]:
    """
    Stream TOUCAN from HuggingFace in shuffled order.

    TOUCAN-1.5M has 4 configs: Kimi-K2, OSS, Qwen3, SFT.
    We default to 'SFT' — it is the standard supervised fine-tuning split
    designed for training, and has the most consistent schema.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: Install datasets: pip install datasets huggingface_hub")
        sys.exit(1)

    print(f"Loading {dataset_name} (config={config}, streaming)...")
    ds = load_dataset(
        dataset_name,
        config,
        split=split,
        streaming=True,
        trust_remote_code=True,
    )
    ds = ds.shuffle(seed=shuffle_seed, buffer_size=10_000)
    yield from ds


# ── Main Conversion Loop ───────────────────────────────────────────────────────

def run(args):
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    repl_config = REPLConfig.for_level(1)

    stats = {
        "seen": 0,
        "parse_fail": 0,
        "registry_size_filter": 0,
        "repl_build_fail": 0,
        "ok": 0,
    }

    out_file = open(output_path, "w")

    try:
        for raw_entry in stream_toucan(args.dataset, args.config, args.split, args.shuffle_seed):
            stats["seen"] += 1

            if stats["seen"] % 1000 == 0:
                pct = stats["ok"] / max(stats["seen"], 1) * 100
                print(
                    f"  [{stats['seen']:,} seen | {stats['ok']:,} ok ({pct:.1f}%)] "
                    f"parse_fail={stats['parse_fail']} "
                    f"size_filter={stats['registry_size_filter']} "
                    f"repl_fail={stats['repl_build_fail']}"
                )

            # ── Parse: returns list of single-tool turns ─────────────────
            turns = parse_toucan_entry(raw_entry)
            if not turns:
                stats["parse_fail"] += 1
                continue

            # All turns from the same entry share the same tools list.
            # Build registry + REPL once per entry, reset REPL per turn.
            tools = turns[0]["tools"]
            n = len(tools)

            # ── Registry size filter ──────────────────────────────────────
            if n < args.min_registry_size or n > args.max_registry_size:
                stats["registry_size_filter"] += 1
                continue

            registry = MCPToolRegistry(tools)
            repl = REPLEngine(
                config=repl_config,
                initial_namespace={"tools_registry": registry},
            )

            # ── Determine registry size bucket (once per entry) ───────────
            if n <= 15:
                reg_label = f"toucan_{n}"
            elif n <= 30:
                reg_label = "toucan_medium"
            elif n <= 60:
                reg_label = "toucan_large"
            else:
                reg_label = "toucan_xlarge"

            for turn in turns:
                # Reset REPL state between turns (fresh registry)
                repl.reset(keep_initial={"tools_registry": registry})

                result = build_repl_trajectory(
                    query=turn["query"],
                    correct_tool=turn["correct_tool"],
                    params=turn["params"],
                    registry=registry,
                    repl=repl,
                    num_tools=n,
                    seed=stats["ok"],
                )

                if result is None:
                    stats["repl_build_fail"] += 1
                    continue

                record = {
                    "source": "toucan",
                    "registry": reg_label,
                    "registry_size": n,
                    "query": turn["query"],
                    "correct_tool": turn["correct_tool"],
                    "predicted_tool": turn["correct_tool"],
                    "final_answer": {"tool": turn["correct_tool"], "params": turn["params"]},
                    "messages": result["messages"],
                    "turns": result["turns"],
                    "tools": tools,
                }

                out_file.write(json.dumps(record) + "\n")
                stats["ok"] += 1

                if stats["ok"] % 500 == 0:
                    out_file.flush()
                    print(f"  >> Saved {stats['ok']:,} trajectories")

                if stats["ok"] >= args.max_examples:
                    break

            if stats["ok"] >= args.max_examples:
                print(f"\nReached target of {args.max_examples:,} examples.")
                break

    finally:
        out_file.close()

    # ── Summary ───────────────────────────────────────────────────────────
    total = stats["seen"]
    ok = stats["ok"]
    print(f"\n{'='*60}")
    print(f"TOUCAN CONVERSION COMPLETE")
    print(f"  Entries seen         : {total:,}")
    print(f"  Converted (ok)       : {ok:,} ({ok/max(total,1)*100:.1f}%)")
    print(f"  Parse failures       : {stats['parse_fail']:,}")
    print(f"  Registry size filter : {stats['registry_size_filter']:,}")
    print(f"  REPL build failures  : {stats['repl_build_fail']:,}")
    print(f"  Output file          : {output_path}")
    print(f"{'='*60}")

    if ok < args.max_examples // 2:
        print(f"\nWARNING: Only got {ok:,} trajectories (target: {args.max_examples:,}).")
        print("  The TOUCAN dataset schema may differ from expected.")
        print("  Check --dataset or run with --debug to inspect raw entries.")


def main():
    parser = argparse.ArgumentParser(
        description="Convert TOUCAN-1.5M MCP trajectories to REPL-based training format"
    )
    parser.add_argument(
        "--dataset", type=str, default="Agent-Ark/Toucan-1.5M",
        help="HuggingFace dataset name (default: Agent-Ark/Toucan-1.5M)",
    )
    parser.add_argument(
        "--config", type=str, default="SFT",
        choices=["SFT", "OSS", "Qwen3", "Kimi-K2"],
        help="TOUCAN dataset config (default: SFT)",
    )
    parser.add_argument(
        "--split", type=str, default="train",
        help="Dataset split to use (default: train)",
    )
    parser.add_argument(
        "--output", type=str, default="data/toucan_repl.jsonl",
        help="Output JSONL file path (relative to project root)",
    )
    parser.add_argument(
        "--max-examples", type=int, default=20_000,
        help="Maximum number of trajectories to generate (default: 20000)",
    )
    parser.add_argument(
        "--min-registry-size", type=int, default=5,
        help="Minimum number of tools in registry (default: 5)",
    )
    parser.add_argument(
        "--max-registry-size", type=int, default=100,
        help="Maximum number of tools in registry (default: 100)",
    )
    parser.add_argument(
        "--shuffle-seed", type=int, default=42,
        help="Seed for dataset shuffling (default: 42)",
    )
    parser.add_argument(
        "--debug", action="store_true",
        help="Print first 3 raw entries and parsed output, then exit",
    )
    args = parser.parse_args()

    if args.debug:
        _debug_mode(args)
        return

    run(args)


def _debug_mode(args):
    """
    Scan TOUCAN entries until we find 3 that parse successfully.
    Also tracks skip reasons across the first 200 entries for diagnostics.
    """
    print("DEBUG: Scanning TOUCAN entries for parseable examples...\n")
    found = 0
    skip_counts: dict[str, int] = {}
    total_seen = 0

    for entry in stream_toucan(args.dataset, args.config, args.split, args.shuffle_seed):
        total_seen += 1

        # Track skip reason
        subset = entry.get("subset_name", "?")
        target = entry.get("target_tools", "")
        question = entry.get("question", "")
        if subset == "irrelevant":
            skip_counts["irrelevant_subset"] = skip_counts.get("irrelevant_subset", 0) + 1
        elif "," in target:
            skip_counts["multi_tool"] = skip_counts.get("multi_tool", 0) + 1
        elif isinstance(question, str) and question.startswith("["):
            skip_counts["multi_question"] = skip_counts.get("multi_question", 0) + 1
        else:
            skip_counts["other"] = skip_counts.get("other", 0) + 1

        turns = parse_toucan_entry(entry)
        for t in turns:
            found += 1
            print(f"{'='*60}")
            print(f"SUCCESS — Entry {total_seen}, turn {found} (subset={subset})")
            print(f"  query:        {t['query'][:100]}")
            print(f"  correct_tool: {t['correct_tool']}")
            print(f"  params:       {t['params']}")
            print(f"  num_tools:    {len(t['tools'])}")
            print(f"  tool names:   {[x['name'] for x in t['tools'][:4]]}...")
            if found >= 3:
                break
        if found >= 3:
            break

        if total_seen >= 500:
            print(f"\n[stopped after {total_seen} entries — only {found} parseable]")
            break

    print(f"\nScan summary ({total_seen} entries):")
    for k, v in sorted(skip_counts.items(), key=lambda x: -x[1]):
        print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
