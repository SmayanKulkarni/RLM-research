"""
augment_trajectories.py — Offline Trajectory Augmentation (NO API calls).

Takes existing verified trajectories and multiplies them ~3-4× via:
1. Cross-registry re-grounding  — replay code blocks against larger registries
2. Template-based generation   — create trajectories for uncovered tools
3. Query paraphrasing          — diversify natural-language queries

All augmented trajectories are re-executed against the REAL REPL and verified.

Usage:
    conda run -n astro python -m src.augment_trajectories \
        --input data/trajectories_verified.jsonl \
        --output data/trajectories_augmented.jsonl
"""

from __future__ import annotations

import argparse
import copy
import json
import random
import re
import sys
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.tool_registry import MCPToolRegistry
from src.repl_engine import REPLEngine, REPLConfig

# ── Paraphrase templates ──────────────────────────────────────────────────────
# Each maps a pattern to a list of rephrasings.
# {THING} placeholders are filled from the original query via regex extraction.

PARAPHRASE_TEMPLATES = {
    # Action-oriented queries
    r"(?:Can you |Please |I want to |I need to |Help me )?(.+)": [
        "Could you {0}",
        "I'd like to {0}",
        "Please help me {0}",
        "{0}",
    ],
}

# Simple word-level synonyms for common query verbs/nouns
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


# ── Registry + REPL helpers ──────────────────────────────────────────────────

REPL_CONFIG = REPLConfig.for_level(2)

REGISTRIES: dict[str, MCPToolRegistry] = {}
REGISTRY_ORDER = ["small_10", "medium_25", "large_50"]


def get_registry(name: str) -> MCPToolRegistry:
    if name not in REGISTRIES:
        path = PROJECT_ROOT / "test_data" / "tool_registries" / f"{name}.json"
        REGISTRIES[name] = MCPToolRegistry.from_json_file(path)
    return REGISTRIES[name]


def load_system_prompt(level: int = 2, num_tools: int = 25) -> str:
    path = PROJECT_ROOT / "prompts" / f"level{level}.txt"
    return path.read_text().replace("{num_tools}", str(num_tools))


# ── 1. Cross-registry re-grounding ──────────────────────────────────────────

def reground_trajectory(
    traj: dict,
    target_registry_name: str,
) -> dict | None:
    """
    Re-execute a trajectory's code blocks against a different (larger) registry.

    Returns a new trajectory dict with updated REPL outputs and system prompt,
    or None if any code block fails or the tool can't be found.
    """
    target_reg = get_registry(target_registry_name)

    # Check tool exists in target registry
    if traj["correct_tool"] not in set(target_reg.list_names()):
        return None

    num_tools = target_reg.count()
    system_prompt = load_system_prompt(level=2, num_tools=num_tools)

    repl = REPLEngine(
        config=REPL_CONFIG,
        initial_namespace={"tools_registry": target_reg},
    )

    new_messages = []
    for msg in traj["messages"]:
        if msg["role"] == "system":
            new_messages.append({"role": "system", "content": system_prompt})
        elif msg["role"] == "user" and msg["content"].startswith("[REPL OUTPUT]"):
            # This is a REPL output; will be replaced by the previous code execution
            # (already appended during assistant processing below)
            continue
        elif msg["role"] == "assistant":
            new_messages.append(msg)

            # Execute any code blocks against the new registry
            code_blocks = REPLEngine.extract_code_blocks(msg["content"])
            for code in code_blocks:
                try:
                    result = repl.execute(code)
                    new_messages.append({
                        "role": "user",
                        "content": f"[REPL OUTPUT]:\n{result.output}"
                    })
                except Exception:
                    return None  # Code failed → invalid augmentation
        else:
            new_messages.append(msg)

    new_traj = copy.deepcopy(traj)
    new_traj["messages"] = new_messages
    new_traj["registry"] = target_registry_name
    new_traj["augmentation"] = f"reground:{traj['registry']}→{target_registry_name}"
    return new_traj


# ── 2. Template-based trajectory generation ──────────────────────────────────

# Standard REPL trajectory template: search → get_schema → FINAL
TEMPLATE_THOUGHTS = [
    # Step 1: keyword search
    (
        'THOUGHT: I need to find a tool that can help with this request. '
        'Let me search for relevant tools using keywords.\n'
        '<code>\nprint(tools_registry.search("{keyword1}"))\n</code>'
    ),
    # Step 2: get schema
    (
        'THOUGHT: The search returned some promising results. '
        'Let me look at the full schema of the most relevant tool.\n'
        '<code>\nprint(tools_registry.get_schema("{tool_name}"))\n</code>'
    ),
    # Step 3: FINAL
    (
        'THOUGHT: Based on the schema, "{tool_name}" is the right tool for this task. '
        'I can extract the required parameters from the user\'s query.\n'
        'FINAL({{"tool": "{tool_name}", "params": {params_json}}})'
    ),
]

# Alternative: category filter → search → FINAL
TEMPLATE_THOUGHTS_ALT = [
    # Step 1: category filter
    (
        'THOUGHT: Let me filter tools by a relevant category to narrow down the options.\n'
        '<code>\nprint(tools_registry.filter_by_category("{category}"))\n</code>'
    ),
    # Step 2: get schema of match
    (
        'THOUGHT: I found some tools in this category. '
        'Let me get the detailed schema to find the best match.\n'
        '<code>\nprint(tools_registry.get_schema("{tool_name}"))\n</code>'
    ),
    # Step 3: FINAL
    (
        'THOUGHT: The tool "{tool_name}" matches the user\'s request perfectly. '
        'I\'ll select it with the appropriate parameters.\n'
        'FINAL({{"tool": "{tool_name}", "params": {params_json}}})'
    ),
]

# Map tool names to likely search keywords and categories
TOOL_KEYWORDS: dict[str, dict] = {
    "resize_image":         {"keyword1": "image resize", "category": "media"},
    "summarize_text":       {"keyword1": "summarize text", "category": "ml_analytics"},
    "git_clone":            {"keyword1": "git clone", "category": "development"},
    "git_commit":           {"keyword1": "git commit", "category": "development"},
    "search_code":          {"keyword1": "search code", "category": "development"},
    "generate_image":       {"keyword1": "generate image", "category": "media"},
    "compress_files":       {"keyword1": "compress zip", "category": "filesystem"},
    "lookup_definition":    {"keyword1": "definition lookup", "category": "web_apis"},
    "create_github_issue":  {"keyword1": "github issue", "category": "development"},
    "format_date":          {"keyword1": "format date", "category": "system"},
    "list_tasks":           {"keyword1": "list tasks", "category": "productivity"},
    "hash_text":            {"keyword1": "hash", "category": "system"},
    "get_stock_price":      {"keyword1": "stock price", "category": "web_apis"},
    "create_task":          {"keyword1": "create task", "category": "productivity"},
    "generate_password":    {"keyword1": "generate password", "category": "system"},
    "parse_csv":            {"keyword1": "parse csv", "category": "data_storage"},
    "search_wikipedia":     {"keyword1": "wikipedia search", "category": "web_apis"},
    "convert_currency":     {"keyword1": "convert currency", "category": "web_apis"},
    "ocr_image":            {"keyword1": "ocr image text", "category": "media"},
    "create_spreadsheet":   {"keyword1": "spreadsheet create", "category": "data_storage"},
    "generate_uuid":        {"keyword1": "uuid generate", "category": "system"},
    "convert_pdf_to_text":  {"keyword1": "pdf to text", "category": "media"},
    "validate_json":        {"keyword1": "validate json", "category": "development"},
    "calculate_math":       {"keyword1": "calculate math", "category": "system"},
    # Tools that exist in smaller registries (for alternative templates)
    "send_email":           {"keyword1": "send email", "category": "communication"},
    "get_weather":          {"keyword1": "weather", "category": "web_apis"},
    "create_calendar_event":{"keyword1": "calendar event", "category": "productivity"},
    "read_file":            {"keyword1": "read file", "category": "filesystem"},
    "write_file":           {"keyword1": "write file", "category": "filesystem"},
    "send_sms":             {"keyword1": "send sms text", "category": "communication"},
    "send_slack_message":   {"keyword1": "slack message", "category": "communication"},
    "list_directory":       {"keyword1": "list directory", "category": "filesystem"},
    "get_forecast":         {"keyword1": "forecast weather", "category": "web_apis"},
    "geocode_address":      {"keyword1": "geocode address", "category": "web_apis"},
    "get_current_time":     {"keyword1": "current time", "category": "system"},
    "translate_text":       {"keyword1": "translate", "category": "ml_analytics"},
    "create_chart":         {"keyword1": "chart create", "category": "data_storage"},
    "search_web":           {"keyword1": "search web", "category": "web_apis"},
    "query_database":       {"keyword1": "query database", "category": "data_storage"},
    "analyze_csv":          {"keyword1": "analyze csv", "category": "data_storage"},
    "get_embeddings":       {"keyword1": "embeddings", "category": "ml_analytics"},
    "get_system_info":      {"keyword1": "system info", "category": "system"},
    "list_tables":          {"keyword1": "list tables", "category": "data_storage"},
    "classify_text":        {"keyword1": "classify text", "category": "ml_analytics"},
    "run_inference":        {"keyword1": "run inference", "category": "ml_analytics"},
    "run_shell_command":    {"keyword1": "shell command", "category": "system"},
    "http_post":            {"keyword1": "http post", "category": "web_apis"},
    "scrape_webpage":       {"keyword1": "scrape webpage", "category": "web_apis"},
    "http_get":             {"keyword1": "http get", "category": "web_apis"},
}

# Sample queries for the 24 tools only in large_50
TEMPLATE_QUERIES: dict[str, list[str]] = {
    "resize_image":         ["Resize the image at /photos/pic.jpg to 800x600 pixels",
                             "I need to change the dimensions of an image to 1024x768",
                             "Make the photo /uploads/banner.png smaller, 640x480"],
    "summarize_text":       ["Summarize this article for me in a few sentences",
                             "Can you give me a short summary of this long document?",
                             "I need a brief overview of the following text"],
    "git_clone":            ["Clone the repository at https://github.com/user/project to /home/dev",
                             "I need to download a git repo from https://github.com/org/app",
                             "Pull the code from https://github.com/team/library into /workspace"],
    "git_commit":           ["Commit the changes with message 'Fix bug in login'",
                             "I need to commit my code changes with the message 'Add new feature'",
                             "Make a git commit with message 'Update README'"],
    "search_code":          ["Find all Python files in /src that contain the word 'TODO'",
                             "Search for functions named 'handle' in the /app directory",
                             "Look for import statements in all .py files under /project"],
    "generate_image":       ["Generate an image of a sunset over mountains",
                             "Create a picture of a futuristic city skyline",
                             "I want an AI-generated image of a cat wearing a hat"],
    "compress_files":       ["Compress the files in /backup into a zip archive",
                             "I need to zip up the /documents folder",
                             "Create a compressed archive of /logs/2025"],
    "lookup_definition":    ["What does the word 'ephemeral' mean?",
                             "Look up the definition of 'serendipity'",
                             "Define the word 'ubiquitous' for me"],
    "create_github_issue":  ["Create a GitHub issue titled 'Fix login bug' in the repo user/project",
                             "Open a new issue on GitHub about the memory leak in org/app",
                             "File a bug report on GitHub for the crash in team/service"],
    "format_date":          ["Format the date 2025-03-12 as 'March 12, 2025'",
                             "Convert the timestamp 1710288000 to a readable date",
                             "Reformat '12/03/2025' to 'YYYY-MM-DD' format"],
    "list_tasks":           ["Show me all my pending tasks",
                             "List all tasks with high priority",
                             "What tasks are due this week?"],
    "hash_text":            ["Get the SHA-256 hash of the text 'hello123'",
                             "Compute the MD5 hash of 'password'",
                             "I need a hash of the string 'my-secret-key'"],
    "get_stock_price":      ["What's Apple's current stock price?",
                             "Get the latest stock price for MSFT",
                             "How much is Tesla stock trading at right now?"],
    "create_task":          ["Create a high-priority task called 'Review PR' due tomorrow",
                             "Add a new task: 'Deploy to production' with medium priority",
                             "Make a task titled 'Write documentation' due next Friday"],
    "generate_password":    ["Generate a 16-character secure password",
                             "I need a random password with 20 characters",
                             "Create a strong password for my new account"],
    "parse_csv":            ["Parse the CSV file at /data/sales.csv",
                             "Read and parse /reports/quarterly.csv",
                             "I need to extract data from a CSV file at /tmp/export.csv"],
    "search_wikipedia":     ["Search Wikipedia for information about quantum computing",
                             "Find the Wikipedia article about the French Revolution",
                             "Look up 'machine learning' on Wikipedia"],
    "convert_currency":     ["Convert 100 USD to EUR",
                             "How much is 500 GBP in Japanese yen?",
                             "What's 250 euros in US dollars?"],
    "ocr_image":            ["Extract text from the scanned document at /docs/receipt.jpg",
                             "Read the text in the image /scans/letter.png",
                             "I need OCR on the photo at /uploads/whiteboard.jpg"],
    "create_spreadsheet":   ["Create a new spreadsheet called 'Budget 2025'",
                             "Set up a spreadsheet for tracking monthly expenses",
                             "Make a new Excel file called 'Sales Report'"],
    "generate_uuid":        ["Generate a new UUID",
                             "I need a unique identifier for my database record",
                             "Create a random UUID for the session"],
    "convert_pdf_to_text":  ["Convert the PDF at /docs/report.pdf to text",
                             "Extract text from the PDF file /uploads/invoice.pdf",
                             "I need the text content of /papers/research.pdf"],
    "validate_json":        ["Check if this JSON string is valid: {\"key\": \"value\"}",
                             "Validate the JSON at /config/settings.json",
                             "I need to verify that my JSON data is well-formed"],
    "calculate_math":       ["Calculate the square root of 144",
                             "What is 15% of 350?",
                             "Compute 2^10 + 3^5"],
}


def extract_params_from_query(tool_name: str, query: str, registry: MCPToolRegistry) -> dict:
    """Try to extract plausible parameter values from the query for a tool."""
    schema = registry.get_schema(tool_name)
    if not schema:
        return {}

    props = schema.get("inputSchema", {}).get("properties", {})
    params = {}

    for pname, pschema in props.items():
        ptype = pschema.get("type", "string")
        # Simple heuristic extraction
        if ptype == "string":
            # Use a portion of the query or a placeholder
            if "path" in pname or "file" in pname or "url" in pname or "directory" in pname:
                # Try to extract a path/URL
                path_m = re.search(r'(/[\w/.-]+|https?://[\w/.-]+)', query)
                if path_m:
                    params[pname] = path_m.group(1)
                else:
                    params[pname] = f"/example/{pname}"
            elif "message" in pname or "content" in pname or "text" in pname or "body" in pname:
                params[pname] = query[:80]
            elif "title" in pname or "name" in pname or "subject" in pname:
                title_m = re.search(r"['\"]([^'\"]+)['\"]", query)
                if title_m:
                    params[pname] = title_m.group(1)
                else:
                    params[pname] = "Task from query"
            elif "email" in pname or "to" in pname or "recipient" in pname:
                email_m = re.search(r'[\w.+-]+@[\w.-]+', query)
                if email_m:
                    params[pname] = email_m.group(0)
                else:
                    params[pname] = "user@example.com"
            else:
                params[pname] = pname + "_value"
        elif ptype == "integer" or ptype == "number":
            num_m = re.search(r'\b(\d+)\b', query)
            params[pname] = int(num_m.group(1)) if num_m else 100
        elif ptype == "boolean":
            params[pname] = True

    return params


def generate_template_trajectory(
    tool_name: str,
    query: str,
    registry_name: str,
) -> dict | None:
    """Generate a trajectory from a template (no API call)."""
    registry = get_registry(registry_name)

    if tool_name not in set(registry.list_names()):
        return None

    num_tools = registry.count()
    system_prompt = load_system_prompt(level=2, num_tools=num_tools)

    info = TOOL_KEYWORDS.get(tool_name, {})
    keyword1 = info.get("keyword1", tool_name.replace("_", " "))
    category = info.get("category", "")

    params = extract_params_from_query(tool_name, query, registry)
    params_json = json.dumps(params) if params else "{}"

    # Choose template variant randomly
    use_alt = category and random.random() < 0.4
    templates = TEMPLATE_THOUGHTS_ALT if use_alt else TEMPLATE_THOUGHTS

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": query},
    ]

    repl = REPLEngine(
        config=REPL_CONFIG,
        initial_namespace={"tools_registry": registry},
    )

    for i, tmpl in enumerate(templates):
        filled = tmpl.format(
            keyword1=keyword1,
            tool_name=tool_name,
            category=category,
            params_json=params_json,
        )

        messages.append({"role": "assistant", "content": filled})

        # Execute code blocks
        code_blocks = REPLEngine.extract_code_blocks(filled)
        for code in code_blocks:
            try:
                result = repl.execute(code)
                messages.append({
                    "role": "user",
                    "content": f"[REPL OUTPUT]:\n{result.output}"
                })
            except Exception:
                return None

    return {
        "registry": registry_name,
        "query": query,
        "correct_tool": tool_name,
        "turns": len([m for m in messages if m["role"] == "assistant"
                      and "<code>" in m["content"]]),
        "messages": messages,
        "augmentation": f"template:{registry_name}",
    }


# ── 3. Query paraphrasing ────────────────────────────────────────────────────

def paraphrase_query(query: str) -> list[str]:
    """Generate 1-2 paraphrased variants of a query."""
    variants = []

    # Synonym substitution (pick 1-2)
    words = query.split()
    for _ in range(2):
        new_words = list(words)
        swapped = False
        for i, w in enumerate(new_words):
            w_lower = w.lower().strip(".,!?")
            if w_lower in SYNONYMS and random.random() < 0.5:
                replacement = random.choice(SYNONYMS[w_lower])
                # Preserve case
                if w[0].isupper():
                    replacement = replacement.capitalize()
                new_words[i] = replacement
                swapped = True
        if swapped:
            variant = " ".join(new_words)
            if variant != query and variant not in variants:
                variants.append(variant)

    # Structural rephrase
    prefixes = [
        "Can you ", "Please ", "I need to ", "I'd like to ",
        "Help me ", "I want to ", "Could you ",
    ]
    # Strip existing prefix
    stripped = query
    for p in prefixes:
        if query.lower().startswith(p.lower()):
            stripped = query[len(p):]
            break

    if stripped != query:
        new_prefix = random.choice([p for p in prefixes if p.lower() != query[:len(p)].lower()])
        variant = new_prefix + stripped[0].lower() + stripped[1:]
        if variant not in variants:
            variants.append(variant)
    else:
        new_prefix = random.choice(prefixes)
        variant = new_prefix + query[0].lower() + query[1:]
        if variant not in variants:
            variants.append(variant)

    return variants[:2]  # max 2 variants


# ── Verification ──────────────────────────────────────────────────────────────

def verify_trajectory(traj: dict) -> bool:
    """Quick verification that a trajectory is valid for training."""
    messages = traj.get("messages", [])
    if len(messages) < 4:
        return False

    # Must have system, user, at least one assistant
    roles = [m["role"] for m in messages]
    if roles[0] != "system" or roles[1] != "user":
        return False

    # Last assistant message must contain FINAL
    last_assistant = [m for m in messages if m["role"] == "assistant"]
    if not last_assistant:
        return False

    final_msg = last_assistant[-1]["content"]
    if "FINAL(" not in final_msg:
        return False

    # Correct tool must appear in FINAL
    tool = traj.get("correct_tool", "")
    if tool not in final_msg:
        return False

    return True


# ── Main pipeline ─────────────────────────────────────────────────────────────

def run(args):
    random.seed(args.seed)

    # Load original trajectories
    input_path = PROJECT_ROOT / args.input
    with open(input_path) as f:
        originals = [json.loads(l) for l in f if l.strip()]
    print(f"Loaded {len(originals)} original trajectories")

    all_trajectories = list(originals)  # start with originals
    stats = {"originals": len(originals), "reground": 0, "template": 0, "paraphrase": 0}

    # ── 1. Cross-registry re-grounding ──────────────────────────────────
    print("\n[Stage 1] Cross-registry re-grounding...")
    for traj in originals:
        src_reg = traj["registry"]
        src_idx = REGISTRY_ORDER.index(src_reg) if src_reg in REGISTRY_ORDER else -1

        # Try all larger registries
        for target_reg in REGISTRY_ORDER[src_idx + 1:]:
            new_traj = reground_trajectory(traj, target_reg)
            if new_traj and verify_trajectory(new_traj):
                all_trajectories.append(new_traj)
                stats["reground"] += 1

    print(f"  → {stats['reground']} re-grounded trajectories")

    # ── 2. Template-based generation for uncovered tools ────────────────
    print("\n[Stage 2] Template-based trajectory generation...")

    # Identify covered vs uncovered tools
    covered_tools = {(t["correct_tool"], t["registry"]) for t in all_trajectories}

    for registry_name in REGISTRY_ORDER:
        reg = get_registry(registry_name)
        tool_names = reg.list_names()

        for tool_name in tool_names:
            if (tool_name, registry_name) in covered_tools:
                # Already have trajectories — maybe add 1-2 more for balance
                existing_count = sum(
                    1 for t in all_trajectories
                    if t["correct_tool"] == tool_name and t["registry"] == registry_name
                )
                if existing_count >= 3:
                    continue

            # Get queries for this tool
            queries = TEMPLATE_QUERIES.get(tool_name, [])
            if not queries:
                # Generate simple queries from tool description
                schema = reg.get_schema(tool_name)
                desc = schema.get("description", "") if schema else ""
                queries = [
                    f"I need to use {tool_name.replace('_', ' ')}",
                    f"Help me {desc[:60].lower()}" if desc else f"Use the {tool_name.replace('_', ' ')} tool",
                    f"Can you {desc[:60].lower()}" if desc else f"Run {tool_name.replace('_', ' ')}",
                ]

            for query in queries[:3]:  # max 3 per tool per registry
                if (tool_name, registry_name) in covered_tools:
                    existing_count = sum(
                        1 for t in all_trajectories
                        if t["correct_tool"] == tool_name and t["registry"] == registry_name
                    )
                    if existing_count >= 4:
                        break

                new_traj = generate_template_trajectory(tool_name, query, registry_name)
                if new_traj and verify_trajectory(new_traj):
                    all_trajectories.append(new_traj)
                    covered_tools.add((tool_name, registry_name))
                    stats["template"] += 1

    print(f"  → {stats['template']} template trajectories")

    # ── 3. Query paraphrasing (applied to ALL trajectories so far) ──────
    print("\n[Stage 3] Query paraphrasing...")

    # Paraphrase a subset to avoid ballooning too much
    candidates = list(all_trajectories)
    random.shuffle(candidates)

    # Target total: ~400-500 trajectories
    target = max(400, len(all_trajectories))
    remaining = target - len(all_trajectories)

    for traj in candidates:
        if remaining <= 0:
            break

        query = traj["query"]
        variants = paraphrase_query(query)

        for variant in variants:
            if remaining <= 0:
                break

            new_traj = copy.deepcopy(traj)
            new_traj["query"] = variant
            new_traj["augmentation"] = f"paraphrase:{traj.get('augmentation', 'original')}"

            # Update the user message with the new query
            for msg in new_traj["messages"]:
                if msg["role"] == "user" and msg["content"] == query:
                    msg["content"] = variant
                    break

            # Note: REPL outputs don't change (same code, same registry)
            # The search keywords in assistant messages also stay the same —
            # this is intentional: we want the model to learn that different
            # query phrasings map to the same search strategy.

            if verify_trajectory(new_traj):
                all_trajectories.append(new_traj)
                stats["paraphrase"] += 1
                remaining -= 1

    print(f"  → {stats['paraphrase']} paraphrased trajectories")

    # ── Deduplicate ─────────────────────────────────────────────────────
    seen = set()
    deduped = []
    for traj in all_trajectories:
        key = (traj["query"], traj["correct_tool"], traj["registry"])
        if key not in seen:
            seen.add(key)
            deduped.append(traj)

    removed = len(all_trajectories) - len(deduped)
    all_trajectories = deduped

    # ── Save ────────────────────────────────────────────────────────────
    output_path = PROJECT_ROOT / args.output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        for traj in all_trajectories:
            f.write(json.dumps(traj) + "\n")

    # ── Summary ─────────────────────────────────────────────────────────
    from collections import Counter
    reg_counts = Counter(t["registry"] for t in all_trajectories)
    tool_counts = Counter(t["correct_tool"] for t in all_trajectories)

    print(f"\n{'='*60}")
    print(f"AUGMENTATION COMPLETE")
    print(f"  Originals     : {stats['originals']}")
    print(f"  Re-grounded   : {stats['reground']}")
    print(f"  Templates     : {stats['template']}")
    print(f"  Paraphrased   : {stats['paraphrase']}")
    print(f"  Duplicates    : {removed}")
    print(f"  TOTAL         : {len(all_trajectories)}")
    print(f"\nPer registry:")
    for reg in REGISTRY_ORDER:
        print(f"  {reg}: {reg_counts.get(reg, 0)}")
    print(f"\nUnique tools covered: {len(tool_counts)}")
    print(f"Min/Max per tool: {min(tool_counts.values())}/{max(tool_counts.values())}")
    print(f"\nSaved to: {output_path}")
    print(f"{'='*60}")


def main():
    parser = argparse.ArgumentParser(description="Offline trajectory augmentation")
    parser.add_argument("--input", default="data/trajectories_verified.jsonl")
    parser.add_argument("--output", default="data/trajectories_augmented.jsonl")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
