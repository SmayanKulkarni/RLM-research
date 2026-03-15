# Implementation POA: Clarification Answers + Comprehensive Plan

**Date:** 2026-03-10  
**Scope:** Detailed answers to 4 implementation questions + full implementation plan of action  
**Method:** RLM PDF deep extraction (system prompts, REPL architecture), APIGen PDF analysis, 4 web searches on MCP test servers, Python SDK, and evaluation metrics

---

## Q1: MCP Test Server — What to Use?

### Decision: Build a **Synthetic MCP Tool Registry** (NOT a full MCP server)

**Why NOT a full MCP server at this stage:**
- We don't need end-to-end MCP communication right now
- What we're testing is the SLM's ability to **select the right tool from tool descriptions** using the REPL
- The actual MCP protocol (JSON-RPC handshake, transport layer) is irrelevant at this research stage
- Building a minimal test harness is faster and more controlled for experimentation

**What we actually need:**
A Python file containing realistic MCP tool description sets of varying sizes (10, 25, 50, 100 tools) that we can load into the REPL as the `context` variable.

### Recommended Approach: Hybrid Synthetic + Real

#### Step 1: Collect Real MCP Tool Descriptions (from existing servers)
🟢 **EMPIRICAL** — The official MCP specification provides reference servers with real tool definitions:

| Source | Tools Available | How to Get |
|--------|----------------|------------|
| **MCP Reference Servers** ([github.com/modelcontextprotocol/servers](https://github.com/modelcontextprotocol/servers)) | Filesystem (11 tools), GitHub (18+ tools), PostgreSQL (5+ tools), Memory (4 tools) | Clone repo, extract tool schemas |
| **Anthropic MCP Examples** | Weather, Puppeteer, EverArt, etc. | From MCP docs |
| **Community MCP Servers** (1000+ on MCP Marketplace) | Diverse (Notion, Slack, Google Drive, etc.) | Via MCP Inspector or direct API |

**How to extract tool schemas programmatically using MCP Python SDK:**

```python
from mcp import FastMCP

# Or if connecting to an existing server:
# Use MCP Inspector (https://github.com/modelcontextprotocol/inspector)
# to extract tool definitions from any running MCP server
```

#### Step 2: Create Synthetic Tool Descriptions (for scale testing)
Generate additional tools to reach our test sizes (50, 100 tools). Use this template based on the real MCP JSON schema:

```python
# Template for synthetic MCP tool descriptions
TOOL_TEMPLATE = {
    "name": "tool_name",           # lowercase, underscored
    "description": "What this tool does in 1-2 sentences",
    "inputSchema": {
        "type": "object",
        "properties": {
            "param1": {"type": "string", "description": "..."},
            "param2": {"type": "integer", "description": "..."}
        },
        "required": ["param1"]
    }
}
```

Generate synthetic tools across **diverse categories** (inspired by APIGen's 21 categories):
- **Data & Storage**: read_file, write_file, query_database, list_tables
- **Communication**: send_email, send_slack_message, create_calendar_event
- **Web & APIs**: http_get, scrape_webpage, search_web
- **ML & Analytics**: run_inference, get_embeddings, classify_text
- **System**: get_time, run_command, get_system_info
- **Domain-specific**: get_weather, translate_text, generate_image, get_stock_price

#### Is APIGen relevant for us?

🟢 **EMPIRICAL** — After reading the [APIGen paper (Zhang et al., 2024)](https://arxiv.org/abs/2406.18518):

**APIGen is NOT directly for MCP** — it's a pipeline for generating function-calling training data from REST APIs. BUT several components are highly relevant:

| APIGen Component | Relevance to Us | How to Adapt |
|-----------------|-----------------|--------------|
| **3-stage verification** (Format → Execution → Semantic) | ✅ Very High | Use the same pipeline to verify our synthetic trajectory quality |
| **21 API categories** | ✅ High | Use as template for diversifying our synthetic MCP tools |
| **Multi-style queries** (Simple, Multiple, Parallel, Parallel-Multiple) | ✅ High | Design our test queries to cover all styles |
| **60K verified examples** | ⚠️ Medium | We can't use their dataset directly (REST APIs ≠ MCP tools), but the FORMAT is instructive |
| **Quality filtering stats** | ✅ High | Lower-quality LLMs produce 34% valid data vs 84% for stronger ones — sets expectations for our trajectory generation |

**Key takeaway from APIGen:** Use the **verification pipeline** (not the dataset) — apply format→execution→semantic checking to our own MCP trajectory generation later.

### Final Test Data Structure

```
test_data/
├── tool_registries/
│   ├── small_10.json      # 10 tools — simple selection
│   ├── medium_25.json     # 25 tools — moderate selection
│   ├── large_50.json      # 50 tools — realistic MCP deployment
│   └── xlarge_100.json    # 100 tools — stress test
├── queries/
│   ├── simple.json        # "What's the weather?" → 1 tool needed
│   ├── multi_select.json  # "Send email about stock prices" → 2 tools
│   └── discovery.json     # "What tools can help me analyze data?"
└── ground_truth/
    └── expected_selections.json  # Correct tool(s) per query
```

---

## Q2: How to Design the REPL / Constrained REPL Environment

### Understanding the RLM Paper's REPL Architecture

🟢 **EMPIRICAL** — From [Zhang et al., 2025, Appendix D]:

The RLM REPL has these core components:

```
┌───────────────────────────────────────────────────┐
│                   ROOT LLM                        │
│  (Generates text + code in ```repl blocks)        │
├───────────────────────────────────────────────────┤
│              REPL ENVIRONMENT                     │
│                                                   │
│  Pre-loaded variables:                            │
│    • context: str/List[str] — the prompt text     │
│                                                   │
│  Available functions:                             │
│    • print() — output to LLM                      │
│    • llm_query(prompt) — sub-LM call (depth≥1)    │
│                                                   │
│  Execution:                                       │
│    • exec() on code from ```repl blocks           │
│    • Variables PERSIST across turns                │
│    • Output is TRUNCATED to fit context window     │
│                                                   │
│  Output mechanism:                                │
│    • FINAL(answer) or FINAL_VAR(variable_name)    │
└───────────────────────────────────────────────────┘
```

**Key RLM implementation details:**
1. The model outputs text freely, interleaving reasoning with code blocks
2. Code blocks are marked with ` ```repl ` and executed by the environment
3. All output from `print()` is captured and sent back to the model as part of its next input
4. Print output is **truncated** to prevent blowing up the context window
5. Variables created in one `exec()` persist — the model can build state across turns
6. For depth=0 (no sub-calls), the `llm_query()` function is not injected

### Our Constrained REPL Design (3 Levels)

#### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                  MCP-REPL SCAFFOLD                      │
│                                                         │
│  ┌─────────────┐     ┌──────────────────┐              │
│  │   SLM        │◄───►│  REPL Engine     │              │
│  │ (Qwen3.5    │     │                  │              │
│  │  Coder 3B)  │     │  Pre-loaded:     │              │
│  │             │     │  • tools_registry│              │
│  │  Generates: │     │  • helper funcs  │              │
│  │  • Thought  │     │                  │              │
│  │  • Code     │     │  Executes:       │              │
│  │  • FINAL()  │     │  • ```repl code  │              │
│  └─────────────┘     │  • Returns stdout│              │
│                      └──────────────────┘              │
│                                                         │
│  Query Flow:                                            │
│  User Query → SLM thinks → writes ```repl code →       │
│  REPL executes → stdout sent back to SLM →             │
│  SLM reasons more → FINAL(tool_call)                   │
└─────────────────────────────────────────────────────────┘
```

#### Level 1: Constrained REPL (Start Here)

The SLM can ONLY call **predefined helper functions** — it cannot write arbitrary Python.

```python
# === REPL ENVIRONMENT: PRE-LOADED FUNCTIONS ===

class MCPToolRegistry:
    """Loaded into REPL as `tools_registry`"""
    
    def __init__(self, tools: list[dict]):
        self._tools = tools  # List of MCP tool JSON schemas
    
    def list_names(self) -> list[str]:
        """Return just the names of all available tools"""
        return [t["name"] for t in self._tools]
    
    def search(self, query: str) -> list[dict]:
        """Search tools by keyword in name + description.
        Returns matching tools with name + description (no full schema).
        """
        query_lower = query.lower()
        results = []
        for t in self._tools:
            if (query_lower in t["name"].lower() or 
                query_lower in t["description"].lower()):
                results.append({
                    "name": t["name"],
                    "description": t["description"]
                })
        return results
    
    def get_schema(self, tool_name: str) -> dict | None:
        """Get the full JSON schema for a specific tool by name"""
        for t in self._tools:
            if t["name"] == tool_name:
                return t
        return None
    
    def filter_by_category(self, category: str) -> list[dict]:
        """Filter tools by category tag (if present in annotations)"""
        results = []
        for t in self._tools:
            annotations = t.get("annotations", {})
            if category.lower() in str(annotations).lower():
                results.append({"name": t["name"], "description": t["description"]})
        return results
    
    def count(self) -> int:
        """Return total number of available tools"""
        return len(self._tools)


# === REPL SESSION CONFIG ===
REPL_CONFIG = {
    "max_output_chars": 2000,    # Truncate print() output for SLM context
    "max_turns": 10,             # Max REPL interaction turns
    "timeout_seconds": 30,       # Per-exec timeout
    "allowed_imports": [],       # Level 1: no imports allowed
    "allowed_builtins": ["print", "len", "str", "int", "list", "dict"],
}
```

**What the SLM does at Level 1:**
```
User query: "I need to find the current weather in Paris"

SLM output:
THINK: I need to find a weather-related tool. Let me search.
```repl
results = tools_registry.search("weather")
print(results)
```

[REPL OUTPUT]: [{"name": "get_weather", "description": "Get current weather..."}]

THINK: Found get_weather. Let me get its full schema.
```repl
schema = tools_registry.get_schema("get_weather")
print(schema)
```

[REPL OUTPUT]: {"name": "get_weather", "inputSchema": {"properties": {"city": ...}}}

FINAL({"tool": "get_weather", "params": {"city": "Paris"}})
```

#### Level 2: Simple Python REPL (After Fine-Tuning)

Extends Level 1 — SLM can write simple Python using the helper functions + basic operations.

```python
# What changes from Level 1:
REPL_CONFIG_L2 = {
    "allowed_imports": ["re", "json"],  # Allow regex and json
    "allowed_builtins": ["print", "len", "str", "int", "list", "dict", 
                         "enumerate", "sorted", "filter", "any", "all"],
}
# Everything else from Level 1 remains available
```

**What the SLM does at Level 2:**
```
```repl
import re
# Search for tools related to data analysis
names = tools_registry.list_names()
data_tools = [n for n in names if re.search(r"data|analy|chart|plot", n)]
for tool_name in data_tools:
    schema = tools_registry.get_schema(tool_name)
    print(f"{tool_name}: {schema['description']}")
```
```

#### Level 3: Full Python REPL (Stretch Goal)

SLM writes unrestricted Python. Same as RLM paper's approach but applied to MCP tool descriptions.

### Implementation: The REPL Engine

```python
# === repl_engine.py ===
import io
import sys
import traceback
from contextlib import redirect_stdout, redirect_stderr

class REPLEngine:
    """Execute SLM-generated code in a controlled Python REPL."""
    
    def __init__(self, tools_registry: MCPToolRegistry, config: dict):
        self.config = config
        self.namespace = {
            "tools_registry": tools_registry,  # Pre-loaded!
            "__builtins__": {name: __builtins__[name] 
                           for name in config.get("allowed_builtins", [])},
        }
        self.turn_count = 0
        self.history = []  # Track all interactions
    
    def execute(self, code: str) -> str:
        """Execute code block and return truncated stdout."""
        self.turn_count += 1
        if self.turn_count > self.config["max_turns"]:
            return "[ERROR: Maximum REPL turns exceeded]"
        
        stdout_capture = io.StringIO()
        try:
            with redirect_stdout(stdout_capture):
                exec(code, self.namespace)
            output = stdout_capture.getvalue()
        except Exception as e:
            output = f"[ERROR: {type(e).__name__}: {str(e)}]"
        
        # Truncate output to fit SLM context
        max_chars = self.config.get("max_output_chars", 2000)
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n... [TRUNCATED, {len(output)} total chars]"
        
        self.history.append({"code": code, "output": output, "turn": self.turn_count})
        return output
    
    def extract_final(self, text: str) -> str | None:
        """Extract FINAL() or FINAL_VAR() from SLM output."""
        import re
        # Check for FINAL(answer)
        match = re.search(r'FINAL\((.+?)\)', text, re.DOTALL)
        if match:
            return match.group(1)
        # Check for FINAL_VAR(variable_name)
        match = re.search(r'FINAL_VAR\((\w+)\)', text)
        if match:
            var_name = match.group(1)
            return str(self.namespace.get(var_name, f"[ERROR: Variable '{var_name}' not found]"))
        return None
```

### The Interaction Loop (Orchestrator)

```python
# === scaffold.py ===
import re

class MCPRLMScaffold:
    """Main orchestrator: SLM + REPL interaction loop."""
    
    def __init__(self, slm, repl_engine: REPLEngine, system_prompt: str):
        self.slm = slm  # Any LLM interface (Ollama, HuggingFace, etc.)
        self.repl = repl_engine
        self.system_prompt = system_prompt
    
    def run(self, user_query: str) -> dict:
        """Run a single MCP tool selection task."""
        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query}
        ]
        
        for turn in range(self.repl.config["max_turns"]):
            # Get SLM response
            response = self.slm.generate(conversation)
            
            # Check for FINAL answer
            final = self.repl.extract_final(response)
            if final:
                return {
                    "answer": final,
                    "turns": turn + 1,
                    "history": self.repl.history,
                    "success": True
                }
            
            # Extract and execute ```repl code blocks
            code_blocks = re.findall(r'```repl\n(.*?)```', response, re.DOTALL)
            
            repl_output = ""
            for code in code_blocks:
                output = self.repl.execute(code.strip())
                repl_output += f"\n[REPL OUTPUT]:\n{output}\n"
            
            # Append SLM response and REPL output to conversation
            conversation.append({"role": "assistant", "content": response})
            if repl_output:
                conversation.append({"role": "user", "content": repl_output})
        
        return {"answer": None, "turns": self.repl.config["max_turns"],
                "history": self.repl.history, "success": False}
```

---

## Q3: System Prompt Design for Qwen3.5-4B

### RLM Paper's System Prompt Structure (Extracted from Appendix D)

🟢 **EMPIRICAL** — The RLM paper uses this structure (verified from PDF pages 25–26):

1. **Role & task** — "You are tasked with answering a query with associated context"
2. **Context description** — "Your context is a {context_type} with {total_length} chars"
3. **REPL capabilities** — What's available (`context` variable, `print()`, `llm_query()`)
4. **Constraints** — Truncation warning, output format rules
5. **Examples** — 2–3 worked examples showing the pattern (keyword search, chunking, regex)
6. **Output format** — FINAL(answer) or FINAL_VAR(variable)

### Our Adapted System Prompt (Level 1: Constrained REPL)

```
You are an AI assistant that selects MCP tools to answer user requests.

== YOUR ENVIRONMENT ==
You have access to a REPL environment with a `tools_registry` object pre-loaded 
with {num_tools} available MCP tools. You cannot see all tools at once — use the 
registry functions to find the right one.

Available functions in the REPL:
- tools_registry.list_names() → list of all tool names
- tools_registry.search(query) → find tools matching a keyword
- tools_registry.get_schema(tool_name) → get full tool definition  
- tools_registry.filter_by_category(category) → filter by category
- tools_registry.count() → total number of tools

== HOW TO INTERACT ==
1. Think about what the user needs
2. Write Python code in ```repl blocks to explore available tools
3. You will see the output of your code, then continue reasoning
4. When you've found the right tool, provide your final answer

== EXAMPLES ==

User: "What's the weather in Tokyo?"

THINK: The user wants weather info. Let me search for weather tools.
```repl
results = tools_registry.search("weather")
print(results)
```

[REPL OUTPUT]:
[{"name": "get_weather", "description": "Get current weather for a city"}]

THINK: Found get_weather. Let me get the full schema to set the parameters correctly.
```repl
schema = tools_registry.get_schema("get_weather")
print(schema)
```

[REPL OUTPUT]:
{"name": "get_weather", "description": "Get current weather for a city", 
 "inputSchema": {"type": "object", "properties": {"city": {"type": "string"}}, 
 "required": ["city"]}}

FINAL({"tool": "get_weather", "params": {"city": "Tokyo"}})

---

User: "What tools do I have for data analysis?"

THINK: The user is asking a discovery question. Let me search broadly.
```repl
results = tools_registry.search("data")
print(f"Found {len(results)} data-related tools:")
for r in results:
    print(f"  - {r['name']}: {r['description']}")
```

[REPL OUTPUT]:
Found 3 data-related tools:
  - query_database: Execute SQL queries on connected databases
  - analyze_csv: Analyze a CSV file and return summary statistics
  - create_chart: Create a chart from data

FINAL(Here are the data-related tools:\n1. query_database — Execute SQL queries\n2. analyze_csv — Analyze CSV files\n3. create_chart — Create charts from data)

== IMPORTANT RULES ==
- Always use the REPL to explore tools before answering
- Never guess tool names — always search first
- Use FINAL(...) when you have your answer
- Keep your REPL code simple — use the provided functions


### Should You Iterate the System Prompt Across Testing Levels?

**YES — the prompt evolves with each level:**

| Level | System Prompt Changes |
|-------|----------------------|
| **Level 0 (zero-shot, no REPL)** | No REPL instructions. Just: "Here are the tools: {full_list}. Select the right tool for: {query}" |
| **Level 1 (constrained REPL)** | Full prompt above — only predefined functions |
| **Level 2 (simple Python)** | Add: "You can also write basic Python (loops, conditionals, `re` module) to process results" |
| **Level 3 (full REPL)** | Use RLM paper's original prompt adapted for MCP context |

**Iteration strategy:**
1. Test prompt V1 → analyze failure modes
2. If SLM fails to use `search()`, add more examples of search patterns
3. If SLM generates code outside allowed functions, add explicit constraint
4. If SLM forgets to use FINAL(), add emphasis + more examples
5. Track prompt versions in a `prompts/` directory for reproducibility

---

## Q4: How to Evaluate — Metrics and Decision Framework

### Evaluation Metrics

| Metric | What It Measures | How to Compute | Threshold for "Success" |
|--------|-----------------|----------------|------------------------|
| **Tool Selection Accuracy (TSA)** | Did the SLM pick the correct tool? | `correct_selections / total_queries` | ≥70% = promising; ≥85% = excellent |
| **Parameter Correctness (PC)** | Are the tool call parameters correct? | Exact match or semantic match of params | ≥60% = acceptable; ≥80% = excellent |
| **REPL Code Validity (RCV)** | Does the generated code execute without errors? | `successful_executions / total_code_blocks` | ≥80% = Level 2 viable; ≥95% = fine-tuning not critical |
| **End-to-End Task Completion (E2E)** | Full pipeline: correct tool + correct params | `fully_correct / total_queries` | ≥50% = publishable; ≥70% = strong paper |
| **REPL Turns Used** | Efficiency — how many turns to reach answer | Average turns across all queries | ≤3 = efficient; >5 = needs optimization |
| **Context Tokens Saved** | How much context does REPL approach save vs. loading all tool descriptions? | `1 - (tokens_used / total_tool_description_tokens)` | ≥50% = meaningful savings |
| **Timeout/Failure Rate** | How often does the SLM fail to produce FINAL() | `timeout_runs / total_runs` | ≤20% = acceptable; ≤5% = excellent |

### Decision Framework: "Do We Need Fine-Tuning?"

```
┌─────────────────────────────────────────┐
│  Run Level 0 (zero-shot, no REPL)       │
│  TSA ≥ 70%?                             │
│    YES → REPL may not be needed for     │
│          simple selection. Test scaling. │
│    NO  → Expected. Move to Level 1.     │
├─────────────────────────────────────────┤
│  Run Level 1 (constrained REPL)         │
│  RCV ≥ 80% AND TSA ≥ 50%?              │
│    YES → SLM can use constrained REPL!  │
│          Try Level 2 next.              │
│    NO  → Check: is it code failures     │
│          (RCV low) or wrong tools       │
│          (TSA low)?                      │
│          → Code failures: need fine-    │
│            tuning on REPL trajectories  │
│          → Wrong tools: need better     │
│            prompt or few-shot examples  │
├─────────────────────────────────────────┤
│  Run Level 2 (few-shot REPL)            │
│  RCV ≥ 80% AND TSA ≥ 60%?              │
│    YES → Few-shot is sufficient for     │
│          baseline. Fine-tune for boost. │
│    NO  → Fine-tuning is REQUIRED.       │
│          Proceed to SFT pipeline.       │
├─────────────────────────────────────────┤
│  After Fine-Tuning:                     │
│  TSA ≥ 85% AND E2E ≥ 70%?              │
│    YES → Strong result. Try depth=1.    │
│    NO  → Consider GRPO refinement or    │
│          DisCIPL planner-follower.       │
└─────────────────────────────────────────┘
```

### Evaluation Script Design

```python
# === evaluator.py (skeleton) ===

class MCPToolEvaluator:
    """Evaluate SLM performance on MCP tool selection tasks."""
    
    def __init__(self, scaffold: MCPRLMScaffold, test_data: dict):
        self.scaffold = scaffold
        self.test_data = test_data  # queries + ground truth
    
    def run_evaluation(self) -> dict:
        results = []
        for item in self.test_data:
            query = item["query"]
            expected_tool = item["expected_tool"]
            expected_params = item.get("expected_params", {})
            
            output = self.scaffold.run(query)
            
            result = {
                "query": query,
                "expected_tool": expected_tool,
                "predicted": output["answer"],
                "turns": output["turns"],
                "success": output["success"],
                "code_validity": self._check_code_validity(output["history"]),
                "tool_correct": self._check_tool_selection(output["answer"], expected_tool),
                "params_correct": self._check_params(output["answer"], expected_params),
            }
            results.append(result)
        
        return self._compute_metrics(results)
    
    def _compute_metrics(self, results: list) -> dict:
        n = len(results)
        return {
            "tool_selection_accuracy": sum(r["tool_correct"] for r in results) / n,
            "parameter_correctness": sum(r["params_correct"] for r in results) / n,
            "repl_code_validity": sum(r["code_validity"] for r in results) / n,
            "end_to_end": sum(r["tool_correct"] and r["params_correct"] for r in results) / n,
            "avg_turns": sum(r["turns"] for r in results) / n,
            "failure_rate": sum(not r["success"] for r in results) / n,
            "total_queries": n,
        }
```

---

## Part II: Comprehensive Implementation POA

### Phase 0: Environment Setup (Days 1–2)

| Task | Details | Deliverable |
|------|---------|-------------|
| Install Ollama + Qwen3.5-4B | `ollama pull qwen3.5:4b` | Working local inference |
| Set up Python project structure | See directory structure below | Clean codebase |
| Install dependencies | `pip install mcp pydantic` (minimal) | requirements.txt |
| Create synthetic tool registries | 10/25/50/100 tool JSON files | `test_data/tool_registries/` |
| Create test queries + ground truth | 50+ queries across query styles | `test_data/queries/`, `test_data/ground_truth/` |

**Project structure:**
```
rlm/
├── src/
│   ├── repl_engine.py       # REPL execution engine
│   ├── scaffold.py          # SLM ↔ REPL interaction loop
│   ├── tool_registry.py     # MCPToolRegistry class
│   ├── evaluator.py         # Evaluation metrics
│   └── slm_interface.py     # Ollama/HuggingFace wrapper
├── prompts/
│   ├── level0_zero_shot.txt
│   ├── level1_constrained_repl.txt
│   ├── level2_simple_python.txt
│   └── level3_full_repl.txt
├── test_data/
│   ├── tool_registries/     # JSON files with tool descriptions
│   ├── queries/             # Test queries
│   └── ground_truth/        # Expected tool selections
├── results/
│   └── (evaluation outputs)
├── notebooks/
│   └── (analysis notebooks)
└── configs/
    └── repl_config.yaml     # Configurable parameters
```

### Phase 1: Baseline Testing (Days 3–7)

**Step 1: Level 0 — Zero-Shot (No REPL)**
- Load ALL tool descriptions directly into Qwen3.5-4B's context
- Run all test queries, measure TSA + PC
- Test with 10, 25, 50 tools (find where it breaks)
- **Expected outcome:** Works decently with 10 tools; degrades sharply at 25+

**Step 2: Level 1 — Constrained REPL (Zero-Shot)**
- Use the Level 1 system prompt (with examples)
- Run the scaffold with MCPToolRegistry + predefined functions
- Measure TSA + RCV + E2E + turns
- **Expected outcome:** RCV may be low (SLM may not follow the code pattern well without fine-tuning); some correct selections via search()

**Step 3: Level 2 — Few-Shot REPL**
- Add 3–5 worked examples into the system prompt
- Re-run Level 1 tests
- **Expected outcome:** Improvement over Level 1, especially for simple queries

**Record all baselines in `results/baselines.json`**

### Phase 2: Analysis & Iteration (Days 7–10)

- Analyze failure modes: Is the SLM failing on **code generation** or **tool selection reasoning**?
- Iterate system prompt based on observed failures
- Try with larger context (if using Phi-3-mini with 128K context, test without REPL)
- Run same tests on a second SLM (e.g., Phi-3-mini) for comparison
- Document findings in `research/RESEARCH_LOG.md`

### Phase 3: Decision Point — Fine-Tune or Not? (Day 10)

Apply the decision framework from Q4 above. If fine-tuning is needed:

### Phase 4: Trajectory Generation (Days 11–15)

- Use Groq free tier (Llama-3.3-70B) to generate 5,000–10,000 REPL-MCP trajectories
- Format: `system_prompt → user_query → [THINK → ```repl → output]* → FINAL()`
- Apply APIGen-inspired 3-stage verification:
  1. **Format check**: Is the trajectory well-formed? Has FINAL()?
  2. **Execution check**: Does the REPL code actually execute?
  3. **Semantic check**: Does the selected tool match the query intent?
- Target: 3,000–5,000 verified trajectories (expect ~60% pass rate)

### Phase 5: Fine-Tuning (Days 15–20)

- LoRA fine-tune Qwen3.5-4B on verified trajectories using Unsloth
- Hardware: Colab T4 (primary) / RTX 4070Ti (secondary)
- Hyperparameters: LoRA rank=32, lr=2e-4, epochs=3
- Save checkpoints per epoch, evaluate on held-out test set
- Optionally: GRPO refinement with verifiable rewards

### Phase 6: Evaluation & Scaling (Days 20–25)

- Run all metrics on fine-tuned model across all tool registry sizes
- Compare: baseline vs. fine-tuned vs. depth=0 REPL vs. no REPL
- Test with depth=1 if depth=0 results are strong
- Generate tables and figures for the paper

---

## Key Design Principles (for Future-Proofing)

1. **Everything is configurable**: REPL level, model, tool registry, prompt — all swappable via config
2. **Evaluation is modular**: Same evaluator works across all levels and models
3. **Trajectories are reusable**: Generated trajectories can be used for SFT, GRPO, or distillation
4. **Code is the training data**: The REPL history IS the training trajectory — no separate format needed
5. **Scale incrementally**: 10 → 25 → 50 → 100 tools; Level 1 → 2 → 3; depth=0 → depth=1

---

## Sources

- [Zhang et al., 2025] "Recursive Language Models" — `references/methods/RLM.pdf`, Appendix D (system prompts, REPL architecture)
- [Zhang et al., 2024] "APIGen: Automated Pipeline for Generating Verifiable and Diverse Function-Calling Datasets" — `references/methods/Apigen.pdf`, [arXiv:2406.18518](https://arxiv.org/abs/2406.18518)
- [MCP Python SDK](https://github.com/modelcontextprotocol/python-sdk) — Official Python SDK with FastMCP
- [MCP Reference Servers](https://github.com/modelcontextprotocol/servers) — Official reference MCP servers (Filesystem, GitHub, etc.)
- [MCP Inspector](https://github.com/modelcontextprotocol/inspector) — Tool inspection and debugging
- [MockMCP](https://mockmcp.com) — Mock MCP server generator
- [SLM-Bench](https://arxiv.org/abs/2405.06615) — Comprehensive SLM benchmarking framework
- [MCP Specification — Tools](https://modelcontextprotocol.io/specification/2025-03-26/server/tools) — Official tool schema spec
