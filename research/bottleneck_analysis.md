# Bottleneck Analysis: Scaling MCP Access for SLMs via RLM

**Date:** 2026-03-01  
**Scope:** Comprehensive analysis of all issues, bottlenecks, and proposed solutions for adapting the RLM framework to enable SLMs to access MCP servers.

---

## 1. The Core Problem Statement

SLMs (1B–7B parameters) cannot effectively use MCP (Model Context Protocol) servers because:

1. **Context window overflow:** MCP tool descriptions (JSON schemas with name, parameters, descriptions) consume 400–500 tokens *per tool*. With 50 tools connected, that's **20,000–25,000 tokens** — exceeding most SLM context windows entirely. 🟢 **EMPIRICAL** — [GitHub/MCP community reports, Anthropic docs]

2. **No room for reasoning:** Even if descriptions fit, the SLM has almost no remaining context for actual conversation, memory, or chain-of-thought reasoning.

3. **Insufficient tool selection capability:** SLMs struggle to select the correct tool from large flat lists, even when they have enough context space. 🟢 **EMPIRICAL** — [BFCL V4, Berkeley Function Calling Leaderboard]

---

## 2. Issues We Aim to Address

### Issue 1: Context Window Saturation from MCP Tools
🟢 **EMPIRICAL**

**The problem:** An MCP server handshake returns all tool definitions upfront. For SLMs with 2K–8K token context windows, the tool descriptions alone fill or exceed capacity.

**Quantified impact:**
- Typical MCP tool description: 400–500 tokens (name + description + inputSchema with JSON Schema)
- 10 tools: ~4,000–5,000 tokens → fills a 4K context entirely
- 50 tools: ~20,000–25,000 tokens → even 32K context models are severely constrained
- The "MCP tax" (system prompt + tool definitions + memory) can consume >16% of context before any user interaction

**RLM-based solution concept:**
- Store ALL tool descriptions as a string variable in the REPL environment
- SLM writes code to search/filter relevant tools (e.g., keyword matching, regex)
- Only the selected tool descriptions are loaded into the SLM's working context
- This is architecturally identical to how RLMs handle BrowseComp+ (6M–11M token corpora)

### Issue 2: SLM Reasoning Limitations for Tool Use
🟢 **EMPIRICAL**

**The problem:** Even with tools fitting in context, SLMs produce lower-quality tool calls than LLMs.

**Evidence:**
- BFCL V4 (Berkeley Function Calling Leaderboard): Smaller models consistently underperform on complex multi-tool scenarios
- TAU-Bench: Even state-of-the-art agents struggle with compound requests requiring policy adherence
- SLMs particularly struggle with: parallel function calls, nested tool dependencies, and relevance detection

**Why this matters:** Our solution doesn't just need to fit tools in context — we need the SLM to actually *reason* about which tools to call and with what parameters.

### Issue 3: Effective Context Utilization
🟡 **ANALYTICAL**

**The problem:** Simply extending an SLM's context window (e.g., via position interpolation, YaRN) doesn't solve the quality issue. Longer contexts suffer from "context rot" — performance degrades as context length increases.

**Evidence:**
- [Hong et al., 2025] "Context Rot" — performance degrades even for frontier models at long contexts
- RLM paper Figure 1: GPT-5 performance degrades significantly beyond 2^14 tokens; the problem is worse for SLMs
- Training short-context models to handle longer contexts typically requires expensive continued pretraining

---

## 3. Bottlenecks We Will Face

### Bottleneck 1: SLMs Lack Coding Capability for REPL Interaction ⭐ (PRIMARY)
🟢 **EMPIRICAL**

**The problem:** The RLM framework requires the model to write Python code to interact with its context. SLMs generally have weaker code generation capabilities.

**Evidence:**
- RLM paper Section 5: "Models without sufficient coding capabilities struggle as RLMs"
- Phi-1 (1.3B): 50% on HumanEval — basic Python only
- Phi-3 Mini (3.8B): Stronger, but still limited for complex REPL interactions
- Qwen2.5 (7B): Better coding but at the upper end of "small"
- Models below 3B: Typically <40% on HumanEval — insufficient for reliable REPL code

**Specific failure modes in RLM context:**
- Generating syntactically incorrect Python
- Incorrect regex patterns for context filtering
- Failure to handle edge cases in string manipulation
- Inability to decompose problems into sub-tasks programmatically

### Bottleneck 2: SLMs Cannot Create Proper Chain-of-Thought for RLM Navigation
🟢 **EMPIRICAL**

**The problem:** The RLM scaffold requires the model to reason about *what* to search for, *how* to chunk context, and *when* to call sub-LMs. This requires meta-cognitive capabilities that SLMs typically lack.

**Evidence:**
- The RLM paper shows that even Qwen3-Coder-480B makes suboptimal decisions (e.g., excessive sub-calls, redundant verification)
- SLMs typically struggle with multi-step planning and self-monitoring
- Without proper CoT, the SLM may enter infinite loops, make redundant operations, or miss relevant information

### Bottleneck 3: Sub-LM Calling Adds Another Layer of Complexity
🟡 **ANALYTICAL**

**The problem:** Full RLM requires the model to decide when to delegate to sub-LMs and what to query them about. This is a higher-order reasoning task even frontier models struggle with.

**Mitigation:** Use the "no sub-calls" ablation as a starting point — this already shows strong performance gains (3.3× for Qwen3-Coder on CodeQA) without requiring sub-call reasoning.

### Bottleneck 4: Generating Synthetic Training Data for Fine-Tuning
🟡 **ANALYTICAL**

**The problem:** To fine-tune an SLM for REPL-based MCP interaction, we need training data consisting of correct RLM trajectories. This data doesn't exist for the MCP use case.

**Challenges:**
- Need a frontier model (GPT-5, Claude) to generate correct trajectory examples
- Each trajectory includes: thought → code → execution output → next thought → ... → final answer
- MCP-specific trajectories need realistic tool schemas and user queries
- API costs for generating thousands of trajectories could be significant

**Mitigation strategies:**
- Use free-tier APIs (Groq, Together.ai) for generating simpler trajectories
- Focus on a narrow domain (e.g., 5–10 specific MCP tools rather than arbitrary tools)
- Use the DisCIPL "Planner-Follower" pattern: have a frontier model create a *plan template* that the SLM follows

### Bottleneck 5: Evaluation and Benchmarking
🟡 **ANALYTICAL**

**The problem:** There is no existing benchmark specifically for "SLM + MCP via REPL." We need to create evaluation protocols.

**Options:**
- Adapt existing tool-use benchmarks (BFCL, ToolBench) to our REPL-based setup
- Create a custom benchmark with varying numbers of MCP tools and query complexity
- Measure: accuracy of tool selection, parameter generation correctness, end-to-end task completion

### Bottleneck 6: Hardware and Resource Constraints
🟢 **EMPIRICAL**

**Available resources:**
- M1 Mac Air (8GB) — can run quantized models up to ~3B via Ollama
- RTX 4070Ti (12GB VRAM) — can fine-tune models up to ~3B with QLoRA via Unsloth
- Google Colab free tier (T4, 16GB VRAM) — can fine-tune up to ~3B with QLoRA
- No cloud compute budget

**Practical limits:**
- **Max model size for fine-tuning:** ~3B parameters (QLoRA with Unsloth)
- **Max model size for inference:** ~7B (4-bit quantized on RTX 4070Ti)
- **Training time:** ~2–4 hours for 3B model with 10K examples on T4 (Unsloth benchmarks)
- **API costs for data generation:** Must use free-tier APIs (Groq: Llama-3.1-70B free, Together.ai limited free)

---

## 4. Evaluation of User's Proposed Solutions

### Solution A: Fine-Tune SLM on Synthetic RLM Trajectories
🔵 **INTUITION** (user's idea) → 🟡 ANALYTICAL (our assessment)

**Concept:** Use a frontier model to generate correct RLM trajectories for MCP use cases, then fine-tune an SLM on these trajectories so it learns the proper CoT pattern.

**Assessment: ✅ PROMISING — This is likely our best approach**

**Supporting evidence:**
- DisCIPL (Grand et al., 2025) showed Llama-3.2-1B went from 4% → 87% with planner-generated programs — proves SLMs *can* learn structured interaction patterns
- THREAD (Schroeder et al., 2025) showed 10–50% gains for small models (Llama-3-8b, CodeLlama-7b)
- Fine-tuned OPT-350M achieved 77.55% on ToolBench, surpassing ChatGPT — proves tiny models can learn tool use
- Unsloth + QLoRA on T4 can fine-tune 3B models on 10K+ examples in ~2–4 hours — feasible with our hardware
- RLM paper Section 5 explicitly suggests: "training models to be used as RLMs... could provide additional performance improvements"

**Practical implementation path:**
1. Use Groq free tier (Llama-3.1-70B) to generate ~5,000–10,000 RLM trajectories for MCP tool selection
2. Format as instruction-response pairs with the "Thought + Code" pattern
3. Fine-tune Phi-3-mini (3.8B) or Qwen2.5-3B using QLoRA with Unsloth on Colab T4
4. Evaluate on custom MCP tool selection benchmark

**Risks:**
- Quality of synthetic trajectories depends on the frontier model's quality
- Domain coverage: narrow trajectories may not generalize well
- Overfitting risk with small datasets

### Solution B: Predefined Python Scripts (No Open-Ended Code Generation)
🔵 **INTUITION** (user's idea) → 🟡 ANALYTICAL (our assessment)

**Concept:** Instead of having the SLM write arbitrary Python code, provide predefined scripts/functions that the SLM can call with specific arguments. The SLM would choose which script to call, not write code from scratch.

**Assessment: ✅ PROMISING — Good fallback / hybrid approach**

**Supporting evidence:**
- This is conceptually similar to the "Tool Search Tool" pattern Anthropic introduced for MCP
- Reduces the coding requirement to function *selection* rather than function *generation*
- Aligns with how BFCL benchmarks evaluate tool calling — select the right function with right params

**Practical implementation:**
- Create a library of pre-built REPL helper functions:
  - `search_tools(keyword)` → returns tool descriptions matching keyword
  - `filter_tools(category)` → returns tools in a specific category
  - `get_tool_schema(tool_name)` → returns full schema for a specific tool
  - `execute_tool(tool_name, params)` → calls the tool with parameters
- SLM only needs to generate function call instructions, not raw Python code
- This is a **constrained version** of the RLM approach — easier for SLMs to learn

**Risks:**
- Less flexible than full RLM — can't handle novel patterns
- Requires upfront engineering of the helper function library
- May not generalize well to arbitrary MCP servers

### Solution C (NEW): DisCIPL-Style Planner-Follower Architecture
🟡 **ANALYTICAL** — synthesized from our research

**Concept:** Use a larger model as a "Planner" that generates a structured inference program (plan) for MCP tool interaction, and the SLM as a "Follower" that executes this plan step-by-step.

**Supporting evidence:**
- DisCIPL showed Llama-3.2-1B went from 4% → 87% success with this pattern
- The plan constrains the SLM's behavior, preventing failure modes like infinite loops
- The plan can be generated once per MCP server configuration and reused

**Practical advantage:** The planning step doesn't need to be real-time — it can be done offline with a free-tier API when setting up the MCP connection.

---

## 5. Additional Issues/Bottlenecks Identified Beyond User's List

### 5a. MCP Handshake Protocol Limitations
🟡 **ANALYTICAL**

The current MCP spec sends ALL tool definitions during the handshake. There's no built-in mechanism for lazy loading or pagination of tools. Our RLM-based approach needs to intercept this and load tools into the REPL rather than the model's context.

### 5b. Dynamic Tool Discovery vs. Static Tool Caching
🟡 **ANALYTICAL**

MCP tools can change at runtime (servers add/remove tools). Our REPL-based approach needs to handle this — either by periodically refreshing the REPL variable or by event-driven updates.

### 5c. Multi-Turn Context Accumulation
🟡 **ANALYTICAL**

In a real MCP conversation, the context grows over multiple turns (conversation history + tool results). The REPL approach helps with tool descriptions, but conversation history still fills the context. We may need:
- ReSum-style periodic summarization of conversation history
- MemGPT-style tiered memory management

### 5d. Latency Overhead of REPL Interaction
🟡 **ANALYTICAL**

Each REPL interaction (code generation → execution → output reading) adds latency compared to direct tool calling. For SLMs that are already fast at inference, the REPL overhead may negate speed advantages.

### 5e. Security of Code Execution
🟡 **ANALYTICAL**

Running SLM-generated code in a REPL environment has security implications. SLMs' weaker code quality may produce code that:
- Accesses unintended files/resources
- Enters infinite loops consuming resources
- Generates malformed API calls to MCP servers

Sandboxing (as noted in the RLM paper) is essential but adds complexity and cost.

### 5f. The "Goldilocks Zone" for Model Size
🟡 **ANALYTICAL**

There's likely a **minimum model size** below which the RLM approach provides no benefit because the model simply can't generate useful code. Based on available evidence:
- **< 1B params:** Almost certainly too small for any meaningful REPL interaction
- **1B–3B params:** Marginal — may work with heavy constrained generation (predefined scripts) or fine-tuning
- **3B–7B params:** Most promising range — models like Phi-3 Mini (3.8B), Qwen2.5-3B show adequate coding
- **> 7B params:** Increasingly capable but pushing the boundary of "small"

Our paper should empirically determine this threshold.

---

## 6. Solution Priority Ranking

| Priority | Solution | Feasibility | Expected Impact | Resource Cost |
|----------|----------|-------------|-----------------|---------------|
| 1️⃣ | Fine-tune SLM on synthetic RLM trajectories (Solution A) | ✅ High | 🟢 High | Medium (API costs for data gen) |
| 2️⃣ | Predefined helper scripts in REPL (Solution B) | ✅ High | 🟡 Medium | Low (engineering effort) |
| 3️⃣ | DisCIPL-style Planner-Follower (Solution C) | ⚠️ Medium | 🟢 High | Medium |
| 4️⃣ | Hybrid: B+A (scripts + fine-tuning for script selection) | ✅ High | 🟢 High | Medium |
| 5️⃣ | Direct RLM with no modifications | ❌ Low for SLMs | 🔴 Low | Low |

**Recommended approach: Hybrid (Solution B + A)**
1. Build a predefined function library for MCP REPL interaction (Solution B)
2. Fine-tune the SLM to call these functions correctly (Solution A)
3. Optionally add DisCIPL-style planning for complex multi-tool scenarios (Solution C)

---

## Sources
- [Zhang et al., 2025] "Recursive Language Models" — references/RLM.pdf
- [Grand et al., 2025] "Self-steering language models (DisCIPL)" — arXiv:2504.07081
- [Schroeder et al., 2025] "THREAD: Thinking Deeper with Recursive Spawning" — arXiv:2405.17402
- [Hong et al., 2025] "Context Rot" — research.trychroma.com
- [Packer et al., 2024] "MemGPT" — arXiv:2310.08560
- [Wu et al., 2025] "ReSum" — Referenced in RLM paper
- Anthropic MCP Documentation — modelcontextprotocol.io
- Berkeley Function Calling Leaderboard (BFCL V4) — berkeley.edu
- Unsloth Documentation — unsloth.ai
- NVIDIA Synthetic Data Generation — developer.nvidia.com
