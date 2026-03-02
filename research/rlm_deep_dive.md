# Deep Dive: Recursive Language Models (RLMs)
**Paper:** "Recursive Language Models" — Alex L. Zhang

---

## TL;DR
RLMs offload an LLM's input prompt into a Python REPL environment as a variable, letting the model write code to interact with its context (regex search, chunking, filtering) and recursively sub-call LMs on relevant slices. This enables processing of **10M+ tokens** with performance **up to 2× better** than base models and existing scaffolds, while keeping costs comparable. The key limitation: **models without sufficient coding capabilities struggle as RLMs** — which is precisely the challenge for SLMs.

---

## 1. Technical Architecture

### 1a. Core Abstraction: RLM = LM + Environment
The formal definition from the paper:
- A language model `M` takes input `x` and produces output `y = M(x)`
- An RLM additionally has an **environment** `E`, so `y = RLM(M, E, x)`
- The environment `E` is a **Python REPL** where:
  - The input prompt is stored as a string variable (`prompt_text` or similar)
  - The model can execute arbitrary Python code to interact with it
  - A sub-LM calling function is available: `query_model(sub_prompt)` → calls a potentially cheaper/smaller LM

### 1b. The REPL Environment — Implementation Details
🟢 **EMPIRICAL** (from Appendix D of the paper):
- **Environment setup:** The prompt text is loaded into the REPL as a Python string variable before the root LM begins
- **Code execution:** The root LM generates Python code blocks that are executed via `exec()` in the REPL
- **Persistent state:** Variables persist across execution steps — the model can build up intermediate results
- **Sub-LM calls:** A function is injected into the REPL that calls a sub-LM (e.g., GPT-5-mini) on arbitrary text
- **Two types of environments:**
  - **Non-isolated:** Code runs on the same machine via Python `exec()` — suitable for benchmarking
  - **Isolated (sandboxed):** Uses cloud sandboxes (Prime/Modal) for security in production

### 1c. The Root LM vs. Sub-LM Split
🟢 **EMPIRICAL:**
- **Root LM:** The main model that orchestrates the entire process (e.g., GPT-5)
- **Sub-LM:** A potentially cheaper model called recursively on context slices (e.g., GPT-5-mini)
- The paper used **depth=1 recursion** — sub-calls are plain LMs, not RLMs themselves
- The paper suggests deeper recursion as future work but didn't test it
- **Cost optimization:** Using GPT-5 + GPT-5-mini was the sweet spot (capable root + cheap sub-calls)

### 1d. System Prompt
🟢 **EMPIRICAL** (from Appendix D):
- The system prompt is **fixed across ALL experiments** for each model — not tuned per benchmark
- It instructs the model that its prompt is available as a variable in the REPL
- It explains that the model can write Python code to interact with the context
- It explains the sub-LM calling function
- For Qwen3-Coder, an **extra line was added** warning against using too many sub-calls (because Qwen was too liberal with sub-calling)

---

## 2. Key Results — Full Numerical Breakdown

### 2a. Main Benchmark Results (Table 1 from paper)

| Model | Method | CodeQA (23K–4.2M tok) | BrowseComp+ 1K (6M–11M tok) | OOLONG (131K tok) | OOLONG-Pairs (32K tok) |
|-------|--------|-----------|----------------|---------|--------------|
| **Qwen3-Coder** | Base Model | 20.00* | 0.00* | 36.00 | 0.06 |
| | CodeAct+BM25 | 24.00* | 12.66 | 38.00 | 0.28 |
| | Summary Agent | 50.00 | 38.00 | 44.06 | 0.31 |
| | **RLM** | **56.00** ($0.92) | **44.66** ($0.84) | **48.00** ($0.61) | **23.11** ($1.02) |
| | RLM no sub-calls | **66.00** ($0.18) | **46.00** ($0.82) | 43.50 ($0.32) | 17.34 ($1.77) |
| **GPT-5** | Base Model | 24.00* | 0.00* | 44.00 | 0.04 |
| | CodeAct+BM25 | 22.00* | 51.00 | 38.00 | 24.67 |
| | Summary Agent | 58.00 | 70.47 | 46.00 | 0.01 |
| | **RLM** | **62.00** ($0.11) | **91.33** ($0.99) | **56.50** ($0.43) | **58.00** ($0.33) |
| | RLM no sub-calls | 58.00 ($0.18) | 88.00 ($0.44) | 36.00 | 14.00 |

*\* = hit context limits*

### 2b. Critical Ablation: "RLM No Sub-Calls"
🟢 **EMPIRICAL** — This is the most relevant ablation for our SLM project:
- **What it is:** The REPL environment with context as a variable, but NO ability to call sub-LMs
- **Key finding:** It **still outperforms base models significantly** on long context tasks
  - Qwen3-Coder: CodeQA 66.00 vs 20.00 (base) — **3.3× improvement**
  - Qwen3-Coder: BrowseComp+ 46.00 vs 0.00 (base) — from impossible to functional
- **Where it falls short:** Information-dense tasks (OOLONG-Pairs): 17.34 vs 23.11 (with sub-calls)
- **Why this matters for us:** If we can get an SLM to interact with the REPL *at all*, even without sub-calls, we may get significant gains over the base SLM

### 2c. Cost Analysis
🟢 **EMPIRICAL:**
- Median RLM cost is **comparable or cheaper** than base model for very long contexts
- GPT-5 RLM median on BrowseComp+ 1K: **$0.99** vs theoretical $1.50–$2.75 for base model ingesting 6–11M tokens
- **High variance in tail:** RLM costs can spike dramatically for outlier trajectories (long loops, excessive sub-calls)
- Costs scale proportionally to task complexity, not input length

### 2d. Scaling Behavior (Figure 1 analysis)
🟢 **EMPIRICAL:**
- Base LM performance **degrades significantly** as input grows beyond 2^14 tokens
- RLM performance **degrades much slower** — maintains higher accuracy at 10M+ tokens
- The degradation is worse for more complex tasks (constant < linear < quadratic processing cost)
- **Crossover point:** RLMs consistently outperform base models beyond **~16K tokens (2^14)**

---

## 3. Emergent Patterns in RLM Trajectories

These patterns emerge **without explicit training** — the models develop these strategies through the scaffold alone:

### 3a. Filtering Input with Code + Model Priors
🟢 **EMPIRICAL:**
- RLMs use regex queries to search for keywords based on their priors (e.g., searching for "festival", "La Union")
- Common pattern: probe context by printing a few lines → filter based on observations
- This is the key mechanism that keeps costs low — selective context viewing

### 3b. Chunking and Recursive Sub-Calling
🟢 **EMPIRICAL:**
- RLMs chunk their context (e.g., by newlines, by fixed size) and sub-call LMs on each chunk
- **No sophisticated partitioning observed** — just uniform chunking or keyword-based splitting
- Qwen3-Coder tends to make **far more sub-calls** than GPT-5 (thousands vs. selective)

### 3c. Answer Verification via Sub-LMs
🟢 **EMPIRICAL:**
- Some RLMs verify answers by making additional sub-LM calls with small contexts
- This can be beneficial (catches context rot errors) or wasteful (redundant verification loops)
- **Failure mode observed:** Qwen3-Coder on OOLONG reproduced its correct answer 5+ times before choosing the wrong one

### 3d. Variable-Based Output Assembly
🟢 **EMPIRICAL:**
- For long-output tasks, RLMs store sub-LM outputs in REPL variables and stitch them together
- This enables **unbounded output length** beyond the base model's token limit
- Heavily used in OOLONG-Pairs tasks

---

## 4. Relevance to SLM × MCP Project

### 4a. What We Can Reuse Directly
| Component | Adaptability for SLMs | Notes |
|-----------|----------------------|-------|
| REPL as context storage | ✅ High | Core mechanism, doesn't depend on model size |
| Context as a variable | ✅ High | Just a Python string — any model can be told about it |
| Sub-LM calling mechanism | ⚠️ Medium | SLMs may not generate correct sub-call code |
| System prompt structure | ⚠️ Medium | Need simplified version for SLMs |
| Code-based filtering | ❌ Low (without fine-tuning) | Requires model to write correct Python |
| Emergent chunking strategies | ❌ Low (without fine-tuning) | Requires reasoning about problem decomposition |

### 4b. Critical Insight: "No Sub-Calls" May Be Our Best Starting Point
🟡 **ANALYTICAL:**
- The RLM "no sub-calls" ablation already shows massive improvements
- This variant only requires the model to write Python code for context interaction
- **Reduced requirements:** No need to reason about "when to call sub-LMs" — just code execution
- **Implication:** If we can teach an SLM to write basic Python for string manipulation, this alone could enable MCP tool access

### 4c. The Coding Capability Bottleneck — Direct Paper Evidence
🟢 **EMPIRICAL** (Section 5 "Limitations"):
> "Models without sufficient coding capabilities struggle as RLMs"

- The paper explicitly identifies this as a limitation
- They didn't test any SLMs — only GPT-5 and Qwen3-Coder-480B (both frontier models)
- This creates a **clear research gap** that our paper can fill

### 4d. Mapping to MCP Use Case
🟡 **ANALYTICAL:**
- **MCP tool descriptions** are JSON schemas describing available tools (name, parameters, descriptions)
- 50 tool definitions can consume **20,000–25,000 tokens** (based on web search findings)
- For an SLM with 4K context window, this makes MCP completely unusable
- **RLM approach:** Store tool descriptions as a variable in the REPL → SLM writes code to search/filter relevant tools → only loads selected tool descriptions into its working context
- This is architecturally identical to how RLMs handle large corpora in BrowseComp+

---

## 5. Papers to Read Next (from RLM references, prioritized)

| Priority | Paper | Relevance to Our Project |
|----------|-------|--------------------------|
| 🔴 Critical | THREAD (Schroeder et al., 2025) | Recursive spawning for small models — showed 10-50% gains on Llama-3-8b and CodeLlama-7b |
| 🔴 Critical | DisCIPL (Grand et al., 2025) | Planner-Follower pattern — Llama-3.2-1B went from 4% to 87% success |
| 🟠 High | MemGPT (Packer et al., 2024) | OS-inspired memory hierarchy for LLM agents |
| 🟠 High | ReSum (Wu et al., 2025) | Summarization tool for periodic context compression in agents |
| 🟡 Medium | Context Folding (Sun et al., 2025) | Context-folding for long-horizon LLM agents |
| 🟡 Medium | AgentFold (Ye et al., 2025) | Related context management technique |
| 🟡 Medium | CodeAct (Wang et al., 2024) | Executable code actions for LLM agents |
| 🟢 Lower | MemWalker (Chen et al., 2023) | Tree-based memory navigation |

---

## 6. Sources
- [Zhang et al., 2025] "Recursive Language Models" — Primary reference (references/RLM.pdf)
- [Schroeder et al., 2025] "THREAD: Thinking Deeper with Recursive Spawning" — arXiv:2405.17402
- [Grand et al., 2025] "Self-steering language models" — arXiv:2504.07081
- [Packer et al., 2024] "MemGPT: Towards LLMs as Operating Systems" — arXiv:2310.08560
- [Wu et al., 2025] "ReSum: Unlocking Long-Horizon..." — Referenced in RLM paper
- [Wang et al., 2024] "Executable Code Actions Elicit Better LLM Agents" — arXiv:2402.01030
- [Chen et al., 2023] "Walking Down the Memory Maze" — arXiv:2310.05029
