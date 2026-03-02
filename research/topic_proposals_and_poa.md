# Topic Proposals, Plan of Action, and Feasibility Assessment

**Date:** 2026-03-01  
**Project:** Scaling MCP Access for SLMs via the REPL Approach  
**Team:** TY Engineering Students (M1 Mac Air + RTX 4070Ti + Colab T4)

---

## Part I: Paper Topic Proposals

### Topic 1 (RECOMMENDED): "REPL-Augmented Tool Access for Small Language Models: Enabling MCP Interaction Beyond Context Limits"

**One-liner:** Fine-tune SLMs to use a REPL environment with predefined helper functions to access MCP-server tools that would otherwise overflow their context windows.

**Why this is a sure-shot topic:**
| Criterion | Evidence |
|-----------|----------|
| **Clear research gap** | RLM paper tested only frontier models (GPT-5, Qwen3-Coder-480B). No one has tested SLMs. |
| **Strong prior work to build on** | RLM's "no sub-calls" ablation already shows 3.3× gains — provides baseline architecture |
| **Feasible with our resources** | QLoRA fine-tuning of 3B models possible on T4/RTX 4070Ti via Unsloth |
| **High impact** | MCP is a rapidly growing standard; SLMs are the practical deployment target |
| **Empirically testable** | Can measure tool selection accuracy, context utilization, and task completion |
| **Timely** | MCP launched late 2024, RLM paper is 2025 — this is cutting-edge |

**Core contributions the paper would make:**
1. **First study** of RLM-based context externalization for SLMs in the tool-use domain
2. **REPL-MCP bridge architecture** — predefined helper functions for SLMs to navigate MCP tool descriptions
3. **Synthetic trajectory generation pipeline** for training SLMs on REPL-based tool interaction
4. **Empirical analysis** of the minimum model size threshold for effective REPL interaction
5. **Comparison** across multiple SLM families (Phi, Qwen, Gemma) and sizes (1B, 3B, 7B)

---

### Topic 2 (Alternative): "Constrained REPL Interaction Patterns for Resource-Efficient MCP Agents"

**One-liner:** Design a constrained set of REPL interaction templates that SLMs can navigate without generating open-ended code, enabling MCP tool use on edge devices.

**Differentiation from Topic 1:** This focuses more on the **template/script design** rather than fine-tuning. More systems-oriented, less ML-heavy.

**Pros:** Lower API cost (no synthetic data generation needed), more engineering-focused  
**Cons:** Less novel from a research perspective, harder to publish in top ML venues

---

### Topic 3 (Alternative): "Planner-Follower Architectures for SLM Tool Access: Bridging the Capability Gap with Offline Planning"

**One-liner:** Use a larger "Planner" model (offline) to generate structured interaction plans that a small "Follower" SLM executes for MCP tool access, inspired by DisCIPL.

**Differentiation:** This is purely about the planner-follower split, not about REPL environments specifically. Builds on DisCIPL rather than RLM.

**Pros:** Strong theoretical backing (DisCIPL: 4% → 87%), elegant architecture  
**Cons:** Less directly connected to RLM paper, "offline planning" may not generalize well

---

### **Verdict: Go with Topic 1**
Topic 1 is the strongest because it:
- Directly fills the gap identified in the RLM paper
- Combines multiple proven techniques (REPL externalization + predefined scripts + fine-tuning)
- Has clear, measurable experiments
- Is novel yet grounded in established work
- Can incorporate elements of Topics 2 and 3 as ablations/experiments within the paper

---

## Part II: Plan of Action (POA) Skeleton

### Phase 0: Literature & Background (Weeks 1–2)
- [ ] Complete reading of all papers from the "Papers to Read Next" table in `rlm_deep_dive.md`
- [ ] Summarize each paper's relevance in `research/RESEARCH_LOG.md`
- [ ] Finalize the exact paper framing and scope
- [ ] Document the related work section outline

### Phase 1: Architecture Design (Week 3)
- [ ] Design the REPL-MCP bridge architecture
  - Define the Python REPL environment setup
  - Design the helper function library (`search_tools`, `filter_tools`, `get_tool_schema`, `execute_tool`)
  - Define the interaction protocol (how SLM communicates with REPL)
- [ ] Define the system prompt for SLMs (simplified version of RLM paper's prompt)
- [ ] Design the synthetic trajectory format (input format for fine-tuning)
- [ ] Select target SLMs: Phi-3-mini (3.8B), Qwen2.5-3B, Gemma-2-2B (minimum 3 model families)

### Phase 2: Synthetic Data Generation (Weeks 4–5)
- [ ] Set up MCP server instance(s) with diverse tool configurations
  - Option A: Use existing open-source MCP servers (GitHub MCP server, filesystem MCP, etc.)
  - Option B: Create synthetic tool description sets of varying sizes (10, 25, 50, 100 tools)
- [ ] Generate synthetic trajectories using a frontier model (via free-tier APIs)
  - Use Groq (Llama-3.1-70B) or Together.ai for trajectory generation
  - Target: 5,000–10,000 trajectories covering:
    - Simple tool selection (pick 1 tool from N)
    - Multi-step tool chains (tool A → tool B → result)
    - Tool discovery queries ("which tools can help me with X?")
  - Each trajectory format: `[system_prompt, user_query] → [thought → code → output → ... → final_answer]`
- [ ] Validate trajectory quality: manually inspect 100+ trajectories, measure code correctness
- [ ] Create train/validation/test splits

### Phase 3: Fine-Tuning (Weeks 5–6)
- [ ] Set up Unsloth + QLoRA pipeline on:
  - Google Colab T4 (primary)
  - RTX 4070Ti (validation/secondary)
- [ ] Fine-tune target models:
  - Phi-3-mini (3.8B) — Microsoft's coding-focused SLM
  - Qwen2.5-3B — Strong multilingual + coding
  - Gemma-2-2B — Google's efficient SLM (minimum size test)
- [ ] Hyperparameter search: LoRA rank (16, 32, 64), learning rate, epochs
- [ ] Save checkpoints and evaluate on validation set after each epoch

### Phase 4: Evaluation & Benchmarking (Weeks 6–7)
- [ ] Design evaluation benchmark:
  - **Tool Selection Accuracy:** Given a query and N tool descriptions, does the SLM select the correct tool?
  - **Parameter Generation:** Does it generate correct tool call parameters?
  - **End-to-End Task Completion:** Can it complete realistic MCP tasks?
  - **Context Efficiency:** How many tokens does the REPL approach save vs. direct loading?
  - **Scaling Analysis:** Performance as a function of tool count (10 → 25 → 50 → 100)
- [ ] Run experiments:
  - Baseline: Direct SLM (all tools in context)
  - Baseline: SLM + BM25 retrieval (retrieve relevant tools)
  - Ours: SLM + REPL (no fine-tuning)
  - Ours: SLM + REPL (with fine-tuning)
  - Ours: SLM + REPL + predefined scripts (with fine-tuning)
  - Ablation: Effect of model size (2B vs 3B vs 7B)
  - Ablation: Effect of training data size (1K vs 5K vs 10K trajectories)
- [ ] Compute cost analysis: tokens used, API cost, latency

### Phase 5: Paper Writing (Weeks 7–9)
- [ ] Draft paper sections:
  - Abstract + Introduction
  - Related Work (RLM, MCP, SLM tool use, context management)
  - Architecture (REPL-MCP bridge, helper functions, system prompt)
  - Data Generation Pipeline
  - Experiments & Results
  - Analysis & Discussion
  - Conclusion + Future Work
- [ ] Create figures and tables
- [ ] Internal review + revisions
- [ ] Target venue: EMNLP / NAACL / ACL Workshop / NeurIPS Workshop

---

## Part III: Feasibility Assessment

### Hardware Resource Analysis

| Resource | Available | Use Case | Sufficient? |
|----------|-----------|----------|-------------|
| M1 Mac Air (8GB) | ✅ | Running quantized models for quick testing, writing code | ✅ For development |
| RTX 4070Ti (12GB) | ✅ | Fine-tuning 3B models, inference of 7B models | ✅ For fine-tuning |
| Colab T4 (16GB free) | ✅ | Primary fine-tuning platform with Unsloth | ✅ For fine-tuning |
| Cloud compute | ❌ | Would be ideal for larger experiments | ⚠️ Limits to 3B models |

**Fine-tuning feasibility (validated):**
- Unsloth + QLoRA can fine-tune 3B models on T4 with 8GB VRAM usage
- Training time: ~2–4 hours for 10K examples
- Supports Phi-3, Qwen2.5, Llama-3.2, Gemma-2

### API Cost Estimation

| Task | API | Model | Estimated Cost |
|------|-----|-------|----------------|
| Trajectory generation (10K) | Groq free tier | Llama-3.1-70B | **$0** (free tier: 14.4K req/day) |
| Trajectory validation | Groq free tier | Llama-3.1-70B | **$0** |
| Frontier model baseline | Together.ai free tier | Llama-3.1-405B | **$0** (limited) |
| RLM trajectory gen (backup) | OpenRouter | Various | **~$5–15** |

**Total estimated cost: $0–$15** 🟢 Feasible

### Software and Tools

| Tool | Purpose | Cost |
|------|---------|------|
| Unsloth | QLoRA fine-tuning | Free (open source) |
| Ollama | Local SLM inference | Free (open source) |
| transformers / PEFT | Model loading, LoRA | Free (open source) |
| MCP SDK (Python) | MCP server setup | Free (open source) |
| Groq API | Trajectory generation | Free tier |
| Weights & Biases | Experiment tracking | Free tier |

### Timeline Feasibility

| Phase | Duration | Parallelizable? | Risk Level |
|-------|----------|----------------|------------|
| Literature review | 2 weeks | No (sequential) | 🟢 Low |
| Architecture design | 1 week | No | 🟢 Low |
| Data generation | 2 weeks | Yes (with fine-tuning setup) | 🟡 Medium |
| Fine-tuning | 1-2 weeks | No | 🟡 Medium |
| Evaluation | 1-2 weeks | Partially | 🟡 Medium |
| Paper writing | 2-3 weeks | Yes (with experiments) | 🟢 Low |
| **Total** | **~8-10 weeks** | | |

### Risk Matrix

| Risk | Probability | Impact | Mitigation |
|------|------------|--------|------------|
| SLM can't learn REPL interaction even after fine-tuning | Medium | High | Fallback to predefined scripts (Solution B) — still publishable |
| Free API tier insufficient for data generation | Low | Medium | Use local Ollama with Llama-3.1-8B as teacher (lower quality) |
| Fine-tuning produces overfitted model | Medium | Medium | Careful data diversity, cross-validation, multiple checkpoints |
| No significant improvement over baselines | Low | High | The "no sub-calls" ablation showed 3.3× improvement even without training — finding holds |
| Hardware fails during training | Low | Low | Colab provides free restart, 4070Ti is backup |

### Key Assumptions
1. Groq free tier remains available (currently: 14,400 requests/day, Llama-3.1-70B)
2. Unsloth supports chosen models (Phi-3, Qwen2.5, Gemma-2 already confirmed)
3. MCP Python SDK allows programmatic tool description extraction
4. The team can allocate ~20 hours/week to the project for 8–10 weeks

---

## Summary Verdict

> **This project is feasible, timely, and addresses a clear research gap.** The combination of the RLM paper's explicit limitation statement ("models without sufficient coding capabilities struggle"), the emergence of MCP as a standard, and recent advances in SLM fine-tuning (DisCIPL, Unsloth, synthetic trajectory training) creates a strong foundation for a publishable paper. The resource requirements (free APIs + consumer GPU + open-source tools) are well within the team's constraints.

---

## Sources
- All sources cited in `rlm_deep_dive.md` and `bottleneck_analysis.md`
- Unsloth documentation — unsloth.ai
- Groq API documentation — groq.com/docs
- MCP Python SDK — github.com/modelcontextprotocol/python-sdk
- BFCL V4 — gorilla.cs.berkeley.edu/leaderboard
