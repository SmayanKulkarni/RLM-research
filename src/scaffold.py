"""
MCPRLMScaffold — Orchestrates the SLM ↔ REPL interaction loop.

This is the main entry point that connects:
  1. An SLM (via SLMInterface)
  2. A REPL engine (REPLEngine)
  3. A system prompt (loaded from file)

The scaffold runs the iterative loop:
  User query → SLM generates thought + code → REPL executes →
  stdout feedback → SLM reasons more → ... → FINAL(answer)

Design Reference: implementation_poa.md § Q2 (Interaction Loop)
RLM Paper Reference: Zhang et al., 2025, Section 3 & Appendix D
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .repl_engine import REPLEngine, REPLConfig
from .slm_interface import SLMInterface, SLMConfig
from .tool_registry import MCPToolRegistry


@dataclass
class ScaffoldResult:
    """Result of a single scaffold run."""
    query: str
    answer: str | None
    turns: int
    success: bool
    repl_history: list[dict]
    conversation: list[dict]
    elapsed_seconds: float
    raw_slm_outputs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "query": self.query,
            "answer": self.answer,
            "turns": self.turns,
            "success": self.success,
            "repl_history": [
                {"turn": h.turn, "code": h.code, "output": h.output, "success": h.success}
                for h in self.repl_history
            ],
            "elapsed_seconds": round(self.elapsed_seconds, 2),
        }


class MCPRLMScaffold:
    """
    Main orchestrator: SLM + REPL interaction loop.

    Supports three testing modes:
        Level 0 (zero-shot):        No REPL — all tools in context
        Level 1 (constrained REPL): REPL with predefined functions only
        Level 2 (few-shot REPL):    REPL with examples in system prompt
    """

    def __init__(
        self,
        slm: SLMInterface,
        repl_engine: REPLEngine,
        system_prompt: str,
    ):
        self.slm = slm
        self.repl = repl_engine
        self.system_prompt = system_prompt

    @classmethod
    def create(
        cls,
        tools_registry: MCPToolRegistry,
        system_prompt: str,
        repl_level: int = 1,
        slm_config: SLMConfig | None = None,
    ) -> MCPRLMScaffold:
        """
        Factory to create a fully configured scaffold.

        Args:
            tools_registry: Pre-built MCPToolRegistry.
            system_prompt: System prompt string (with {num_tools} placeholder).
            repl_level: REPL level (1, 2, or 3).
            slm_config: Optional SLM configuration.
        """
        # Format the system prompt with tool count
        formatted_prompt = system_prompt.replace(
            "{num_tools}", str(tools_registry.count())
        )

        # Create REPL engine with level-appropriate config
        repl_config = REPLConfig.for_level(repl_level)
        repl = REPLEngine(
            config=repl_config,
            initial_namespace={"tools_registry": tools_registry},
        )

        # Create SLM interface
        slm = SLMInterface(config=slm_config or SLMConfig())

        return cls(slm=slm, repl_engine=repl, system_prompt=formatted_prompt)

    def run(self, user_query: str) -> ScaffoldResult:
        """
        Run a single MCP tool selection task.

        The loop:
          1. Send system prompt + user query to SLM
          2. SLM generates text (possibly with ```repl code blocks)
          3. Extract and execute code blocks in REPL
          4. Send REPL output back to SLM as a user message
          5. Repeat until FINAL() is found or max turns reached

        Args:
            user_query: The user's natural language request.

        Returns:
            ScaffoldResult with answer, history, and metrics.
        """
        start_time = time.time()

        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": user_query},
        ]
        raw_outputs = []
        discovered = False
        verified = False
        recovery_hint_count = 0
        repeated_code_count = 0
        last_code_block = None

        for turn in range(self.repl.config.max_turns):
            # Get SLM response
            response = self.slm.generate(conversation)
            raw_outputs.append(response)

            # Step 1: Check for FINAL in the raw response text first
            # (handles pure text responses with FINAL and no code)
            final = REPLEngine.extract_final(response, self.repl.namespace)

            # Step 2: Extract code blocks
            code_blocks = REPLEngine.extract_code_blocks(response)

            # Step 3: If code blocks exist, check if any contain FINAL()
            # (SLM sometimes puts FINAL inside <code> tags)
            if code_blocks:
                for code in code_blocks:
                    if self._is_discovery_code(code):
                        discovered = True
                    if self._is_verify_code(code):
                        verified = True

                    if last_code_block is not None and code.strip() == last_code_block.strip():
                        repeated_code_count += 1
                    else:
                        repeated_code_count = 0
                    last_code_block = code

                for code in code_blocks:
                    code_final = REPLEngine.extract_final(code, self.repl.namespace)
                    if code_final is not None:
                        if not verified and recovery_hint_count == 0:
                            conversation.append({"role": "assistant", "content": response})
                            conversation.append({
                                "role": "user",
                                "content": (
                                    "Before FINAL, verify candidate tool schema using "
                                    "tools_registry.get_schema(tool_name). Then provide FINAL with JSON."
                                ),
                            })
                            recovery_hint_count += 1
                            break

                        return ScaffoldResult(
                            query=user_query,
                            answer=code_final,
                            turns=turn + 1,
                            success=True,
                            repl_history=self.repl.history,
                            conversation=conversation,
                            elapsed_seconds=time.time() - start_time,
                            raw_slm_outputs=raw_outputs,
                        )

                # No FINAL in code → execute code blocks in REPL
                repl_output = ""
                for code in code_blocks:
                    result = self.repl.execute(code)
                    repl_output += f"\n[REPL OUTPUT]:\n{result.output}\n"

                if self._is_empty_or_low_signal_output(repl_output) and recovery_hint_count < 2:
                    conversation.append({"role": "assistant", "content": response})
                    conversation.append({
                        "role": "user",
                        "content": (
                            "Your previous step returned low-signal output. Use a narrower keyword search, "
                            "then run get_schema on the top candidate before FINAL."
                        ),
                    })
                    recovery_hint_count += 1
                    continue

                # Append SLM response + REPL output to conversation
                conversation.append({"role": "assistant", "content": response})
                if repl_output.strip():
                    conversation.append({"role": "user", "content": repl_output.strip()})

                if repeated_code_count >= 2 and recovery_hint_count < 2:
                    conversation.append({
                        "role": "user",
                        "content": (
                            "You are repeating the same code. Move to VERIFY: call get_schema on one "
                            "candidate and then DECIDE with FINAL JSON."
                        ),
                    })
                    recovery_hint_count += 1
                    continue

            elif final is not None:
                # No code blocks, but FINAL found → return answer
                if not verified and recovery_hint_count == 0:
                    conversation.append({"role": "assistant", "content": response})
                    conversation.append({
                        "role": "user",
                        "content": (
                            "Before FINAL, complete VERIFY by checking tool schema with get_schema. "
                            "Then return FINAL with JSON."
                        ),
                    })
                    recovery_hint_count += 1
                    continue

                return ScaffoldResult(
                    query=user_query,
                    answer=final,
                    turns=turn + 1,
                    success=True,
                    repl_history=self.repl.history,
                    conversation=conversation,
                    elapsed_seconds=time.time() - start_time,
                    raw_slm_outputs=raw_outputs,
                )
            else:
                # No code blocks and no FINAL — nudge the SLM
                conversation.append({"role": "assistant", "content": response})
                conversation.append({
                    "role": "user",
                    "content": (
                        "Please write code inside <code></code> tags to search for tools, "
                        "or provide your final answer with FINAL(...)."
                    ),
                })

            if turn >= 5 and recovery_hint_count < 2 and not verified:
                conversation.append({
                    "role": "user",
                    "content": (
                        "You are in DISCOVER mode too long. Transition now: pick top candidate, "
                        "verify with get_schema, then output FINAL JSON."
                    ),
                })
                recovery_hint_count += 1

            if turn >= 7 and not verified and recovery_hint_count >= 2:
                break

        # Max turns reached without FINAL
        return ScaffoldResult(
            query=user_query,
            answer=None,
            turns=self.repl.config.max_turns,
            success=False,
            repl_history=self.repl.history,
            conversation=conversation,
            elapsed_seconds=time.time() - start_time,
            raw_slm_outputs=raw_outputs,
        )

    def run_zero_shot(
        self,
        user_query: str,
        tools_registry: MCPToolRegistry,
    ) -> ScaffoldResult:
        """
        Run Level 0: Zero-shot tool selection (NO REPL).

        All tool descriptions are loaded directly into the prompt.
        This tests whether the SLM can select tools with direct context.
        """
        start_time = time.time()

        # Build tool list for direct context
        tool_summaries = tools_registry.get_tool_names_with_descriptions()
        tools_text = "\n".join(
            f"- {t.name}: {t.description}" for t in tool_summaries
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
            {
                "role": "user",
                "content": (
                    f"Available tools:\n{tools_text}\n\n"
                    f"User request: {user_query}\n\n"
                    f"Select the best tool and provide the parameters. "
                    f"Respond with FINAL({{\"tool\": \"tool_name\", \"params\": {{...}}}})"
                ),
            },
        ]

        response = self.slm.generate(messages)
        final = REPLEngine.extract_final(response, None)

        return ScaffoldResult(
            query=user_query,
            answer=final if final else response,
            turns=1,
            success=final is not None,
            repl_history=[],
            conversation=messages + [{"role": "assistant", "content": response}],
            elapsed_seconds=time.time() - start_time,
            raw_slm_outputs=[response],
        )

    def reset(self, tools_registry: MCPToolRegistry):
        """Reset the REPL for a new task (preserves scaffold config)."""
        self.repl.reset(keep_initial={"tools_registry": tools_registry})

    @staticmethod
    def _is_discovery_code(code: str) -> bool:
        code_lc = code.lower()
        return (
            "tools_registry.search(" in code_lc
            or "tools_registry.list_names(" in code_lc
            or "tools_registry.filter_by_category(" in code_lc
        )

    @staticmethod
    def _is_verify_code(code: str) -> bool:
        return "tools_registry.get_schema(" in code.lower()

    @staticmethod
    def _is_empty_or_low_signal_output(repl_output: str) -> bool:
        text = repl_output.lower()
        if not text.strip():
            return True
        low_signal_tokens = [
            "[]",
            "none",
            "(no output)",
            "0 results",
            "not found",
        ]
        return any(token in text for token in low_signal_tokens)
