"""
REPLEngine — Executes SLM-generated code in a controlled Python REPL.

Adapted from the RLM paper's REPL architecture (Zhang et al., 2025, Appendix D):
- Variables persist across turns (exec in shared namespace)
- Output is truncated to fit SLM context window
- Code blocks extracted from <code></code> XML tags (CodeAct pattern)
  or ```repl fenced blocks (fallback)
- FINAL() / FINAL_VAR() extraction for output

Design Reference: implementation_poa.md § Q2 (REPL Design)
CodeAct Reference: Wang et al., ICML 2024 — XML delimiters for turn-taking
"""

from __future__ import annotations

import io
import re
import signal
import traceback
from contextlib import redirect_stdout, redirect_stderr
from dataclasses import dataclass, field
from typing import Any


@dataclass
class REPLTurn:
    """Record of a single REPL execution turn."""
    turn: int
    code: str
    output: str
    success: bool
    error: str | None = None


@dataclass
class REPLConfig:
    """Configuration for the REPL engine."""
    max_output_chars: int = 2000      # Truncate print() output
    max_turns: int = 10               # Max interaction turns
    timeout_seconds: int = 30         # Per-execution timeout
    allowed_imports: list[str] = field(default_factory=list)
    allowed_builtins: list[str] = field(default_factory=lambda: [
        "print", "len", "str", "int", "float", "list", "dict",
        "tuple", "set", "bool", "type", "isinstance", "range",
        "enumerate", "sorted", "reversed", "zip", "map", "filter",
        "any", "all", "min", "max", "sum", "abs", "round",
        "True", "False", "None",
    ])

    @classmethod
    def for_level(cls, level: int) -> REPLConfig:
        """Create config appropriate for the given REPL level."""
        if level == 1:
            return cls(
                allowed_imports=[],
                allowed_builtins=[
                    "print", "len", "str", "int", "float", "list", "dict",
                    "True", "False", "None",
                ],
            )
        elif level == 2:
            return cls(
                allowed_imports=["re", "json"],
                allowed_builtins=[
                    "print", "len", "str", "int", "float", "list", "dict",
                    "tuple", "set", "bool", "type", "isinstance", "range",
                    "enumerate", "sorted", "reversed", "zip", "map", "filter",
                    "any", "all", "min", "max", "sum", "abs", "round",
                    "True", "False", "None",
                ],
            )
        else:  # Level 3: full REPL
            return cls(
                allowed_imports=["re", "json", "math", "collections",
                                 "itertools", "functools", "string"],
                max_output_chars=4000,
            )


class REPLEngine:
    """
    Execute SLM-generated code in a controlled Python REPL environment.

    The engine maintains a persistent namespace across turns, allowing
    the SLM to build up state incrementally. Output from print() is
    captured and truncated to fit within the SLM's context budget.
    """

    def __init__(self, config: REPLConfig, initial_namespace: dict[str, Any] | None = None):
        """
        Args:
            config: REPL configuration (level, limits, allowed imports).
            initial_namespace: Variables to pre-load (e.g., tools_registry).
        """
        self.config = config
        self.turn_count = 0
        self.history: list[REPLTurn] = []

        # Build the execution namespace
        # Start with safe builtins
        safe_builtins = {}
        import builtins
        for name in config.allowed_builtins:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)

        self.namespace: dict[str, Any] = {
            "__builtins__": safe_builtins,
        }

        # Pre-load allowed imports into namespace
        for module_name in config.allowed_imports:
            try:
                self.namespace[module_name] = __import__(module_name)
            except ImportError:
                pass

        # Pre-load any initial variables (e.g., tools_registry)
        if initial_namespace:
            self.namespace.update(initial_namespace)

    def execute(self, code: str) -> REPLTurn:
        """
        Execute a code block and return the result.

        Args:
            code: Python code string to execute.

        Returns:
            REPLTurn with output, success status, and optional error.
        """
        self.turn_count += 1

        if self.turn_count > self.config.max_turns:
            turn = REPLTurn(
                turn=self.turn_count,
                code=code,
                output="",
                success=False,
                error="Maximum REPL turns exceeded"
            )
            self.history.append(turn)
            return turn

        stdout_capture = io.StringIO()
        error_msg = None
        success = True

        try:
            with redirect_stdout(stdout_capture):
                exec(code, self.namespace)
            output = stdout_capture.getvalue()
        except Exception as e:
            output = stdout_capture.getvalue()  # Capture any partial output
            error_msg = f"{type(e).__name__}: {str(e)}"
            success = False

        # Truncate output to fit SLM context
        max_chars = self.config.max_output_chars
        if len(output) > max_chars:
            output = output[:max_chars] + f"\n... [TRUNCATED, {len(output)} total chars]"

        if error_msg:
            output += f"\n[ERROR: {error_msg}]"

        turn = REPLTurn(
            turn=self.turn_count,
            code=code,
            output=output,
            success=success,
            error=error_msg,
        )
        self.history.append(turn)
        return turn

    def reset(self, keep_initial: dict[str, Any] | None = None):
        """Reset the REPL state for a new task (preserves config)."""
        self.turn_count = 0
        self.history = []
        # Rebuild namespace
        import builtins
        safe_builtins = {}
        for name in self.config.allowed_builtins:
            if hasattr(builtins, name):
                safe_builtins[name] = getattr(builtins, name)
        self.namespace = {"__builtins__": safe_builtins}
        for module_name in self.config.allowed_imports:
            try:
                self.namespace[module_name] = __import__(module_name)
            except ImportError:
                pass
        if keep_initial:
            self.namespace.update(keep_initial)

    # ── Code Extraction ──────────────────────────────────────────────

    @staticmethod
    def extract_code_blocks(text: str) -> list[str]:
        """
        Extract code blocks from SLM output text.

        Tries XML <code></code> tags first (CodeAct pattern),
        then falls back to ```repl markdown fences.

        Note: When using Ollama stop sequences, the closing </code>
        tag may be stripped from the response. We handle both cases.
        """
        # Primary: XML <code>...</code> tags — with or without closing tag
        # (Ollama's stop sequence consumes </code> from the response)
        xml_pattern = r'<code>\s*\n?(.*?)(?:</code>|$)'
        blocks = re.findall(xml_pattern, text, re.DOTALL)
        if blocks:
            return [block.strip() for block in blocks if block.strip()]

        # Fallback: markdown ```repl/python fences
        md_pattern = r'```(?:repl|python)?\n(.*?)(?:```|$)'
        blocks = re.findall(md_pattern, text, re.DOTALL)
        return [block.strip() for block in blocks if block.strip()]

    @staticmethod
    def extract_final(text: str, namespace: dict | None = None) -> str | None:
        """
        Extract FINAL() or FINAL_VAR() from SLM output.

        Args:
            text: The SLM's output text.
            namespace: REPL namespace for FINAL_VAR resolution.

        Returns:
            The final answer string, or None if not found.
        """
        # Check for FINAL(answer) — greedy to capture full content
        match = re.search(r'FINAL\((.+)\)\s*$', text, re.DOTALL | re.MULTILINE)
        if match:
            return match.group(1).strip()

        # Check for FINAL_VAR(variable_name)
        match = re.search(r'FINAL_VAR\((\w+)\)', text)
        if match and namespace:
            var_name = match.group(1)
            if var_name in namespace:
                return str(namespace[var_name])
            return f"[ERROR: Variable '{var_name}' not found]"

        return None

    # ── Metrics ──────────────────────────────────────────────────────

    def get_code_validity_rate(self) -> float:
        """Fraction of executed code blocks that succeeded."""
        if not self.history:
            return 0.0
        return sum(1 for t in self.history if t.success) / len(self.history)

    def get_total_turns(self) -> int:
        """Total number of REPL turns used."""
        return self.turn_count
