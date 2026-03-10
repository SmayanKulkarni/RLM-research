"""
MCPToolRegistry — Manages and queries MCP tool descriptions.

This is the core data structure pre-loaded into the REPL environment.
The SLM interacts with tools exclusively through this registry's methods.

Design Reference: implementation_poa.md § Q2 (REPL Design)
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from dataclasses import dataclass, field


@dataclass
class ToolSummary:
    """Lightweight view of a tool (name + description only)."""
    name: str
    description: str

    def __repr__(self) -> str:
        return f'{{"name": "{self.name}", "description": "{self.description}"}}'


class MCPToolRegistry:
    """
    Registry of MCP tool definitions, pre-loaded into the REPL as `tools_registry`.

    Provides constrained methods for the SLM to discover, search, and inspect
    tools WITHOUT loading all tool schemas into context at once.

    Levels of access:
        Level 1 (Constrained REPL): SLM calls these methods directly
        Level 2 (Simple Python):    SLM combines these with basic Python
        Level 3 (Full REPL):        SLM can also access self._tools directly
    """

    def __init__(self, tools: list[dict]):
        """
        Args:
            tools: List of MCP tool JSON schema dicts, each containing
                   at minimum: 'name', 'description', 'inputSchema'.
        """
        self._tools = tools
        # Build name index for O(1) lookups
        self._name_index: dict[str, dict] = {t["name"]: t for t in tools}

    # ── Factory Methods ──────────────────────────────────────────────

    @classmethod
    def from_json_file(cls, path: str | Path) -> MCPToolRegistry:
        """Load a tool registry from a JSON file."""
        with open(path) as f:
            data = json.load(f)
        tools = data if isinstance(data, list) else data.get("tools", [])
        return cls(tools)

    @classmethod
    def from_json_string(cls, json_str: str) -> MCPToolRegistry:
        """Load a tool registry from a JSON string."""
        data = json.loads(json_str)
        tools = data if isinstance(data, list) else data.get("tools", [])
        return cls(tools)

    # ── Level 1: Constrained Functions ───────────────────────────────

    def list_names(self) -> list[str]:
        """Return the names of all available tools."""
        return [t["name"] for t in self._tools]

    def count(self) -> int:
        """Return total number of available tools."""
        return len(self._tools)

    def search(self, query: str) -> list[ToolSummary]:
        """
        Search tools by keyword matching in name + description.

        Args:
            query: Search keyword (case-insensitive).

        Returns:
            List of ToolSummary objects for matching tools.
        """
        query_lower = query.lower()
        results = []
        for t in self._tools:
            searchable = f"{t['name']} {t.get('description', '')}".lower()
            if query_lower in searchable:
                results.append(ToolSummary(
                    name=t["name"],
                    description=t.get("description", "")
                ))
        return results

    def get_schema(self, tool_name: str) -> dict | None:
        """
        Get the full JSON schema for a specific tool by exact name.

        Args:
            tool_name: Exact tool name (case-sensitive).

        Returns:
            Full tool definition dict, or None if not found.
        """
        return self._name_index.get(tool_name)

    def filter_by_category(self, category: str) -> list[ToolSummary]:
        """
        Filter tools by category tag (checks annotations and description).

        Args:
            category: Category keyword (case-insensitive).

        Returns:
            List of ToolSummary objects for matching tools.
        """
        cat_lower = category.lower()
        results = []
        for t in self._tools:
            # Check annotations if present
            annotations = t.get("annotations", {})
            desc = t.get("description", "")
            if cat_lower in str(annotations).lower() or cat_lower in desc.lower():
                results.append(ToolSummary(
                    name=t["name"],
                    description=desc
                ))
        return results

    def get_tool_names_with_descriptions(self) -> list[ToolSummary]:
        """Return name + description for ALL tools (for Level 0 / zero-shot)."""
        return [
            ToolSummary(name=t["name"], description=t.get("description", ""))
            for t in self._tools
        ]

    # ── Utility ──────────────────────────────────────────────────────

    def get_total_tokens_estimate(self, chars_per_token: float = 4.0) -> int:
        """Estimate total tokens if all tool schemas were loaded at once."""
        total_chars = sum(len(json.dumps(t)) for t in self._tools)
        return int(total_chars / chars_per_token)

    def __repr__(self) -> str:
        return f"MCPToolRegistry({self.count()} tools)"
