"""
SLMInterface — Wrapper for interacting with SLMs via Ollama.

Provides a clean interface for the scaffold to generate completions
from any Ollama-hosted model. Designed to be swappable with HuggingFace
or API-based backends later.
"""

from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass, field
from typing import Any


@dataclass
class SLMConfig:
    """Configuration for the SLM."""
    model_name: str = "qwen2.5-coder:3b"
    temperature: float = 0.1       # Low temp for deterministic tool selection
    max_tokens: int = 2048
    top_p: float = 0.9
    repeat_penalty: float = 1.1
    base_url: str = "http://localhost:11434"


class SLMInterface:
    """
    Interface to a local SLM running via Ollama.

    Uses Ollama's REST API for chat completions. Falls back to
    subprocess if the `ollama` Python package is not installed.
    """

    def __init__(self, config: SLMConfig | None = None):
        self.config = config or SLMConfig()
        self._client = None
        self._init_client()

    def _init_client(self):
        """Try to initialize the Ollama Python client."""
        try:
            import ollama
            self._client = ollama.Client(host=self.config.base_url)
            # Verify the model is available
            try:
                self._client.show(self.config.model_name)
            except Exception:
                print(f"[WARN] Model '{self.config.model_name}' not found. "
                      f"Run: ollama pull {self.config.model_name}")
        except ImportError:
            self._client = None
            print("[WARN] 'ollama' package not installed. "
                  "Install with: pip install ollama")

    def generate(self, messages: list[dict[str, str]]) -> str:
        """
        Generate a completion from the SLM.

        Args:
            messages: Chat messages in OpenAI format:
                      [{"role": "system"|"user"|"assistant", "content": "..."}]

        Returns:
            The assistant's response text.
        """
        if self._client is not None:
            return self._generate_ollama_client(messages)
        else:
            return self._generate_ollama_subprocess(messages)

    def _generate_ollama_client(self, messages: list[dict]) -> str:
        """Generate using the ollama Python client."""
        response = self._client.chat(
            model=self.config.model_name,
            messages=messages,
            options={
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
                "repeat_penalty": self.config.repeat_penalty,
            },
        )
        return response["message"]["content"]

    def _generate_ollama_subprocess(self, messages: list[dict]) -> str:
        """Fallback: generate using ollama CLI via subprocess."""
        import urllib.request

        payload = {
            "model": self.config.model_name,
            "messages": messages,
            "stream": False,
            "options": {
                "temperature": self.config.temperature,
                "num_predict": self.config.max_tokens,
                "top_p": self.config.top_p,
                "repeat_penalty": self.config.repeat_penalty,
            },
        }

        req = urllib.request.Request(
            f"{self.config.base_url}/api/chat",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                result = json.loads(resp.read().decode("utf-8"))
                return result["message"]["content"]
        except Exception as e:
            return f"[SLM ERROR: {type(e).__name__}: {str(e)}]"

    def __repr__(self) -> str:
        return f"SLMInterface(model={self.config.model_name})"
