"""Multi-model provider abstraction for FORGE.

Supports OpenAI, Anthropic, Ollama, and a built-in demo mode.
Auto-detects available providers from environment variables.
"""

from __future__ import annotations

import json
import os
import time
from abc import ABC, abstractmethod
from typing import Any

import httpx

from .models import PERSPECTIVE_PROMPTS, GENERATION_PROMPT, REVIEW_SYSTEM_PROMPT


class Provider(ABC):
    """Abstract provider for code generation and review."""

    name: str

    @abstractmethod
    async def generate(
        self, perspective: str, challenge_desc: str,
        language: str, test_hint: str,
    ) -> tuple[str, str]:
        """Generate a solution. Returns (code, explanation)."""

    @abstractmethod
    async def review(
        self, reviewer_perspective: str, code: str,
        challenge_desc: str, language: str,
    ) -> dict[str, Any]:
        """Review a solution. Returns {score, strengths, weaknesses, verdict}."""


class OpenAIProvider(Provider):
    """Uses OpenAI chat completions API."""

    name = "openai"

    def __init__(self, api_key: str | None = None, model: str = "gpt-4o-mini") -> None:
        self._key = api_key or os.environ.get("OPENAI_API_KEY", "")
        self._model = model
        self._base = os.environ.get("OPENAI_API_BASE", "https://api.openai.com/v1")

    @property
    def available(self) -> bool:
        return bool(self._key)

    async def _chat(self, system: str, user: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                f"{self._base}/chat/completions",
                headers={"Authorization": f"Bearer {self._key}"},
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "temperature": 0.7,
                    "max_tokens": 4096,
                },
            )
            resp.raise_for_status()
            return resp.json()["choices"][0]["message"]["content"]

    async def generate(
        self, perspective: str, challenge_desc: str,
        language: str, test_hint: str,
    ) -> tuple[str, str]:
        system = PERSPECTIVE_PROMPTS.get(perspective, PERSPECTIVE_PROMPTS["pragmatist"])
        prompt = GENERATION_PROMPT.format(
            language=language, description=challenge_desc, test_hint=test_hint,
        )
        code = await self._chat(system, prompt)
        code = _strip_markdown_fences(code)
        return code, ""

    async def review(
        self, reviewer_perspective: str, code: str,
        challenge_desc: str, language: str,
    ) -> dict[str, Any]:
        system = REVIEW_SYSTEM_PROMPT.format(perspective=reviewer_perspective)
        prompt = (
            f"Challenge ({language}): {challenge_desc}\n\n"
            f"Solution to review:\n```{language}\n{code}\n```"
        )
        raw = await self._chat(system, prompt)
        return _parse_review_json(raw)


class AnthropicProvider(Provider):
    """Uses Anthropic messages API."""

    name = "anthropic"

    def __init__(self, api_key: str | None = None, model: str = "claude-sonnet-4-5-20250929") -> None:
        self._key = api_key or os.environ.get("ANTHROPIC_API_KEY", "")
        self._model = model

    @property
    def available(self) -> bool:
        return bool(self._key)

    async def _chat(self, system: str, user: str) -> str:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self._key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self._model,
                    "max_tokens": 4096,
                    "system": system,
                    "messages": [{"role": "user", "content": user}],
                },
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

    async def generate(
        self, perspective: str, challenge_desc: str,
        language: str, test_hint: str,
    ) -> tuple[str, str]:
        system = PERSPECTIVE_PROMPTS.get(perspective, PERSPECTIVE_PROMPTS["pragmatist"])
        prompt = GENERATION_PROMPT.format(
            language=language, description=challenge_desc, test_hint=test_hint,
        )
        code = await self._chat(system, prompt)
        code = _strip_markdown_fences(code)
        return code, ""

    async def review(
        self, reviewer_perspective: str, code: str,
        challenge_desc: str, language: str,
    ) -> dict[str, Any]:
        system = REVIEW_SYSTEM_PROMPT.format(perspective=reviewer_perspective)
        prompt = (
            f"Challenge ({language}): {challenge_desc}\n\n"
            f"Solution to review:\n```{language}\n{code}\n```"
        )
        raw = await self._chat(system, prompt)
        return _parse_review_json(raw)


class OllamaProvider(Provider):
    """Uses local Ollama API."""

    name = "ollama"

    def __init__(self, model: str = "llama3.1:8b", base_url: str = "http://localhost:11434") -> None:
        self._model = model
        self._base = base_url

    @property
    async def available(self) -> bool:
        try:
            async with httpx.AsyncClient(timeout=3.0) as client:
                resp = await client.get(f"{self._base}/api/tags")
                return resp.status_code == 200
        except Exception:
            return False

    async def _chat(self, system: str, user: str) -> str:
        async with httpx.AsyncClient(timeout=120.0) as client:
            resp = await client.post(
                f"{self._base}/api/chat",
                json={
                    "model": self._model,
                    "messages": [
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                    "stream": False,
                },
            )
            resp.raise_for_status()
            return resp.json()["message"]["content"]

    async def generate(
        self, perspective: str, challenge_desc: str,
        language: str, test_hint: str,
    ) -> tuple[str, str]:
        system = PERSPECTIVE_PROMPTS.get(perspective, PERSPECTIVE_PROMPTS["pragmatist"])
        prompt = GENERATION_PROMPT.format(
            language=language, description=challenge_desc, test_hint=test_hint,
        )
        code = await self._chat(system, prompt)
        code = _strip_markdown_fences(code)
        return code, ""

    async def review(
        self, reviewer_perspective: str, code: str,
        challenge_desc: str, language: str,
    ) -> dict[str, Any]:
        system = REVIEW_SYSTEM_PROMPT.format(perspective=reviewer_perspective)
        prompt = (
            f"Challenge ({language}): {challenge_desc}\n\n"
            f"Solution to review:\n```{language}\n{code}\n```"
        )
        raw = await self._chat(system, prompt)
        return _parse_review_json(raw)


class DemoProvider(Provider):
    """Built-in demo mode that generates template-based solutions.

    Each perspective produces genuinely different code approaches.
    No API keys needed.
    """

    name = "demo"

    _TEMPLATES: dict[str, str] = {
        "architect": '''\
"""Architect's solution — clean abstractions and extensibility."""

from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class Solution(ABC, Generic[T]):
    """Abstract base for the solution pattern."""

    @abstractmethod
    def execute(self, *args, **kwargs) -> T:
        ...

    def validate(self, *args, **kwargs) -> bool:
        return True


class {name}(Solution[{rtype}]):
    """Concrete implementation with clean separation of concerns."""

    def __init__(self):
        self._cache = {{}}

    def execute(self, {params}) -> {rtype}:
        """Main entry point with validation and caching."""
        key = ({param_names},)
        if key in self._cache:
            return self._cache[key]
        if not self.validate({param_names}):
            raise ValueError("Invalid input")
        result = self._compute({param_names})
        self._cache[key] = result
        return result

    def _compute(self, {params}) -> {rtype}:
{body_architect}

    def validate(self, {params}) -> bool:
{validation}


def {func_name}({params}) -> {rtype}:
    """Public API — delegates to Solution pattern."""
    return {name}().execute({param_names})
''',
        "hacker": '''\
"""Hacker's solution — maximum performance, minimal code."""

from functools import lru_cache


@lru_cache(maxsize=None)
def {func_name}({params}) -> {rtype}:
{body_hacker}
''',
        "scientist": '''\
"""Scientist's solution — correctness first, with invariant checks."""

from typing import Final


def {func_name}({params}) -> {rtype}:
    """
    Formally verified implementation.

    Preconditions: {preconditions}
    Postconditions: result satisfies {postconditions}
    Invariant: {invariant}
    """
    # Precondition checks
{precondition_checks}

    # Core computation
{body_scientist}

    # Postcondition verification
{postcondition_checks}
    return result
''',
        "pragmatist": '''\
"""Pragmatist's solution — readable, boring, correct."""


def {func_name}({params}) -> {rtype}:
    """{docstring}"""
{body_pragmatist}
''',
    }

    _DEMO_SOLUTIONS: dict[str, dict[str, dict[str, str]]] = {
        "default": {
            "architect": {
                "name": "Solver",
                "rtype": "object",
                "params": "data",
                "param_names": "data",
                "body_architect": "        return data  # Extensible computation",
                "validation": "        return data is not None",
            },
            "hacker": {
                "body_hacker": "    return data  # O(1) baby",
            },
            "scientist": {
                "preconditions": "data is not None",
                "postconditions": "output type matches input",
                "invariant": "referential transparency",
                "precondition_checks": "    assert data is not None, 'Precondition: data must not be None'",
                "body_scientist": "    result = data",
                "postcondition_checks": "    assert result is not None, 'Postcondition: result must not be None'",
            },
            "pragmatist": {
                "docstring": "Simple, readable implementation.",
                "body_pragmatist": "    # Just do the obvious thing\n    return data",
            },
        }
    }

    async def generate(
        self, perspective: str, challenge_desc: str,
        language: str, test_hint: str,
    ) -> tuple[str, str]:
        # Extract function signature hints from challenge
        func_name = _extract_func_name(challenge_desc)
        params = _extract_params(challenge_desc)
        rtype = _extract_rtype(challenge_desc)

        template = self._TEMPLATES.get(perspective, self._TEMPLATES["pragmatist"])
        defaults = self._DEMO_SOLUTIONS["default"].get(perspective, {})

        code = template.format(
            func_name=func_name,
            name=func_name.title().replace("_", ""),
            params=params,
            param_names=params.split(":")[0].strip() if ":" in params else params,
            rtype=rtype,
            **defaults,
        )
        explanation = f"Generated by {perspective} perspective (demo mode)"
        return code, explanation

    async def review(
        self, reviewer_perspective: str, code: str,
        challenge_desc: str, language: str,
    ) -> dict[str, Any]:
        import random
        base_score = random.uniform(4.0, 9.0)

        perspective_bias = {
            "architect": {"strengths": ["Good structure", "Clean abstractions"],
                          "weaknesses": ["Could use more design patterns", "Missing interface documentation"]},
            "hacker": {"strengths": ["Concise implementation", "Good performance"],
                       "weaknesses": ["Could be shorter", "Missing optimization opportunities"]},
            "scientist": {"strengths": ["Handles edge cases", "Good assertions"],
                          "weaknesses": ["Missing formal invariants", "Needs more boundary checks"]},
            "pragmatist": {"strengths": ["Readable code", "Good naming"],
                           "weaknesses": ["Overly complex", "Could be simpler"]},
        }

        bias = perspective_bias.get(reviewer_perspective, perspective_bias["pragmatist"])
        return {
            "score": round(base_score, 1),
            "strengths": bias["strengths"],
            "weaknesses": bias["weaknesses"],
            "verdict": f"Score: {base_score:.1f}/10 — {reviewer_perspective}'s assessment",
        }


# ── Provider Auto-Detection ────────────────────────────────────

async def detect_providers() -> list[Provider]:
    """Auto-detect available providers. Always includes demo as fallback."""
    providers: list[Provider] = []

    openai = OpenAIProvider()
    if openai.available:
        providers.append(openai)

    anthropic = AnthropicProvider()
    if anthropic.available:
        providers.append(anthropic)

    # Ollama check is async
    ollama = OllamaProvider()
    try:
        async with httpx.AsyncClient(timeout=2.0) as client:
            resp = await client.get("http://localhost:11434/api/tags")
            if resp.status_code == 200:
                providers.append(ollama)
    except Exception:
        pass

    # Always have demo as fallback
    providers.append(DemoProvider())
    return providers


def get_provider(providers: list[Provider], preferred: str = "openai") -> Provider:
    """Get the best available provider."""
    for p in providers:
        if p.name == preferred:
            return p
    return providers[-1]  # fallback to demo


# ── Helpers ─────────────────────────────────────────────────────

def _strip_markdown_fences(text: str) -> str:
    """Remove markdown code fences from LLM output."""
    lines = text.strip().splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].strip() == "```":
        lines = lines[:-1]
    return "\n".join(lines)


def _parse_review_json(raw: str) -> dict[str, Any]:
    """Parse JSON review output from LLM, with fallback."""
    raw = raw.strip()
    # Try to find JSON in the response
    start = raw.find("{")
    end = raw.rfind("}") + 1
    if start >= 0 and end > start:
        try:
            return json.loads(raw[start:end])
        except json.JSONDecodeError:
            pass
    return {
        "score": 5.0,
        "strengths": ["Could not parse structured review"],
        "weaknesses": ["Review format was not valid JSON"],
        "verdict": raw[:200],
    }


def _extract_func_name(desc: str) -> str:
    """Extract a likely function name from challenge description."""
    desc_lower = desc.lower()
    for keyword in ("implement", "write", "create", "build", "make"):
        if keyword in desc_lower:
            idx = desc_lower.index(keyword)
            rest = desc[idx:].split()
            if len(rest) >= 3:
                name = rest[2].strip(".,!?()\"'`")
                return name.lower().replace(" ", "_").replace("-", "_")
    return "solve"


def _extract_params(desc: str) -> str:
    return "data"


def _extract_rtype(desc: str) -> str:
    return "object"
