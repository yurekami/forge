"""Data models for the FORGE arena."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal


def _now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _id() -> str:
    return uuid.uuid4().hex[:12]


class Status(str, Enum):
    PENDING = "pending"
    GENERATING = "generating"
    REVIEWING = "reviewing"
    VOTING = "voting"
    VERIFYING = "verifying"
    COMPLETE = "complete"
    FAILED = "failed"


PERSPECTIVES = (
    "architect",
    "hacker",
    "scientist",
    "pragmatist",
)

PERSPECTIVE_PROMPTS: dict[str, str] = {
    "architect": (
        "You are The Architect. You prioritize clean design, SOLID principles, "
        "clear abstractions, and extensibility. Your code is elegant and well-structured. "
        "You favor design patterns when they clarify intent, and you always think about "
        "how the code will evolve. Type hints and docstrings are non-negotiable."
    ),
    "hacker": (
        "You are The Hacker. You prioritize cleverness, performance, and minimal code. "
        "You find the most efficient solution possible, even if it uses advanced language "
        "features others might not know. You optimize for speed and memory. "
        "Your code is terse but correct. You love one-liners and bit tricks."
    ),
    "scientist": (
        "You are The Scientist. You prioritize correctness, edge case handling, and "
        "formal reasoning. You think about invariants, preconditions, and postconditions. "
        "You add assertions and handle every possible input. Your code reads like a proof. "
        "You always consider: what could go wrong? What are the boundary conditions?"
    ),
    "pragmatist": (
        "You are The Pragmatist. You prioritize readability, maintainability, and "
        "simplicity above all else. You write code that a junior developer can understand "
        "immediately. You use standard library solutions, avoid clever tricks, and prefer "
        "boring, reliable approaches. If the standard library does it, you use it."
    ),
}

REVIEW_SYSTEM_PROMPT = (
    "You are a code reviewer for the FORGE arena. You are reviewing from the "
    "perspective of: {perspective}. Score the solution 0-10 and provide specific "
    "strengths and weaknesses. Be honest and critical. Output ONLY valid JSON:\n"
    '{{"score": <0-10>, "strengths": ["..."], "weaknesses": ["..."], "verdict": "..."}}'
)

GENERATION_PROMPT = (
    "Solve this coding challenge. Output ONLY the code solution — no markdown, "
    "no explanation, no backticks. Just pure, runnable {language} code.\n\n"
    "Challenge: {description}\n\n"
    "Requirements:\n{test_hint}"
)


@dataclass(frozen=True)
class Challenge:
    id: str
    title: str
    description: str
    language: str = "python"
    test_code: str = ""
    status: Status = Status.PENDING
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "title": self.title, "description": self.description,
            "language": self.language, "test_code": self.test_code,
            "status": self.status.value, "created_at": self.created_at,
        }


@dataclass(frozen=True)
class Solution:
    id: str
    challenge_id: str
    perspective: str
    code: str
    explanation: str = ""
    generation_time_ms: float = 0.0
    score: float | None = None
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "challenge_id": self.challenge_id,
            "perspective": self.perspective, "code": self.code,
            "explanation": self.explanation,
            "generation_time_ms": self.generation_time_ms,
            "score": self.score, "created_at": self.created_at,
        }


@dataclass(frozen=True)
class Review:
    id: str
    solution_id: str
    reviewer_perspective: str
    score: float
    strengths: list[str] = field(default_factory=list)
    weaknesses: list[str] = field(default_factory=list)
    verdict: str = ""
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "solution_id": self.solution_id,
            "reviewer_perspective": self.reviewer_perspective,
            "score": self.score, "strengths": self.strengths,
            "weaknesses": self.weaknesses, "verdict": self.verdict,
            "created_at": self.created_at,
        }


@dataclass(frozen=True)
class VerificationResult:
    solution_id: str
    passed: bool
    output: str
    error: str = ""
    execution_time_ms: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "solution_id": self.solution_id, "passed": self.passed,
            "output": self.output, "error": self.error,
            "execution_time_ms": self.execution_time_ms,
        }


@dataclass(frozen=True)
class Verdict:
    id: str
    challenge_id: str
    winning_solution_id: str | None
    final_code: str
    reasoning: str
    verification: VerificationResult | None = None
    created_at: str = field(default_factory=_now)

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id, "challenge_id": self.challenge_id,
            "winning_solution_id": self.winning_solution_id,
            "final_code": self.final_code, "reasoning": self.reasoning,
            "verification": self.verification.to_dict() if self.verification else None,
            "created_at": self.created_at,
        }


@dataclass
class EloRating:
    perspective: str
    elo: float = 1500.0
    wins: int = 0
    losses: int = 0
    draws: int = 0
    challenges_entered: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "perspective": self.perspective, "elo": self.elo,
            "wins": self.wins, "losses": self.losses,
            "draws": self.draws, "challenges_entered": self.challenges_entered,
        }


EventType = Literal[
    "challenge_started",
    "generation_started",
    "solution_generated",
    "review_started",
    "review_complete",
    "voting_started",
    "vote_result",
    "verification_started",
    "verification_complete",
    "forge_complete",
    "forge_error",
]


@dataclass(frozen=True)
class ForgeEvent:
    type: EventType
    data: dict[str, Any]
    challenge_id: str
    timestamp: str = field(default_factory=_now)

    def to_sse(self) -> str:
        import json
        payload = {
            "type": self.type,
            "data": self.data,
            "challenge_id": self.challenge_id,
            "timestamp": self.timestamp,
        }
        return f"data: {json.dumps(payload)}\n\n"
