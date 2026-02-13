"""Core FORGE engine — orchestrates the adversarial code synthesis pipeline.

Flow: Challenge → Generate → Adversarial Review → Vote → Verify → ELO Update
Each step emits ForgeEvents streamed to the dashboard via SSE.
"""

from __future__ import annotations

import asyncio
import subprocess
import sys
import tempfile
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import Any

from .db import ForgeDB
from .models import (
    PERSPECTIVES,
    Challenge,
    EloRating,
    ForgeEvent,
    Review,
    Solution,
    Status,
    Verdict,
    VerificationResult,
    _id,
    _now,
)
from .providers import Provider


class ForgeEngine:
    """Orchestrates the full adversarial code forge pipeline."""

    def __init__(self, db: ForgeDB, provider: Provider) -> None:
        self._db = db
        self._provider = provider

    async def forge(self, challenge: Challenge) -> AsyncGenerator[ForgeEvent, None]:
        """Run a full forge on a challenge, yielding events as they occur."""
        cid = challenge.id

        # ── Stage 1: Challenge Start ──────────────────────────
        await self._db.save_challenge(challenge)
        yield ForgeEvent(
            type="challenge_started",
            data=challenge.to_dict(),
            challenge_id=cid,
        )

        try:
            # ── Stage 2: Generate Solutions ───────────────────
            await self._db.update_challenge_status(cid, Status.GENERATING)
            yield ForgeEvent(
                type="generation_started",
                data={"perspectives": list(PERSPECTIVES)},
                challenge_id=cid,
            )

            solutions: list[Solution] = []
            gen_tasks = [
                self._generate_solution(challenge, perspective)
                for perspective in PERSPECTIVES
            ]
            results = await asyncio.gather(*gen_tasks, return_exceptions=True)

            for result in results:
                if isinstance(result, Exception):
                    yield ForgeEvent(
                        type="forge_error",
                        data={"error": str(result), "stage": "generation"},
                        challenge_id=cid,
                    )
                    continue
                solution = result
                solutions.append(solution)
                await self._db.save_solution(solution)
                yield ForgeEvent(
                    type="solution_generated",
                    data=solution.to_dict(),
                    challenge_id=cid,
                )

            if len(solutions) < 2:
                yield ForgeEvent(
                    type="forge_error",
                    data={"error": "Need at least 2 solutions to proceed"},
                    challenge_id=cid,
                )
                await self._db.update_challenge_status(cid, Status.FAILED)
                return

            # ── Stage 3: Adversarial Review ───────────────────
            await self._db.update_challenge_status(cid, Status.REVIEWING)
            yield ForgeEvent(
                type="review_started",
                data={"solution_count": len(solutions)},
                challenge_id=cid,
            )

            all_reviews: dict[str, list[Review]] = {s.id: [] for s in solutions}
            review_tasks = []

            for solution in solutions:
                for reviewer_perspective in PERSPECTIVES:
                    if reviewer_perspective == solution.perspective:
                        continue  # Don't review your own solution
                    review_tasks.append(
                        self._review_solution(
                            challenge, solution, reviewer_perspective,
                        )
                    )

            review_results = await asyncio.gather(
                *review_tasks, return_exceptions=True,
            )

            for result in review_results:
                if isinstance(result, Exception):
                    continue
                review = result
                all_reviews[review.solution_id].append(review)
                await self._db.save_review(review)
                yield ForgeEvent(
                    type="review_complete",
                    data=review.to_dict(),
                    challenge_id=cid,
                )

            # ── Stage 4: Vote / Score Aggregation ─────────────
            await self._db.update_challenge_status(cid, Status.VOTING)
            yield ForgeEvent(
                type="voting_started",
                data={},
                challenge_id=cid,
            )

            scored_solutions = []
            for solution in solutions:
                reviews = all_reviews.get(solution.id, [])
                if reviews:
                    avg_score = sum(r.score for r in reviews) / len(reviews)
                else:
                    avg_score = 0.0
                await self._db.update_solution_score(solution.id, avg_score)
                scored_solutions.append((solution, avg_score))

            scored_solutions.sort(key=lambda x: x[1], reverse=True)

            vote_data = [
                {
                    "solution_id": s.id,
                    "perspective": s.perspective,
                    "avg_score": round(score, 2),
                    "rank": i + 1,
                }
                for i, (s, score) in enumerate(scored_solutions)
            ]
            yield ForgeEvent(
                type="vote_result",
                data={"rankings": vote_data},
                challenge_id=cid,
            )

            winner_solution, winner_score = scored_solutions[0]

            # ── Stage 5: Verification ─────────────────────────
            await self._db.update_challenge_status(cid, Status.VERIFYING)
            yield ForgeEvent(
                type="verification_started",
                data={"solution_id": winner_solution.id},
                challenge_id=cid,
            )

            verification = await self._verify_solution(
                winner_solution, challenge,
            )
            yield ForgeEvent(
                type="verification_complete",
                data=verification.to_dict(),
                challenge_id=cid,
            )

            # ── Stage 6: Verdict & ELO Update ─────────────────
            verdict = Verdict(
                id=_id(),
                challenge_id=cid,
                winning_solution_id=winner_solution.id,
                final_code=winner_solution.code,
                reasoning=(
                    f"{winner_solution.perspective} wins with avg score "
                    f"{winner_score:.1f}/10. "
                    f"Verification: {'PASSED' if verification.passed else 'FAILED'}."
                ),
                verification=verification,
            )
            await self._db.save_verdict(verdict)

            # Update ELO: winner beats all others
            losers = [s for s, _ in scored_solutions[1:]]
            for loser in losers:
                await self._db.update_elo(
                    winner_solution.perspective,
                    loser.perspective,
                )
            # Record participation for all
            await self._db.record_participation(
                *(s.perspective for s in solutions),
            )

            await self._db.update_challenge_status(cid, Status.COMPLETE)
            yield ForgeEvent(
                type="forge_complete",
                data={
                    "verdict": verdict.to_dict(),
                    "rankings": vote_data,
                    "winner": winner_solution.perspective,
                    "winner_score": round(winner_score, 2),
                },
                challenge_id=cid,
            )

        except Exception as exc:
            await self._db.update_challenge_status(cid, Status.FAILED)
            yield ForgeEvent(
                type="forge_error",
                data={"error": str(exc), "stage": "unknown"},
                challenge_id=cid,
            )

    async def _generate_solution(
        self, challenge: Challenge, perspective: str,
    ) -> Solution:
        """Generate a single solution from a given perspective."""
        t0 = time.perf_counter()

        test_hint = ""
        if challenge.test_code:
            test_hint = (
                "Your solution must pass these tests:\n"
                f"```{challenge.language}\n{challenge.test_code}\n```"
            )

        code, explanation = await self._provider.generate(
            perspective=perspective,
            challenge_desc=challenge.description,
            language=challenge.language,
            test_hint=test_hint,
        )

        elapsed = (time.perf_counter() - t0) * 1000

        return Solution(
            id=_id(),
            challenge_id=challenge.id,
            perspective=perspective,
            code=code,
            explanation=explanation,
            generation_time_ms=round(elapsed, 1),
        )

    async def _review_solution(
        self, challenge: Challenge, solution: Solution,
        reviewer_perspective: str,
    ) -> Review:
        """Have one perspective review another's solution."""
        result = await self._provider.review(
            reviewer_perspective=reviewer_perspective,
            code=solution.code,
            challenge_desc=challenge.description,
            language=challenge.language,
        )

        return Review(
            id=_id(),
            solution_id=solution.id,
            reviewer_perspective=reviewer_perspective,
            score=float(result.get("score", 5.0)),
            strengths=result.get("strengths", []),
            weaknesses=result.get("weaknesses", []),
            verdict=result.get("verdict", ""),
        )

    async def _verify_solution(
        self, solution: Solution, challenge: Challenge,
    ) -> VerificationResult:
        """Verify a solution by executing it with test code in a sandbox."""
        if challenge.language != "python":
            return VerificationResult(
                solution_id=solution.id,
                passed=True,
                output="Verification skipped (non-Python)",
            )

        if not challenge.test_code.strip():
            return VerificationResult(
                solution_id=solution.id,
                passed=True,
                output="No test code provided — verification skipped",
            )

        # Combine solution + test code
        full_code = f"{solution.code}\n\n{challenge.test_code}"

        t0 = time.perf_counter()
        try:
            result = await asyncio.to_thread(
                self._run_sandboxed, full_code,
            )
            elapsed = (time.perf_counter() - t0) * 1000
            return VerificationResult(
                solution_id=solution.id,
                passed=result[0],
                output=result[1],
                error=result[2],
                execution_time_ms=round(elapsed, 1),
            )
        except Exception as exc:
            elapsed = (time.perf_counter() - t0) * 1000
            return VerificationResult(
                solution_id=solution.id,
                passed=False,
                output="",
                error=str(exc),
                execution_time_ms=round(elapsed, 1),
            )

    @staticmethod
    def _run_sandboxed(code: str) -> tuple[bool, str, str]:
        """Run code in a subprocess with timeout and safety restrictions."""
        # Block dangerous imports
        dangerous_patterns = [
            "import os", "from os", "import subprocess", "from subprocess",
            "import shutil", "from shutil", "import socket", "from socket",
            "import ctypes", "from ctypes", "import sys", "from sys",
            "__import__", "exec(", "eval(", "compile(",
            "open(", "import pathlib", "from pathlib",
        ]
        code_lower = code.lower()
        for pattern in dangerous_patterns:
            if pattern.lower() in code_lower:
                return False, "", f"Blocked: dangerous pattern '{pattern}' detected"

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".py", delete=False,
        ) as f:
            f.write(code)
            f.flush()
            tmp_path = f.name

        try:
            kwargs: dict[str, Any] = {
                "capture_output": True,
                "text": True,
                "timeout": 30,
                "cwd": tempfile.gettempdir(),
            }

            # Add resource limits on Unix
            if sys.platform != "win32":
                import resource

                def _limit_resources() -> None:
                    # 256 MB memory limit
                    resource.setrlimit(
                        resource.RLIMIT_AS, (256 * 1024 * 1024, 256 * 1024 * 1024),
                    )
                    # 10 second CPU time limit
                    resource.setrlimit(resource.RLIMIT_CPU, (10, 10))

                kwargs["preexec_fn"] = _limit_resources

            result = subprocess.run(
                [sys.executable, "-I", tmp_path],  # -I for isolated mode
                **kwargs,
            )
            passed = result.returncode == 0
            return passed, result.stdout[:5000], result.stderr[:5000]
        except subprocess.TimeoutExpired:
            return False, "", "Execution timed out (30s limit)"
        finally:
            Path(tmp_path).unlink(missing_ok=True)
