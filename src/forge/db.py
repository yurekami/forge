"""Async SQLite database layer for FORGE."""

from __future__ import annotations

import json
from pathlib import Path

import aiosqlite

from .models import (
    Challenge, EloRating, Review, Solution, Status, Verdict, VerificationResult,
)

SCHEMA = """\
CREATE TABLE IF NOT EXISTS challenges (
    id TEXT PRIMARY KEY,
    title TEXT NOT NULL,
    description TEXT NOT NULL,
    language TEXT NOT NULL DEFAULT 'python',
    test_code TEXT NOT NULL DEFAULT '',
    status TEXT NOT NULL DEFAULT 'pending',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS solutions (
    id TEXT PRIMARY KEY,
    challenge_id TEXT NOT NULL REFERENCES challenges(id),
    perspective TEXT NOT NULL,
    code TEXT NOT NULL,
    explanation TEXT NOT NULL DEFAULT '',
    generation_time_ms REAL DEFAULT 0,
    score REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS reviews (
    id TEXT PRIMARY KEY,
    solution_id TEXT NOT NULL REFERENCES solutions(id),
    reviewer_perspective TEXT NOT NULL,
    score REAL NOT NULL,
    strengths TEXT NOT NULL DEFAULT '[]',
    weaknesses TEXT NOT NULL DEFAULT '[]',
    verdict TEXT NOT NULL DEFAULT '',
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS verdicts (
    id TEXT PRIMARY KEY,
    challenge_id TEXT NOT NULL REFERENCES challenges(id),
    winning_solution_id TEXT,
    final_code TEXT NOT NULL,
    reasoning TEXT NOT NULL,
    verification_passed INTEGER,
    verification_output TEXT,
    verification_error TEXT,
    verification_time_ms REAL,
    created_at TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS elo_ratings (
    perspective TEXT PRIMARY KEY,
    elo REAL NOT NULL DEFAULT 1500.0,
    wins INTEGER NOT NULL DEFAULT 0,
    losses INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,
    challenges_entered INTEGER NOT NULL DEFAULT 0
);
"""


class ForgeDB:
    def __init__(self, db_path: str | Path = "forge.db") -> None:
        self._path = str(db_path)
        self._conn: aiosqlite.Connection | None = None

    async def init(self) -> None:
        self._conn = await aiosqlite.connect(self._path)
        await self._conn.executescript(SCHEMA)
        await self._conn.commit()

    async def close(self) -> None:
        if self._conn:
            await self._conn.close()
            self._conn = None

    @property
    def _db(self) -> aiosqlite.Connection:
        if self._conn is None:
            raise RuntimeError("Database not initialized. Call init() first.")
        return self._conn

    # ── Challenges ──────────────────────────────────────────────

    async def save_challenge(self, c: Challenge) -> None:
        db = self._db
        await db.execute(
            "INSERT INTO challenges VALUES (?,?,?,?,?,?,?)",
            (c.id, c.title, c.description, c.language,
             c.test_code, c.status.value, c.created_at),
        )
        await db.commit()

    async def update_challenge_status(self, cid: str, status: Status) -> None:
        db = self._db
        await db.execute(
            "UPDATE challenges SET status=? WHERE id=?",
            (status.value, cid),
        )
        await db.commit()

    async def get_challenge(self, cid: str) -> Challenge | None:
        db = self._db
        db.row_factory = aiosqlite.Row
        cur = await db.execute("SELECT * FROM challenges WHERE id=?", (cid,))
        row = await cur.fetchone()
        if not row:
            return None
        return Challenge(
            id=row["id"], title=row["title"],
            description=row["description"], language=row["language"],
            test_code=row["test_code"], status=Status(row["status"]),
            created_at=row["created_at"],
        )

    async def list_challenges(self, limit: int = 50) -> list[Challenge]:
        db = self._db
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM challenges ORDER BY created_at DESC LIMIT ?",
            (limit,),
        )
        return [
            Challenge(
                id=r["id"], title=r["title"],
                description=r["description"], language=r["language"],
                test_code=r["test_code"], status=Status(r["status"]),
                created_at=r["created_at"],
            )
            for r in await cur.fetchall()
        ]

    # ── Solutions ───────────────────────────────────────────────

    async def save_solution(self, s: Solution) -> None:
        db = self._db
        await db.execute(
            "INSERT INTO solutions VALUES (?,?,?,?,?,?,?,?)",
            (s.id, s.challenge_id, s.perspective, s.code,
             s.explanation, s.generation_time_ms, s.score, s.created_at),
        )
        await db.commit()

    async def get_solutions(self, challenge_id: str) -> list[Solution]:
        db = self._db
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM solutions WHERE challenge_id=? ORDER BY created_at",
            (challenge_id,),
        )
        return [
            Solution(
                id=r["id"], challenge_id=r["challenge_id"],
                perspective=r["perspective"], code=r["code"],
                explanation=r["explanation"],
                generation_time_ms=r["generation_time_ms"],
                score=r["score"], created_at=r["created_at"],
            )
            for r in await cur.fetchall()
        ]

    async def update_solution_score(self, sid: str, score: float) -> None:
        db = self._db
        await db.execute("UPDATE solutions SET score=? WHERE id=?", (score, sid))
        await db.commit()

    # ── Reviews ─────────────────────────────────────────────────

    async def save_review(self, r: Review) -> None:
        db = self._db
        await db.execute(
            "INSERT INTO reviews VALUES (?,?,?,?,?,?,?,?)",
            (r.id, r.solution_id, r.reviewer_perspective, r.score,
             json.dumps(r.strengths), json.dumps(r.weaknesses),
             r.verdict, r.created_at),
        )
        await db.commit()

    async def get_reviews(self, solution_id: str) -> list[Review]:
        db = self._db
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM reviews WHERE solution_id=? ORDER BY created_at",
            (solution_id,),
        )
        return [
            Review(
                id=r["id"], solution_id=r["solution_id"],
                reviewer_perspective=r["reviewer_perspective"],
                score=r["score"],
                strengths=json.loads(r["strengths"]),
                weaknesses=json.loads(r["weaknesses"]),
                verdict=r["verdict"], created_at=r["created_at"],
            )
            for r in await cur.fetchall()
        ]

    # ── Verdicts ────────────────────────────────────────────────

    async def save_verdict(self, v: Verdict) -> None:
        vr = v.verification
        db = self._db
        await db.execute(
            "INSERT INTO verdicts VALUES (?,?,?,?,?,?,?,?,?,?)",
            (v.id, v.challenge_id, v.winning_solution_id,
             v.final_code, v.reasoning,
             int(vr.passed) if vr else None,
             vr.output if vr else None,
             vr.error if vr else None,
             vr.execution_time_ms if vr else None,
             v.created_at),
        )
        await db.commit()

    async def get_verdict(self, challenge_id: str) -> Verdict | None:
        db = self._db
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM verdicts WHERE challenge_id=? "
            "ORDER BY created_at DESC LIMIT 1",
            (challenge_id,),
        )
        row = await cur.fetchone()
        if not row:
            return None
        vr = None
        if row["verification_passed"] is not None:
            vr = VerificationResult(
                solution_id=row["winning_solution_id"] or "",
                passed=bool(row["verification_passed"]),
                output=row["verification_output"] or "",
                error=row["verification_error"] or "",
                execution_time_ms=row["verification_time_ms"] or 0.0,
            )
        return Verdict(
            id=row["id"], challenge_id=row["challenge_id"],
            winning_solution_id=row["winning_solution_id"],
            final_code=row["final_code"], reasoning=row["reasoning"],
            verification=vr, created_at=row["created_at"],
        )

    # ── ELO Ratings ─────────────────────────────────────────────

    async def get_elo_ratings(self) -> list[EloRating]:
        db = self._db
        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT * FROM elo_ratings ORDER BY elo DESC",
        )
        return [
            EloRating(
                perspective=r["perspective"], elo=r["elo"],
                wins=r["wins"], losses=r["losses"],
                draws=r["draws"],
                challenges_entered=r["challenges_entered"],
            )
            for r in await cur.fetchall()
        ]

    async def _ensure_perspective(
        self, db: aiosqlite.Connection, perspective: str,
    ) -> None:
        await db.execute(
            "INSERT OR IGNORE INTO elo_ratings "
            "(perspective,elo,wins,losses,draws,challenges_entered) "
            "VALUES (?,1500.0,0,0,0,0)",
            (perspective,),
        )

    async def update_elo(
        self, winner: str, loser: str, k: float = 32.0,
    ) -> None:
        db = self._db
        for p in (winner, loser):
            await self._ensure_perspective(db, p)

        db.row_factory = aiosqlite.Row
        cur = await db.execute(
            "SELECT elo FROM elo_ratings WHERE perspective=?", (winner,),
        )
        w_elo = (await cur.fetchone())["elo"]
        cur = await db.execute(
            "SELECT elo FROM elo_ratings WHERE perspective=?", (loser,),
        )
        l_elo = (await cur.fetchone())["elo"]

        expected_winner = 1.0 / (1.0 + 10.0 ** ((l_elo - w_elo) / 400.0))
        expected_loser = 1.0 - expected_winner
        new_w = w_elo + k * (1.0 - expected_winner)
        new_l = l_elo + k * (0.0 - expected_loser)

        await db.execute(
            "UPDATE elo_ratings SET elo=?, wins=wins+1, "
            "challenges_entered=challenges_entered+1 WHERE perspective=?",
            (new_w, winner),
        )
        await db.execute(
            "UPDATE elo_ratings SET elo=?, losses=losses+1, "
            "challenges_entered=challenges_entered+1 WHERE perspective=?",
            (new_l, loser),
        )
        await db.commit()

    async def record_participation(self, *perspectives: str) -> None:
        db = self._db
        for p in perspectives:
            await self._ensure_perspective(db, p)
        await db.commit()
