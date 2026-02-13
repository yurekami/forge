"""FastAPI web server for FORGE arena."""

from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from .db import ForgeDB
from .engine import ForgeEngine
from .models import Challenge, ForgeEvent, Status, _id, _now
from .providers import DemoProvider, Provider, detect_providers, get_provider

STATIC_DIR = Path(__file__).parent / "static"


def create_app(
    db_path: str = "forge.db",
    provider_name: str = "auto",
) -> FastAPI:
    # ── State ───────────────────────────────────────────────
    state: dict[str, Any] = {
        "db": ForgeDB(db_path),
        "provider": None,
        "providers": [],
        "active_streams": {},  # challenge_id -> list[asyncio.Queue]
    }

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        # Startup
        await state["db"].init()
        state["providers"] = await detect_providers()
        if provider_name == "auto":
            state["provider"] = state["providers"][0]
        else:
            state["provider"] = get_provider(state["providers"], provider_name)
        pnames = [p.name for p in state["providers"]]
        print(f"  FORGE providers detected: {pnames}")
        print(f"  Active provider: {state['provider'].name}")
        yield
        # Shutdown
        await state["db"].close()

    app = FastAPI(title="FORGE Arena", version="0.1.0", lifespan=lifespan)

    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    def db() -> ForgeDB:
        return state["db"]

    def provider() -> Provider:
        return state["provider"]

    def get_queues(cid: str) -> list[asyncio.Queue]:
        if cid not in state["active_streams"]:
            state["active_streams"][cid] = []
        return state["active_streams"][cid]

    # ── Dashboard ───────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def dashboard() -> HTMLResponse:
        index = STATIC_DIR / "index.html"
        if not index.exists():
            return HTMLResponse(
                "<h1>FORGE</h1><p>Dashboard not found. "
                "Place index.html in src/forge/static/</p>",
                status_code=200,
            )
        return HTMLResponse(index.read_text(encoding="utf-8"))

    # ── API Endpoints ───────────────────────────────────────

    @app.post("/api/challenges")
    async def create_challenge(request: Request) -> JSONResponse:
        body = await request.json()
        title = body.get("title", "Untitled Challenge")
        description = body.get("description", "")
        language = body.get("language", "python")
        test_code = body.get("test_code", "")

        if not description.strip():
            raise HTTPException(400, "Description is required")
        if len(description) > 10000:
            raise HTTPException(400, "Description too long (max 10000 characters)")
        if len(test_code) > 50000:
            raise HTTPException(400, "Test code too long (max 50000 characters)")

        challenge = Challenge(
            id=_id(),
            title=title,
            description=description,
            language=language,
            test_code=test_code,
        )

        # Start the forge in background
        async def run_forge() -> None:
            engine = ForgeEngine(db(), provider())
            async for event in engine.forge(challenge):
                # Push event to all SSE subscribers
                for q in get_queues(challenge.id):
                    await q.put(event)
                # Small delay for UI animation
                await asyncio.sleep(0.15)
            # Signal end
            for q in get_queues(challenge.id):
                await q.put(None)
            # Cleanup stream queues
            state["active_streams"].pop(challenge.id, None)

        task = asyncio.create_task(run_forge())
        task.add_done_callback(
            lambda t: t.exception() if not t.cancelled() and t.exception() else None
        )

        return JSONResponse(
            {"challenge": challenge.to_dict()},
            status_code=201,
        )

    @app.get("/api/challenges")
    async def list_challenges() -> JSONResponse:
        challenges = await db().list_challenges()
        return JSONResponse(
            {"challenges": [c.to_dict() for c in challenges]},
        )

    @app.get("/api/challenges/{challenge_id}")
    async def get_challenge(challenge_id: str) -> JSONResponse:
        challenge = await db().get_challenge(challenge_id)
        if not challenge:
            raise HTTPException(404, "Challenge not found")

        solutions = await db().get_solutions(challenge_id)
        reviews_map: dict[str, list] = {}
        for s in solutions:
            reviews = await db().get_reviews(s.id)
            reviews_map[s.id] = [r.to_dict() for r in reviews]

        verdict = await db().get_verdict(challenge_id)

        return JSONResponse({
            "challenge": challenge.to_dict(),
            "solutions": [s.to_dict() for s in solutions],
            "reviews": reviews_map,
            "verdict": verdict.to_dict() if verdict else None,
        })

    @app.get("/api/challenges/{challenge_id}/stream")
    async def stream_challenge(
        request: Request, challenge_id: str,
    ) -> EventSourceResponse:
        queue: asyncio.Queue[ForgeEvent | None] = asyncio.Queue()
        get_queues(challenge_id).append(queue)

        async def event_generator():
            try:
                while True:
                    if await request.is_disconnected():
                        break
                    try:
                        event = await asyncio.wait_for(queue.get(), timeout=30.0)
                    except asyncio.TimeoutError:
                        # Send keepalive
                        yield {"event": "keepalive", "data": "ping"}
                        continue

                    if event is None:
                        break
                    yield {
                        "event": event.type,
                        "data": json.dumps({
                            "type": event.type,
                            "data": event.data,
                            "challenge_id": event.challenge_id,
                            "timestamp": event.timestamp,
                        }),
                    }
            finally:
                queues = get_queues(challenge_id)
                if queue in queues:
                    queues.remove(queue)

        return EventSourceResponse(event_generator())

    @app.get("/api/stats")
    async def get_stats() -> JSONResponse:
        ratings = await db().get_elo_ratings()
        challenges = await db().list_challenges()
        return JSONResponse({
            "leaderboard": [r.to_dict() for r in ratings],
            "total_challenges": len(challenges),
            "completed": sum(
                1 for c in challenges if c.status == Status.COMPLETE
            ),
            "provider": state["provider"].name,
            "available_providers": [p.name for p in state["providers"]],
        })

    @app.get("/api/health")
    async def health() -> JSONResponse:
        return JSONResponse({"status": "ok", "version": "0.1.0"})

    return app
