"""FastAPI server with REST endpoints and WebSocket for real-time updates."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from src.core.config import SimulationConfig
from src.core.engine import SimulationEngine
from src.timeline.fork import ParallelWorldEngine
from src.timeline.manager import TimelineManager

logger = logging.getLogger(__name__)

# --- Pydantic request/response models ---


class NewsRequest(BaseModel):
    headline: str
    summary: str = ""
    sentiment: float = 0.0
    topic_id: str | None = None
    create_fork: bool = True


class TopicRequest(BaseModel):
    name: str


class SimulationStartRequest(BaseModel):
    agents: int = 50
    steps: int = 14
    topic: str = "AI regulation"
    seed: int = 42
    speed: float = 1.0  # Seconds between steps


# --- Global state ---

_engine: SimulationEngine | None = None
_pw_engine: ParallelWorldEngine | None = None
_tm: TimelineManager | None = None
_running = False
_ws_clients: list[WebSocket] = []


def _get_engine() -> SimulationEngine:
    if _engine is None:
        raise RuntimeError("Simulation not initialized. POST /api/simulation/start first.")
    return _engine


def _get_pw() -> ParallelWorldEngine:
    if _pw_engine is None:
        raise RuntimeError("Simulation not initialized.")
    return _pw_engine


# --- App ---

@asynccontextmanager
async def lifespan(app: FastAPI):
    yield
    # Cleanup
    if _engine is not None:
        _engine.db.close()


app = FastAPI(
    title="Parallel World Simulator API",
    description="Real-time social network simulation with timeline forking",
    version="0.2.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# --- REST Endpoints ---


@app.get("/api/status")
def get_status():
    """Get current simulation status."""
    if _engine is None:
        return {"status": "not_started", "message": "POST /api/simulation/start to begin"}

    engine = _get_engine()
    return {
        "status": "running" if _running else "paused",
        "step": engine.time.step,
        "sim_date": engine.time.sim_date_str,
        "agent_count": len(engine.profiles),
        "topics": engine.topic_ids,
        "total_posts": sum(s.post_count + s.reply_count for s in engine.states.values()),
        "network": engine.graph.stats,
        "timelines": len(_tm.timelines) if _tm else 1,
    }


@app.post("/api/simulation/start")
def start_simulation(req: SimulationStartRequest):
    """Initialize and start a new simulation."""
    global _engine, _pw_engine, _tm

    config = SimulationConfig(
        agent_count=req.agents,
        seed=req.seed,
        initial_topics=[req.topic],
    )

    _engine = SimulationEngine(config=config, start_time=datetime(2026, 3, 22))
    _engine.initialize()

    _tm = TimelineManager(_engine.db)
    _pw_engine = ParallelWorldEngine(_engine, _tm)

    return {
        "status": "initialized",
        "agents": len(_engine.profiles),
        "topic": req.topic,
        "network": _engine.graph.stats,
    }


@app.post("/api/simulation/step")
async def step_simulation():
    """Advance simulation by one step."""
    pw = _get_pw()
    result = pw.step_all()

    # Broadcast to WebSocket clients
    await _broadcast({"type": "step", "data": result})

    return result


@app.post("/api/simulation/run")
async def run_simulation(steps: int = 10, speed: float = 1.0):
    """Run multiple steps with optional delay."""
    global _running
    pw = _get_pw()
    _running = True

    results = []
    for i in range(steps):
        if not _running:
            break
        result = pw.step_all()
        results.append(result)
        await _broadcast({"type": "step", "data": result})
        if speed > 0 and i < steps - 1:
            await asyncio.sleep(speed)

    _running = False
    return {"steps_completed": len(results), "final": results[-1] if results else None}


@app.post("/api/simulation/stop")
def stop_simulation():
    """Stop a running simulation."""
    global _running
    _running = False
    return {"status": "stopped"}


@app.get("/api/agents")
def get_agents():
    """Get all agent profiles with current state."""
    engine = _get_engine()
    agents = []
    for pid, profile in engine.profiles.items():
        state = engine.states[pid]
        agents.append({
            "id": pid,
            "name": profile.name,
            "age_group": profile.age_group,
            "occupation": profile.occupation,
            "region": profile.region,
            "personality": {
                "openness": profile.personality.openness,
                "conscientiousness": profile.personality.conscientiousness,
                "extraversion": profile.personality.extraversion,
                "agreeableness": profile.personality.agreeableness,
                "neuroticism": profile.personality.neuroticism,
            },
            "opinions": state.opinions,
            "post_count": state.post_count,
            "reply_count": state.reply_count,
            "emotional_valence": state.emotional_state.valence,
        })
    return {"agents": agents, "count": len(agents)}


@app.get("/api/agents/{agent_id}")
def get_agent(agent_id: str):
    """Get detailed info for a single agent."""
    engine = _get_engine()
    if agent_id not in engine.profiles:
        return {"error": "Agent not found"}

    profile = engine.profiles[agent_id]
    state = engine.states[agent_id]
    return {
        "profile": {
            "id": profile.id,
            "name": profile.name,
            "age_group": profile.age_group,
            "occupation": profile.occupation,
            "region": profile.region,
            "core_values": profile.core_values,
        },
        "state": {
            "opinions": state.opinions,
            "memory": state.memory[-5:],
            "post_count": state.post_count,
            "reply_count": state.reply_count,
            "emotional_state": {
                "anger": state.emotional_state.anger,
                "anxiety": state.emotional_state.anxiety,
                "hope": state.emotional_state.hope,
                "frustration": state.emotional_state.frustration,
                "enthusiasm": state.emotional_state.enthusiasm,
                "valence": state.emotional_state.valence,
            },
        },
        "neighbors": engine.graph.get_neighbors(agent_id),
    }


@app.get("/api/opinions")
def get_opinions(topic_id: str | None = None):
    """Get opinion distribution."""
    engine = _get_engine()
    dist = engine.get_opinion_distribution(topic_id)
    all_ops = [
        s.opinions.get(topic_id or engine.topic_id, 0.0)
        for s in engine.states.values()
    ]
    return {
        "topic": topic_id or engine.topic_id,
        "distribution": dist,
        "mean": sum(all_ops) / len(all_ops) if all_ops else 0,
        "std": _std(all_ops),
        "min": min(all_ops) if all_ops else 0,
        "max": max(all_ops) if all_ops else 0,
    }


@app.post("/api/news/inject")
async def inject_news(req: NewsRequest):
    """Inject a news item, optionally creating a timeline fork."""
    pw = _get_pw()
    engine = _get_engine()

    if req.topic_id and req.topic_id not in engine.topic_ids:
        engine.add_topic(req.topic_id)

    post, fork = pw.inject_news_with_fork(
        headline=req.headline,
        summary=req.summary or req.headline,
        sentiment=req.sentiment,
        create_counterfactual=req.create_fork,
    )

    result = {
        "post_id": post.id,
        "headline": req.headline,
        "fork_created": fork is not None,
    }
    if fork:
        result["fork"] = {
            "timeline_id": fork.timeline.id,
            "description": fork.timeline.description,
        }

    await _broadcast({"type": "news_injected", "data": result})
    return result


@app.post("/api/topics")
def add_topic(req: TopicRequest):
    """Add a new discussion topic."""
    engine = _get_engine()
    topic_id = engine.add_topic(req.name)
    return {"topic_id": topic_id, "all_topics": engine.topic_ids}


@app.get("/api/timelines")
def get_timelines():
    """Get all timelines (parallel worlds)."""
    if _tm is None:
        return {"timelines": {}}
    pw = _get_pw()
    return {"worlds": pw.get_world_summary(), "tree": _tm.get_timeline_tree()}


@app.get("/api/stats")
def get_stats():
    """Get step-by-step statistics."""
    engine = _get_engine()
    return {"stats": engine.step_stats[-50:]}  # Last 50 steps


# --- WebSocket ---


@app.websocket("/ws/live")
async def websocket_live(ws: WebSocket):
    """Real-time simulation updates via WebSocket."""
    await ws.accept()
    _ws_clients.append(ws)
    logger.info("WebSocket client connected (%d total)", len(_ws_clients))

    try:
        # Send initial state
        if _engine is not None:
            await ws.send_json({
                "type": "init",
                "data": {
                    "step": _engine.time.step,
                    "sim_date": _engine.time.sim_date_str,
                    "agents": len(_engine.profiles),
                    "topics": _engine.topic_ids,
                },
            })

        # Keep alive and receive commands
        while True:
            data = await ws.receive_text()
            msg = json.loads(data)

            if msg.get("type") == "step":
                result = _get_pw().step_all()
                await _broadcast({"type": "step", "data": result})
            elif msg.get("type") == "inject_news":
                pw = _get_pw()
                post, fork = pw.inject_news_with_fork(
                    headline=msg.get("headline", ""),
                    summary=msg.get("summary", ""),
                    sentiment=msg.get("sentiment", 0.0),
                )
                await _broadcast({
                    "type": "news_injected",
                    "data": {"headline": msg.get("headline"), "fork": fork is not None},
                })

    except WebSocketDisconnect:
        _ws_clients.remove(ws)
        logger.info("WebSocket client disconnected (%d remaining)", len(_ws_clients))


async def _broadcast(message: dict) -> None:
    """Send a message to all connected WebSocket clients."""
    disconnected = []
    for ws in _ws_clients:
        try:
            await ws.send_json(message)
        except Exception:
            disconnected.append(ws)
    for ws in disconnected:
        _ws_clients.remove(ws)


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
