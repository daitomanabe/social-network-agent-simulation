"""FastAPI server with REST endpoints and WebSocket for real-time updates."""

from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
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

# Static files
_static_dir = Path(__file__).parent / "static"


@app.get("/")
def serve_index():
    """Serve the web dashboard."""
    return FileResponse(_static_dir / "index.html")


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


@app.get("/api/polarization")
def get_polarization(topic_id: str | None = None):
    """Get polarization analysis and echo chamber detection."""
    engine = _get_engine()
    tid = topic_id or engine.topic_id

    pol = engine.dynamics.compute_polarization(engine.states, tid)
    chambers = engine.dynamics.detect_echo_chambers(engine.states, tid)

    return {
        "topic": tid,
        "polarization": {
            "index": pol.polarization_index,
            "bimodality": pol.bimodality_coefficient,
            "modularity": pol.modularity,
            "kurtosis": pol.opinion_kurtosis,
            "cross_cluster_edges": pol.cross_cluster_edges_ratio,
        },
        "echo_chambers": [
            {
                "id": ch.id,
                "size": ch.size,
                "mean_opinion": ch.mean_opinion,
                "opinion_std": ch.opinion_std,
                "density": ch.internal_density,
                "label": ch.label,
                "agent_ids": ch.agent_ids[:10],  # Limit for response size
            }
            for ch in chambers
        ],
        "chamber_count": len(chambers),
        "severity": (
            "high" if pol.polarization_index > 0.7
            else "moderate" if pol.polarization_index > 0.4
            else "low"
        ),
    }


@app.get("/api/network/graph")
def get_network_graph():
    """Export the network graph in D3.js-compatible format.

    Returns nodes with agent data and links with weights.
    """
    engine = _get_engine()
    G = engine.graph.graph

    nodes = []
    for node_id in G.nodes():
        profile = engine.profiles.get(node_id)
        state = engine.states.get(node_id)
        if not profile or not state:
            continue
        opinion = state.opinions.get(engine.topic_id, 0.0)
        nodes.append({
            "id": node_id,
            "name": profile.name,
            "group": _opinion_group(opinion),
            "opinion": round(opinion, 3),
            "extraversion": round(profile.personality.extraversion, 2),
            "post_count": state.post_count + state.reply_count,
            "age_group": profile.age_group,
            "occupation": profile.occupation,
        })

    links = []
    for u, v, data in G.edges(data=True):
        links.append({
            "source": u,
            "target": v,
            "weight": round(data.get("weight", 0.5), 3),
            "interactions": data.get("interaction_count", 0),
        })

    return {
        "nodes": nodes,
        "links": links,
        "stats": engine.graph.stats,
    }


@app.get("/api/network/communities")
def get_communities():
    """Get community structure of the network."""
    engine = _get_engine()
    chambers = engine.dynamics.detect_echo_chambers(engine.states, engine.topic_id)

    communities = {}
    for ch in chambers:
        communities[ch.id] = {
            "agent_ids": ch.agent_ids,
            "size": ch.size,
            "mean_opinion": ch.mean_opinion,
            "label": ch.label,
            "density": ch.internal_density,
        }

    return {"communities": communities, "count": len(communities)}


@app.get("/api/memory/{agent_id}")
def get_agent_memory(agent_id: str, n: int = 10):
    """Get an agent's memory stream with reflections."""
    engine = _get_engine()
    if agent_id not in engine.memory_streams:
        return {"error": "Agent not found"}

    stream = engine.memory_streams[agent_id]
    retrieved = stream.retrieve(engine.time.step, engine.topic_id, n=n)

    return {
        "agent_id": agent_id,
        "total_items": len(stream.items),
        "observations": len(stream.observations),
        "reflections_count": len(stream.reflections),
        "memories": [
            {
                "content": item.content,
                "step": item.step,
                "importance": item.importance,
                "is_reflection": item.is_reflection,
                "topic": item.topic_id,
            }
            for item in retrieved
        ],
        "all_reflections": [
            {"content": r.content, "step": r.step}
            for r in stream.reflections
        ],
    }


@app.post("/api/simulation/evolve")
def evolve_network():
    """Trigger one round of network evolution."""
    engine = _get_engine()
    changes = engine.dynamics.evolve_network(engine.states, engine.topic_id, seed=engine.time.step)
    return {"changes": changes}


def _opinion_group(opinion: float) -> int:
    """Map opinion to a group index for visualization coloring."""
    if opinion < -0.6:
        return 0  # strong against
    elif opinion < -0.2:
        return 1  # against
    elif opinion < 0.2:
        return 2  # neutral
    elif opinion < 0.6:
        return 3  # for
    else:
        return 4  # strong for


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
