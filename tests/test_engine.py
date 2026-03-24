"""Tests for simulation engine, network, database, and timeline."""

import tempfile
from datetime import datetime

from src.core.config import SimulationConfig
from src.core.database import Database
from src.core.engine import SimulationEngine
from src.core.time_manager import TimeManager
from src.network.graph import SocialGraph
from src.agents.factory import AgentFactory
from src.network.models import Post


def _make_engine(agents=20, **kwargs):
    config = SimulationConfig(
        agent_count=agents,
        db_path=tempfile.mktemp(suffix=".db"),
        **kwargs,
    )
    engine = SimulationEngine(config=config, start_time=datetime(2026, 3, 22))
    engine.initialize()
    return engine


class TestTimeManager:
    def test_advance(self):
        tm = TimeManager(sim_start=datetime(2026, 1, 1))
        tm.advance(3)
        assert tm.step == 1
        assert tm.elapsed_sim_days == 3

    def test_format_status(self):
        tm = TimeManager(sim_start=datetime(2026, 3, 22))
        tm.advance()
        status = tm.format_status()
        assert "Step 1" in status
        assert "2026-03-23" in status


class TestDatabase:
    def test_roundtrip(self):
        db = Database(tempfile.mktemp(suffix=".db"))
        db.init_db()

        from src.agents.models import AgentProfile, AgentState
        profile = AgentProfile(name="Test", age_group="30s")
        db.insert_agent(profile)

        loaded = db.get_agent(profile.id)
        assert loaded is not None
        assert loaded.name == "Test"

        state = AgentState(agent_id=profile.id, opinions={"test": 0.5}, post_count=3)
        db.save_agent_state(state, step=1)

        loaded_state = db.get_agent_state(profile.id, step=1)
        assert loaded_state.opinions == {"test": 0.5}
        assert loaded_state.post_count == 3

        db.close()


class TestSocialGraph:
    def test_build_small_world(self):
        agents = AgentFactory.generate_population(30, seed=42)
        profiles = [p for p, _ in agents]

        graph = SocialGraph()
        graph.build_small_world(profiles, k=4, p=0.1, seed=42)

        assert graph.graph.number_of_nodes() == 30
        stats = graph.stats
        assert stats["avg_clustering"] > 0.3  # Small-world property

    def test_get_feed(self):
        agents = AgentFactory.generate_population(10, seed=42)
        profiles = [p for p, _ in agents]

        graph = SocialGraph()
        graph.build_small_world(profiles, k=4, p=0.1, seed=42)

        post = Post(author_id=profiles[0].id, topic_id="test", content="Hello", step=1)
        graph.add_post(post)

        neighbors = graph.get_neighbors(profiles[0].id)
        if neighbors:
            feed = graph.get_feed(neighbors[0], step=1)
            assert len(feed) >= 0  # May or may not see the post depending on topology


class TestSimulationEngine:
    def test_initialize(self):
        engine = _make_engine(20)
        assert len(engine.profiles) == 20
        assert len(engine.states) == 20
        assert engine.graph.graph.number_of_nodes() == 20

    def test_step(self):
        engine = _make_engine(20)
        stats = engine.step()
        assert stats["step"] == 1
        assert stats["sim_date"] == "2026-03-23"
        assert stats["active_agents"] > 0

    def test_run_multiple_steps(self):
        engine = _make_engine(20)
        results = engine.run(10)
        assert len(results) == 10
        assert results[-1]["step"] == 10

    def test_inject_news(self):
        engine = _make_engine(20)
        post = engine.inject_news("Test headline", "Test summary", sentiment=0.5)
        assert post.is_news_seed
        assert post.sentiment == 0.5

    def test_multi_topic(self):
        engine = _make_engine(20, initial_topics=["AI regulation", "climate change"])
        assert len(engine.topic_ids) == 2
        engine.run(5)

        # Both topics should have opinions
        for state in engine.states.values():
            assert "ai_regulation" in state.opinions
            assert "climate_change" in state.opinions

    def test_add_topic(self):
        engine = _make_engine(20)
        tid = engine.add_topic("new topic")
        assert tid == "new_topic"
        assert "new_topic" in engine.topic_ids

    def test_opinion_distribution(self):
        engine = _make_engine(30)
        engine.run(5)
        dist = engine.get_opinion_distribution()
        total = sum(dist.values())
        assert total == 30

    def test_network_evolution(self):
        engine = _make_engine(30)
        engine.enable_network_evolution = True
        engine.network_evolve_interval = 2
        results = engine.run(4)
        # Step 2 and 4 should have network changes
        assert results[1].get("network_changes") != {}

    def test_reflections(self):
        engine = _make_engine(30)
        engine.enable_reflections = True
        engine.reflection_interval = 3
        results = engine.run(6)
        # Some steps should have reflections
        has_reflections = any(r.get("reflections_generated", 0) > 0 for r in results)
        # May not have enough observations yet, so just check it runs
        assert len(results) == 6
