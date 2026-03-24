"""Tests for timeline management, forking, and Reality Diff."""

import tempfile
from datetime import datetime

from src.core.config import SimulationConfig
from src.core.database import Database
from src.core.engine import SimulationEngine
from src.timeline.fork import ParallelWorldEngine, ForkRunner
from src.timeline.manager import TimelineManager
from src.timeline.reality_diff import RealityDiffEngine


def _make_system(agents=20):
    config = SimulationConfig(agent_count=agents, db_path=tempfile.mktemp(suffix=".db"))
    engine = SimulationEngine(config=config, start_time=datetime(2026, 3, 22))
    engine.initialize()
    tm = TimelineManager(engine.db)
    pw = ParallelWorldEngine(engine, tm)
    return engine, tm, pw


class TestTimelineManager:
    def test_create_snapshot(self):
        engine, tm, pw = _make_system()
        engine.run(5)

        snapshot = tm.create_snapshot(
            tm.main_timeline_id, engine.time.step,
            engine.time.sim_date_str, engine.states, engine.graph,
        )
        assert snapshot.step == 5

    def test_restore_snapshot(self):
        engine, tm, pw = _make_system()
        engine.run(5)

        snapshot = tm.create_snapshot(
            tm.main_timeline_id, engine.time.step,
            engine.time.sim_date_str, engine.states, engine.graph,
        )

        states, edges = tm.restore_snapshot(snapshot.id, engine.profiles)
        assert len(states) == len(engine.states)

        # Verify state matches
        for aid in engine.states:
            assert states[aid].opinions == engine.states[aid].opinions

    def test_create_fork(self):
        engine, tm, pw = _make_system()
        engine.run(5)

        snapshot = tm.create_snapshot(
            tm.main_timeline_id, engine.time.step,
            engine.time.sim_date_str, engine.states, engine.graph,
        )

        fork = tm.create_fork(
            tm.main_timeline_id, engine.time.step,
            engine.time.sim_date_str, "Test event", snapshot.id,
        )
        assert fork.fork_step == 5
        assert len(tm.timelines) == 2


class TestParallelWorldEngine:
    def test_inject_news_with_fork(self):
        engine, tm, pw = _make_system()
        engine.run(5)

        post, fork = pw.inject_news_with_fork("Test news", "Summary", 0.5)
        assert post.is_news_seed
        assert fork is not None
        assert len(pw.forks) == 1

    def test_step_all(self):
        engine, tm, pw = _make_system()
        engine.run(5)

        pw.inject_news_with_fork("Test news", "Summary", 0.3)
        result = pw.step_all()

        assert "main" in result
        assert "forks" in result
        assert "comparisons" in result
        assert len(result["forks"]) == 1

    def test_divergence_increases(self):
        engine, tm, pw = _make_system(30)
        engine.run(3)

        pw.inject_news_with_fork("Major event", "Big news", 0.8)

        divs = []
        for _ in range(10):
            result = pw.step_all()
            for comp in result["comparisons"].values():
                divs.append(comp["divergence"])

        # Divergence should generally increase over time
        assert divs[-1] >= divs[0]

    def test_world_summary(self):
        engine, tm, pw = _make_system()
        engine.run(3)
        pw.inject_news_with_fork("News", "Summary", 0.3)
        pw.step_all()

        summary = pw.get_world_summary()
        assert "main" in summary
        assert len(summary) == 2


class TestRealityDiff:
    def test_record_prediction(self):
        db = Database(tempfile.mktemp(suffix=".db"))
        db.init_db()
        rde = RealityDiffEngine(db)

        pred = rde.record_prediction(
            topic_id="ai_regulation",
            predicted_for_date="2026-06-22",
            recorded_at_date="2026-03-22",
            recorded_at_step=0,
            mean_opinion=0.15,
            opinion_std=0.35,
            distribution={"for": 10, "against": 5},
        )
        assert pred.predicted_mean == 0.15
        db.close()

    def test_compute_diff(self):
        db = Database(tempfile.mktemp(suffix=".db"))
        db.init_db()
        rde = RealityDiffEngine(db)

        pred = rde.record_prediction(
            topic_id="test", predicted_for_date="2026-06-01",
            recorded_at_date="2026-03-01", recorded_at_step=0,
            mean_opinion=0.2, opinion_std=0.3,
            distribution={"for": 20, "against": 10},
        )

        outcome = rde.record_reality(
            topic_id="test", date="2026-06-01",
            actual_sentiment=0.25,
            actual_distribution={"for": 22, "against": 8},
        )

        report = rde.compute_diff(pred, outcome)
        assert 0 <= report.accuracy_score <= 100
        assert report.mean_opinion_error >= 0
        assert report.days_ahead == 92
        db.close()
