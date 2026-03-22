"""World Runner: continuous simulation with auto news, predictions, and forks.

This is the top-level orchestrator that combines all subsystems:
- SimulationEngine (agent behavior, opinions)
- ParallelWorldEngine (forks, comparisons)
- NetworkDynamics (echo chambers, rewiring)
- NewsScheduler (RSS polling)
- RealityDiffEngine (prediction recording)

It runs the simulation continuously, periodically:
1. Polling news feeds for new stories
2. Injecting news and creating timeline forks
3. Recording predictions at milestones
4. Evolving the network
5. Generating agent reflections
6. Tracking divergence between parallel worlds
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import datetime

from src.core.config import SimulationConfig
from src.core.engine import SimulationEngine
from src.network.dynamics import NetworkDynamics
from src.news.scheduler import NewsScheduler
from src.timeline.fork import ParallelWorldEngine
from src.timeline.manager import TimelineManager
from src.timeline.reality_diff import RealityDiffEngine

logger = logging.getLogger(__name__)


@dataclass
class WorldRunnerConfig:
    """Configuration for continuous world running."""

    # Simulation
    sim_config: SimulationConfig = field(default_factory=SimulationConfig)
    start_time: datetime = field(default_factory=lambda: datetime(2026, 3, 22))

    # Timing
    step_delay_seconds: float = 1.0  # Real-time delay between steps
    max_steps: int = 0               # 0 = unlimited

    # Features
    auto_news: bool = False           # Auto-fetch news from RSS
    news_poll_interval: int = 10      # Steps between news polls
    auto_fork: bool = True            # Auto-fork on news injection
    auto_evolve: bool = True          # Network evolution
    evolve_interval: int = 5          # Steps between evolution
    auto_predict: bool = True         # Record predictions
    predict_interval: int = 7         # Steps between prediction snapshots
    max_forks: int = 5                # Max parallel timelines

    # Display
    verbose: bool = True


class WorldRunner:
    """Orchestrates continuous simulation with all subsystems."""

    def __init__(self, config: WorldRunnerConfig | None = None) -> None:
        self.config = config or WorldRunnerConfig()

        # Core
        self.engine = SimulationEngine(
            config=self.config.sim_config,
            start_time=self.config.start_time,
        )
        self.engine.enable_network_evolution = self.config.auto_evolve
        self.engine.network_evolve_interval = self.config.evolve_interval
        self.engine.enable_reflections = True

        # Timeline
        self._tm: TimelineManager | None = None
        self._pw: ParallelWorldEngine | None = None

        # News
        self.news_scheduler = NewsScheduler(auto_inject=False)

        # Reality Diff
        self._rde: RealityDiffEngine | None = None

        # State
        self._initialized = False
        self._running = False
        self._step_count = 0
        self._events: list[dict] = []

        # Callbacks
        self._on_step: list = []
        self._on_news: list = []
        self._on_prediction: list = []

    @property
    def tm(self) -> TimelineManager:
        if self._tm is None:
            raise RuntimeError("Not initialized")
        return self._tm

    @property
    def pw(self) -> ParallelWorldEngine:
        if self._pw is None:
            raise RuntimeError("Not initialized")
        return self._pw

    @property
    def rde(self) -> RealityDiffEngine:
        if self._rde is None:
            raise RuntimeError("Not initialized")
        return self._rde

    def initialize(self) -> None:
        """Initialize all subsystems."""
        self.engine.initialize()
        self._tm = TimelineManager(self.engine.db)
        self._pw = ParallelWorldEngine(self.engine, self._tm)
        self._pw.max_forks = self.config.max_forks
        self._rde = RealityDiffEngine(self.engine.db)

        # Wire up news scheduler
        if self.config.auto_news:
            self.news_scheduler.set_inject_callback(self._on_news_received)

        self._initialized = True
        self._log_event("initialized", f"{len(self.engine.profiles)} agents ready")

    def run(self, n_steps: int = 0) -> list[dict]:
        """Run the simulation.

        Args:
            n_steps: Number of steps (0 = use config.max_steps, 0 there = unlimited)
        """
        if not self._initialized:
            self.initialize()

        target = n_steps or self.config.max_steps
        self._running = True
        results = []

        try:
            i = 0
            while self._running:
                if target > 0 and i >= target:
                    break

                result = self.step()
                results.append(result)
                i += 1

                if self.config.step_delay_seconds > 0:
                    time.sleep(self.config.step_delay_seconds)

        except KeyboardInterrupt:
            self._running = False
            self._log_event("interrupted", f"Stopped at step {self._step_count}")

        return results

    def step(self) -> dict:
        """Execute one full simulation step with all subsystems."""
        if not self._initialized:
            self.initialize()

        self._step_count += 1
        step_num = self._step_count

        # 1. Poll news (periodically)
        news_items = []
        if self.config.auto_news and step_num % self.config.news_poll_interval == 0:
            news_items = self.news_scheduler.poll_all()

        # 2. Advance all timelines
        result = self.pw.step_all()
        main_stats = result["main"]

        # 3. Record prediction (periodically)
        prediction = None
        if self.config.auto_predict and step_num % self.config.predict_interval == 0:
            prediction = self._record_prediction(main_stats)

        # 4. Build step result
        step_result = {
            "step": step_num,
            "sim_date": main_stats["sim_date"],
            "main": main_stats,
            "forks": result.get("forks", {}),
            "comparisons": result.get("comparisons", {}),
            "news_injected": len(news_items),
            "prediction_recorded": prediction is not None,
            "active_forks": len(self.pw.forks),
        }

        # 5. Notify callbacks
        for cb in self._on_step:
            cb(step_result)

        return step_result

    def inject_news(
        self,
        headline: str,
        summary: str = "",
        sentiment: float = 0.0,
        create_fork: bool | None = None,
    ) -> dict:
        """Manually inject a news item."""
        fork_enabled = create_fork if create_fork is not None else self.config.auto_fork
        post, fork = self.pw.inject_news_with_fork(
            headline, summary or headline, sentiment,
            create_counterfactual=fork_enabled,
        )

        event = {
            "type": "news_injected",
            "headline": headline,
            "sentiment": sentiment,
            "fork_created": fork is not None,
            "fork_description": fork.timeline.description if fork else None,
            "step": self.engine.time.step,
        }
        self._events.append(event)

        for cb in self._on_news:
            cb(event)

        return event

    def _on_news_received(self, headline: str, summary: str, sentiment: float, topic_id: str | None) -> None:
        """Callback from news scheduler when news is auto-fetched."""
        if topic_id and topic_id not in self.engine.topic_ids:
            self.engine.add_topic(topic_id)

        self.inject_news(headline, summary, sentiment)

    def _record_prediction(self, stats: dict) -> dict | None:
        """Record a prediction for the current simulation state."""
        # Predict 30 sim-days ahead
        from datetime import timedelta
        current = self.engine.time.current
        target_date = (current + timedelta(days=30)).strftime("%Y-%m-%d")

        dist = self.engine.get_opinion_distribution()

        # Detect trends
        trends = self._detect_trends()

        pred = self.rde.record_prediction(
            topic_id=self.engine.topic_id,
            predicted_for_date=target_date,
            recorded_at_date=current.strftime("%Y-%m-%d"),
            recorded_at_step=stats["step"],
            mean_opinion=stats["mean_opinion"],
            opinion_std=stats["opinion_std"],
            distribution=dist,
            trends=trends,
            confidence=0.5,
        )

        event = {
            "type": "prediction_recorded",
            "target_date": target_date,
            "mean_opinion": stats["mean_opinion"],
            "trends": trends,
            "step": stats["step"],
        }
        self._events.append(event)

        for cb in self._on_prediction:
            cb(event)

        return event

    def _detect_trends(self) -> list[str]:
        """Detect trends from recent step stats."""
        trends = []
        stats = self.engine.step_stats

        if len(stats) < 5:
            return ["データ不足のため傾向分析不可"]

        recent = stats[-5:]
        older = stats[-10:-5] if len(stats) >= 10 else stats[:5]

        # Opinion direction
        recent_mean = sum(s["mean_opinion"] for s in recent) / len(recent)
        older_mean = sum(s["mean_opinion"] for s in older) / len(older)
        delta = recent_mean - older_mean

        if delta > 0.02:
            trends.append("世論が賛成方向に移動中")
        elif delta < -0.02:
            trends.append("世論が反対方向に移動中")
        else:
            trends.append("世論は安定している")

        # Polarization trend
        recent_std = sum(s["opinion_std"] for s in recent) / len(recent)
        older_std = sum(s["opinion_std"] for s in older) / len(older)
        std_delta = recent_std - older_std

        if std_delta > 0.01:
            trends.append("意見の分極化が進行中")
        elif std_delta < -0.01:
            trends.append("意見の収束傾向がある")

        # Activity trend
        recent_posts = sum(s["posts"] + s["replies"] for s in recent)
        older_posts = sum(s["posts"] + s["replies"] for s in older)
        if recent_posts > older_posts * 1.3:
            trends.append("議論が活発化している")
        elif recent_posts < older_posts * 0.7:
            trends.append("議論が沈静化している")

        return trends

    def get_world_summary(self) -> dict:
        """Get comprehensive summary of all parallel worlds."""
        summary = self.pw.get_world_summary()

        # Add polarization
        pol = self.engine.dynamics.compute_polarization(
            self.engine.states, self.engine.topic_id
        )

        summary["polarization"] = {
            "index": pol.polarization_index,
            "echo_chambers": pol.echo_chamber_count,
            "modularity": pol.modularity,
        }
        summary["predictions"] = len(self.rde.predictions)
        summary["events"] = self._events[-10:]  # Last 10 events
        summary["trends"] = self._detect_trends()

        return summary

    def on_step(self, callback) -> None:
        self._on_step.append(callback)

    def on_news(self, callback) -> None:
        self._on_news.append(callback)

    def on_prediction(self, callback) -> None:
        self._on_prediction.append(callback)

    def stop(self) -> None:
        self._running = False

    def _log_event(self, event_type: str, description: str) -> None:
        event = {
            "type": event_type,
            "description": description,
            "step": self._step_count,
            "time": datetime.now().isoformat(),
        }
        self._events.append(event)
        if self.config.verbose:
            logger.info("[%s] %s", event_type, description)
