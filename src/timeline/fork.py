"""Fork runner: manages parallel timeline execution."""

from __future__ import annotations

import copy
import logging
from datetime import datetime, timedelta

from src.agents.behavior import RuleBasedBehavior, ActionType
from src.agents.memory import MemoryManager
from src.agents.models import AgentProfile, AgentState
from src.core.config import SimulationConfig
from src.network.graph import SocialGraph
from src.network.models import Post
from src.timeline.manager import TimelineManager
from src.timeline.models import Timeline, TimelineComparison

logger = logging.getLogger(__name__)


class ForkRunner:
    """Runs a forked timeline independently from the main simulation.

    A ForkRunner holds its own copy of agent states and graph,
    running steps with the same behavior engine but without the
    triggering news event.
    """

    def __init__(
        self,
        timeline: Timeline,
        profiles: dict[str, AgentProfile],
        initial_states: dict[str, AgentState],
        initial_edges: list[dict],
        config: SimulationConfig,
        fork_step: int,
        fork_sim_date: datetime,
        topic_id: str,
    ) -> None:
        self.timeline = timeline
        self.profiles = profiles
        self.config = config
        self.topic_id = topic_id

        # Deep copy states so they evolve independently
        self.states: dict[str, AgentState] = {}
        for aid, state in initial_states.items():
            self.states[aid] = AgentState(
                agent_id=state.agent_id,
                opinions=dict(state.opinions),
                emotional_state=copy.deepcopy(state.emotional_state),
                memory=list(state.memory),
                trust_scores=dict(state.trust_scores),
                relationships=dict(state.relationships),
                post_count=state.post_count,
                reply_count=state.reply_count,
            )

        # Rebuild graph from edges
        self.graph = SocialGraph()
        self.graph.build_small_world(
            list(profiles.values()),
            k=config.network_k,
            p=config.network_p,
            seed=config.seed,
        )
        # Restore edge weights from snapshot
        for edge in initial_edges:
            u, v = edge["u"], edge["v"]
            if self.graph.graph.has_edge(u, v):
                self.graph.graph[u][v]["weight"] = edge.get("weight", 0.5)
                self.graph.graph[u][v]["interaction_count"] = edge.get("interaction_count", 0)

        self.behavior = RuleBasedBehavior(seed=config.seed + hash(timeline.id))
        self.memory_mgr = MemoryManager()

        self._current_step = fork_step
        self._sim_date = fork_sim_date
        self.step_stats: list[dict] = []

    @property
    def current_step(self) -> int:
        return self._current_step

    @property
    def sim_date(self) -> datetime:
        return self._sim_date

    def step(self) -> dict:
        """Execute one step in this forked timeline."""
        import random

        self._current_step += 1
        self._sim_date += timedelta(days=1)

        rng = random.Random(hash((self.config.seed, self.timeline.id, self._current_step)))

        # Select active agents
        all_ids = list(self.profiles.keys())
        n_active = max(1, int(len(all_ids) * self.config.activity_rate))
        active_ids = rng.sample(all_ids, n_active)

        posts_this_step: list[Post] = []
        actions = {"idle": 0, "post": 0, "reply": 0}
        opinion_shifts: list[float] = []

        for agent_id in active_ids:
            profile = self.profiles[agent_id]
            state = self.states[agent_id]

            feed = self.graph.get_feed(agent_id, self._current_step)
            action = self.behavior.decide_action(
                profile, state, feed, self.topic_id,
                self._current_step,
            )

            actions[action.action_type.value] += 1

            if action.post is not None:
                action.post.sim_time = self._sim_date
                self.graph.add_post(action.post)
                posts_this_step.append(action.post)
                if action.action_type == ActionType.POST:
                    state.post_count += 1
                else:
                    state.reply_count += 1

            if action.opinion_deltas:
                for topic, delta in action.opinion_deltas.items():
                    old = state.opinions.get(topic, 0.0)
                    state.opinions[topic] = max(-1.0, min(1.0, old + delta))
                    opinion_shifts.append(abs(delta))

            if action.memory_entry:
                state.memory = self.memory_mgr.add(state.memory, action.memory_entry)

        # Compute stats
        all_opinions = [s.opinions.get(self.topic_id, 0.0) for s in self.states.values()]
        stats = {
            "timeline_id": self.timeline.id,
            "timeline_name": self.timeline.name,
            "step": self._current_step,
            "sim_date": self._sim_date.strftime("%Y-%m-%d"),
            "active_agents": n_active,
            "posts": actions["post"],
            "replies": actions["reply"],
            "mean_opinion": sum(all_opinions) / len(all_opinions) if all_opinions else 0,
            "opinion_std": _std(all_opinions),
        }
        self.step_stats.append(stats)
        return stats

    def run(self, n_steps: int) -> list[dict]:
        """Run N steps in this fork."""
        return [self.step() for _ in range(n_steps)]

    def get_opinion_distribution(self) -> dict[str, int]:
        """Get opinion distribution buckets."""
        buckets = {"strong_against": 0, "against": 0, "neutral": 0, "for": 0, "strong_for": 0}
        for state in self.states.values():
            op = state.opinions.get(self.topic_id, 0.0)
            if op < -0.6:
                buckets["strong_against"] += 1
            elif op < -0.2:
                buckets["against"] += 1
            elif op < 0.2:
                buckets["neutral"] += 1
            elif op < 0.6:
                buckets["for"] += 1
            else:
                buckets["strong_for"] += 1
        return buckets


class ParallelWorldEngine:
    """Manages the main simulation and all forked timelines together.

    Orchestrates:
    1. Main timeline runs with real news
    2. On news injection → snapshot + fork (counterfactual)
    3. All active forks advance in lockstep with main
    4. Comparisons generated after each step
    """

    def __init__(self, engine, timeline_manager: TimelineManager) -> None:
        from src.core.engine import SimulationEngine
        self.engine: SimulationEngine = engine
        self.tm = timeline_manager
        self.forks: dict[str, ForkRunner] = {}  # timeline_id -> ForkRunner
        self._comparisons: list[TimelineComparison] = []

        # Max concurrent forks
        self.max_forks = 5

    def inject_news_with_fork(
        self,
        headline: str,
        summary: str,
        sentiment: float = 0.0,
        create_counterfactual: bool = True,
    ) -> tuple[Post, ForkRunner | None]:
        """Inject news into the main timeline and optionally create a fork.

        The fork represents "what if this news didn't happen".
        """
        # Snapshot current state BEFORE news injection
        snapshot = self.tm.create_snapshot(
            timeline_id=self.tm.main_timeline_id,
            step=self.engine.time.step,
            sim_date=self.engine.time.sim_date_str,
            agent_states=self.engine.states,
            graph=self.engine.graph,
            opinion_dist=self.engine.get_opinion_distribution(),
        )

        # Inject news into main timeline
        post = self.engine.inject_news(headline, summary, sentiment)

        fork_runner = None
        if create_counterfactual and len(self.forks) < self.max_forks:
            # Create fork
            timeline = self.tm.create_fork(
                parent_timeline_id=self.tm.main_timeline_id,
                fork_step=self.engine.time.step,
                sim_date=self.engine.time.sim_date_str,
                trigger_event=headline,
                snapshot_id=snapshot.id,
            )

            # Restore state for the fork
            restored_states, edges = self.tm.restore_snapshot(
                snapshot.id, self.engine.profiles
            )

            fork_runner = ForkRunner(
                timeline=timeline,
                profiles=self.engine.profiles,
                initial_states=restored_states,
                initial_edges=edges,
                config=self.engine.config,
                fork_step=self.engine.time.step,
                fork_sim_date=self.engine.time.current,
                topic_id=self.engine.topic_id,
            )
            self.forks[timeline.id] = fork_runner

        return post, fork_runner

    def step_all(self) -> dict:
        """Advance main timeline and all forks by one step.

        Returns a summary with main stats and fork comparisons.
        """
        # Step main timeline
        main_stats = self.engine.step()

        # Step all active forks
        fork_stats = {}
        for tid, fork in list(self.forks.items()):
            if self.tm.timelines[tid].status.value != "active":
                continue
            stats = fork.step()
            fork_stats[tid] = stats
            self.tm.update_timeline_step(tid, fork.current_step)

        # Compare each fork with main
        comparisons = {}
        for tid, fork in self.forks.items():
            comp = self.tm.compare_timelines(
                self.engine.states,
                fork.states,
                self.tm.main_timeline_id,
                tid,
                topic_id=self.engine.topic_id,
                step=self.engine.time.step,
                sim_date=self.engine.time.sim_date_str,
            )
            comparisons[tid] = comp
            self._comparisons.append(comp)

        return {
            "main": main_stats,
            "forks": fork_stats,
            "comparisons": {
                tid: {
                    "divergence": c.divergence_score,
                    "mean_diff": c.opinion_mean_diff,
                    "description": self.tm.timelines[tid].description,
                }
                for tid, c in comparisons.items()
            },
        }

    def run_all(self, n_steps: int) -> list[dict]:
        """Run N steps across all timelines."""
        return [self.step_all() for _ in range(n_steps)]

    def get_world_summary(self) -> dict:
        """Get a summary of all parallel worlds."""
        main_dist = self.engine.get_opinion_distribution()
        worlds = {
            "main": {
                "name": "Main Timeline",
                "step": self.engine.time.step,
                "sim_date": self.engine.time.sim_date_str,
                "opinion_distribution": main_dist,
                "mean_opinion": self.engine.step_stats[-1]["mean_opinion"]
                if self.engine.step_stats else 0,
            }
        }

        for tid, fork in self.forks.items():
            tl = self.tm.timelines[tid]
            worlds[tid] = {
                "name": tl.name,
                "description": tl.description,
                "fork_step": tl.fork_step,
                "fork_reason": tl.fork_reason,
                "step": fork.current_step,
                "sim_date": fork.sim_date.strftime("%Y-%m-%d"),
                "opinion_distribution": fork.get_opinion_distribution(),
                "mean_opinion": fork.step_stats[-1]["mean_opinion"]
                if fork.step_stats else 0,
            }

        return worlds


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
