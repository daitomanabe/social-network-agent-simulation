"""Core simulation engine."""

from __future__ import annotations

import random
from datetime import datetime

from src.agents.behavior import RuleBasedBehavior, ActionType
from src.agents.factory import AgentFactory
from src.agents.memory import MemoryManager
from src.agents.models import AgentProfile, AgentState
from src.core.config import SimulationConfig
from src.core.database import Database
from src.core.time_manager import TimeManager
from src.network.graph import SocialGraph
from src.network.models import Post


class SimulationEngine:
    """Orchestrates the simulation loop."""

    def __init__(
        self,
        config: SimulationConfig | None = None,
        start_time: datetime | None = None,
    ) -> None:
        self.config = config or SimulationConfig()
        self.db = Database(self.config.db_path)
        self.db.init_db()

        start = start_time or datetime.now()
        self.time = TimeManager(
            sim_start=start,
            sim_days_per_real_hour=self.config.sim_days_per_real_hour,
        )

        self.graph = SocialGraph()
        self.behavior = RuleBasedBehavior(seed=self.config.seed)
        self.memory_mgr = MemoryManager()

        self.profiles: dict[str, AgentProfile] = {}
        self.states: dict[str, AgentState] = {}
        self.topic_id: str = ""

        # Stats per step
        self.step_stats: list[dict] = []

        # Callbacks for visualization
        self._on_step_complete: list = []

        self._initialized = False

    def initialize(self) -> None:
        """Set up agents, network, and initial state."""
        topic_names = self.config.initial_topics
        self.topic_id = topic_names[0].lower().replace(" ", "_")

        # Generate population
        agents = AgentFactory.generate_population(
            n=self.config.agent_count,
            seed=self.config.seed,
            initial_topics=[self.topic_id],
        )

        for profile, state in agents:
            self.profiles[profile.id] = profile
            self.states[profile.id] = state

        # Build network
        profiles_list = list(self.profiles.values())
        self.graph.build_small_world(
            profiles_list,
            k=self.config.network_k,
            p=self.config.network_p,
            seed=self.config.seed,
        )

        # Persist initial state
        self.db.insert_agents_batch(profiles_list)
        self.db.save_agent_states_batch(list(self.states.values()), step=0)

        self._initialized = True

    def step(self) -> dict:
        """Execute one simulation step (= 1 sim day)."""
        if not self._initialized:
            self.initialize()

        step_num = self.time.step + 1
        self.time.advance()

        rng = random.Random(hash((self.config.seed, step_num)))

        # Select active agents
        all_ids = list(self.profiles.keys())
        n_active = max(1, int(len(all_ids) * self.config.activity_rate))
        active_ids = rng.sample(all_ids, n_active)

        posts_this_step: list[Post] = []
        actions_taken = {"idle": 0, "post": 0, "reply": 0}
        opinion_shifts: list[float] = []

        for agent_id in active_ids:
            profile = self.profiles[agent_id]
            state = self.states[agent_id]

            # Get feed
            feed = self.graph.get_feed(agent_id, step_num)

            # Decide action
            action = self.behavior.decide_action(
                profile, state, feed, self.topic_id, step_num,
                sim_time_str=self.time.sim_date_str,
            )

            actions_taken[action.action_type.value] += 1

            # Apply action
            if action.post is not None:
                action.post.sim_time = self.time.current
                self.graph.add_post(action.post)
                posts_this_step.append(action.post)

                if action.action_type == ActionType.POST:
                    state.post_count += 1
                else:
                    state.reply_count += 1

            # Apply opinion shifts
            if action.opinion_deltas:
                for topic, delta in action.opinion_deltas.items():
                    old = state.opinions.get(topic, 0.0)
                    new = max(-1.0, min(1.0, old + delta))
                    state.opinions[topic] = new
                    opinion_shifts.append(abs(delta))

            # Update memory
            if action.memory_entry:
                state.memory = self.memory_mgr.add(state.memory, action.memory_entry)

        # Persist posts
        for post in posts_this_step:
            self.db.insert_post(post)

        # Periodically save state (every 5 steps)
        if step_num % 5 == 0:
            self.db.save_agent_states_batch(list(self.states.values()), step=step_num)

        # Compute stats
        all_opinions = [
            s.opinions.get(self.topic_id, 0.0) for s in self.states.values()
        ]
        stats = {
            "step": step_num,
            "sim_date": self.time.sim_date_str,
            "active_agents": n_active,
            "posts": actions_taken["post"],
            "replies": actions_taken["reply"],
            "idle": actions_taken["idle"],
            "mean_opinion": sum(all_opinions) / len(all_opinions) if all_opinions else 0,
            "opinion_std": _std(all_opinions),
            "opinion_min": min(all_opinions) if all_opinions else 0,
            "opinion_max": max(all_opinions) if all_opinions else 0,
            "avg_shift": sum(opinion_shifts) / len(opinion_shifts) if opinion_shifts else 0,
            "total_posts": sum(s.post_count + s.reply_count for s in self.states.values()),
        }
        self.step_stats.append(stats)

        # Notify callbacks
        for cb in self._on_step_complete:
            cb(stats, posts_this_step)

        return stats

    def run(self, n_steps: int) -> list[dict]:
        """Run N simulation steps."""
        results = []
        for _ in range(n_steps):
            stats = self.step()
            results.append(stats)
        return results

    def inject_news(self, headline: str, summary: str, sentiment: float = 0.0) -> Post:
        """Inject a news item as a seed post."""
        post = Post(
            author_id="NEWS",
            topic_id=self.topic_id,
            content=f"📰 {headline}\n{summary}",
            step=self.time.step + 1,
            sim_time=self.time.current,
            sentiment=sentiment,
            is_news_seed=True,
        )
        self.graph.add_post(post)
        self.db.insert_post(post)
        return post

    def on_step_complete(self, callback) -> None:
        """Register a callback for step completion."""
        self._on_step_complete.append(callback)

    def get_opinion_distribution(self) -> dict:
        """Get current opinion distribution in buckets."""
        buckets = {
            "strong_against (-1.0 ~ -0.6)": 0,
            "against (-0.6 ~ -0.2)": 0,
            "neutral (-0.2 ~ 0.2)": 0,
            "for (0.2 ~ 0.6)": 0,
            "strong_for (0.6 ~ 1.0)": 0,
        }
        for state in self.states.values():
            op = state.opinions.get(self.topic_id, 0.0)
            if op < -0.6:
                buckets["strong_against (-1.0 ~ -0.6)"] += 1
            elif op < -0.2:
                buckets["against (-0.6 ~ -0.2)"] += 1
            elif op < 0.2:
                buckets["neutral (-0.2 ~ 0.2)"] += 1
            elif op < 0.6:
                buckets["for (0.2 ~ 0.6)"] += 1
            else:
                buckets["strong_for (0.6 ~ 1.0)"] += 1
        return buckets


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    variance = sum((x - mean) ** 2 for x in values) / len(values)
    return variance**0.5
