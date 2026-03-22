"""Core simulation engine."""

from __future__ import annotations

import asyncio
import logging
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

logger = logging.getLogger(__name__)


class SimulationEngine:
    """Orchestrates the simulation loop.

    Supports two modes:
    - Rule-based (default): fast, no API costs, good for prototyping
    - Hybrid (use_llm=True): top agents use Claude API, rest use rules
    """

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

        # LLM brain (lazy-initialized)
        self._llm_brain = None

        self.profiles: dict[str, AgentProfile] = {}
        self.states: dict[str, AgentState] = {}
        self.topic_id: str = ""      # Primary topic (backwards compat)
        self.topic_ids: list[str] = []  # All active topics

        # Stats per step
        self.step_stats: list[dict] = []

        # Callbacks for visualization
        self._on_step_complete: list = []

        self._initialized = False

    @property
    def llm_brain(self):
        """Lazy-init LLM brain only when needed."""
        if self._llm_brain is None:
            from src.agents.llm_brain import LLMBrain
            from src.core.llm_client import LLMClient
            client = LLMClient(
                model=self.config.llm.model,
                max_tokens=self.config.llm.max_tokens,
                temperature=self.config.llm.temperature,
                max_concurrent=self.config.llm.max_concurrent_calls,
                cost_limit_usd=self.config.llm.cost_limit_daily_usd,
            )
            self._llm_brain = LLMBrain(client)
        return self._llm_brain

    def initialize(self) -> None:
        """Set up agents, network, and initial state."""
        topic_names = self.config.initial_topics
        self.topic_ids = [t.lower().replace(" ", "_") for t in topic_names]
        self.topic_id = self.topic_ids[0]  # Primary topic for backwards compat

        # Generate population
        agents = AgentFactory.generate_population(
            n=self.config.agent_count,
            seed=self.config.seed,
            initial_topics=self.topic_ids,
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

        # Split into LLM and rule-based agents
        if self.config.use_llm:
            llm_ids, rule_ids = self._split_by_priority(active_ids)
        else:
            llm_ids, rule_ids = [], active_ids

        posts_this_step: list[Post] = []
        actions_taken = {"idle": 0, "post": 0, "reply": 0}
        opinion_shifts: list[float] = []
        llm_calls = 0

        # Process LLM agents (async batch)
        if llm_ids:
            llm_actions = self._run_llm_batch(llm_ids, step_num)
            llm_calls = len(llm_ids)
            for agent_id, action in zip(llm_ids, llm_actions):
                self._apply_action(
                    agent_id, action, step_num, posts_this_step,
                    actions_taken, opinion_shifts,
                )

        # Process rule-based agents
        for agent_id in rule_ids:
            profile = self.profiles[agent_id]
            state = self.states[agent_id]
            feed = self.graph.get_feed(agent_id, step_num)

            # Pick topic: agent discusses the topic they have strongest opinion on,
            # or the one with most feed activity
            topic = self._pick_topic(agent_id, feed, rng)

            action = self.behavior.decide_action(
                profile, state, feed, topic, step_num,
                sim_time_str=self.time.sim_date_str,
            )
            self._apply_action(
                agent_id, action, step_num, posts_this_step,
                actions_taken, opinion_shifts,
            )

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
            "llm_calls": llm_calls,
        }
        self.step_stats.append(stats)

        # Notify callbacks
        for cb in self._on_step_complete:
            cb(stats, posts_this_step)

        return stats

    def _split_by_priority(self, active_ids: list[str]) -> tuple[list[str], list[str]]:
        """Split active agents into LLM and rule-based groups.

        Priority for LLM: higher extraversion (more active in discussions),
        plus agents who recently had opinion shifts or received news.
        """
        n_llm = max(1, int(len(active_ids) * self.config.llm_agent_ratio))

        # Score agents by "discussion importance"
        scored = []
        for aid in active_ids:
            profile = self.profiles[aid]
            state = self.states[aid]

            score = profile.personality.extraversion * 0.4
            score += profile.personality.openness * 0.2
            # Agents with strong opinions are more interesting for LLM
            opinion = abs(state.opinions.get(self.topic_id, 0.0))
            score += opinion * 0.2
            # Agents with more social connections
            n_neighbors = len(self.graph.get_neighbors(aid))
            score += min(n_neighbors / 10.0, 0.2)

            scored.append((aid, score))

        scored.sort(key=lambda x: x[1], reverse=True)

        llm_ids = [aid for aid, _ in scored[:n_llm]]
        rule_ids = [aid for aid, _ in scored[n_llm:]]

        return llm_ids, rule_ids

    def _run_llm_batch(self, agent_ids: list[str], step_num: int) -> list:
        """Run LLM brain for a batch of agents."""
        from src.agents.behavior import AgentAction

        agent_names = {pid: p.name for pid, p in self.profiles.items()}
        topic_name = self.topic_id.replace("_", " ").title()

        batch = []
        for aid in agent_ids:
            profile = self.profiles[aid]
            state = self.states[aid]
            feed = self.graph.get_feed(aid, step_num)
            batch.append((profile, state, feed))

        try:
            loop = asyncio.new_event_loop()
            actions = loop.run_until_complete(
                self.llm_brain.think_batch(batch, topic_name, step_num, agent_names)
            )
            loop.close()
            return actions
        except Exception as e:
            logger.error("LLM batch failed: %s, falling back to rules", e)
            # Fallback to rule-based
            actions = []
            for profile, state, feed in batch:
                action = self.behavior.decide_action(
                    profile, state, feed, self.topic_id, step_num,
                )
                actions.append(action)
            return actions

    def _apply_action(
        self,
        agent_id: str,
        action,
        step_num: int,
        posts_this_step: list[Post],
        actions_taken: dict[str, int],
        opinion_shifts: list[float],
    ) -> None:
        """Apply an agent's action to the simulation state."""
        state = self.states[agent_id]
        actions_taken[action.action_type.value] += 1

        if action.post is not None:
            action.post.sim_time = self.time.current
            self.graph.add_post(action.post)
            posts_this_step.append(action.post)

            if action.action_type == ActionType.POST:
                state.post_count += 1
            else:
                state.reply_count += 1

        if action.opinion_deltas:
            for topic, delta in action.opinion_deltas.items():
                old = state.opinions.get(topic, 0.0)
                new = max(-1.0, min(1.0, old + delta))
                state.opinions[topic] = new
                opinion_shifts.append(abs(delta))

        if action.memory_entry:
            state.memory = self.memory_mgr.add(state.memory, action.memory_entry)

    def run(self, n_steps: int) -> list[dict]:
        """Run N simulation steps."""
        results = []
        for _ in range(n_steps):
            stats = self.step()
            results.append(stats)
        return results

    def _pick_topic(self, agent_id: str, feed: list[Post], rng: random.Random) -> str:
        """Pick which topic an agent discusses this step."""
        if len(self.topic_ids) == 1:
            return self.topic_ids[0]

        state = self.states[agent_id]

        # Weight by: opinion strength (agents care more about strong opinions)
        # + feed activity (topics with more posts in feed)
        topic_scores: dict[str, float] = {}
        for tid in self.topic_ids:
            opinion_strength = abs(state.opinions.get(tid, 0.0))
            feed_count = sum(1 for p in feed if p.topic_id == tid)
            topic_scores[tid] = opinion_strength * 0.6 + feed_count * 0.3 + 0.1  # Floor

        topics = list(topic_scores.keys())
        weights = [topic_scores[t] for t in topics]
        return rng.choices(topics, weights=weights, k=1)[0]

    def add_topic(self, topic_name: str) -> str:
        """Add a new discussion topic to the simulation."""
        topic_id = topic_name.lower().replace(" ", "_")
        if topic_id not in self.topic_ids:
            self.topic_ids.append(topic_id)
            # Initialize opinions for all agents
            rng = random.Random(hash(topic_id))
            for state in self.states.values():
                profile = self.profiles[state.agent_id]
                base = (profile.personality.openness - 0.5) * 0.4
                state.opinions[topic_id] = max(-1.0, min(1.0, base + rng.gauss(0, 0.25)))
        return topic_id

    def inject_news(self, headline: str, summary: str, sentiment: float = 0.0, topic_id: str | None = None) -> Post:
        """Inject a news item as a seed post."""
        target_topic = topic_id or self.topic_id
        post = Post(
            author_id="NEWS",
            topic_id=target_topic,
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

    def get_opinion_distribution(self, topic_id: str | None = None) -> dict:
        """Get current opinion distribution in buckets."""
        buckets = {
            "strong_against (-1.0 ~ -0.6)": 0,
            "against (-0.6 ~ -0.2)": 0,
            "neutral (-0.2 ~ 0.2)": 0,
            "for (0.2 ~ 0.6)": 0,
            "strong_for (0.6 ~ 1.0)": 0,
        }
        tid = topic_id or self.topic_id
        for state in self.states.values():
            op = state.opinions.get(tid, 0.0)
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
