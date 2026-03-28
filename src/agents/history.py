"""Agent opinion history and conversation thread tracking.

Records how each agent's opinions evolve over time and tracks
conversation threads (chains of posts and replies).
"""

from __future__ import annotations

from dataclasses import dataclass, field

from src.network.models import Post


@dataclass
class OpinionSnapshot:
    """A single opinion data point in time."""

    step: int
    sim_date: str
    topic_id: str
    opinion: float
    event: str = ""  # What caused the change (e.g., "read news", "discussion")


@dataclass
class AgentHistory:
    """Complete history of an agent's opinion evolution."""

    agent_id: str
    snapshots: list[OpinionSnapshot] = field(default_factory=list)

    def add_snapshot(self, step: int, sim_date: str, topic_id: str, opinion: float, event: str = "") -> None:
        self.snapshots.append(OpinionSnapshot(
            step=step, sim_date=sim_date, topic_id=topic_id,
            opinion=opinion, event=event,
        ))

    def get_timeline(self, topic_id: str | None = None) -> list[dict]:
        """Get opinion timeline for visualization."""
        filtered = self.snapshots
        if topic_id:
            filtered = [s for s in filtered if s.topic_id == topic_id]
        return [
            {"step": s.step, "date": s.sim_date, "opinion": s.opinion, "event": s.event}
            for s in filtered
        ]

    def opinion_delta(self, topic_id: str) -> float:
        """Total opinion change from start to current."""
        relevant = [s for s in self.snapshots if s.topic_id == topic_id]
        if len(relevant) < 2:
            return 0.0
        return relevant[-1].opinion - relevant[0].opinion

    def volatility(self, topic_id: str) -> float:
        """How much the agent's opinion has fluctuated."""
        relevant = [s.opinion for s in self.snapshots if s.topic_id == topic_id]
        if len(relevant) < 2:
            return 0.0
        changes = [abs(relevant[i] - relevant[i-1]) for i in range(1, len(relevant))]
        return sum(changes) / len(changes)


@dataclass
class ConversationThread:
    """A chain of posts and replies forming a conversation."""

    id: str
    topic_id: str
    root_post: Post
    replies: list[Post] = field(default_factory=list)
    participants: set[str] = field(default_factory=set)
    start_step: int = 0
    end_step: int = 0

    @property
    def depth(self) -> int:
        return len(self.replies)

    @property
    def participant_count(self) -> int:
        return len(self.participants)

    def add_reply(self, post: Post) -> None:
        self.replies.append(post)
        self.participants.add(post.author_id)
        self.end_step = max(self.end_step, post.step)


class HistoryTracker:
    """Tracks agent opinion histories and conversation threads."""

    def __init__(self) -> None:
        self.agent_histories: dict[str, AgentHistory] = {}
        self.threads: dict[str, ConversationThread] = {}
        self._post_to_thread: dict[str, str] = {}  # post_id -> thread_id

    def record_opinion(
        self,
        agent_id: str,
        step: int,
        sim_date: str,
        topic_id: str,
        opinion: float,
        event: str = "",
    ) -> None:
        """Record an agent's opinion at a point in time."""
        if agent_id not in self.agent_histories:
            self.agent_histories[agent_id] = AgentHistory(agent_id=agent_id)
        self.agent_histories[agent_id].add_snapshot(step, sim_date, topic_id, opinion, event)

    def record_post(self, post: Post) -> None:
        """Record a post, creating or extending a conversation thread."""
        if post.reply_to and post.reply_to in self._post_to_thread:
            # This is a reply to an existing thread
            thread_id = self._post_to_thread[post.reply_to]
            thread = self.threads[thread_id]
            thread.add_reply(post)
            self._post_to_thread[post.id] = thread_id
        else:
            # New thread
            thread = ConversationThread(
                id=post.id,
                topic_id=post.topic_id,
                root_post=post,
                participants={post.author_id},
                start_step=post.step,
                end_step=post.step,
            )
            self.threads[post.id] = thread
            self._post_to_thread[post.id] = post.id

    def get_agent_timeline(self, agent_id: str, topic_id: str | None = None) -> list[dict]:
        """Get an agent's opinion timeline."""
        history = self.agent_histories.get(agent_id)
        if not history:
            return []
        return history.get_timeline(topic_id)

    def get_most_changed_agents(self, topic_id: str, top_n: int = 10) -> list[dict]:
        """Find agents whose opinions changed the most."""
        changes = []
        for aid, history in self.agent_histories.items():
            delta = history.opinion_delta(topic_id)
            vol = history.volatility(topic_id)
            changes.append({
                "agent_id": aid,
                "total_delta": round(delta, 4),
                "volatility": round(vol, 4),
                "abs_delta": round(abs(delta), 4),
            })
        changes.sort(key=lambda x: x["abs_delta"], reverse=True)
        return changes[:top_n]

    def get_longest_threads(self, top_n: int = 10) -> list[dict]:
        """Get the longest conversation threads."""
        sorted_threads = sorted(self.threads.values(), key=lambda t: t.depth, reverse=True)
        return [
            {
                "thread_id": t.id,
                "topic": t.topic_id,
                "depth": t.depth,
                "participants": t.participant_count,
                "start_step": t.start_step,
                "end_step": t.end_step,
                "root_content": t.root_post.content[:100],
            }
            for t in sorted_threads[:top_n]
        ]

    def get_stats(self) -> dict:
        """Get tracking statistics."""
        return {
            "agents_tracked": len(self.agent_histories),
            "total_snapshots": sum(len(h.snapshots) for h in self.agent_histories.values()),
            "threads": len(self.threads),
            "threads_with_replies": sum(1 for t in self.threads.values() if t.depth > 0),
            "avg_thread_depth": (
                sum(t.depth for t in self.threads.values()) / len(self.threads)
                if self.threads else 0
            ),
        }
