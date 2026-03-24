"""Influence propagation tracker.

Tracks how news and opinions spread through the social network,
recording which agents were exposed, when they reacted, and how
their opinions shifted. This creates a "cascade" visualization
showing information rippling through the network.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class PropagationEvent:
    """A single event in a propagation cascade."""

    agent_id: str
    step: int
    event_type: str  # "exposed", "reacted", "opinion_shifted", "amplified"
    opinion_before: float = 0.0
    opinion_after: float = 0.0
    post_id: str | None = None
    source_agent_id: str | None = None  # Who influenced this agent


@dataclass
class PropagationCascade:
    """Tracks how a single piece of news propagates through the network."""

    id: str = ""
    trigger_headline: str = ""
    trigger_step: int = 0
    trigger_sentiment: float = 0.0
    topic_id: str = ""

    events: list[PropagationEvent] = field(default_factory=list)

    # Summary stats (updated as cascade grows)
    total_exposed: int = 0
    total_reacted: int = 0
    total_opinion_shifts: int = 0
    avg_opinion_shift: float = 0.0
    max_depth: int = 0  # How many hops from source
    peak_step: int = 0  # Step with most activity

    # Per-step activity
    activity_per_step: dict[int, int] = field(default_factory=dict)

    def add_event(self, event: PropagationEvent) -> None:
        self.events.append(event)
        step = event.step
        self.activity_per_step[step] = self.activity_per_step.get(step, 0) + 1

        if event.event_type == "exposed":
            self.total_exposed += 1
        elif event.event_type in ("reacted", "amplified"):
            self.total_reacted += 1
        elif event.event_type == "opinion_shifted":
            self.total_opinion_shifts += 1

    def compute_summary(self) -> None:
        """Recompute summary statistics."""
        shifts = [
            abs(e.opinion_after - e.opinion_before)
            for e in self.events if e.event_type == "opinion_shifted"
        ]
        self.avg_opinion_shift = sum(shifts) / len(shifts) if shifts else 0.0

        if self.activity_per_step:
            self.peak_step = max(self.activity_per_step, key=self.activity_per_step.get)

    def get_timeline(self) -> list[dict]:
        """Get cascade events as a timeline for visualization."""
        timeline = []
        for step in sorted(self.activity_per_step.keys()):
            step_events = [e for e in self.events if e.step == step]
            timeline.append({
                "step": step,
                "steps_after_trigger": step - self.trigger_step,
                "activity": len(step_events),
                "exposed": sum(1 for e in step_events if e.event_type == "exposed"),
                "reacted": sum(1 for e in step_events if e.event_type in ("reacted", "amplified")),
                "shifted": sum(1 for e in step_events if e.event_type == "opinion_shifted"),
            })
        return timeline

    def get_agent_chain(self) -> list[list[str]]:
        """Get the chain of influence (who influenced whom)."""
        chains: list[list[str]] = []
        for e in self.events:
            if e.source_agent_id and e.event_type in ("reacted", "amplified"):
                chains.append([e.source_agent_id, e.agent_id])
        return chains


class PropagationTracker:
    """Tracks multiple propagation cascades across the simulation."""

    def __init__(self) -> None:
        self.cascades: dict[str, PropagationCascade] = {}
        self._active_cascade_id: str | None = None

        # Track which agents have seen which news
        self._agent_exposure: dict[str, set[str]] = {}  # agent_id -> set of cascade_ids

    def start_cascade(
        self,
        cascade_id: str,
        headline: str,
        step: int,
        sentiment: float = 0.0,
        topic_id: str = "",
    ) -> PropagationCascade:
        """Start tracking a new propagation cascade."""
        cascade = PropagationCascade(
            id=cascade_id,
            trigger_headline=headline,
            trigger_step=step,
            trigger_sentiment=sentiment,
            topic_id=topic_id,
        )
        self.cascades[cascade_id] = cascade
        self._active_cascade_id = cascade_id
        return cascade

    def record_exposure(
        self,
        cascade_id: str,
        agent_id: str,
        step: int,
        source_agent_id: str | None = None,
    ) -> None:
        """Record that an agent was exposed to news from this cascade."""
        if cascade_id not in self.cascades:
            return

        # Avoid duplicate exposure
        if agent_id not in self._agent_exposure:
            self._agent_exposure[agent_id] = set()
        if cascade_id in self._agent_exposure[agent_id]:
            return
        self._agent_exposure[agent_id].add(cascade_id)

        self.cascades[cascade_id].add_event(PropagationEvent(
            agent_id=agent_id,
            step=step,
            event_type="exposed",
            source_agent_id=source_agent_id,
        ))

    def record_reaction(
        self,
        cascade_id: str,
        agent_id: str,
        step: int,
        post_id: str | None = None,
        source_agent_id: str | None = None,
        is_amplification: bool = False,
    ) -> None:
        """Record that an agent reacted (posted/replied) about this cascade's topic."""
        if cascade_id not in self.cascades:
            return

        event_type = "amplified" if is_amplification else "reacted"
        self.cascades[cascade_id].add_event(PropagationEvent(
            agent_id=agent_id,
            step=step,
            event_type=event_type,
            post_id=post_id,
            source_agent_id=source_agent_id,
        ))

    def record_opinion_shift(
        self,
        cascade_id: str,
        agent_id: str,
        step: int,
        opinion_before: float,
        opinion_after: float,
    ) -> None:
        """Record that an agent's opinion shifted due to cascade exposure."""
        if cascade_id not in self.cascades:
            return

        self.cascades[cascade_id].add_event(PropagationEvent(
            agent_id=agent_id,
            step=step,
            event_type="opinion_shifted",
            opinion_before=opinion_before,
            opinion_after=opinion_after,
        ))

    def get_cascade_summary(self, cascade_id: str) -> dict | None:
        """Get summary of a cascade."""
        cascade = self.cascades.get(cascade_id)
        if not cascade:
            return None

        cascade.compute_summary()
        return {
            "id": cascade.id,
            "headline": cascade.trigger_headline,
            "trigger_step": cascade.trigger_step,
            "sentiment": cascade.trigger_sentiment,
            "total_exposed": cascade.total_exposed,
            "total_reacted": cascade.total_reacted,
            "total_opinion_shifts": cascade.total_opinion_shifts,
            "avg_opinion_shift": round(cascade.avg_opinion_shift, 4),
            "peak_step": cascade.peak_step,
            "timeline": cascade.get_timeline(),
            "influence_chains": cascade.get_agent_chain()[:20],  # Limit
        }

    def get_all_summaries(self) -> list[dict]:
        """Get summaries of all cascades."""
        return [
            self.get_cascade_summary(cid)
            for cid in self.cascades
            if self.get_cascade_summary(cid) is not None
        ]

    def get_most_influential_agents(self, cascade_id: str, top_n: int = 10) -> list[dict]:
        """Find agents who caused the most reactions in a cascade."""
        cascade = self.cascades.get(cascade_id)
        if not cascade:
            return []

        influence_count: dict[str, int] = {}
        for e in cascade.events:
            if e.source_agent_id and e.event_type in ("reacted", "amplified"):
                influence_count[e.source_agent_id] = influence_count.get(e.source_agent_id, 0) + 1

        sorted_agents = sorted(influence_count.items(), key=lambda x: x[1], reverse=True)
        return [{"agent_id": aid, "influence_count": count} for aid, count in sorted_agents[:top_n]]
