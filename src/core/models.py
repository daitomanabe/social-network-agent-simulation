"""Core simulation models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


class EventType(Enum):
    """Types of simulation events."""

    NEWS_INJECTED = "news_injected"
    AGENT_POSTED = "agent_posted"
    AGENT_REPLIED = "agent_replied"
    OPINION_SHIFTED = "opinion_shifted"
    FORK_CREATED = "fork_created"
    STEP_COMPLETED = "step_completed"


@dataclass
class SimTime:
    """Tracks simulation time independently of wall-clock time."""

    current: datetime  # Simulated datetime
    step: int = 0  # Current step number
    start: datetime | None = None  # When simulation began (sim time)

    def __post_init__(self) -> None:
        if self.start is None:
            self.start = self.current

    def advance_days(self, days: int = 1) -> None:
        """Advance simulation time by N days."""
        from datetime import timedelta

        self.current += timedelta(days=days)
        self.step += 1

    @property
    def elapsed_days(self) -> int:
        assert self.start is not None
        return (self.current - self.start).days


@dataclass
class SimEvent:
    """A discrete event in the simulation."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    event_type: EventType = EventType.STEP_COMPLETED
    sim_time: datetime | None = None
    agent_id: str | None = None
    data: dict = field(default_factory=dict)
    description: str = ""


@dataclass
class Topic:
    """A discussion topic in the simulation."""

    id: str = field(default_factory=lambda: uuid4().hex[:8])
    name: str = ""
    description: str = ""
    temperature: float = 0.5  # Activity level 0.0-1.0
    related_topics: list[str] = field(default_factory=list)  # Topic IDs
