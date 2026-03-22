"""Timeline data models for parallel world management."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from uuid import uuid4


class TimelineStatus(Enum):
    ACTIVE = "active"      # Currently running
    FROZEN = "frozen"      # Paused, can be resumed
    ARCHIVED = "archived"  # Read-only, kept for comparison


@dataclass
class Timeline:
    """A single timeline (branch) of the simulation."""

    id: str = field(default_factory=lambda: uuid4().hex[:8])
    name: str = ""
    parent_id: str | None = None       # ID of the parent timeline (None for main)
    fork_step: int = 0                 # Step at which this timeline branched
    fork_reason: str = ""              # Why this fork was created
    status: TimelineStatus = TimelineStatus.ACTIVE
    created_at: datetime = field(default_factory=datetime.now)

    # Metadata
    current_step: int = 0
    news_injected: list[str] = field(default_factory=list)  # News IDs in this timeline
    description: str = ""              # "What if X didn't happen" etc.

    @property
    def is_main(self) -> bool:
        return self.parent_id is None


@dataclass
class ForkPoint:
    """A point where a timeline branches."""

    id: str = field(default_factory=lambda: uuid4().hex[:8])
    step: int = 0
    sim_date: str = ""
    trigger_event: str = ""            # News headline or event that caused the fork
    parent_timeline_id: str = ""
    child_timeline_ids: list[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class Snapshot:
    """A complete state snapshot at a specific step."""

    id: str = field(default_factory=lambda: uuid4().hex[:8])
    timeline_id: str = ""
    step: int = 0
    sim_date: str = ""

    # Serialized state
    agent_states_json: str = ""        # JSON-serialized dict of all agent states
    graph_edges_json: str = ""         # JSON-serialized edge list with weights
    opinion_distribution: dict = field(default_factory=dict)

    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class TimelineComparison:
    """Result of comparing two timelines at the same sim-date."""

    timeline_a_id: str = ""
    timeline_b_id: str = ""
    compared_at_step: int = 0
    sim_date: str = ""

    # Divergence metrics
    opinion_mean_diff: float = 0.0     # Difference in mean opinion
    opinion_std_diff: float = 0.0      # Difference in opinion spread
    polarization_diff: float = 0.0     # Difference in polarization
    distribution_diff: dict = field(default_factory=dict)  # Per-bucket differences

    # Detailed
    most_changed_agents: list[str] = field(default_factory=list)  # Agent IDs
    divergence_score: float = 0.0      # 0.0 = identical, 1.0 = maximally different
