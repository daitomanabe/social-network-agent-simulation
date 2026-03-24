"""Network and content models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class Post:
    """A post in the simulated social network."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    author_id: str = ""
    topic_id: str = ""
    content: str = ""
    sim_time: datetime | None = None
    step: int = 0
    sentiment: float = 0.0  # -1.0 to +1.0
    is_news_seed: bool = False  # True if injected from real news
    reply_to: str | None = None  # Post ID if this is a reply
    reactions: dict[str, int] = field(default_factory=dict)
    # e.g. {"agree": 5, "disagree": 3}


@dataclass
class Relationship:
    """An edge in the social network."""

    agent_a: str = ""
    agent_b: str = ""
    weight: float = 0.5  # 0.0 (weak) to 1.0 (strong)
    interaction_count: int = 0
    opinion_alignment: float = 0.0  # How aligned their opinions are
