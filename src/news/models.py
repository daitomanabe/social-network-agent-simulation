"""News data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from uuid import uuid4


@dataclass
class NewsItem:
    """A raw news article."""

    id: str = field(default_factory=lambda: uuid4().hex[:12])
    title: str = ""
    summary: str = ""
    source: str = ""
    url: str = ""
    published: datetime | None = None
    raw_content: str = ""


@dataclass
class TopicSeed:
    """A processed news item ready for injection into the simulation."""

    id: str = field(default_factory=lambda: uuid4().hex[:8])
    headline: str = ""
    summary: str = ""
    topic_id: str = ""
    sentiment: float = 0.0  # -1.0 to +1.0
    key_claims: list[str] = field(default_factory=list)
    source_news_id: str = ""
