"""Agent data models."""

from __future__ import annotations

from dataclasses import dataclass, field
from uuid import uuid4


@dataclass
class BigFive:
    """Big Five personality traits, each 0.0-1.0."""

    openness: float = 0.5
    conscientiousness: float = 0.5
    extraversion: float = 0.5
    agreeableness: float = 0.5
    neuroticism: float = 0.5


@dataclass
class CognitiveBiases:
    """Cognitive bias strengths, each 0.0-1.0."""

    confirmation_bias: float = 0.5  # Tendency to seek confirming info
    authority_bias: float = 0.5  # Deference to authority figures
    bandwagon_effect: float = 0.3  # Susceptibility to majority opinion
    anchoring: float = 0.4  # Sticking to initial information


@dataclass
class AgentProfile:
    """Immutable (or slowly changing) agent identity."""

    id: str = field(default_factory=lambda: uuid4().hex[:10])
    name: str = ""
    age_group: str = ""  # "20s", "30s", "40s", "50s", "60s+"
    occupation: str = ""
    region: str = ""  # Fictional region
    personality: BigFive = field(default_factory=BigFive)
    biases: CognitiveBiases = field(default_factory=CognitiveBiases)
    core_values: list[str] = field(default_factory=list)
    # e.g. ["freedom", "safety", "equality", "tradition", "innovation"]


@dataclass
class EmotionalState:
    """Current emotional state of an agent."""

    anger: float = 0.0
    anxiety: float = 0.0
    hope: float = 0.5
    frustration: float = 0.0
    enthusiasm: float = 0.3

    @property
    def valence(self) -> float:
        """Overall emotional valence: -1 (negative) to +1 (positive)."""
        positive = self.hope + self.enthusiasm
        negative = self.anger + self.anxiety + self.frustration
        total = positive + negative
        if total == 0:
            return 0.0
        return (positive - negative) / total


@dataclass
class AgentState:
    """Dynamic, mutable agent state that changes each step."""

    agent_id: str = ""
    opinions: dict[str, float] = field(default_factory=dict)
    # topic_id -> opinion score (-1.0 to +1.0)

    emotional_state: EmotionalState = field(default_factory=EmotionalState)
    memory: list[str] = field(default_factory=list)  # Recent event summaries
    trust_scores: dict[str, float] = field(default_factory=dict)
    # source_id -> trust (0.0 to 1.0)

    relationships: dict[str, float] = field(default_factory=dict)
    # agent_id -> relationship strength (-1.0 to +1.0)

    post_count: int = 0
    reply_count: int = 0
