"""Simulation configuration."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class LLMConfig:
    """LLM API settings."""

    model: str = "claude-sonnet-4-6"
    max_tokens: int = 1024
    temperature: float = 0.8
    max_concurrent_calls: int = 5
    cost_limit_daily_usd: float = 10.0


@dataclass
class SimulationConfig:
    """Top-level simulation configuration."""

    # Population
    agent_count: int = 50
    seed: int = 42

    # Time acceleration: how many sim-days per real hour
    sim_days_per_real_hour: float = 12.0  # ~2 weeks per real day

    # Network
    network_k: int = 6  # Watts-Strogatz: each node connected to k nearest
    network_p: float = 0.1  # Rewiring probability

    # Agent activity
    activity_rate: float = 0.4  # Fraction of agents active per step

    # LLM
    llm: LLMConfig = field(default_factory=LLMConfig)

    # Storage
    db_path: Path = Path("data/simulation.db")

    # Topics
    initial_topics: list[str] = field(default_factory=lambda: ["AI regulation"])

    # Hybrid mode: use LLM for top-N most active agents, rule-based for rest
    use_llm: bool = False  # Enable LLM mode
    llm_agent_ratio: float = 0.3  # Fraction of active agents that use LLM
    llm_activity_threshold: float = 0.5  # Extraversion threshold for LLM priority
