"""Tests for data models."""

from datetime import datetime

from src.agents.models import AgentProfile, AgentState, BigFive, CognitiveBiases, EmotionalState
from src.core.config import SimulationConfig
from src.core.models import SimTime, Topic
from src.network.models import Post


def test_simulation_config_defaults():
    config = SimulationConfig()
    assert config.agent_count == 50
    assert config.seed == 42
    assert config.activity_rate == 0.4
    assert config.use_llm is False


def test_sim_time_advance():
    st = SimTime(current=datetime(2026, 3, 22))
    st.advance_days(7)
    assert st.step == 1
    assert st.elapsed_days == 7
    assert st.current.day == 29


def test_agent_profile_creation():
    p = AgentProfile(name="Test Agent", age_group="30s", occupation="engineer")
    assert p.name == "Test Agent"
    assert len(p.id) == 10
    assert p.personality.openness == 0.5  # default


def test_agent_state_defaults():
    s = AgentState(agent_id="test123")
    assert s.opinions == {}
    assert s.post_count == 0
    # Default emotional state has hope=0.5, enthusiasm=0.3 → positive valence
    assert s.emotional_state.valence > 0


def test_emotional_valence():
    e = EmotionalState(anger=0.0, anxiety=0.0, hope=1.0, enthusiasm=1.0, frustration=0.0)
    assert e.valence > 0  # Positive emotions dominate

    e2 = EmotionalState(anger=1.0, anxiety=1.0, hope=0.0, enthusiasm=0.0, frustration=1.0)
    assert e2.valence < 0  # Negative emotions dominate


def test_post_creation():
    p = Post(author_id="a1", topic_id="ai_regulation", content="Test post", step=1)
    assert p.author_id == "a1"
    assert p.is_news_seed is False
    assert len(p.id) == 12


def test_topic():
    t = Topic(name="AI Regulation", temperature=0.8)
    assert t.temperature == 0.8
    assert len(t.id) == 8
