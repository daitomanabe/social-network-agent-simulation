"""Tests for agent factory, behavior, and memory."""

from src.agents.factory import AgentFactory
from src.agents.behavior import RuleBasedBehavior, ActionType
from src.agents.memory import MemoryManager, MemoryStream, generate_reflection_simple, MemoryItem
from src.agents.models import AgentProfile, AgentState, BigFive
from src.agents.prompts import build_system_prompt, build_action_prompt, parse_llm_response
from src.network.models import Post


class TestAgentFactory:
    def test_generate_population(self):
        agents = AgentFactory.generate_population(30, seed=42)
        assert len(agents) == 30
        profiles = [p for p, _ in agents]
        states = [s for _, s in agents]

        # All have unique IDs
        ids = [p.id for p in profiles]
        assert len(set(ids)) == 30

        # All have names
        assert all(p.name for p in profiles)

    def test_reproducibility(self):
        a1 = AgentFactory.generate_population(10, seed=99)
        a2 = AgentFactory.generate_population(10, seed=99)
        for (p1, _), (p2, _) in zip(a1, a2):
            assert p1.name == p2.name
            assert p1.personality.openness == p2.personality.openness

    def test_diverse_opinions(self):
        agents = AgentFactory.generate_population(50, seed=42, initial_topics=["ai_regulation"])
        opinions = [s.opinions["ai_regulation"] for _, s in agents]
        assert min(opinions) < -0.1
        assert max(opinions) > 0.1
        assert -1.0 <= min(opinions) and max(opinions) <= 1.0


class TestRuleBasedBehavior:
    def test_decide_action(self):
        behavior = RuleBasedBehavior(seed=42)
        profile = AgentProfile(
            name="Test", personality=BigFive(extraversion=0.9)  # Very active
        )
        state = AgentState(agent_id=profile.id, opinions={"ai_regulation": 0.5})
        feed = [Post(author_id="other", content="AI is great", sentiment=0.5, step=1)]

        action = behavior.decide_action(profile, state, feed, "ai_regulation", step=1)
        assert action.action_type in (ActionType.POST, ActionType.REPLY, ActionType.IDLE)

    def test_opinion_shift_bounded(self):
        behavior = RuleBasedBehavior(seed=42)
        profile = AgentProfile(name="Test")
        state = AgentState(agent_id=profile.id, opinions={"ai_regulation": 0.9})

        # Feed with opposing views
        feed = [Post(author_id=f"a{i}", content="Against", sentiment=-0.8, step=1) for i in range(5)]
        action = behavior.decide_action(profile, state, feed, "ai_regulation", step=1)

        if action.opinion_deltas:
            delta = action.opinion_deltas.get("ai_regulation", 0)
            assert -0.15 <= delta <= 0.15  # Bounded


class TestMemory:
    def test_memory_manager_add(self):
        mgr = MemoryManager(max_items=5)
        mem = []
        for i in range(10):
            mem = mgr.add(mem, f"Event {i}")
        assert len(mem) <= 5

    def test_memory_stream(self):
        stream = MemoryStream(reflection_interval=3)
        for i in range(5):
            stream.add_observation(f"Event {i}", step=i)
        assert len(stream.observations) == 5
        assert stream.should_reflect  # 5 > 3

    def test_reflection_generation(self):
        items = [
            MemoryItem(content="AI規制について投稿した", step=i)
            for i in range(10)
        ]
        reflections = generate_reflection_simple(items)
        assert isinstance(reflections, list)

    def test_memory_retrieval(self):
        stream = MemoryStream()
        stream.add_observation("Old event", step=0, importance=0.3)
        stream.add_observation("Recent important", step=10, importance=0.9)
        stream.add_reflection("Key insight", step=10)

        retrieved = stream.retrieve(current_step=10, n=3)
        assert len(retrieved) <= 3
        # Reflection should be high-priority
        assert any(m.is_reflection for m in retrieved)


class TestPrompts:
    def test_system_prompt(self):
        profile = AgentProfile(
            name="佐藤", age_group="30s", occupation="エンジニア",
            personality=BigFive(openness=0.9), core_values=["革新", "自由"]
        )
        prompt = build_system_prompt(profile)
        assert "佐藤" in prompt
        assert "エンジニア" in prompt

    def test_parse_valid_response(self):
        resp = '{"action": "post", "content": "テスト投稿", "opinion_change": 0.05, "emotional_reaction": "hope", "reasoning": "理由"}'
        parsed = parse_llm_response(resp)
        assert parsed["action"] == "post"
        assert parsed["content"] == "テスト投稿"
        assert parsed["opinion_change"] == 0.05

    def test_parse_invalid_response(self):
        parsed = parse_llm_response("not json at all")
        assert parsed["action"] == "idle"

    def test_parse_clamped_opinion(self):
        resp = '{"action": "idle", "opinion_change": 999}'
        parsed = parse_llm_response(resp)
        assert parsed["opinion_change"] == 0.1  # Clamped to max
