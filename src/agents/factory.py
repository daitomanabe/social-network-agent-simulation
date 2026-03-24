"""Agent population generator."""

from __future__ import annotations

import random

from src.agents.models import AgentProfile, AgentState, BigFive, CognitiveBiases

# Name pools (fictional Japanese-style names)
_FAMILY_NAMES = [
    "佐藤", "鈴木", "高橋", "田中", "伊藤", "渡辺", "山本", "中村", "小林", "加藤",
    "吉田", "山田", "松本", "井上", "木村", "林", "清水", "山口", "阿部", "池田",
    "橋本", "石川", "前田", "藤田", "岡田", "後藤", "長谷川", "村上", "近藤", "石井",
]
_GIVEN_NAMES = [
    "翔太", "蓮", "大翔", "悠真", "陽太", "結衣", "咲良", "陽菜", "凛", "葵",
    "健太", "直樹", "裕子", "美咲", "遥", "拓海", "大輝", "愛", "真央", "優花",
]

_AGE_GROUPS = ["20s", "30s", "40s", "50s", "60s+"]
_AGE_WEIGHTS = [0.20, 0.25, 0.25, 0.18, 0.12]

_OCCUPATIONS = [
    "ソフトウェアエンジニア", "教師", "医療従事者", "会社員", "公務員",
    "自営業", "研究者", "学生", "クリエイター", "営業職",
    "介護福祉士", "農業", "フリーランス", "経営者", "主婦/主夫",
]

_REGIONS = ["北部地区", "南部地区", "東部地区", "西部地区", "中央地区"]

_ALL_VALUES = [
    "自由", "安全", "平等", "伝統", "革新",
    "効率", "公正", "共感", "自立", "協調",
]


def _clamp(v: float) -> float:
    return max(0.0, min(1.0, v))


def _random_big_five(rng: random.Random) -> BigFive:
    """Generate personality with varied distributions (not all centrist)."""
    # Use beta distribution for more natural spread
    def trait() -> float:
        return _clamp(rng.betavariate(2.0, 2.0))

    return BigFive(
        openness=trait(),
        conscientiousness=trait(),
        extraversion=trait(),
        agreeableness=trait(),
        neuroticism=trait(),
    )


def _random_biases(rng: random.Random, personality: BigFive) -> CognitiveBiases:
    """Generate cognitive biases influenced by personality."""
    # Higher neuroticism → stronger confirmation bias
    # Lower openness → stronger authority bias
    # Higher agreeableness → stronger bandwagon effect
    return CognitiveBiases(
        confirmation_bias=_clamp(0.3 + personality.neuroticism * 0.4 + rng.gauss(0, 0.1)),
        authority_bias=_clamp(0.6 - personality.openness * 0.4 + rng.gauss(0, 0.1)),
        bandwagon_effect=_clamp(personality.agreeableness * 0.5 + rng.gauss(0, 0.1)),
        anchoring=_clamp(0.3 + rng.gauss(0, 0.15)),
    )


def _pick_values(rng: random.Random, personality: BigFive) -> list[str]:
    """Pick 2-4 core values influenced by personality."""
    # Weight values based on personality
    weights = {
        "自由": personality.openness,
        "安全": 1.0 - personality.openness + personality.neuroticism,
        "平等": personality.agreeableness,
        "伝統": 1.0 - personality.openness,
        "革新": personality.openness,
        "効率": personality.conscientiousness,
        "公正": personality.agreeableness * 0.5 + 0.3,
        "共感": personality.agreeableness,
        "自立": personality.extraversion * 0.3 + (1 - personality.agreeableness) * 0.5,
        "協調": personality.agreeableness * 0.7 + personality.extraversion * 0.3,
    }
    n = rng.randint(2, 4)
    values_list = list(weights.keys())
    w = [weights[v] + 0.1 for v in values_list]  # Add floor to avoid zero weights
    chosen = []
    for _ in range(n):
        selected = rng.choices(values_list, weights=w, k=1)[0]
        if selected not in chosen:
            chosen.append(selected)
            idx = values_list.index(selected)
            w[idx] = 0  # Don't pick same value twice
    return chosen


class AgentFactory:
    """Generates a diverse, reproducible population of agents."""

    @staticmethod
    def generate_population(
        n: int = 50,
        seed: int = 42,
        initial_topics: list[str] | None = None,
    ) -> list[tuple[AgentProfile, AgentState]]:
        """Generate n agents with diverse attributes.

        Returns list of (profile, initial_state) tuples.
        """
        rng = random.Random(seed)
        initial_topics = initial_topics or ["ai_regulation"]
        agents: list[tuple[AgentProfile, AgentState]] = []

        used_names: set[str] = set()

        for _ in range(n):
            # Generate unique name
            while True:
                name = f"{rng.choice(_FAMILY_NAMES)} {rng.choice(_GIVEN_NAMES)}"
                if name not in used_names:
                    used_names.add(name)
                    break

            personality = _random_big_five(rng)
            biases = _random_biases(rng, personality)
            values = _pick_values(rng, personality)
            age_group = rng.choices(_AGE_GROUPS, weights=_AGE_WEIGHTS, k=1)[0]

            profile = AgentProfile(
                name=name,
                age_group=age_group,
                occupation=rng.choice(_OCCUPATIONS),
                region=rng.choice(_REGIONS),
                personality=personality,
                biases=biases,
                core_values=values,
            )

            # Initial opinions: slightly random, influenced by personality
            opinions = {}
            for topic in initial_topics:
                # Openness correlates with progressive opinions
                base = (personality.openness - 0.5) * 0.6
                noise = rng.gauss(0, 0.3)
                opinions[topic] = max(-1.0, min(1.0, base + noise))

            state = AgentState(
                agent_id=profile.id,
                opinions=opinions,
            )

            agents.append((profile, state))

        return agents
