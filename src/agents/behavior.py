"""Rule-based agent behavior (no LLM needed)."""

from __future__ import annotations

import random
from dataclasses import dataclass
from enum import Enum

from src.agents.models import AgentProfile, AgentState
from src.network.models import Post


class ActionType(Enum):
    IDLE = "idle"
    POST = "post"
    REPLY = "reply"


@dataclass
class AgentAction:
    """The result of an agent's decision this step."""

    action_type: ActionType = ActionType.IDLE
    post: Post | None = None
    opinion_deltas: dict[str, float] | None = None  # topic_id -> delta
    memory_entry: str | None = None


# Post templates by sentiment direction
_PRO_TEMPLATES = [
    "{topic}について、私は賛成です。{reason}",
    "{topic}は社会にとって必要だと思います。{reason}",
    "{topic}を推進すべきです。{reason}",
    "多くの人が見落としていますが、{topic}には大きなメリットがあります。{reason}",
]

_CON_TEMPLATES = [
    "{topic}について、懸念があります。{reason}",
    "{topic}は慎重に検討すべきです。{reason}",
    "{topic}のリスクを過小評価すべきではありません。{reason}",
    "{topic}には反対です。{reason}",
]

_NEUTRAL_TEMPLATES = [
    "{topic}について、両方の意見を理解できます。",
    "{topic}は複雑な問題で、簡単には判断できません。",
    "皆さんの{topic}についての意見を聞きたいです。",
]

_REPLY_TEMPLATES_AGREE = [
    "同感です。{original_point}",
    "その通りだと思います。",
    "良い視点ですね。私も同じように感じています。",
]

_REPLY_TEMPLATES_DISAGREE = [
    "それは少し違うのではないでしょうか。{counter_point}",
    "その意見には賛成できません。{counter_point}",
    "別の角度から考えると、{counter_point}",
]

_REASONS = {
    "ai_regulation": [
        "AIの発展には適切なルールが必要です",
        "イノベーションを阻害しない範囲で規制すべきです",
        "安全性を確保しながら発展を促進できます",
        "技術の進歩は止められません",
        "市民の権利を守る必要があります",
        "経済的な影響も考慮すべきです",
    ],
}

_DEFAULT_REASONS = [
    "社会全体のバランスを考える必要があります",
    "将来世代のことも考慮すべきです",
    "多様な視点から検討する必要があります",
]


class RuleBasedBehavior:
    """Agent decision-making using simple heuristics (no LLM)."""

    def __init__(self, seed: int | None = None) -> None:
        self.rng = random.Random(seed)
        self._step_seed = 0

    def decide_action(
        self,
        profile: AgentProfile,
        state: AgentState,
        feed: list[Post],
        topic_id: str,
        step: int,
        sim_time_str: str = "",
    ) -> AgentAction:
        """Decide what the agent does this step."""
        # Use step as additional seed for variety
        self.rng.seed(hash((profile.id, step)))

        # Activity probability based on extraversion
        activity_prob = 0.2 + profile.personality.extraversion * 0.5
        if self.rng.random() > activity_prob:
            return AgentAction(action_type=ActionType.IDLE)

        opinion = state.opinions.get(topic_id, 0.0)

        # Process feed: update opinion based on what we see
        opinion_delta = self._compute_opinion_shift(profile, state, feed, topic_id)

        # Decide: post or reply?
        if feed and self.rng.random() < 0.4:
            return self._make_reply(profile, state, feed, topic_id, step, opinion_delta)
        else:
            return self._make_post(profile, state, topic_id, step, opinion, opinion_delta)

    def _compute_opinion_shift(
        self,
        profile: AgentProfile,
        state: AgentState,
        feed: list[Post],
        topic_id: str,
    ) -> float:
        """Bounded confidence model: only influenced by posts within tolerance range."""
        if not feed:
            return 0.0

        my_opinion = state.opinions.get(topic_id, 0.0)
        # Tolerance: how different an opinion can be and still influence us
        # Higher openness = wider tolerance
        tolerance = 0.3 + profile.personality.openness * 0.5

        total_influence = 0.0
        count = 0

        for post in feed:
            post_opinion = post.sentiment
            distance = abs(my_opinion - post_opinion)

            if distance < tolerance:
                # Influenced by this post
                # Confirmation bias: more influenced by agreeing posts
                if (my_opinion > 0 and post_opinion > 0) or (my_opinion < 0 and post_opinion < 0):
                    weight = 1.0 + profile.biases.confirmation_bias * 0.5
                else:
                    weight = 1.0 - profile.biases.confirmation_bias * 0.3

                influence = (post_opinion - my_opinion) * 0.1 * weight

                # Bandwagon: stronger influence if many posts share this opinion
                if profile.biases.bandwagon_effect > 0.5:
                    influence *= 1.2

                total_influence += influence
                count += 1

        if count == 0:
            return 0.0

        # Dampen total shift
        return max(-0.15, min(0.15, total_influence / count))

    def _make_post(
        self,
        profile: AgentProfile,
        state: AgentState,
        topic_id: str,
        step: int,
        opinion: float,
        opinion_delta: float,
    ) -> AgentAction:
        """Generate a new post."""
        reasons = _REASONS.get(topic_id, _DEFAULT_REASONS)
        reason = self.rng.choice(reasons)
        topic_name = topic_id.replace("_", " ").title()

        if opinion > 0.2:
            template = self.rng.choice(_PRO_TEMPLATES)
        elif opinion < -0.2:
            template = self.rng.choice(_CON_TEMPLATES)
        else:
            template = self.rng.choice(_NEUTRAL_TEMPLATES)

        content = template.format(topic=topic_name, reason=reason)

        post = Post(
            author_id=profile.id,
            topic_id=topic_id,
            content=content,
            step=step,
            sentiment=opinion,
        )

        return AgentAction(
            action_type=ActionType.POST,
            post=post,
            opinion_deltas={topic_id: opinion_delta} if opinion_delta != 0 else None,
            memory_entry=f"Step {step}: {topic_name}について投稿した（意見: {opinion:.2f}）",
        )

    def _make_reply(
        self,
        profile: AgentProfile,
        state: AgentState,
        feed: list[Post],
        topic_id: str,
        step: int,
        opinion_delta: float,
    ) -> AgentAction:
        """Generate a reply to a feed post."""
        target = self.rng.choice(feed)
        my_opinion = state.opinions.get(topic_id, 0.0)
        opinion_diff = my_opinion - target.sentiment

        if abs(opinion_diff) < 0.3:
            template = self.rng.choice(_REPLY_TEMPLATES_AGREE)
            content = template.format(original_point="特にその点は重要だと思います")
        else:
            template = self.rng.choice(_REPLY_TEMPLATES_DISAGREE)
            content = template.format(counter_point="異なる視点も検討すべきではないでしょうか")

        post = Post(
            author_id=profile.id,
            topic_id=topic_id,
            content=content,
            step=step,
            sentiment=my_opinion,
            reply_to=target.id,
        )

        return AgentAction(
            action_type=ActionType.REPLY,
            post=post,
            opinion_deltas={topic_id: opinion_delta} if opinion_delta != 0 else None,
            memory_entry=f"Step {step}: 他のエージェントの投稿に返信した",
        )
