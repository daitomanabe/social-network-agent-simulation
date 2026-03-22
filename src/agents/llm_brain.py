"""LLM-powered agent cognition."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from src.agents.behavior import ActionType, AgentAction
from src.agents.models import AgentProfile, AgentState
from src.agents.prompts import build_action_prompt, build_system_prompt, parse_llm_response
from src.core.llm_client import LLMClient
from src.network.models import Post


class LLMBrain:
    """Agent brain powered by Claude API."""

    def __init__(self, client: LLMClient | None = None) -> None:
        self.client = client or LLMClient()

    async def think(
        self,
        profile: AgentProfile,
        state: AgentState,
        feed: list[Post],
        topic_name: str,
        step: int,
        agent_names: dict[str, str] | None = None,
    ) -> AgentAction:
        """Process feed and decide on an action using LLM."""
        system = build_system_prompt(profile)
        user = build_action_prompt(state, feed, topic_name, agent_names)

        response_text = await self.client.complete(system, user, use_cache=False)
        parsed = parse_llm_response(response_text)

        # Build the action
        action_type = {
            "post": ActionType.POST,
            "reply": ActionType.REPLY,
            "idle": ActionType.IDLE,
        }.get(parsed["action"], ActionType.IDLE)

        post = None
        topic_id = topic_name.lower().replace(" ", "_")

        if action_type in (ActionType.POST, ActionType.REPLY) and parsed["content"]:
            reply_to = None
            if action_type == ActionType.REPLY and parsed["reply_to_index"] is not None:
                idx = parsed["reply_to_index"]
                if 0 <= idx < len(feed):
                    reply_to = feed[idx].id

            post = Post(
                author_id=profile.id,
                topic_id=topic_id,
                content=parsed["content"],
                step=step,
                sentiment=state.opinions.get(topic_id, 0.0),
                reply_to=reply_to,
            )

        opinion_deltas = None
        if parsed["opinion_change"] != 0:
            opinion_deltas = {topic_id: parsed["opinion_change"]}

        # Update emotional state
        emo = parsed.get("emotional_reaction", "neutral")
        _update_emotion(state, emo)

        return AgentAction(
            action_type=action_type,
            post=post,
            opinion_deltas=opinion_deltas,
            memory_entry=(
                f"Step {step}: {parsed.get('reasoning', '')}"
                if parsed.get("reasoning")
                else None
            ),
        )

    async def think_batch(
        self,
        agents: list[tuple[AgentProfile, AgentState, list[Post]]],
        topic_name: str,
        step: int,
        agent_names: dict[str, str] | None = None,
    ) -> list[AgentAction]:
        """Process multiple agents concurrently."""
        tasks = [
            self.think(profile, state, feed, topic_name, step, agent_names)
            for profile, state, feed in agents
        ]
        return await asyncio.gather(*tasks)


def _update_emotion(state: AgentState, reaction: str) -> None:
    """Update emotional state based on LLM-reported reaction."""
    decay = 0.9  # Existing emotions decay slightly
    emo = state.emotional_state

    emo.anger *= decay
    emo.anxiety *= decay
    emo.hope *= decay
    emo.frustration *= decay
    emo.enthusiasm *= decay

    boost = 0.2
    if reaction == "anger":
        emo.anger = min(1.0, emo.anger + boost)
    elif reaction == "anxiety":
        emo.anxiety = min(1.0, emo.anxiety + boost)
    elif reaction == "hope":
        emo.hope = min(1.0, emo.hope + boost)
    elif reaction == "enthusiasm":
        emo.enthusiasm = min(1.0, emo.enthusiasm + boost)
    elif reaction == "frustration":
        emo.frustration = min(1.0, emo.frustration + boost)
