"""Prompt templates for LLM-powered agent cognition."""

from __future__ import annotations

from src.agents.models import AgentProfile, AgentState
from src.network.models import Post

JSON_TEMPLATE = """```json
{
  "action": "post" or "reply" or "idle",
  "content": "投稿内容（actionがidleの場合は空文字）",
  "reply_to_index": null or 返信先の投稿インデックス（0始まり）,
  "opinion_change": -0.1〜+0.1の範囲で意見の変化量,
  "emotional_reaction": "hope" / "anger" / "anxiety" / "enthusiasm" / "frustration" / "neutral",
  "reasoning": "この行動を取った理由（1文）"
}
```"""


def build_system_prompt(profile: AgentProfile) -> str:
    """Build the system prompt that defines an agent's persona."""
    p = profile.personality
    b = profile.biases

    personality_desc = []
    if p.openness > 0.6:
        personality_desc.append("新しいアイデアに開放的")
    elif p.openness < 0.4:
        personality_desc.append("慎重で伝統を重視")

    if p.extraversion > 0.6:
        personality_desc.append("社交的で積極的に発言")
    elif p.extraversion < 0.4:
        personality_desc.append("内向的で慎重に発言")

    if p.agreeableness > 0.6:
        personality_desc.append("協調的で他者の意見を尊重")
    elif p.agreeableness < 0.4:
        personality_desc.append("批判的で自己主張が強い")

    if p.neuroticism > 0.6:
        personality_desc.append("不安を感じやすく慎重")
    elif p.neuroticism < 0.4:
        personality_desc.append("感情的に安定している")

    if p.conscientiousness > 0.6:
        personality_desc.append("論理的で根拠を重視")

    personality_str = "、".join(personality_desc) if personality_desc else "バランスの取れた性格"
    values_str = "、".join(profile.core_values) if profile.core_values else "特になし"

    bias_hints = []
    if b.confirmation_bias > 0.6:
        bias_hints.append("自分の既存の考えを確認する情報に注目しやすい")
    if b.authority_bias > 0.6:
        bias_hints.append("権威ある情報源の意見を重視する")
    if b.bandwagon_effect > 0.6:
        bias_hints.append("多数派の意見に影響されやすい")
    bias_str = "。".join(bias_hints) if bias_hints else ""

    return f"""あなたは架空のソーシャルネットワーク上のユーザーです。以下の設定に従って振る舞ってください。

## あなたのプロフィール
- 名前: {profile.name}
- 年齢層: {profile.age_group}
- 職業: {profile.occupation}
- 地域: {profile.region}
- 性格: {personality_str}
- 大切にする価値観: {values_str}
{f"- 思考傾向: {bias_str}" if bias_str else ""}

## ルール
- SNSの短い投稿（1〜3文）として自然な日本語で応答してください
- あなたの性格・価値観・職業に基づいた視点で意見を述べてください
- 感情的になりすぎず、かつ人間らしく振る舞ってください
- 他者の投稿への返信は、相手の意見を踏まえた上で応答してください"""


def build_action_prompt(
    state: AgentState,
    feed: list[Post],
    topic_name: str,
    agent_names: dict[str, str] | None = None,
) -> str:
    """Build the user prompt for a single action decision."""
    names = agent_names or {}

    # Format feed
    feed_lines = []
    for post in feed[-5:]:  # Last 5 posts
        author = names.get(post.author_id, post.author_id[:8])
        prefix = "📰 ニュース" if post.is_news_seed else f"@{author}"
        feed_lines.append(f"  {prefix}: {post.content}")

    feed_str = "\n".join(feed_lines) if feed_lines else "  (フィードに投稿なし)"

    # Current opinion
    opinion = state.opinions.get(topic_name.lower().replace(" ", "_"), 0.0)
    if opinion > 0.3:
        opinion_desc = "どちらかと言えば賛成"
    elif opinion < -0.3:
        opinion_desc = "どちらかと言えば反対"
    else:
        opinion_desc = "中立〜やや迷っている"

    # Recent memories
    memory_section = ""
    if state.memory:
        recent = state.memory[-3:]
        items = "\n".join(f"  - {m}" for m in recent)
        memory_section = f"## 最近の記憶\n{items}"

    return f"""## 現在の話題: {topic_name}

## あなたの現在の立場
{opinion_desc}（スコア: {opinion:+.2f}）

## フィードの最近の投稿
{feed_str}

{memory_section}

## タスク
上記のフィードを見て、以下のJSON形式で応答してください：

{JSON_TEMPLATE}"""


def parse_llm_response(response_text: str) -> dict:
    """Parse the LLM's JSON response into an action dict.

    Returns a dict with keys: action, content, reply_to_index, opinion_change,
    emotional_reaction, reasoning.
    Falls back to idle on parse failure.
    """
    import json
    import re

    # Try to extract JSON from response
    json_match = re.search(r"\{[^{}]*\}", response_text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group())
            # Validate required fields
            action = data.get("action", "idle")
            if action not in ("post", "reply", "idle"):
                action = "idle"
            return {
                "action": action,
                "content": str(data.get("content", "")),
                "reply_to_index": data.get("reply_to_index"),
                "opinion_change": max(-0.1, min(0.1, float(data.get("opinion_change", 0)))),
                "emotional_reaction": data.get("emotional_reaction", "neutral"),
                "reasoning": str(data.get("reasoning", "")),
            }
        except (json.JSONDecodeError, ValueError, TypeError):
            pass

    return {
        "action": "idle",
        "content": "",
        "reply_to_index": None,
        "opinion_change": 0.0,
        "emotional_reaction": "neutral",
        "reasoning": "Failed to parse LLM response",
    }
