"""Agent memory management with reflection and summarization.

Inspired by Generative Agents (Park et al., 2023):
- Observation: raw events are stored
- Reflection: periodically synthesize higher-level insights
- Retrieval: weighted by recency, importance, and relevance
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class MemoryItem:
    """A single memory entry with metadata."""

    content: str
    step: int = 0
    importance: float = 0.5  # 0.0 (trivial) to 1.0 (critical)
    is_reflection: bool = False  # True if this is a synthesized insight
    topic_id: str = ""
    created_at: str = ""

    def __str__(self) -> str:
        prefix = "💡" if self.is_reflection else "📝"
        return f"{prefix} [Step {self.step}] {self.content}"


@dataclass
class MemoryStream:
    """An agent's full memory stream with observations and reflections."""

    items: list[MemoryItem] = field(default_factory=list)
    max_observations: int = 30
    reflection_interval: int = 10  # Generate reflection every N observations
    _obs_since_reflection: int = 0

    def add_observation(
        self,
        content: str,
        step: int,
        topic_id: str = "",
        importance: float = 0.5,
    ) -> None:
        """Add a raw observation."""
        item = MemoryItem(
            content=content,
            step=step,
            importance=importance,
            is_reflection=False,
            topic_id=topic_id,
        )
        self.items.append(item)
        self._obs_since_reflection += 1

        # Compact if too many observations
        if len(self.observations) > self.max_observations:
            self._compact_observations()

    def add_reflection(self, content: str, step: int, topic_id: str = "") -> None:
        """Add a synthesized reflection (higher-level insight)."""
        item = MemoryItem(
            content=content,
            step=step,
            importance=0.8,  # Reflections are always high importance
            is_reflection=True,
            topic_id=topic_id,
        )
        self.items.append(item)
        self._obs_since_reflection = 0

    @property
    def observations(self) -> list[MemoryItem]:
        return [m for m in self.items if not m.is_reflection]

    @property
    def reflections(self) -> list[MemoryItem]:
        return [m for m in self.items if m.is_reflection]

    @property
    def should_reflect(self) -> bool:
        return self._obs_since_reflection >= self.reflection_interval

    def retrieve(
        self,
        current_step: int,
        topic_id: str = "",
        n: int = 5,
    ) -> list[MemoryItem]:
        """Retrieve most relevant memories using weighted scoring.

        Score = recency * 0.4 + importance * 0.4 + relevance * 0.2
        """
        if not self.items:
            return []

        scored = []
        for item in self.items:
            # Recency: exponential decay
            age = current_step - item.step
            recency = 1.0 / (1.0 + age * 0.1)

            # Relevance: topic match
            relevance = 1.0 if (topic_id and item.topic_id == topic_id) else 0.3

            # Combined score
            score = recency * 0.4 + item.importance * 0.4 + relevance * 0.2

            # Reflections get a bonus
            if item.is_reflection:
                score *= 1.3

            scored.append((item, score))

        scored.sort(key=lambda x: x[1], reverse=True)
        return [item for item, _ in scored[:n]]

    def _compact_observations(self) -> None:
        """Remove old, low-importance observations."""
        obs = self.observations
        reflections = self.reflections

        # Keep reflections and recent/important observations
        obs.sort(key=lambda m: m.importance + (1.0 if m.step > obs[-1].step - 10 else 0), reverse=True)
        keep_n = max(10, self.max_observations // 2)
        kept_obs = obs[:keep_n]

        self.items = reflections + kept_obs
        self.items.sort(key=lambda m: m.step)

    def to_legacy_list(self) -> list[str]:
        """Convert to the legacy list[str] format for backwards compatibility."""
        return [item.content for item in self.items]

    @staticmethod
    def from_legacy_list(memories: list[str]) -> "MemoryStream":
        """Create a MemoryStream from legacy list format."""
        stream = MemoryStream()
        for i, content in enumerate(memories):
            stream.items.append(MemoryItem(content=content, step=i))
        return stream

    def format_for_prompt(self, current_step: int, topic_id: str = "", max_items: int = 5) -> str:
        """Format retrieved memories for LLM prompt context."""
        relevant = self.retrieve(current_step, topic_id, n=max_items)
        if not relevant:
            return "（記憶なし）"
        return "\n".join(f"- {item}" for item in relevant)


def generate_reflection_prompt(observations: list[MemoryItem]) -> str:
    """Build a prompt for LLM-based reflection generation."""
    obs_text = "\n".join(f"- {obs.content}" for obs in observations[-10:])
    return f"""以下はあなたの最近の経験・観察です：

{obs_text}

上記の経験を振り返り、以下の形式で1〜2つの高次の洞察（リフレクション）を生成してください：

```json
{{
  "reflections": [
    "洞察1: ...",
    "洞察2: ..."
  ]
}}
```

洞察は具体的なイベントではなく、パターンや傾向についてのものにしてください。"""


def generate_reflection_simple(observations: list[MemoryItem]) -> list[str]:
    """Generate reflections without LLM (rule-based heuristic).

    Identifies patterns in recent observations to create insights.
    """
    if len(observations) < 3:
        return []

    recent = observations[-10:]
    reflections = []

    # Pattern 1: Opinion consistency
    opinion_mentions = [obs for obs in recent if "意見" in obs.content or "投稿" in obs.content]
    if len(opinion_mentions) >= 3:
        reflections.append("最近の議論で自分の意見が一貫している。立場が固まりつつある。")

    # Pattern 2: Active participation
    post_count = sum(1 for obs in recent if "投稿した" in obs.content)
    reply_count = sum(1 for obs in recent if "返信" in obs.content)
    if post_count + reply_count >= 5:
        reflections.append("議論への参加が活発になっている。このトピックへの関心が高まっている。")
    elif post_count + reply_count <= 1:
        reflections.append("最近はあまり議論に参加していない。関心が薄れているかもしれない。")

    # Pattern 3: Exposure to diverse opinions
    diverse = [obs for obs in recent if "異なる" in obs.content or "反対" in obs.content or "disagree" in obs.content.lower()]
    if len(diverse) >= 2:
        reflections.append("異なる意見に触れる機会が多い。視野が広がっている可能性がある。")

    return reflections[:2]


# Backwards-compatible wrapper
@dataclass
class MemoryManager:
    """Manages an agent's memory of recent events and discussions.

    Supports both legacy list[str] format and new MemoryStream format.
    """

    max_items: int = 20

    def add(self, memory_list: list[str], event: str) -> list[str]:
        """Add an event to memory, truncating if needed (legacy interface)."""
        memory_list.append(event)
        if len(memory_list) > self.max_items:
            memory_list = self._compact(memory_list)
        return memory_list

    def get_recent(self, memory_list: list[str], n: int = 5) -> list[str]:
        """Get the N most recent memories."""
        return memory_list[-n:]

    def _compact(self, memory_list: list[str]) -> list[str]:
        """Simple compaction: keep last 75% of items."""
        keep = max(5, int(self.max_items * 0.75))
        return memory_list[-keep:]

    def format_for_prompt(self, memory_list: list[str], max_items: int = 5) -> str:
        """Format recent memories for LLM prompt context."""
        recent = self.get_recent(memory_list, max_items)
        if not recent:
            return "（記憶なし）"
        return "\n".join(f"- {m}" for m in recent)
