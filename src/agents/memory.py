"""Agent memory management."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class MemoryManager:
    """Manages an agent's memory of recent events and discussions."""

    max_items: int = 20  # Max memory entries before summarization

    def add(self, memory_list: list[str], event: str) -> list[str]:
        """Add an event to memory, truncating if needed."""
        memory_list.append(event)
        if len(memory_list) > self.max_items:
            # Keep most recent items, summarize old ones
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
