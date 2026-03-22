"""Rich CLI dashboard for simulation visualization."""

from __future__ import annotations

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from src.network.models import Post


class CLIDashboard:
    """Real-time CLI visualization using Rich."""

    def __init__(self) -> None:
        self.console = Console()
        self._latest_stats: dict = {}
        self._recent_posts: list[Post] = []
        self._opinion_dist: dict = {}
        self._max_recent_posts = 8
        self._agent_names: dict[str, str] = {}  # id -> name

    def set_agent_names(self, names: dict[str, str]) -> None:
        self._agent_names = names

    def update(self, stats: dict, posts: list[Post]) -> None:
        """Called after each step with new data."""
        self._latest_stats = stats
        self._recent_posts = (self._recent_posts + posts)[-self._max_recent_posts:]

    def render(self) -> str:
        """Render the full dashboard as a string for Rich Live."""
        layout = Layout()
        layout.split_column(
            Layout(name="header", size=3),
            Layout(name="body"),
            Layout(name="footer", size=3),
        )
        layout["body"].split_row(
            Layout(name="left", ratio=1),
            Layout(name="right", ratio=1),
        )

        layout["header"].update(self._render_header())
        layout["left"].update(self._render_opinions())
        layout["right"].update(self._render_posts())
        layout["footer"].update(self._render_stats())

        return layout

    def _render_header(self) -> Panel:
        s = self._latest_stats
        if not s:
            return Panel("⏳ Initializing...", title="Parallel World Simulator")

        title = Text()
        title.append("🌍 Parallel World Simulator", style="bold cyan")
        title.append(f"  |  Step {s.get('step', 0)}", style="white")
        title.append(f"  |  📅 {s.get('sim_date', '?')}", style="yellow")
        return Panel(title, style="blue")

    def _render_opinions(self) -> Panel:
        s = self._latest_stats
        if not s:
            return Panel("...", title="Opinion Distribution")

        table = Table(show_header=False, expand=True, padding=(0, 1))
        table.add_column("Range", style="bold", width=12)
        table.add_column("Bar", ratio=1)
        table.add_column("N", width=4, justify="right")

        # Build distribution from stats
        # We'll compute it from the opinion buckets
        dist = self._opinion_dist
        if not dist:
            return Panel("Waiting for data...", title="📊 Opinion Distribution")

        max_count = max(dist.values()) if dist.values() else 1
        colors = ["red", "yellow", "white", "green", "bright_green"]
        labels = ["強く反対", "反対", "中立", "賛成", "強く賛成"]

        for (bucket, count), color, label in zip(dist.items(), colors, labels):
            bar_len = int((count / max(max_count, 1)) * 30)
            bar = "█" * bar_len
            table.add_row(label, Text(bar, style=color), str(count))

        mean = s.get("mean_opinion", 0)
        std = s.get("opinion_std", 0)
        subtitle = f"mean={mean:+.3f}  std={std:.3f}"

        return Panel(table, title="📊 Opinion Distribution", subtitle=subtitle)

    def _render_posts(self) -> Panel:
        if not self._recent_posts:
            return Panel("No posts yet...", title="💬 Recent Posts")

        table = Table(show_header=False, expand=True, padding=(0, 1))
        table.add_column("Post", ratio=1)

        for post in reversed(self._recent_posts[-6:]):
            author = self._agent_names.get(post.author_id, post.author_id[:8])
            if post.is_news_seed:
                style = "bold red"
                prefix = "📰"
            elif post.reply_to:
                style = "dim"
                prefix = "↪"
            else:
                style = "white"
                prefix = "💭"

            text = Text()
            text.append(f"{prefix} ", style=style)
            text.append(f"[{author}] ", style="bold cyan")
            content = post.content[:80] + ("..." if len(post.content) > 80 else "")
            text.append(content, style=style)
            table.add_row(text)

        return Panel(table, title="💬 Recent Posts")

    def _render_stats(self) -> Panel:
        s = self._latest_stats
        if not s:
            return Panel("...", title="Stats")

        text = Text()
        text.append(f"Active: {s.get('active_agents', 0)}", style="green")
        text.append(f"  |  Posts: {s.get('posts', 0)}", style="yellow")
        text.append(f"  |  Replies: {s.get('replies', 0)}", style="cyan")
        text.append(f"  |  Total: {s.get('total_posts', 0)}", style="white")
        text.append(f"  |  Avg Shift: {s.get('avg_shift', 0):.4f}", style="magenta")
        return Panel(text, style="dim")

    def print_step_summary(self, stats: dict) -> None:
        """Print a one-line summary for non-live mode."""
        s = stats
        self.console.print(
            f"[cyan]Step {s['step']}[/] ({s['sim_date']}) | "
            f"Posts: {s['posts']} Replies: {s['replies']} | "
            f"Opinion: {s['mean_opinion']:+.3f} ± {s['opinion_std']:.3f}"
        )
