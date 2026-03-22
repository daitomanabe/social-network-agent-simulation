"""Main entry point for the Parallel World Simulator."""

from __future__ import annotations

import argparse
import signal
import sys
import time
from datetime import datetime

from rich.console import Console
from rich.live import Live

from src.core.config import SimulationConfig
from src.core.engine import SimulationEngine
from src.news.ingestion import NewsIngester, TopicExtractor
from src.visualization.cli import CLIDashboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel World Social Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main run                           # Run with defaults
  python -m src.main run --agents 100 --steps 30   # Custom run
  python -m src.main run --no-live --speed 0       # Fast, no dashboard
  python -m src.main inject "AI法案が可決"           # Inject news then run
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Run command
    run_parser = subparsers.add_parser("run", help="Run the simulation")
    run_parser.add_argument("--agents", type=int, default=50)
    run_parser.add_argument("--steps", type=int, default=14)
    run_parser.add_argument("--topic", type=str, default="AI regulation")
    run_parser.add_argument("--seed", type=int, default=42)
    run_parser.add_argument("--speed", type=float, default=1.0, help="Delay between steps (seconds)")
    run_parser.add_argument("--no-live", action="store_true")
    run_parser.add_argument("--db", type=str, default="data/simulation.db")
    run_parser.add_argument("--news", type=str, nargs="*", help="News headlines to inject before run")

    # Inject command
    inject_parser = subparsers.add_parser("inject", help="Inject news and run")
    inject_parser.add_argument("headline", type=str, help="News headline")
    inject_parser.add_argument("--summary", type=str, default="", help="News summary")
    inject_parser.add_argument("--sentiment", type=float, default=0.0, help="Sentiment (-1 to 1)")
    inject_parser.add_argument("--agents", type=int, default=50)
    inject_parser.add_argument("--steps", type=int, default=14)
    inject_parser.add_argument("--seed", type=int, default=42)
    inject_parser.add_argument("--speed", type=float, default=1.0)
    inject_parser.add_argument("--no-live", action="store_true")
    inject_parser.add_argument("--db", type=str, default="data/simulation.db")

    # Status command
    subparsers.add_parser("status", help="Show simulation status from DB")

    args = parser.parse_args()
    console = Console()

    if args.command is None:
        # Default: run
        args.command = "run"
        args.agents = 50
        args.steps = 14
        args.topic = "AI regulation"
        args.seed = 42
        args.speed = 1.0
        args.no_live = False
        args.db = "data/simulation.db"
        args.news = None

    if args.command == "status":
        _show_status(console, "data/simulation.db")
        return

    if args.command == "inject":
        args.topic = "AI regulation"
        args.news = None

    config = SimulationConfig(
        agent_count=args.agents,
        seed=args.seed,
        initial_topics=[args.topic] if hasattr(args, "topic") else ["AI regulation"],
        db_path=args.db,
    )

    console.print(f"\n[bold cyan]🌍 Parallel World Simulator[/]")
    console.print(f"  Agents: {config.agent_count} | Topic: {args.topic} | Steps: {args.steps}")
    console.print(f"  Seed: {args.seed} | DB: {args.db}")

    engine = SimulationEngine(config=config, start_time=datetime(2026, 3, 22))
    engine.initialize()

    console.print(f"[green]  ✓ Initialized {len(engine.profiles)} agents[/]")
    console.print(f"[green]  ✓ Network: {engine.graph.stats}[/]")

    # Inject news if provided
    if args.command == "inject":
        news = engine.inject_news(args.headline, args.summary or args.headline, args.sentiment)
        console.print(f"[red]  📰 News injected: {args.headline}[/]")
    elif hasattr(args, "news") and args.news:
        for headline in args.news:
            engine.inject_news(headline, headline)
            console.print(f"[red]  📰 News injected: {headline}[/]")

    console.print()

    dashboard = CLIDashboard()
    dashboard.set_agent_names({pid: p.name for pid, p in engine.profiles.items()})

    # Handle Ctrl+C
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, signal_handler)

    if args.no_live:
        _run_simple(engine, dashboard, args.steps, args.speed, lambda: interrupted)
    else:
        _run_live(engine, dashboard, args.steps, args.speed, lambda: interrupted)

    # Final summary
    _print_summary(console, engine)


def _run_simple(engine, dashboard, n_steps, speed, interrupted_check):
    """Run without live display."""
    for i in range(n_steps):
        if interrupted_check():
            break
        stats = engine.step()
        dashboard.print_step_summary(stats)
        if speed > 0 and i < n_steps - 1:
            time.sleep(speed)


def _run_live(engine, dashboard, n_steps, speed, interrupted_check):
    """Run with Rich Live dashboard."""
    def on_step(stats, posts):
        dashboard.update(stats, posts)
        dashboard._opinion_dist = engine.get_opinion_distribution()

    engine.on_step_complete(on_step)

    with Live(dashboard.render(), refresh_per_second=2, console=dashboard.console) as live:
        for i in range(n_steps):
            if interrupted_check():
                break
            engine.step()
            live.update(dashboard.render())
            if speed > 0 and i < n_steps - 1:
                time.sleep(speed)


def _print_summary(console, engine):
    """Print final simulation summary."""
    eq = "=" * 60
    console.print(f"\n[bold cyan]{eq}[/]")
    console.print("[bold]Simulation Complete[/]")
    console.print(f"  Final sim date: {engine.time.sim_date_str}")
    console.print(f"  Total steps: {engine.time.step}")
    console.print(f"  Real time: {engine.time.elapsed_real_seconds:.1f}s")
    total = sum(s.post_count + s.reply_count for s in engine.states.values())
    console.print(f"  Total posts: {total}")

    console.print(f"\n[bold]Final Opinion Distribution:[/]")
    dist = engine.get_opinion_distribution()
    labels = ["強く反対", "反対", "中立", "賛成", "強く賛成"]
    colors = ["red", "yellow", "white", "green", "bright_green"]
    for (bucket, count), label, color in zip(dist.items(), labels, colors):
        bar = "█" * count
        console.print(f"  [{color}]{label:6s} {bar} ({count})[/]")

    # Show step-by-step opinion trend
    if engine.step_stats:
        console.print(f"\n[bold]Opinion Trend:[/]")
        for s in engine.step_stats[::max(1, len(engine.step_stats) // 10)]:
            m = s["mean_opinion"]
            indicator = "+" if m > 0 else ""
            console.print(
                f"  {s['sim_date']} | mean={indicator}{m:.3f} std={s['opinion_std']:.3f} "
                f"| posts={s['posts']} replies={s['replies']}"
            )

    console.print(f"\n  DB saved to: {engine.config.db_path}")


def _show_status(console, db_path):
    """Show status from existing DB."""
    from src.core.database import Database
    from pathlib import Path

    if not Path(db_path).exists():
        console.print("[red]No simulation database found.[/]")
        return

    db = Database(db_path)
    agents = db.get_all_agents()
    console.print(f"Agents: {len(agents)}")
    state = db.get_state("last_step")
    console.print(f"Last step: {state}")
    db.close()


if __name__ == "__main__":
    main()
