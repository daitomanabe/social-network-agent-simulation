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
from src.visualization.cli import CLIDashboard


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Parallel World Social Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python -m src.main                        # Run with defaults (50 agents, 14 steps)
  python -m src.main --agents 100 --steps 30
  python -m src.main --topic "climate change" --no-live
        """,
    )
    parser.add_argument("--agents", type=int, default=50, help="Number of agents (default: 50)")
    parser.add_argument("--steps", type=int, default=14, help="Steps to run (default: 14 = 2 weeks)")
    parser.add_argument("--topic", type=str, default="AI regulation", help="Initial topic")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--speed", type=float, default=1.0, help="Delay between steps in seconds")
    parser.add_argument("--no-live", action="store_true", help="Disable live dashboard")
    parser.add_argument("--db", type=str, default="data/simulation.db", help="Database path")
    args = parser.parse_args()

    console = Console()

    config = SimulationConfig(
        agent_count=args.agents,
        seed=args.seed,
        initial_topics=[args.topic],
        db_path=args.db,
    )

    console.print(f"\n[bold cyan]🌍 Parallel World Simulator[/]")
    console.print(f"  Agents: {config.agent_count} | Topic: {args.topic} | Steps: {args.steps}")
    console.print(f"  Seed: {args.seed} | DB: {args.db}\n")

    engine = SimulationEngine(config=config, start_time=datetime(2026, 3, 22))
    engine.initialize()

    console.print(f"[green]✓ Initialized {len(engine.profiles)} agents[/]")
    console.print(f"[green]✓ Network: {engine.graph.stats}[/]\n")

    dashboard = CLIDashboard()
    dashboard.set_agent_names({
        pid: p.name for pid, p in engine.profiles.items()
    })

    # Handle Ctrl+C gracefully
    interrupted = False

    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True

    signal.signal(signal.SIGINT, signal_handler)

    if args.no_live:
        _run_simple(engine, dashboard, args.steps, args.speed, interrupted_check=lambda: interrupted)
    else:
        _run_live(engine, dashboard, args.steps, args.speed, interrupted_check=lambda: interrupted)

    # Final summary
    console.print(f"\n[bold cyan]{'='*60}[/]")
    console.print(f"[bold]Simulation Complete[/]")
    console.print(f"  Final sim date: {engine.time.sim_date_str}")
    console.print(f"  Total steps: {engine.time.step}")
    console.print(f"  Real time: {engine.time.elapsed_real_seconds:.1f}s")
    console.print(f"\n[bold]Final Opinion Distribution:[/]")
    dist = engine.get_opinion_distribution()
    labels = ["強く反対", "反対", "中立", "賛成", "強く賛成"]
    colors = ["red", "yellow", "white", "green", "bright_green"]
    for (bucket, count), label, color in zip(dist.items(), labels, colors):
        bar = "█" * count
        console.print(f"  [{color}]{label:6s} {bar} ({count})[/]")

    console.print(f"\n  DB saved to: {config.db_path}")


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


if __name__ == "__main__":
    main()
