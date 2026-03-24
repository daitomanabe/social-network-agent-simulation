"""Main entry point for the Parallel World Simulator."""

from __future__ import annotations

import argparse
import signal
import time
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.table import Table
from rich.text import Text

from src.core.config import SimulationConfig
from src.core.engine import SimulationEngine
from src.network.dynamics import NetworkDynamics
from src.timeline.fork import ParallelWorldEngine
from src.timeline.manager import TimelineManager
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
  python -m src.main inject "AI法案が可決"           # Inject news with fork
  python -m src.main worlds                        # Show parallel worlds
        """,
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # Run command
    run_p = subparsers.add_parser("run", help="Run the simulation")
    _add_common_args(run_p)
    run_p.add_argument("--news", type=str, nargs="*", help="News headlines to inject (creates forks)")
    run_p.add_argument("--news-at", type=int, default=5, help="Step at which to inject news")
    run_p.add_argument("--evolve-network", action="store_true", help="Enable network evolution")

    # Inject command
    inj_p = subparsers.add_parser("inject", help="Inject news with fork and run")
    inj_p.add_argument("headline", type=str, help="News headline")
    inj_p.add_argument("--summary", type=str, default="", help="News summary")
    inj_p.add_argument("--sentiment", type=float, default=0.0)
    inj_p.add_argument("--no-fork", action="store_true", help="Inject without forking")
    _add_common_args(inj_p)

    # Worlds command
    subparsers.add_parser("worlds", help="Show parallel worlds summary")

    # Status command
    subparsers.add_parser("status", help="Show simulation status from DB")

    args = parser.parse_args()
    console = Console()

    if args.command is None:
        args.command = "run"
        _set_defaults(args)

    if args.command == "status":
        _show_status(console)
        return
    if args.command == "worlds":
        console.print("[yellow]Run a simulation first to see parallel worlds.[/]")
        return

    if args.command == "inject":
        if not hasattr(args, "topic"):
            args.topic = "AI regulation"
        if not hasattr(args, "news"):
            args.news = None
        if not hasattr(args, "news_at"):
            args.news_at = 0
        if not hasattr(args, "evolve_network"):
            args.evolve_network = False

    config = SimulationConfig(
        agent_count=args.agents,
        seed=args.seed,
        initial_topics=[args.topic],
        db_path=args.db,
    )

    # Header
    console.print(f"\n[bold cyan]{'='*60}[/]")
    console.print(f"[bold cyan]  🌍 Parallel World Simulator[/]")
    console.print(f"[bold cyan]{'='*60}[/]")
    console.print(f"  Agents: {config.agent_count} | Topic: {args.topic} | Steps: {args.steps}")
    console.print(f"  Seed: {args.seed} | DB: {args.db}")

    # Initialize
    engine = SimulationEngine(config=config, start_time=datetime(2026, 3, 22))
    engine.initialize()

    tm = TimelineManager(engine.db)
    pw = ParallelWorldEngine(engine, tm)
    dynamics = NetworkDynamics(engine.graph)

    console.print(f"[green]  ✓ {len(engine.profiles)} agents initialized[/]")
    console.print(f"[green]  ✓ Network: {engine.graph.stats}[/]")

    # Inject news at start if inject command
    if args.command == "inject":
        post, fork = pw.inject_news_with_fork(
            args.headline,
            args.summary or args.headline,
            args.sentiment,
            create_counterfactual=not args.no_fork,
        )
        console.print(f"[red]  📰 News: {args.headline}[/]")
        if fork:
            console.print(f"[magenta]  🔀 Fork: {fork.timeline.description}[/]")

    console.print()

    # Setup dashboard
    dashboard = CLIDashboard()
    dashboard.set_agent_names({pid: p.name for pid, p in engine.profiles.items()})

    # Handle Ctrl+C
    interrupted = False
    def signal_handler(sig, frame):
        nonlocal interrupted
        interrupted = True
    signal.signal(signal.SIGINT, signal_handler)

    # Prepare news injection at specified step
    news_to_inject = []
    if hasattr(args, "news") and args.news:
        news_to_inject = [(args.news_at, h) for h in args.news]

    # Run
    if args.no_live:
        _run_simple(pw, engine, dynamics, dashboard, console, args, news_to_inject, lambda: interrupted)
    else:
        _run_live(pw, engine, dynamics, dashboard, console, args, news_to_inject, lambda: interrupted)

    # Final summary
    _print_full_summary(console, engine, pw, tm, dynamics)


def _add_common_args(parser):
    parser.add_argument("--agents", type=int, default=50)
    parser.add_argument("--steps", type=int, default=14)
    parser.add_argument("--topic", type=str, default="AI regulation")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--speed", type=float, default=1.0)
    parser.add_argument("--no-live", action="store_true")
    parser.add_argument("--db", type=str, default="data/simulation.db")


def _set_defaults(args):
    args.agents = 50
    args.steps = 14
    args.topic = "AI regulation"
    args.seed = 42
    args.speed = 1.0
    args.no_live = False
    args.db = "data/simulation.db"
    args.news = None
    args.news_at = 5
    args.evolve_network = False


def _run_simple(pw, engine, dynamics, dashboard, console, args, news_to_inject, interrupted_check):
    """Run without live display."""
    evolve = getattr(args, "evolve_network", False)

    for i in range(args.steps):
        if interrupted_check():
            break

        # Check if we need to inject news at this step
        for inject_step, headline in news_to_inject:
            if engine.time.step == inject_step:
                post, fork = pw.inject_news_with_fork(headline, headline, 0.3)
                console.print(f"[red]  📰 Step {inject_step}: {headline}[/]")
                if fork:
                    console.print(f"[magenta]    🔀 Fork: {fork.timeline.description}[/]")

        result = pw.step_all()
        main_stats = result["main"]

        # Evolve network periodically
        if evolve and main_stats["step"] % 5 == 0:
            changes = dynamics.evolve_network(engine.states, engine.topic_id, seed=main_stats["step"])

        # Print step info
        forks_info = ""
        if result["comparisons"]:
            divs = [f'{c["divergence"]:.3f}' for c in result["comparisons"].values()]
            forks_info = f" | Forks: [{', '.join(divs)}]"

        console.print(
            f"[cyan]Step {main_stats['step']}[/] ({main_stats['sim_date']}) | "
            f"Posts: {main_stats['posts']} Replies: {main_stats['replies']} | "
            f"Opinion: {main_stats['mean_opinion']:+.3f} +/- {main_stats['opinion_std']:.3f}"
            f"{forks_info}"
        )

        if args.speed > 0 and i < args.steps - 1:
            time.sleep(args.speed)


def _run_live(pw, engine, dynamics, dashboard, console, args, news_to_inject, interrupted_check):
    """Run with Rich Live dashboard."""
    evolve = getattr(args, "evolve_network", False)

    def on_step(stats, posts):
        dashboard.update(stats, posts)
        dashboard._opinion_dist = engine.get_opinion_distribution()

    engine.on_step_complete(on_step)

    with Live(dashboard.render(), refresh_per_second=2, console=dashboard.console) as live:
        for i in range(args.steps):
            if interrupted_check():
                break

            for inject_step, headline in news_to_inject:
                if engine.time.step == inject_step:
                    pw.inject_news_with_fork(headline, headline, 0.3)

            pw.step_all()

            if evolve and engine.time.step % 5 == 0:
                dynamics.evolve_network(engine.states, engine.topic_id, seed=engine.time.step)

            live.update(dashboard.render())
            if args.speed > 0 and i < args.steps - 1:
                time.sleep(args.speed)


def _print_full_summary(console, engine, pw, tm, dynamics):
    """Print comprehensive final summary."""
    eq = "=" * 60
    console.print(f"\n[bold cyan]{eq}[/]")
    console.print("[bold cyan]  Simulation Complete[/]")
    console.print(f"[bold cyan]{eq}[/]")
    console.print(f"  Sim date: {engine.time.sim_date_str} | Steps: {engine.time.step} | Real: {engine.time.elapsed_real_seconds:.1f}s")
    total = sum(s.post_count + s.reply_count for s in engine.states.values())
    console.print(f"  Total posts: {total}")

    # Opinion distribution
    console.print(f"\n[bold]  Opinion Distribution (Main Timeline):[/]")
    dist = engine.get_opinion_distribution()
    labels = ["強く反対", "反対", "中立", "賛成", "強く賛成"]
    colors = ["red", "yellow", "white", "green", "bright_green"]
    for (bucket, count), label, color in zip(dist.items(), labels, colors):
        bar = "█" * count
        console.print(f"    [{color}]{label:6s} {bar} ({count})[/]")

    # Polarization
    pol = dynamics.compute_polarization(engine.states, engine.topic_id)
    console.print(f"\n[bold]  Polarization Analysis:[/]")
    console.print(f"    Index: {pol.polarization_index:.3f} ", end="")
    if pol.polarization_index > 0.7:
        console.print("[red](highly polarized)[/]")
    elif pol.polarization_index > 0.4:
        console.print("[yellow](moderately polarized)[/]")
    else:
        console.print("[green](low polarization)[/]")
    console.print(f"    Echo chambers: {pol.echo_chamber_count} (avg size: {pol.avg_echo_chamber_size})")
    console.print(f"    Modularity: {pol.modularity:.3f} | Cross-cluster edges: {pol.cross_cluster_edges_ratio:.1%}")

    # Echo chambers detail
    chambers = dynamics.detect_echo_chambers(engine.states, engine.topic_id)
    if chambers:
        console.print(f"\n[bold]  Echo Chambers:[/]")
        for ch in chambers[:5]:
            direction = {"pro": "[green]賛成派[/]", "anti": "[red]反対派[/]", "neutral": "[white]中立[/]"}
            console.print(
                f"    Cluster {ch.id}: {ch.size} agents | "
                f"opinion={ch.mean_opinion:+.3f} | {direction[ch.label]} | "
                f"density={ch.internal_density:.2f}"
            )

    # Parallel worlds
    if pw.forks:
        console.print(f"\n[bold]  Parallel Worlds ({len(pw.forks)} forks):[/]")
        for tid, fork in pw.forks.items():
            tl = tm.timelines[tid]
            fork_dist = fork.get_opinion_distribution()
            main_mean = engine.step_stats[-1]["mean_opinion"] if engine.step_stats else 0
            fork_mean = fork.step_stats[-1]["mean_opinion"] if fork.step_stats else 0
            diff = main_mean - fork_mean

            console.print(f"    [magenta]🔀 {tl.description}[/]")
            console.print(f"       Opinion diff: {diff:+.4f} | Fork step: {tl.fork_step}")

            # Show distribution comparison
            main_dist_vals = list(dist.values())
            fork_dist_vals = list(fork_dist.values())
            for j, (mv, fv, label, color) in enumerate(zip(main_dist_vals, fork_dist_vals, labels, colors)):
                delta = fv - mv
                delta_str = f"({delta:+d})" if delta != 0 else ""
                console.print(f"       [{color}]{label:6s}[/]: main={mv} fork={fv} {delta_str}")

    # Opinion trend
    if engine.step_stats:
        console.print(f"\n[bold]  Opinion Trend:[/]")
        step_interval = max(1, len(engine.step_stats) // 8)
        for s in engine.step_stats[::step_interval]:
            m = s["mean_opinion"]
            bar_pos = int((m + 1) / 2 * 20)  # Map [-1,1] to [0,20]
            bar = " " * bar_pos + "●"
            console.print(
                f"    {s['sim_date']} | {m:+.3f} |{bar}"
            )

    console.print(f"\n  DB: {engine.config.db_path}")
    console.print(f"[bold cyan]{eq}[/]\n")


def _show_status(console):
    """Show status from existing DB."""
    from pathlib import Path
    from src.core.database import Database

    db_path = "data/simulation.db"
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
