#!/usr/bin/env python3
"""Demo: 2026年AI規制をめぐるパラレルワールド

150体のAIエージェントが「AI規制」について議論する社会をシミュレーション。
途中でニュースが投入され、タイムラインが分岐。
「もしAI規制法案が可決されなかったら？」の並行世界が生まれる。

Usage:
    python demo.py              # Full demo with Rich output
    python demo.py --fast       # Fast mode (no delays)
    python demo.py --api        # Start API server for web visualization
"""

from __future__ import annotations

import argparse
import sys
import time
import tempfile
from datetime import datetime

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text


def main():
    parser = argparse.ArgumentParser(description="Parallel World Demo")
    parser.add_argument("--fast", action="store_true", help="No delays")
    parser.add_argument("--api", action="store_true", help="Start API server instead")
    parser.add_argument("--agents", type=int, default=150)
    args = parser.parse_args()

    if args.api:
        run_api_demo(args)
        return

    console = Console()
    speed = 0 if args.fast else 0.3

    # ═══════════════════════════════════════════════════
    # ACT 1: 社会の構築
    # ═══════════════════════════════════════════════════
    console.print(Panel(
        "[bold cyan]ACT 1: 社会の構築[/]\n\n"
        f"{args.agents}人の架空市民を生成し、小世界ネットワークで接続。\n"
        "各市民はBig Five性格特性、認知バイアス、コア価値観を持つ。",
        title="🌍 Parallel World Simulator — Demo",
        border_style="cyan",
    ))
    time.sleep(speed * 3)

    from src.core.world_runner import WorldRunner, WorldRunnerConfig
    from src.core.config import SimulationConfig
    from src.network.dynamics import NetworkDynamics

    config = WorldRunnerConfig(
        sim_config=SimulationConfig(
            agent_count=args.agents,
            seed=2026,
            initial_topics=["AI regulation"],
            db_path=tempfile.mktemp(suffix=".db"),
        ),
        start_time=datetime(2026, 3, 22),
        step_delay_seconds=speed,
        auto_evolve=True,
        evolve_interval=3,
        auto_predict=True,
        predict_interval=7,
    )

    runner = WorldRunner(config)
    runner.initialize()

    dynamics = NetworkDynamics(runner.engine.graph)

    console.print(f"  [green]✓ {args.agents} agents generated[/]")
    console.print(f"  [green]✓ Network: {runner.engine.graph.stats}[/]")

    # Show sample agents
    table = Table(title="Sample Agents", show_lines=False, border_style="dim")
    table.add_column("Name", style="cyan")
    table.add_column("Age")
    table.add_column("Occupation")
    table.add_column("Values")
    table.add_column("Opinion", justify="right")

    profiles = list(runner.engine.profiles.values())[:8]
    for p in profiles:
        s = runner.engine.states[p.id]
        op = s.opinions.get("ai_regulation", 0)
        op_color = "green" if op > 0.2 else "red" if op < -0.2 else "white"
        table.add_row(
            p.name, p.age_group, p.occupation,
            ", ".join(p.core_values[:2]),
            f"[{op_color}]{op:+.2f}[/]",
        )

    console.print(table)
    time.sleep(speed * 2)

    # ═══════════════════════════════════════════════════
    # ACT 2: 平時の議論 (14日間)
    # ═══════════════════════════════════════════════════
    console.print(Panel(
        "[bold yellow]ACT 2: 平時の議論[/]\n\n"
        "ニュースなし。エージェントはネットワーク上で自由に議論。\n"
        "意見の自然な収束・分極化を観察。",
        border_style="yellow",
    ))

    for i in range(14):
        result = runner.step()
        m = result["main"]
        bar = _opinion_bar(m["mean_opinion"])
        console.print(
            f"  [dim]{m['sim_date']}[/] | {bar} | "
            f"posts={m['posts']:2d} replies={m['replies']:2d}"
        )
        time.sleep(speed)

    pol1 = dynamics.compute_polarization(runner.engine.states, "ai_regulation")
    console.print(f"\n  [bold]14日後:[/] Polarization={pol1.polarization_index:.3f} | Chambers={pol1.echo_chamber_count}")
    time.sleep(speed * 2)

    # ═══════════════════════════════════════════════════
    # ACT 3: ニュース投入 → パラレルワールド分岐
    # ═══════════════════════════════════════════════════
    console.print(Panel(
        "[bold red]ACT 3: ニュース投入 → 世界が分岐[/]\n\n"
        "📰「AI規制法案が国会で可決された」\n"
        "→ メインタイムライン: ニュースあり\n"
        "→ フォーク: 「もし可決されなかったら」の並行世界が誕生",
        border_style="red",
    ))
    time.sleep(speed * 2)

    event = runner.inject_news(
        "AI規制法案が国会で賛成多数で可決",
        "AI技術の商用利用に届出義務を課す新法案が、賛成多数で可決された。企業は6ヶ月以内に届出が必要。",
        sentiment=0.4,
    )
    console.print(f"  [red]📰 {event['headline']}[/]")
    console.print(f"  [magenta]🔀 {event['fork_description']}[/]")
    time.sleep(speed * 2)

    # Run 14 more days
    console.print(f"\n  [bold]ニュース後の14日間:[/]")
    for i in range(14):
        result = runner.step()
        m = result["main"]
        comps = result.get("comparisons", {})
        div_str = ""
        if comps:
            div = list(comps.values())[0]["divergence"]
            div_str = f" | [magenta]divergence={div:.3f}[/]"

        bar = _opinion_bar(m["mean_opinion"])
        console.print(
            f"  [dim]{m['sim_date']}[/] | {bar} | "
            f"posts={m['posts']:2d}{div_str}"
        )
        time.sleep(speed)

    # ═══════════════════════════════════════════════════
    # ACT 4: 2つ目のニュース → 2つ目の分岐
    # ═══════════════════════════════════════════════════
    console.print(Panel(
        "[bold magenta]ACT 4: 2つ目の分岐[/]\n\n"
        "📰「AI企業が集団訴訟を提起」\n"
        "→ 新しい並行世界が誕生。3つの世界が同時に走る。",
        border_style="magenta",
    ))

    event2 = runner.inject_news(
        "大手AI企業5社がAI規制法に対して集団訴訟を提起",
        "規制法は技術革新を阻害するとして、大手5社が集団で訴訟を起こした。",
        sentiment=-0.5,
    )
    console.print(f"  [red]📰 {event2['headline']}[/]")
    console.print(f"  [magenta]🔀 {event2['fork_description']}[/]")

    # Run 14 more days with 2 forks
    for i in range(14):
        result = runner.step()
        m = result["main"]
        comps = result.get("comparisons", {})
        if comps and i % 3 == 0:
            divs = " | ".join(f"[magenta]{c['divergence']:.3f}[/]" for c in comps.values())
            bar = _opinion_bar(m["mean_opinion"])
            console.print(
                f"  [dim]{m['sim_date']}[/] | {bar} | forks: {divs}"
            )
        time.sleep(speed * 0.5)

    # ═══════════════════════════════════════════════════
    # ACT 5: 結果の比較
    # ═══════════════════════════════════════════════════
    console.print(Panel(
        "[bold green]ACT 5: パラレルワールド比較[/]\n\n"
        "3つの世界を比較。ニュースがあった世界と、なかった世界で、\n"
        "世論はどう異なるのか？",
        border_style="green",
    ))

    summary = runner.get_world_summary()

    # Main timeline
    main_dist = runner.engine.get_opinion_distribution()
    console.print(f"\n  [bold cyan]Main Timeline ({runner.engine.time.sim_date_str}):[/]")
    _print_dist(console, main_dist)

    # Forks
    for tid, fork in runner.pw.forks.items():
        tl = runner.tm.timelines[tid]
        fork_dist = fork.get_opinion_distribution()
        console.print(f"\n  [bold magenta]🔀 {tl.description}:[/]")
        _print_dist_comparison(console, main_dist, fork_dist)

    # Polarization comparison
    pol_final = dynamics.compute_polarization(runner.engine.states, "ai_regulation")
    console.print(f"\n  [bold]分極化:[/]")
    console.print(f"    開始時: {pol1.polarization_index:.3f}")
    console.print(f"    終了時: {pol_final.polarization_index:.3f}")
    delta = pol_final.polarization_index - pol1.polarization_index
    direction = "↑ 増加" if delta > 0 else "↓ 減少"
    console.print(f"    変化:  {delta:+.3f} ({direction})")

    # Echo chambers
    chambers = dynamics.detect_echo_chambers(runner.engine.states, "ai_regulation")
    if chambers:
        console.print(f"\n  [bold]エコーチェンバー ({len(chambers)}個):[/]")
        for ch in chambers[:4]:
            label = {"pro": "賛成派", "anti": "反対派", "neutral": "中立"}[ch.label]
            console.print(f"    Cluster {ch.id}: {ch.size}人 | {label} | mean={ch.mean_opinion:+.3f}")

    # Trends
    trends = summary.get("trends", [])
    if trends:
        console.print(f"\n  [bold]検出された傾向:[/]")
        for t in trends:
            console.print(f"    → {t}")

    # Predictions
    console.print(f"\n  [bold]予測記録: {summary['predictions']}件[/]")

    # Final
    total_steps = runner.engine.time.step
    real_time = runner.engine.time.elapsed_real_seconds
    total_posts = sum(s.post_count + s.reply_count for s in runner.engine.states.values())

    console.print(Panel(
        f"[bold]Simulation Stats:[/]\n"
        f"  Steps: {total_steps} | Sim span: {runner.engine.time.elapsed_sim_days} days\n"
        f"  Real time: {real_time:.1f}s | Posts: {total_posts}\n"
        f"  Parallel worlds: {1 + len(runner.pw.forks)}\n"
        f"  Predictions: {summary['predictions']} | Echo chambers: {len(chambers)}",
        title="✅ Demo Complete",
        border_style="green",
    ))


def _opinion_bar(mean: float, width: int = 20) -> str:
    """Create a visual bar showing opinion position."""
    pos = int((mean + 1) / 2 * width)
    pos = max(0, min(width - 1, pos))
    bar = list("─" * width)
    bar[width // 2] = "│"
    bar[pos] = "●"
    color = "green" if mean > 0.1 else "red" if mean < -0.1 else "white"
    return f"[{color}]{''.join(bar)}[/] {mean:+.3f}"


def _print_dist(console, dist):
    labels = ["強く反対", "反対", "中立", "賛成", "強く賛成"]
    colors = ["red", "yellow", "white", "green", "bright_green"]
    for (_, count), label, color in zip(dist.items(), labels, colors):
        bar = "█" * count
        console.print(f"    [{color}]{label:6s} {bar} ({count})[/]")


def _print_dist_comparison(console, main_dist, fork_dist):
    labels = ["強く反対", "反対", "中立", "賛成", "強く賛成"]
    colors = ["red", "yellow", "white", "green", "bright_green"]
    main_vals = list(main_dist.values())
    fork_vals = list(fork_dist.values())
    for mv, fv, label, color in zip(main_vals, fork_vals, labels, colors):
        delta = fv - mv
        delta_str = f" ({delta:+d})" if delta != 0 else ""
        console.print(f"    [{color}]{label:6s}[/]: {fv}{delta_str}")


def run_api_demo(args):
    """Start the API server for web-based demo."""
    import uvicorn
    print(f"\n🌍 Parallel World Simulator — Web Demo")
    print(f"   Open http://localhost:8000 in your browser")
    print(f"   Click 'Start' to initialize {args.agents} agents")
    print(f"   Use 'Inject News' to create parallel worlds\n")
    uvicorn.run("src.visualization.api:app", host="0.0.0.0", port=8000)


if __name__ == "__main__":
    main()
