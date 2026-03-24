"""Data export: CSV, JSON, and summary reports."""

from __future__ import annotations

import csv
import io
import json
from datetime import datetime


class DataExporter:
    """Exports simulation data in various formats."""

    @staticmethod
    def step_stats_to_csv(stats: list[dict]) -> str:
        """Export step statistics to CSV."""
        if not stats:
            return ""
        output = io.StringIO()
        fields = ["step", "sim_date", "active_agents", "posts", "replies",
                  "mean_opinion", "opinion_std", "avg_shift", "total_posts", "llm_calls"]
        writer = csv.DictWriter(output, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(stats)
        return output.getvalue()

    @staticmethod
    def agent_opinions_to_csv(profiles: dict, states: dict, topic_ids: list[str]) -> str:
        """Export all agents' current opinions to CSV."""
        output = io.StringIO()
        fields = ["agent_id", "name", "age_group", "occupation", "region"] + topic_ids + ["post_count", "reply_count"]
        writer = csv.writer(output)
        writer.writerow(fields)

        for aid, profile in profiles.items():
            state = states.get(aid)
            if not state:
                continue
            row = [
                aid, profile.name, profile.age_group, profile.occupation, profile.region,
            ]
            for tid in topic_ids:
                row.append(f"{state.opinions.get(tid, 0.0):.4f}")
            row.extend([state.post_count, state.reply_count])
            writer.writerow(row)

        return output.getvalue()

    @staticmethod
    def opinion_history_to_csv(history_tracker) -> str:
        """Export opinion history timeline to CSV."""
        output = io.StringIO()
        writer = csv.writer(output)
        writer.writerow(["agent_id", "step", "sim_date", "topic_id", "opinion", "event"])

        for aid, history in history_tracker.agent_histories.items():
            for snap in history.snapshots:
                writer.writerow([
                    aid, snap.step, snap.sim_date, snap.topic_id,
                    f"{snap.opinion:.4f}", snap.event,
                ])

        return output.getvalue()

    @staticmethod
    def network_to_json(graph, profiles: dict, states: dict, topic_id: str) -> str:
        """Export network graph to JSON (D3.js format)."""
        G = graph.graph
        nodes = []
        for nid in G.nodes():
            p = profiles.get(nid)
            s = states.get(nid)
            if not p or not s:
                continue
            nodes.append({
                "id": nid,
                "name": p.name,
                "opinion": round(s.opinions.get(topic_id, 0.0), 4),
                "age_group": p.age_group,
                "occupation": p.occupation,
                "extraversion": round(p.personality.extraversion, 3),
                "post_count": s.post_count + s.reply_count,
            })

        links = []
        for u, v, d in G.edges(data=True):
            links.append({
                "source": u, "target": v,
                "weight": round(d.get("weight", 0.5), 3),
            })

        return json.dumps({"nodes": nodes, "links": links}, ensure_ascii=False, indent=2)

    @staticmethod
    def full_report_json(engine, pw=None, dynamics=None) -> str:
        """Generate a comprehensive JSON report of the simulation state."""
        report = {
            "metadata": {
                "generated_at": datetime.now().isoformat(),
                "sim_date": engine.time.sim_date_str,
                "total_steps": engine.time.step,
                "agent_count": len(engine.profiles),
                "topics": engine.topic_ids,
            },
            "opinion_distribution": engine.get_opinion_distribution(),
            "step_stats": engine.step_stats,
        }

        # Parallel worlds
        if pw:
            report["parallel_worlds"] = pw.get_world_summary()

        # Polarization
        if dynamics:
            pol = dynamics.compute_polarization(engine.states, engine.topic_id)
            chambers = dynamics.detect_echo_chambers(engine.states, engine.topic_id)
            report["polarization"] = {
                "index": pol.polarization_index,
                "modularity": pol.modularity,
                "echo_chambers": [
                    {"id": ch.id, "size": ch.size, "mean_opinion": ch.mean_opinion, "label": ch.label}
                    for ch in chambers
                ],
            }

        # Propagation
        if engine.propagation.cascades:
            report["cascades"] = engine.propagation.get_all_summaries()

        # History stats
        report["history"] = engine.history.get_stats()
        report["most_changed_agents"] = engine.history.get_most_changed_agents(engine.topic_id, top_n=10)

        return json.dumps(report, ensure_ascii=False, indent=2, default=str)
