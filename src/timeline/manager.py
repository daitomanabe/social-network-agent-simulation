"""Timeline manager: snapshots, forks, and state restoration."""

from __future__ import annotations

import copy
import json
import logging
from dataclasses import asdict

from src.agents.models import AgentProfile, AgentState, BigFive, CognitiveBiases, EmotionalState
from src.core.database import Database
from src.network.graph import SocialGraph
from src.timeline.models import (
    ForkPoint,
    Snapshot,
    Timeline,
    TimelineComparison,
    TimelineStatus,
)

logger = logging.getLogger(__name__)


class TimelineManager:
    """Manages timeline branching, snapshots, and comparisons.

    The main timeline always exists. When news is injected, a fork can
    be created that represents a counterfactual ("what if this didn't happen").
    """

    def __init__(self, db: Database) -> None:
        self.db = db
        self.timelines: dict[str, Timeline] = {}
        self.fork_points: list[ForkPoint] = []
        self.snapshots: dict[str, Snapshot] = {}  # snapshot_id -> Snapshot
        self._init_db_tables()

        # Create main timeline
        main = Timeline(name="main", description="Main timeline with all real news")
        self.timelines[main.id] = main
        self.main_timeline_id = main.id

    def _init_db_tables(self) -> None:
        """Create timeline-specific tables."""
        self.db.conn.executescript("""
            CREATE TABLE IF NOT EXISTS timelines (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                parent_id TEXT,
                fork_step INTEGER DEFAULT 0,
                fork_reason TEXT DEFAULT '',
                status TEXT DEFAULT 'active',
                current_step INTEGER DEFAULT 0,
                description TEXT DEFAULT '',
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS fork_points (
                id TEXT PRIMARY KEY,
                step INTEGER NOT NULL,
                sim_date TEXT,
                trigger_event TEXT,
                parent_timeline_id TEXT,
                child_timeline_ids_json TEXT DEFAULT '[]',
                created_at TEXT
            );

            CREATE TABLE IF NOT EXISTS snapshots (
                id TEXT PRIMARY KEY,
                timeline_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                sim_date TEXT,
                agent_states_json TEXT NOT NULL,
                graph_edges_json TEXT NOT NULL,
                opinion_distribution_json TEXT DEFAULT '{}',
                created_at TEXT
            );

            CREATE INDEX IF NOT EXISTS idx_snapshots_timeline
                ON snapshots(timeline_id, step);
        """)
        self.db.conn.commit()

    # --- Snapshot Management ---

    def create_snapshot(
        self,
        timeline_id: str,
        step: int,
        sim_date: str,
        agent_states: dict[str, AgentState],
        graph: SocialGraph,
        opinion_dist: dict | None = None,
    ) -> Snapshot:
        """Capture the complete simulation state at this moment."""
        # Serialize agent states
        states_dict = {}
        for aid, state in agent_states.items():
            states_dict[aid] = {
                "opinions": state.opinions,
                "emotional_state": {
                    "anger": state.emotional_state.anger,
                    "anxiety": state.emotional_state.anxiety,
                    "hope": state.emotional_state.hope,
                    "frustration": state.emotional_state.frustration,
                    "enthusiasm": state.emotional_state.enthusiasm,
                },
                "memory": state.memory,
                "trust_scores": state.trust_scores,
                "relationships": state.relationships,
                "post_count": state.post_count,
                "reply_count": state.reply_count,
            }

        # Serialize graph edges
        edges = []
        for u, v, data in graph.graph.edges(data=True):
            edges.append({
                "u": u, "v": v,
                "weight": data.get("weight", 0.5),
                "interaction_count": data.get("interaction_count", 0),
            })

        snapshot = Snapshot(
            timeline_id=timeline_id,
            step=step,
            sim_date=sim_date,
            agent_states_json=json.dumps(states_dict, ensure_ascii=False),
            graph_edges_json=json.dumps(edges),
            opinion_distribution=opinion_dist or {},
        )

        # Persist
        self.db.conn.execute(
            "INSERT INTO snapshots VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                snapshot.id, snapshot.timeline_id, snapshot.step,
                snapshot.sim_date, snapshot.agent_states_json,
                snapshot.graph_edges_json,
                json.dumps(snapshot.opinion_distribution),
                snapshot.created_at.isoformat(),
            ),
        )
        self.db.conn.commit()

        self.snapshots[snapshot.id] = snapshot
        logger.info("Snapshot created: %s at step %d", snapshot.id, step)
        return snapshot

    def restore_snapshot(
        self,
        snapshot_id: str,
        profiles: dict[str, AgentProfile],
    ) -> tuple[dict[str, AgentState], list[dict]]:
        """Restore agent states from a snapshot.

        Returns (agent_states, edge_list) where edge_list contains
        dicts with u, v, weight, interaction_count.
        """
        snapshot = self.snapshots.get(snapshot_id)
        if snapshot is None:
            # Try loading from DB
            row = self.db.conn.execute(
                "SELECT * FROM snapshots WHERE id = ?", (snapshot_id,)
            ).fetchone()
            if row is None:
                raise ValueError(f"Snapshot {snapshot_id} not found")
            snapshot = Snapshot(
                id=row["id"],
                timeline_id=row["timeline_id"],
                step=row["step"],
                sim_date=row["sim_date"],
                agent_states_json=row["agent_states_json"],
                graph_edges_json=row["graph_edges_json"],
                opinion_distribution=json.loads(row["opinion_distribution_json"]),
            )

        states_dict = json.loads(snapshot.agent_states_json)
        edges = json.loads(snapshot.graph_edges_json)

        agent_states: dict[str, AgentState] = {}
        for aid, sdata in states_dict.items():
            emo_data = sdata.get("emotional_state", {})
            agent_states[aid] = AgentState(
                agent_id=aid,
                opinions=sdata.get("opinions", {}),
                emotional_state=EmotionalState(**emo_data),
                memory=sdata.get("memory", []),
                trust_scores=sdata.get("trust_scores", {}),
                relationships=sdata.get("relationships", {}),
                post_count=sdata.get("post_count", 0),
                reply_count=sdata.get("reply_count", 0),
            )

        return agent_states, edges

    # --- Fork Management ---

    def create_fork(
        self,
        parent_timeline_id: str,
        fork_step: int,
        sim_date: str,
        trigger_event: str,
        snapshot_id: str,
        name: str = "",
        description: str = "",
    ) -> Timeline:
        """Create a new timeline forking from a parent at a given step.

        The fork starts from the state captured in snapshot_id.
        """
        if not name:
            name = f"fork-{fork_step}-{trigger_event[:20]}"
        if not description:
            description = f'もし「{trigger_event}」が起きなかったら'

        timeline = Timeline(
            name=name,
            parent_id=parent_timeline_id,
            fork_step=fork_step,
            fork_reason=trigger_event,
            status=TimelineStatus.ACTIVE,
            description=description,
            current_step=fork_step,
        )

        # Create fork point record
        fork_point = ForkPoint(
            step=fork_step,
            sim_date=sim_date,
            trigger_event=trigger_event,
            parent_timeline_id=parent_timeline_id,
            child_timeline_ids=[timeline.id],
        )

        # Persist
        self.db.conn.execute(
            "INSERT INTO timelines VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                timeline.id, timeline.name, timeline.parent_id,
                timeline.fork_step, timeline.fork_reason,
                timeline.status.value, timeline.current_step,
                timeline.description, timeline.created_at.isoformat(),
            ),
        )
        self.db.conn.execute(
            "INSERT INTO fork_points VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                fork_point.id, fork_point.step, fork_point.sim_date,
                fork_point.trigger_event, fork_point.parent_timeline_id,
                json.dumps(fork_point.child_timeline_ids),
                fork_point.created_at.isoformat(),
            ),
        )
        self.db.conn.commit()

        self.timelines[timeline.id] = timeline
        self.fork_points.append(fork_point)

        logger.info(
            "Fork created: %s from %s at step %d (%s)",
            timeline.id, parent_timeline_id, fork_step, trigger_event,
        )
        return timeline

    def update_timeline_step(self, timeline_id: str, step: int) -> None:
        """Update the current step of a timeline."""
        if timeline_id in self.timelines:
            self.timelines[timeline_id].current_step = step
        self.db.conn.execute(
            "UPDATE timelines SET current_step = ? WHERE id = ?",
            (step, timeline_id),
        )
        self.db.conn.commit()

    def freeze_timeline(self, timeline_id: str) -> None:
        """Freeze a timeline (stop advancing it)."""
        if timeline_id in self.timelines:
            self.timelines[timeline_id].status = TimelineStatus.FROZEN
        self.db.conn.execute(
            "UPDATE timelines SET status = ? WHERE id = ?",
            (TimelineStatus.FROZEN.value, timeline_id),
        )
        self.db.conn.commit()

    def get_active_timelines(self) -> list[Timeline]:
        """Get all active timelines."""
        return [t for t in self.timelines.values() if t.status == TimelineStatus.ACTIVE]

    # --- Timeline Comparison ---

    def compare_timelines(
        self,
        timeline_a_states: dict[str, AgentState],
        timeline_b_states: dict[str, AgentState],
        timeline_a_id: str,
        timeline_b_id: str,
        topic_id: str,
        step: int,
        sim_date: str = "",
    ) -> TimelineComparison:
        """Compare opinion distributions between two timelines."""
        # Compute opinion stats for each
        ops_a = [s.opinions.get(topic_id, 0.0) for s in timeline_a_states.values()]
        ops_b = [s.opinions.get(topic_id, 0.0) for s in timeline_b_states.values()]

        mean_a = sum(ops_a) / len(ops_a) if ops_a else 0
        mean_b = sum(ops_b) / len(ops_b) if ops_b else 0
        std_a = _std(ops_a)
        std_b = _std(ops_b)

        # Compute per-agent differences
        agent_diffs: list[tuple[str, float]] = []
        common_ids = set(timeline_a_states.keys()) & set(timeline_b_states.keys())
        for aid in common_ids:
            op_a = timeline_a_states[aid].opinions.get(topic_id, 0.0)
            op_b = timeline_b_states[aid].opinions.get(topic_id, 0.0)
            agent_diffs.append((aid, abs(op_a - op_b)))

        agent_diffs.sort(key=lambda x: x[1], reverse=True)

        # Distribution comparison
        dist_a = _opinion_buckets(ops_a)
        dist_b = _opinion_buckets(ops_b)
        dist_diff = {k: dist_a.get(k, 0) - dist_b.get(k, 0) for k in dist_a}

        # Overall divergence score (0-1)
        if agent_diffs:
            avg_diff = sum(d for _, d in agent_diffs) / len(agent_diffs)
            divergence = min(1.0, avg_diff / 0.5)  # Normalize: 0.5 difference = max
        else:
            divergence = 0.0

        return TimelineComparison(
            timeline_a_id=timeline_a_id,
            timeline_b_id=timeline_b_id,
            compared_at_step=step,
            sim_date=sim_date,
            opinion_mean_diff=mean_a - mean_b,
            opinion_std_diff=std_a - std_b,
            polarization_diff=std_a - std_b,
            distribution_diff=dist_diff,
            most_changed_agents=[aid for aid, _ in agent_diffs[:5]],
            divergence_score=round(divergence, 4),
        )

    def get_timeline_tree(self) -> dict:
        """Get the timeline tree structure for visualization."""
        tree = {}
        for tid, timeline in self.timelines.items():
            tree[tid] = {
                "name": timeline.name,
                "parent": timeline.parent_id,
                "fork_step": timeline.fork_step,
                "fork_reason": timeline.fork_reason,
                "current_step": timeline.current_step,
                "status": timeline.status.value,
                "description": timeline.description,
            }
        return tree


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5


def _opinion_buckets(opinions: list[float]) -> dict[str, int]:
    buckets = {"strong_against": 0, "against": 0, "neutral": 0, "for": 0, "strong_for": 0}
    for op in opinions:
        if op < -0.6:
            buckets["strong_against"] += 1
        elif op < -0.2:
            buckets["against"] += 1
        elif op < 0.2:
            buckets["neutral"] += 1
        elif op < 0.6:
            buckets["for"] += 1
        else:
            buckets["strong_for"] += 1
    return buckets
