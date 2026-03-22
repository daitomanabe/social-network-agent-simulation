"""SQLite database layer for simulation persistence."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from pathlib import Path

from src.agents.models import AgentProfile, AgentState, BigFive, CognitiveBiases, EmotionalState
from src.network.models import Post


class Database:
    """SQLite-backed storage for simulation state."""

    def __init__(self, db_path: Path | str = "data/simulation.db") -> None:
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(self.db_path))
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA journal_mode=WAL")
        self.conn.execute("PRAGMA foreign_keys=ON")

    def init_db(self) -> None:
        """Create all tables."""
        self.conn.executescript("""
            CREATE TABLE IF NOT EXISTS agents (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                age_group TEXT,
                occupation TEXT,
                region TEXT,
                personality_json TEXT NOT NULL,
                biases_json TEXT NOT NULL,
                core_values_json TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS agent_states (
                agent_id TEXT NOT NULL,
                step INTEGER NOT NULL,
                opinions_json TEXT NOT NULL,
                emotional_state_json TEXT NOT NULL,
                memory_json TEXT NOT NULL,
                trust_scores_json TEXT NOT NULL,
                relationships_json TEXT NOT NULL,
                post_count INTEGER DEFAULT 0,
                reply_count INTEGER DEFAULT 0,
                PRIMARY KEY (agent_id, step),
                FOREIGN KEY (agent_id) REFERENCES agents(id)
            );

            CREATE TABLE IF NOT EXISTS posts (
                id TEXT PRIMARY KEY,
                author_id TEXT NOT NULL,
                topic_id TEXT NOT NULL,
                content TEXT NOT NULL,
                sim_time TEXT,
                step INTEGER NOT NULL,
                sentiment REAL DEFAULT 0.0,
                is_news_seed INTEGER DEFAULT 0,
                reply_to TEXT,
                reactions_json TEXT DEFAULT '{}',
                FOREIGN KEY (author_id) REFERENCES agents(id)
            );

            CREATE TABLE IF NOT EXISTS simulation_state (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL
            );

            CREATE INDEX IF NOT EXISTS idx_posts_step ON posts(step);
            CREATE INDEX IF NOT EXISTS idx_posts_topic ON posts(topic_id);
            CREATE INDEX IF NOT EXISTS idx_agent_states_step ON agent_states(step);
        """)
        self.conn.commit()

    # --- Agent CRUD ---

    def insert_agent(self, profile: AgentProfile) -> None:
        """Insert an agent profile."""
        self.conn.execute(
            "INSERT OR REPLACE INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                profile.id,
                profile.name,
                profile.age_group,
                profile.occupation,
                profile.region,
                json.dumps(vars(profile.personality)),
                json.dumps(vars(profile.biases)),
                json.dumps(profile.core_values),
            ),
        )
        self.conn.commit()

    def insert_agents_batch(self, profiles: list[AgentProfile]) -> None:
        """Insert multiple agent profiles."""
        rows = [
            (
                p.id, p.name, p.age_group, p.occupation, p.region,
                json.dumps(vars(p.personality)),
                json.dumps(vars(p.biases)),
                json.dumps(p.core_values),
            )
            for p in profiles
        ]
        self.conn.executemany("INSERT OR REPLACE INTO agents VALUES (?, ?, ?, ?, ?, ?, ?, ?)", rows)
        self.conn.commit()

    def get_agent(self, agent_id: str) -> AgentProfile | None:
        """Retrieve an agent profile."""
        row = self.conn.execute("SELECT * FROM agents WHERE id = ?", (agent_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_profile(row)

    def get_all_agents(self) -> list[AgentProfile]:
        """Retrieve all agent profiles."""
        rows = self.conn.execute("SELECT * FROM agents").fetchall()
        return [self._row_to_profile(r) for r in rows]

    def _row_to_profile(self, row: sqlite3.Row) -> AgentProfile:
        p_data = json.loads(row["personality_json"])
        b_data = json.loads(row["biases_json"])
        return AgentProfile(
            id=row["id"],
            name=row["name"],
            age_group=row["age_group"] or "",
            occupation=row["occupation"] or "",
            region=row["region"] or "",
            personality=BigFive(**p_data),
            biases=CognitiveBiases(**b_data),
            core_values=json.loads(row["core_values_json"]),
        )

    # --- Agent State ---

    def save_agent_state(self, state: AgentState, step: int) -> None:
        """Save agent state snapshot for a given step."""
        self.conn.execute(
            "INSERT OR REPLACE INTO agent_states VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                state.agent_id,
                step,
                json.dumps(state.opinions),
                json.dumps(vars(state.emotional_state)),
                json.dumps(state.memory),
                json.dumps(state.trust_scores),
                json.dumps(state.relationships),
                state.post_count,
                state.reply_count,
            ),
        )
        self.conn.commit()

    def save_agent_states_batch(self, states: list[AgentState], step: int) -> None:
        """Save multiple agent states."""
        rows = [
            (
                s.agent_id, step,
                json.dumps(s.opinions),
                json.dumps(vars(s.emotional_state)),
                json.dumps(s.memory),
                json.dumps(s.trust_scores),
                json.dumps(s.relationships),
                s.post_count, s.reply_count,
            )
            for s in states
        ]
        self.conn.executemany(
            "INSERT OR REPLACE INTO agent_states VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)", rows
        )
        self.conn.commit()

    def get_agent_state(self, agent_id: str, step: int) -> AgentState | None:
        """Get agent state at a specific step."""
        row = self.conn.execute(
            "SELECT * FROM agent_states WHERE agent_id = ? AND step = ?",
            (agent_id, step),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_state(row)

    def get_latest_agent_state(self, agent_id: str) -> AgentState | None:
        """Get the most recent state for an agent."""
        row = self.conn.execute(
            "SELECT * FROM agent_states WHERE agent_id = ? ORDER BY step DESC LIMIT 1",
            (agent_id,),
        ).fetchone()
        if row is None:
            return None
        return self._row_to_state(row)

    def _row_to_state(self, row: sqlite3.Row) -> AgentState:
        emo_data = json.loads(row["emotional_state_json"])
        return AgentState(
            agent_id=row["agent_id"],
            opinions=json.loads(row["opinions_json"]),
            emotional_state=EmotionalState(**emo_data),
            memory=json.loads(row["memory_json"]),
            trust_scores=json.loads(row["trust_scores_json"]),
            relationships=json.loads(row["relationships_json"]),
            post_count=row["post_count"],
            reply_count=row["reply_count"],
        )

    # --- Posts ---

    def insert_post(self, post: Post) -> None:
        """Insert a post."""
        self.conn.execute(
            "INSERT OR REPLACE INTO posts VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                post.id,
                post.author_id,
                post.topic_id,
                post.content,
                post.sim_time.isoformat() if post.sim_time else None,
                post.step,
                post.sentiment,
                1 if post.is_news_seed else 0,
                post.reply_to,
                json.dumps(post.reactions),
            ),
        )
        self.conn.commit()

    def get_posts_by_step(self, step: int) -> list[Post]:
        """Get all posts from a specific step."""
        rows = self.conn.execute(
            "SELECT * FROM posts WHERE step = ? ORDER BY rowid", (step,)
        ).fetchall()
        return [self._row_to_post(r) for r in rows]

    def get_posts_by_topic(self, topic_id: str, limit: int = 50) -> list[Post]:
        """Get recent posts for a topic."""
        rows = self.conn.execute(
            "SELECT * FROM posts WHERE topic_id = ? ORDER BY step DESC LIMIT ?",
            (topic_id, limit),
        ).fetchall()
        return [self._row_to_post(r) for r in rows]

    def _row_to_post(self, row: sqlite3.Row) -> Post:
        return Post(
            id=row["id"],
            author_id=row["author_id"],
            topic_id=row["topic_id"],
            content=row["content"],
            sim_time=datetime.fromisoformat(row["sim_time"]) if row["sim_time"] else None,
            step=row["step"],
            sentiment=row["sentiment"],
            is_news_seed=bool(row["is_news_seed"]),
            reply_to=row["reply_to"],
            reactions=json.loads(row["reactions_json"]),
        )

    # --- Simulation State ---

    def set_state(self, key: str, value: str) -> None:
        """Set a simulation state value."""
        self.conn.execute(
            "INSERT OR REPLACE INTO simulation_state VALUES (?, ?)", (key, value)
        )
        self.conn.commit()

    def get_state(self, key: str) -> str | None:
        """Get a simulation state value."""
        row = self.conn.execute(
            "SELECT value FROM simulation_state WHERE key = ?", (key,)
        ).fetchone()
        return row["value"] if row else None

    def close(self) -> None:
        """Close the database connection."""
        self.conn.close()
