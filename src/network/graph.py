"""Social network graph using NetworkX."""

from __future__ import annotations

import random

import networkx as nx

from src.agents.models import AgentProfile
from src.network.models import Post


class SocialGraph:
    """Manages the social network topology and feed generation."""

    def __init__(self) -> None:
        self.graph: nx.Graph = nx.Graph()
        self._posts_by_step: dict[int, list[Post]] = {}

    def build_small_world(
        self,
        agents: list[AgentProfile],
        k: int = 6,
        p: float = 0.1,
        seed: int = 42,
    ) -> None:
        """Build a Watts-Strogatz small-world network from agent list."""
        n = len(agents)
        # Ensure k is even and less than n
        k = min(k, n - 1)
        if k % 2 != 0:
            k -= 1
        k = max(k, 2)

        self.graph = nx.watts_strogatz_graph(n, k, p, seed=seed)

        # Map node indices to agent IDs
        mapping = {i: agents[i].id for i in range(n)}
        self.graph = nx.relabel_nodes(self.graph, mapping)

        # Store agent profiles as node attributes
        for agent in agents:
            self.graph.nodes[agent.id]["profile"] = agent

        # Initialize edge weights
        for u, v in self.graph.edges():
            self.graph[u][v]["weight"] = 0.5
            self.graph[u][v]["interaction_count"] = 0

    def get_neighbors(self, agent_id: str) -> list[str]:
        """Get all connected agent IDs."""
        if agent_id not in self.graph:
            return []
        return list(self.graph.neighbors(agent_id))

    def get_feed(self, agent_id: str, step: int, max_posts: int = 10) -> list[Post]:
        """Get a feed of posts from neighbors for the given step.

        Posts are weighted by edge strength (closer connections appear more).
        """
        neighbors = self.get_neighbors(agent_id)
        if not neighbors:
            return []

        # Collect posts from neighbors and news seeds
        candidate_posts: list[tuple[Post, float]] = []
        step_posts = self._posts_by_step.get(step, [])
        prev_posts = self._posts_by_step.get(step - 1, [])
        all_recent = step_posts + prev_posts

        for post in all_recent:
            if post.is_news_seed:
                # News seeds always appear in feed
                candidate_posts.append((post, 1.0))
            elif post.author_id in neighbors:
                weight = self.graph[agent_id][post.author_id].get("weight", 0.5)
                candidate_posts.append((post, weight))

        if not candidate_posts:
            return []

        # Sort by weight (relevance), take top N
        candidate_posts.sort(key=lambda x: x[1], reverse=True)
        return [p for p, _ in candidate_posts[:max_posts]]

    def add_post(self, post: Post) -> None:
        """Register a post for feed distribution."""
        step = post.step
        if step not in self._posts_by_step:
            self._posts_by_step[step] = []
        self._posts_by_step[step].append(post)

    def update_edge_weight(self, agent_a: str, agent_b: str, delta: float) -> None:
        """Adjust the weight of an edge between two agents."""
        if self.graph.has_edge(agent_a, agent_b):
            current = self.graph[agent_a][agent_b].get("weight", 0.5)
            new_weight = max(0.0, min(1.0, current + delta))
            self.graph[agent_a][agent_b]["weight"] = new_weight
            self.graph[agent_a][agent_b]["interaction_count"] += 1

    def rewire_by_opinion(
        self,
        opinions: dict[str, dict[str, float]],
        topic_id: str,
        rewire_prob: float = 0.02,
        seed: int | None = None,
    ) -> int:
        """Homophily-based rewiring: agents more likely to connect with similar opinions.

        Returns number of edges rewired.
        """
        rng = random.Random(seed)
        nodes = list(self.graph.nodes())
        rewired = 0

        for node in nodes:
            if rng.random() > rewire_prob:
                continue
            if node not in opinions or topic_id not in opinions[node]:
                continue

            neighbors = list(self.graph.neighbors(node))
            if not neighbors:
                continue

            # Find the most opinion-distant neighbor
            my_opinion = opinions[node][topic_id]
            max_dist = -1.0
            distant_neighbor = None
            for nb in neighbors:
                if nb in opinions and topic_id in opinions[nb]:
                    dist = abs(my_opinion - opinions[nb][topic_id])
                    if dist > max_dist:
                        max_dist = dist
                        distant_neighbor = nb

            if distant_neighbor is None:
                continue

            # Find a non-neighbor with similar opinion
            non_neighbors = [n for n in nodes if n != node and n not in neighbors]
            similar_candidates = []
            for nn in non_neighbors:
                if nn in opinions and topic_id in opinions[nn]:
                    dist = abs(my_opinion - opinions[nn][topic_id])
                    if dist < max_dist * 0.5:
                        similar_candidates.append(nn)

            if similar_candidates:
                new_friend = rng.choice(similar_candidates)
                self.graph.remove_edge(node, distant_neighbor)
                self.graph.add_edge(node, new_friend, weight=0.3, interaction_count=0)
                rewired += 1

        return rewired

    @property
    def stats(self) -> dict:
        """Return basic network statistics."""
        if len(self.graph) == 0:
            return {"nodes": 0, "edges": 0}
        return {
            "nodes": self.graph.number_of_nodes(),
            "edges": self.graph.number_of_edges(),
            "avg_clustering": round(nx.average_clustering(self.graph), 3),
            "avg_degree": round(
                sum(d for _, d in self.graph.degree()) / self.graph.number_of_nodes(), 1
            ),
        }
