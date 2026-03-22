"""Network dynamics: rewiring, echo chamber detection, and polarization metrics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import networkx as nx

from src.agents.models import AgentState
from src.network.graph import SocialGraph


@dataclass
class EchoChamber:
    """A detected echo chamber (cluster of like-minded agents)."""

    id: int = 0
    agent_ids: list[str] = field(default_factory=list)
    mean_opinion: float = 0.0
    opinion_std: float = 0.0
    internal_density: float = 0.0  # Edge density within cluster
    avg_edge_weight: float = 0.0
    size: int = 0

    @property
    def label(self) -> str:
        if self.mean_opinion > 0.3:
            return "pro"
        elif self.mean_opinion < -0.3:
            return "anti"
        return "neutral"


@dataclass
class PolarizationMetrics:
    """Metrics describing the level of polarization in the network."""

    # Opinion-based
    bimodality_coefficient: float = 0.0  # >0.555 suggests bimodal distribution
    opinion_variance: float = 0.0
    opinion_kurtosis: float = 0.0  # Negative = flat/bimodal, positive = peaked

    # Network-based
    modularity: float = 0.0  # Community structure strength (0-1)
    echo_chamber_count: int = 0
    avg_echo_chamber_size: float = 0.0
    cross_cluster_edges_ratio: float = 0.0  # Ratio of edges between clusters

    # Combined
    polarization_index: float = 0.0  # 0.0 = consensus, 1.0 = fully polarized

    def format(self) -> str:
        return (
            f"Polarization: {self.polarization_index:.3f} | "
            f"Bimodality: {self.bimodality_coefficient:.3f} | "
            f"Modularity: {self.modularity:.3f} | "
            f"Chambers: {self.echo_chamber_count} | "
            f"Cross-edges: {self.cross_cluster_edges_ratio:.1%}"
        )


class NetworkDynamics:
    """Analyzes and evolves the social network structure."""

    def __init__(self, graph: SocialGraph) -> None:
        self.graph = graph

    def detect_echo_chambers(
        self,
        states: dict[str, AgentState],
        topic_id: str,
        resolution: float = 1.0,
    ) -> list[EchoChamber]:
        """Detect echo chambers using community detection + opinion alignment.

        Uses Louvain community detection, then checks if communities have
        aligned opinions (which would indicate an echo chamber).
        """
        G = self.graph.graph
        if G.number_of_nodes() == 0:
            return []

        # Community detection
        try:
            communities = nx.community.louvain_communities(G, resolution=resolution, seed=42)
        except Exception:
            # Fallback to greedy modularity
            communities = list(nx.community.greedy_modularity_communities(G))

        chambers = []
        for i, community in enumerate(communities):
            if len(community) < 3:
                continue

            agent_ids = list(community)
            opinions = [
                states[aid].opinions.get(topic_id, 0.0)
                for aid in agent_ids if aid in states
            ]

            if not opinions:
                continue

            mean_op = sum(opinions) / len(opinions)
            std_op = _std(opinions)

            # Calculate internal density
            subgraph = G.subgraph(agent_ids)
            possible_edges = len(agent_ids) * (len(agent_ids) - 1) / 2
            internal_density = subgraph.number_of_edges() / possible_edges if possible_edges > 0 else 0

            # Average edge weight
            weights = [d.get("weight", 0.5) for _, _, d in subgraph.edges(data=True)]
            avg_weight = sum(weights) / len(weights) if weights else 0

            # It's an echo chamber if opinions are aligned (low std)
            # and internal connections are strong
            if std_op < 0.4 and internal_density > 0.2:
                chambers.append(EchoChamber(
                    id=i,
                    agent_ids=agent_ids,
                    mean_opinion=round(mean_op, 3),
                    opinion_std=round(std_op, 3),
                    internal_density=round(internal_density, 3),
                    avg_edge_weight=round(avg_weight, 3),
                    size=len(agent_ids),
                ))

        return chambers

    def compute_polarization(
        self,
        states: dict[str, AgentState],
        topic_id: str,
    ) -> PolarizationMetrics:
        """Compute comprehensive polarization metrics."""
        G = self.graph.graph
        opinions = [states[aid].opinions.get(topic_id, 0.0) for aid in G.nodes() if aid in states]

        if len(opinions) < 2:
            return PolarizationMetrics()

        n = len(opinions)
        mean = sum(opinions) / n
        variance = sum((x - mean) ** 2 for x in opinions) / n
        std = variance ** 0.5

        # Kurtosis (excess)
        if std > 0:
            kurtosis = sum(((x - mean) / std) ** 4 for x in opinions) / n - 3.0
        else:
            kurtosis = 0.0

        # Skewness
        if std > 0:
            skewness = sum(((x - mean) / std) ** 3 for x in opinions) / n
        else:
            skewness = 0.0

        # Bimodality coefficient: BC = (skewness^2 + 1) / (kurtosis + 3 * (n-1)^2 / ((n-2)*(n-3)))
        # BC > 0.555 suggests bimodal distribution
        if n > 3:
            correction = 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
            denom = kurtosis + correction
            if denom > 0:
                bimodality = (skewness ** 2 + 1) / denom
            else:
                bimodality = 0.0
        else:
            bimodality = 0.0

        # Network modularity
        chambers = self.detect_echo_chambers(states, topic_id)
        if chambers:
            # Build partition for modularity calculation
            partition = []
            assigned = set()
            for ch in chambers:
                partition.append(set(ch.agent_ids))
                assigned.update(ch.agent_ids)
            # Add remaining nodes as singletons
            remaining = set(G.nodes()) - assigned
            if remaining:
                partition.append(remaining)
            try:
                modularity = nx.community.modularity(G, partition)
            except Exception:
                modularity = 0.0
        else:
            modularity = 0.0

        # Cross-cluster edge ratio
        cross_edges = 0
        total_edges = G.number_of_edges()
        if chambers and total_edges > 0:
            chamber_map = {}
            for ch in chambers:
                for aid in ch.agent_ids:
                    chamber_map[aid] = ch.id
            for u, v in G.edges():
                if u in chamber_map and v in chamber_map:
                    if chamber_map[u] != chamber_map[v]:
                        cross_edges += 1
                else:
                    cross_edges += 1
            cross_ratio = cross_edges / total_edges
        else:
            cross_ratio = 1.0

        # Combined polarization index
        # High variance + low cross-cluster ratio + high bimodality = polarized
        pol_opinion = min(1.0, variance / 0.25)  # Normalize: 0.25 variance = max
        pol_network = 1.0 - cross_ratio
        pol_bimodal = min(1.0, bimodality / 0.555)

        polarization_index = pol_opinion * 0.4 + pol_network * 0.3 + pol_bimodal * 0.3

        return PolarizationMetrics(
            bimodality_coefficient=round(bimodality, 4),
            opinion_variance=round(variance, 4),
            opinion_kurtosis=round(kurtosis, 4),
            modularity=round(modularity, 4),
            echo_chamber_count=len(chambers),
            avg_echo_chamber_size=round(
                sum(ch.size for ch in chambers) / len(chambers), 1
            ) if chambers else 0,
            cross_cluster_edges_ratio=round(cross_ratio, 4),
            polarization_index=round(polarization_index, 4),
        )

    def evolve_network(
        self,
        states: dict[str, AgentState],
        topic_id: str,
        rewire_rate: float = 0.03,
        strengthen_rate: float = 0.01,
        seed: int = 42,
    ) -> dict:
        """Evolve the network based on opinion dynamics.

        - Homophily: strengthen connections between similar-opinion agents
        - Rewiring: replace weak cross-opinion edges with same-opinion ones
        - Weakening: reduce weight of edges between disagreeing agents

        Returns stats about changes made.
        """
        import random
        rng = random.Random(seed)
        G = self.graph.graph

        strengthened = 0
        weakened = 0
        rewired = 0

        nodes = list(G.nodes())

        # 1. Strengthen/weaken edges based on opinion alignment
        for u, v in list(G.edges()):
            if u not in states or v not in states:
                continue
            op_u = states[u].opinions.get(topic_id, 0.0)
            op_v = states[v].opinions.get(topic_id, 0.0)
            diff = abs(op_u - op_v)

            if diff < 0.3:
                # Similar opinions: strengthen
                delta = strengthen_rate * (1.0 - diff)
                self.graph.update_edge_weight(u, v, delta)
                strengthened += 1
            elif diff > 0.6:
                # Very different: weaken
                delta = -strengthen_rate * diff
                self.graph.update_edge_weight(u, v, delta)
                weakened += 1

        # 2. Homophily rewiring
        rewired = self.graph.rewire_by_opinion(
            {aid: s.opinions for aid, s in states.items()},
            topic_id,
            rewire_prob=rewire_rate,
            seed=seed,
        )

        return {
            "strengthened": strengthened,
            "weakened": weakened,
            "rewired": rewired,
        }


def _std(values: list[float]) -> float:
    if len(values) < 2:
        return 0.0
    mean = sum(values) / len(values)
    return (sum((x - mean) ** 2 for x in values) / len(values)) ** 0.5
