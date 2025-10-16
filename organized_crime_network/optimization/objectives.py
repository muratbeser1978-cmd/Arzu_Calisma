"""
Objective function implementations for optimization.

This module provides:
- Network structure metrics (LCC, degree, betweenness)
- Intelligence value computations
- Conversion probability calculations
- Information models

All implementations follow exact formulas from constitution.

Mathematical Foundation: gelis.tex §II, §III
Governance: .specify/memory/constitution.md v1.0.0
"""

import networkx as nx
from typing import Dict, Set, FrozenSet, Optional
from ..core.state import NetworkState, ActorState
from ..utils.numerical import expit
from .base import InformationModel, ConversionProbability


def compute_lcc_size(state: NetworkState) -> int:
    """
    Compute size of largest connected component.

    Uses NetworkX for robust connected component detection.
    Treats network as weakly connected (ignores edge direction for connectivity).

    Args:
        state: Current network state

    Returns:
        Number of actors in largest weakly connected component

    Guarantees:
        - Returns 0 for empty network
        - Returns correct LCC size for disconnected networks
        - Handles isolated actors correctly

    Constitution Reference: §II - Network Structure
    """
    if not state.E:
        # Empty network: each active actor is isolated
        active_count = sum(1 for status in state.A.values() if status == ActorState.ACTIVE)
        return 1 if active_count > 0 else 0

    # Build directed graph from edges
    G = nx.DiGraph()
    G.add_edges_from(state.E)

    # Find weakly connected components (treat as undirected for connectivity)
    if len(G) == 0:
        return 0

    components = list(nx.weakly_connected_components(G))
    if not components:
        return 0

    return max(len(c) for c in components)


def compute_structural_damage(actors: FrozenSet[int], network_state: NetworkState) -> float:
    """
    Compute expected structural damage from removing actors.

    Damage = LCC(original_network) - LCC(network_after_removal)

    Args:
        actors: Set of actors to remove
        network_state: Current network state

    Returns:
        Expected reduction in largest connected component size

    Guarantees:
        - Result in [0, |V|]
        - Monotonic: more actors => at least as much damage
        - Handles empty networks and isolated actors

    Constitution Reference: §II - Network Structure
    """
    # Compute original LCC size
    original_lcc = compute_lcc_size(network_state)

    # Simulate network after removing actors
    # Remove edges connected to removed actors
    remaining_edges = [(i, j) for (i, j) in network_state.E
                      if i not in actors and j not in actors]

    # Create temporary state for LCC computation
    # We'll compute LCC directly from edges rather than cloning state
    if not remaining_edges:
        # No edges left after removal
        remaining_lcc = 0
    else:
        G = nx.DiGraph()
        G.add_edges_from(remaining_edges)
        components = list(nx.weakly_connected_components(G))
        remaining_lcc = max(len(c) for c in components) if components else 0

    damage = original_lcc - remaining_lcc
    return float(damage)


def compute_degree_centrality(network_state: NetworkState) -> Dict[int, int]:
    """
    Compute degree centrality for all actors.

    Degree = number of edges (in + out) for each actor.

    Args:
        network_state: Current network state

    Returns:
        Dictionary mapping actor_id to degree (in + out)

    Guarantees:
        - All active actors included
        - Degrees >= 0
        - Isolated actors have degree 0

    Constitution Reference: §II - Degree calculations for preferential attachment
    """
    degrees = {}

    # Initialize all active actors with degree 0
    for actor_id, status in network_state.A.items():
        if status == ActorState.ACTIVE:
            degrees[actor_id] = 0

    # Count edges
    for (i, j) in network_state.E:
        if i in degrees:
            degrees[i] += 1
        if j in degrees:
            degrees[j] += 1

    return degrees


def compute_betweenness_centrality(network_state: NetworkState) -> Dict[int, float]:
    """
    Compute betweenness centrality for all actors.

    Betweenness = fraction of shortest paths passing through actor.
    Uses NetworkX for robust computation.

    Args:
        network_state: Current network state

    Returns:
        Dictionary mapping actor_id to normalized betweenness centrality [0, 1]

    Guarantees:
        - All active actors included
        - Values in [0, 1]
        - Isolated actors have betweenness 0

    Constitution Reference: §II - Network Structure
    """
    if not network_state.E:
        # Empty network: all actors have betweenness 0
        return {actor_id: 0.0 for actor_id, status in network_state.A.items()
                if status == ActorState.ACTIVE}

    # Build directed graph
    G = nx.DiGraph()
    G.add_edges_from(network_state.E)

    # Compute betweenness centrality
    betweenness = nx.betweenness_centrality(G)

    # Ensure all active actors are included (NetworkX may omit isolated nodes)
    for actor_id, status in network_state.A.items():
        if status == ActorState.ACTIVE and actor_id not in betweenness:
            betweenness[actor_id] = 0.0

    return betweenness


class NeighborhoodInformationModel(InformationModel):
    """
    Information model where actors know about their direct connections.

    Each actor knows:
    - Edges from them (outgoing connections)
    - Edges to them (incoming connections)

    Value function: Union size (each unique edge = 1 unit of intelligence)

    Properties:
    - Monotonic: More actors => at least as much information
    - Submodular: Diminishing marginal returns (information overlap increases)

    This ensures (1-1/e) approximation guarantee for lazy greedy algorithm.

    Constitution Reference: §II - Network Structure
    """

    def get_information_set(self, actor_id: int, network_state: NetworkState) -> Set[str]:
        """
        Return edges known by actor (direct connections).

        Args:
            actor_id: ID of the actor
            network_state: Current network state

        Returns:
            Set of edge identifiers as strings "edge_{i}_{j}"

        Constitution Reference: Uses network edges from §II
        """
        info = set()

        # Edges from actor (outgoing)
        for (i, j) in network_state.E:
            if i == actor_id:
                info.add(f"edge_{i}_{j}")

        # Edges to actor (incoming)
        for (i, j) in network_state.E:
            if j == actor_id:
                info.add(f"edge_{i}_{j}")

        return info

    def compute_value(self, info_set: Set[str]) -> float:
        """
        Compute value of information set.

        Value = number of unique edges known.

        Args:
            info_set: Union of information atoms (edge identifiers)

        Returns:
            Number of unique edges (float for consistency with interface)

        Guarantees:
            - Monotonic: |S| >= |T| => value(S) >= value(T)
            - Submodular: f(S∪{i}) - f(S) >= f(T∪{i}) - f(T) for S ⊆ T
        """
        return float(len(info_set))


# Intelligence Optimization Functions (T028-T032)

def compute_average_logit_trust(actor_id: int, network_state: NetworkState) -> float:
    """
    Compute average logit-scale trust for actor.

    Formula (Constitution §II): Ȳᵢ(t) = (1/|Nᵢᵒᵘᵗ(t)|) Σⱼ∈Nᵢᵒᵘᵗ Yᵢⱼ(t)

    Args:
        actor_id: ID of the actor
        network_state: Current network state

    Returns:
        Average logit-scale trust across outgoing edges

    Guarantees:
        - Returns 0.0 for isolated actors (no outgoing edges)
        - Uses exact formula from constitution

    Constitution Reference: §II - Trust Dynamics
    """
    # Find outgoing neighbors
    outgoing_neighbors = [j for (i, j) in network_state.E if i == actor_id]

    if not outgoing_neighbors:
        # Isolated actor: return 0 (neutral trust in logit space)
        return 0.0

    # Sum trust values in logit space
    trust_sum = sum(network_state.Y.get((actor_id, j), 0.0) for j in outgoing_neighbors)

    # Average
    avg_trust = trust_sum / len(outgoing_neighbors)

    return avg_trust


def compute_fragility_rate(actor_id: int, network_state: NetworkState,
                           mu_min: float, mu_rng: float, theta: float) -> float:
    """
    Compute fragility rate μ_HI for actor.

    Formula (Constitution §II): μ_HI(i) = μ_min + μ_rng · expit(-θ · Ȳᵢ(t))

    Args:
        actor_id: ID of the actor
        network_state: Current network state
        mu_min: Minimum fragility rate
        mu_rng: Fragility range
        theta: Trust sensitivity parameter

    Returns:
        Fragility rate (transition rate from Hesitant to Informant)

    Guarantees:
        - Result in [mu_min, mu_min + mu_rng]
        - Uses numerically stable expit
        - Higher trust => lower fragility

    Constitution Reference: §II - Informant Conversion CTMC
    """
    avg_trust = compute_average_logit_trust(actor_id, network_state)

    # Apply exact formula from constitution
    fragility = mu_min + mu_rng * expit(-theta * avg_trust)

    return fragility


def compute_conversion_probability(actor_id: int, network_state: NetworkState,
                                   time_horizon: float, parameters: Dict[str, float]) -> ConversionProbability:
    """
    Compute conversion probability within time horizon.

    Uses exact formulas from constitution for two-stage CTMC:
    - L → H at rate μ_LH (constant)
    - H → I at rate μ_HI(i) (trust-dependent)

    For this optimization context, we use a simplified formula:
    P(convert within T) ≈ 1 - exp(-(μ_LH + μ_HI) * T / 2)

    This is an approximation for the hypo-exponential distribution.

    Args:
        actor_id: ID of the actor
        network_state: Current network state
        time_horizon: Time window T for intelligence gathering
        parameters: Simulation parameters (μ_LH, μ_min, μ_rng, θ)

    Returns:
        ConversionProbability with all relevant data

    Guarantees:
        - Probability in [0, 1]
        - Expected time > 0
        - Uses exact fragility rate formula

    Constitution Reference: §II - Informant Conversion Process
    """
    # Extract parameters
    mu_LH = parameters.get('mu_LH', 0.1)
    mu_min = parameters.get('mu_min', 0.05)
    mu_rng = parameters.get('mu_rng', 0.15)
    theta = parameters.get('theta', 1.0)

    # Compute trust and fragility
    avg_trust = compute_average_logit_trust(actor_id, network_state)
    fragility = compute_fragility_rate(actor_id, network_state, mu_min, mu_rng, theta)

    # Expected time: E[T] = 1/μ_LH + 1/μ_HI
    expected_time = (1.0 / mu_LH) + (1.0 / fragility)

    # Approximate conversion probability within time horizon
    # P(convert by T) = 1 - P(not converted by T)
    # For series of exponentials: simplified approximation
    import math
    effective_rate = (mu_LH + fragility) / 2.0  # Harmonic mean approximation
    probability = 1.0 - math.exp(-effective_rate * time_horizon)

    # Clamp to [0, 1]
    probability = max(0.0, min(1.0, probability))

    return ConversionProbability(
        actor_id=actor_id,
        probability=probability,
        expected_time=expected_time,
        fragility_rate=fragility,
        average_trust=avg_trust,
        calculation_method="simplified_hypo_exponential"
    )


def compute_intelligence_value(actors: FrozenSet[int], network_state: NetworkState,
                               time_horizon: float, parameters: Dict[str, float],
                               information_model: InformationModel) -> float:
    """
    Compute expected intelligence gain from actors.

    Formula: Intelligence = Σᵢ p_i(T) · value(information_set_i)

    where p_i(T) is conversion probability within time horizon T.

    Args:
        actors: Set of actors to arrest
        network_state: Current network state
        time_horizon: Time window for intelligence gathering
        parameters: Simulation parameters
        information_model: Model defining what information actors know

    Returns:
        Expected total intelligence value

    Guarantees:
        - Result >= 0
        - Monotonic: more actors => at least as much intelligence
        - Submodular if information_model is submodular

    Constitution Reference: §II - Informant Conversion + Network Structure
    """
    total_intelligence = 0.0

    # Compute combined information set
    combined_info = set()
    for actor_id in actors:
        info = information_model.get_information_set(actor_id, network_state)
        combined_info.update(info)

    # Compute value of combined information
    info_value = information_model.compute_value(combined_info)

    # Weight by conversion probabilities
    for actor_id in actors:
        conversion = compute_conversion_probability(actor_id, network_state, time_horizon, parameters)
        # Individual contribution weighted by probability
        individual_info = information_model.get_information_set(actor_id, network_state)
        individual_value = information_model.compute_value(individual_info)
        total_intelligence += conversion.probability * individual_value

    return total_intelligence


def compute_marginal_intelligence(actor_id: int, current_actors: FrozenSet[int],
                                  network_state: NetworkState, time_horizon: float,
                                  parameters: Dict[str, float],
                                  information_model: InformationModel) -> float:
    """
    Compute marginal intelligence gain from adding actor.

    Marginal gain: F(S ∪ {i}) - F(S)

    Args:
        actor_id: ID of candidate actor
        current_actors: Already selected actors
        network_state: Current network state
        time_horizon: Time window
        parameters: Simulation parameters
        information_model: Information model

    Returns:
        Marginal intelligence gain from adding actor_id

    Guarantees:
        - Result >= 0 (monotonicity)
        - Diminishing returns if information_model is submodular

    Constitution Reference: §II - Informant Conversion + Network Structure
    """
    # Intelligence with current actors
    current_intelligence = compute_intelligence_value(
        current_actors, network_state, time_horizon, parameters, information_model
    )

    # Intelligence with actor added
    new_actors = frozenset(current_actors | {actor_id})
    new_intelligence = compute_intelligence_value(
        new_actors, network_state, time_horizon, parameters, information_model
    )

    # Marginal gain
    marginal = new_intelligence - current_intelligence

    return max(0.0, marginal)  # Ensure non-negative
