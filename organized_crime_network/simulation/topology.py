"""
Network topology generators for simulation initialization.

This module provides functions to create organized crime networks
with different structural properties: scale-free, hierarchical, random, etc.

Specification Reference: US-003 (Initialize Custom Network Structures)
"""

from typing import List, Tuple, Optional
import numpy as np
import networkx as nx

from ..core.state import NetworkState, ActorState


def create_scale_free_network(
    n_actors: int,
    m: int = 2,
    seed: Optional[int] = None,
    t_current: float = 0.0,
    P_0: float = 1.0,
    Delta: float = 10.0
) -> NetworkState:
    """
    Create scale-free network using Barabási-Albert model.

    Scale-free networks have degree distribution P(k) ~ k^(-γ), common in
    real criminal networks where key actors have many connections.

    Parameters
    ----------
    n_actors : int
        Number of actors to create
    m : int, default=2
        Number of edges to attach from new node
    seed : int, optional
        Random seed for reproducibility
    t_current : float, default=0.0
        Initial simulation time
    P_0 : float, default=1.0
        Initial law enforcement effectiveness
    Delta : float, default=10.0
        Memory window for exposure

    Returns
    -------
    NetworkState
        Initialized scale-free network

    Examples
    --------
    >>> network = create_scale_free_network(n_actors=50, m=2, seed=42)
    >>> len(network.V)
    50
    >>> len(network.E) > 0
    True

    Notes
    -----
    - All actors assigned to hierarchy level 1 (can be modified after creation)
    - Initial trust values based on Jaccard similarity
    - Network is undirected (edges added in both directions)
    """
    state = NetworkState(t_current=t_current, P_0=P_0, Delta=Delta)

    # Generate scale-free graph using NetworkX
    G = nx.barabasi_albert_graph(n=n_actors, m=m, seed=seed)

    # Add all actors with default hierarchy
    for node_id in G.nodes():
        state.add_actor(node_id, hierarchy_level=1, state=ActorState.ACTIVE)

    # Add edges with Jaccard-based initial trust
    rng = np.random.default_rng(seed)
    for (i, j) in G.edges():
        # Add both directions for undirected network
        trust_ij = rng.uniform(0.4, 0.8)
        trust_ji = rng.uniform(0.4, 0.8)

        state.add_edge(i, j, initial_trust=trust_ij)
        state.add_edge(j, i, initial_trust=trust_ji)

    return state


def create_hierarchical_network(
    level_sizes: List[int],
    connections_per_level: int = 2,
    seed: Optional[int] = None,
    t_current: float = 0.0,
    P_0: float = 1.0,
    Delta: float = 10.0
) -> NetworkState:
    """
    Create hierarchical network with multiple tiers.

    Hierarchical structure common in organized crime: operatives → mid-level → leaders

    Parameters
    ----------
    level_sizes : List[int]
        Number of actors at each hierarchy level [L1, L2, L3, ...]
        Level 1 = lowest (operatives), higher = more senior
    connections_per_level : int, default=2
        How many actors from level L connect to level L+1
    seed : int, optional
        Random seed for reproducibility
    t_current : float, default=0.0
        Initial simulation time
    P_0 : float, default=1.0
        Initial law enforcement effectiveness
    Delta : float, default=10.0
        Memory window for exposure

    Returns
    -------
    NetworkState
        Initialized hierarchical network

    Examples
    --------
    >>> # Create network: 20 operatives, 10 mid-level, 5 leaders
    >>> network = create_hierarchical_network(
    ...     level_sizes=[20, 10, 5],
    ...     connections_per_level=2,
    ...     seed=42
    ... )
    >>> len(network.V)
    35

    Notes
    -----
    - Lower hierarchy actors connect upward to higher levels
    - Trust increases with hierarchy level (higher = more trust)
    - Leaders fully connected to each other (high trust)
    """
    state = NetworkState(t_current=t_current, P_0=P_0, Delta=Delta)
    rng = np.random.default_rng(seed)

    # Create actors for each level
    actor_id = 0
    level_actors = []

    for level_idx, size in enumerate(level_sizes):
        hierarchy_level = level_idx + 1
        level_group = []

        for _ in range(size):
            state.add_actor(actor_id, hierarchy_level=hierarchy_level, state=ActorState.ACTIVE)
            level_group.append(actor_id)
            actor_id += 1

        level_actors.append(level_group)

    # Connect levels hierarchically
    for level_idx in range(len(level_actors) - 1):
        lower_level = level_actors[level_idx]
        upper_level = level_actors[level_idx + 1]

        # Base trust increases with hierarchy level
        base_trust = 0.6 + 0.1 * level_idx

        for actor in lower_level:
            # Connect to multiple actors in next level
            targets = rng.choice(upper_level, size=min(connections_per_level, len(upper_level)), replace=False)

            for target in targets:
                trust_up = rng.uniform(base_trust, base_trust + 0.2)
                trust_down = rng.uniform(base_trust, base_trust + 0.2)

                state.add_edge(actor, target, initial_trust=trust_up)
                state.add_edge(target, actor, initial_trust=trust_down)

    # Connect actors within highest level (leaders know each other)
    if len(level_actors) > 0:
        leaders = level_actors[-1]
        leader_trust = 0.85

        for i, leader1 in enumerate(leaders):
            for leader2 in leaders[i+1:]:
                trust = rng.uniform(leader_trust, leader_trust + 0.1)
                state.add_edge(leader1, leader2, initial_trust=trust)
                state.add_edge(leader2, leader1, initial_trust=trust)

    return state


def create_random_network(
    n_actors: int,
    edge_probability: float = 0.1,
    seed: Optional[int] = None,
    t_current: float = 0.0,
    P_0: float = 1.0,
    Delta: float = 10.0
) -> NetworkState:
    """
    Create random (Erdős-Rényi) network.

    Each possible edge exists independently with probability p.

    Parameters
    ----------
    n_actors : int
        Number of actors to create
    edge_probability : float, default=0.1
        Probability of edge formation ∈ (0, 1)
    seed : int, optional
        Random seed for reproducibility
    t_current : float, default=0.0
        Initial simulation time
    P_0 : float, default=1.0
        Initial law enforcement effectiveness
    Delta : float, default=10.0
        Memory window for exposure

    Returns
    -------
    NetworkState
        Initialized random network

    Examples
    --------
    >>> network = create_random_network(n_actors=30, edge_probability=0.15, seed=42)
    >>> len(network.V)
    30

    Notes
    -----
    - All actors assigned to hierarchy level 1
    - Useful as baseline/null model for comparison
    - Expected number of edges: n * (n-1) * p / 2
    """
    state = NetworkState(t_current=t_current, P_0=P_0, Delta=Delta)

    # Generate random graph using NetworkX
    G = nx.erdos_renyi_graph(n=n_actors, p=edge_probability, seed=seed, directed=False)

    # Add all actors
    for node_id in G.nodes():
        state.add_actor(node_id, hierarchy_level=1, state=ActorState.ACTIVE)

    # Add edges with random trust
    rng = np.random.default_rng(seed)
    for (i, j) in G.edges():
        # Add both directions
        trust_ij = rng.uniform(0.3, 0.7)
        trust_ji = rng.uniform(0.3, 0.7)

        state.add_edge(i, j, initial_trust=trust_ij)
        state.add_edge(j, i, initial_trust=trust_ji)

    return state


def create_small_world_network(
    n_actors: int,
    k: int = 4,
    p: float = 0.1,
    seed: Optional[int] = None,
    t_current: float = 0.0,
    P_0: float = 1.0,
    Delta: float = 10.0
) -> NetworkState:
    """
    Create small-world network using Watts-Strogatz model.

    Small-world networks have high clustering but short path lengths,
    characteristic of social networks including criminal organizations.

    Parameters
    ----------
    n_actors : int
        Number of actors to create
    k : int, default=4
        Each node connected to k nearest neighbors in ring
    p : float, default=0.1
        Probability of rewiring each edge
    seed : int, optional
        Random seed for reproducibility
    t_current : float, default=0.0
        Initial simulation time
    P_0 : float, default=1.0
        Initial law enforcement effectiveness
    Delta : float, default=10.0
        Memory window for exposure

    Returns
    -------
    NetworkState
        Initialized small-world network

    Examples
    --------
    >>> network = create_small_world_network(n_actors=40, k=4, p=0.1, seed=42)
    >>> len(network.V)
    40

    Notes
    -----
    - High local clustering (friends of friends are friends)
    - Short average path length (small-world property)
    - All actors assigned to hierarchy level 1
    """
    state = NetworkState(t_current=t_current, P_0=P_0, Delta=Delta)

    # Generate small-world graph using NetworkX
    G = nx.watts_strogatz_graph(n=n_actors, k=k, p=p, seed=seed)

    # Add all actors
    for node_id in G.nodes():
        state.add_actor(node_id, hierarchy_level=1, state=ActorState.ACTIVE)

    # Add edges with trust
    rng = np.random.default_rng(seed)
    for (i, j) in G.edges():
        # Add both directions
        trust_ij = rng.uniform(0.4, 0.75)
        trust_ji = rng.uniform(0.4, 0.75)

        state.add_edge(i, j, initial_trust=trust_ij)
        state.add_edge(j, i, initial_trust=trust_ji)

    return state


def create_core_periphery_network(
    n_core: int,
    n_periphery: int,
    core_density: float = 0.8,
    periphery_to_core_ratio: float = 2.0,
    seed: Optional[int] = None,
    t_current: float = 0.0,
    P_0: float = 1.0,
    Delta: float = 10.0
) -> NetworkState:
    """
    Create core-periphery network structure.

    Core actors are densely connected, periphery actors connect mainly to core.
    Common in criminal networks: core = key operators, periphery = associates.

    Parameters
    ----------
    n_core : int
        Number of core actors
    n_periphery : int
        Number of periphery actors
    core_density : float, default=0.8
        Edge density within core (0-1)
    periphery_to_core_ratio : float, default=2.0
        Average number of core connections per periphery actor
    seed : int, optional
        Random seed for reproducibility
    t_current : float, default=0.0
        Initial simulation time
    P_0 : float, default=1.0
        Initial law enforcement effectiveness
    Delta : float, default=10.0
        Memory window for exposure

    Returns
    -------
    NetworkState
        Initialized core-periphery network

    Examples
    --------
    >>> network = create_core_periphery_network(
    ...     n_core=10,
    ...     n_periphery=30,
    ...     core_density=0.8,
    ...     seed=42
    ... )
    >>> len(network.V)
    40

    Notes
    -----
    - Core actors assigned hierarchy level 2
    - Periphery actors assigned hierarchy level 1
    - Core has high trust, periphery-core has moderate trust
    """
    state = NetworkState(t_current=t_current, P_0=P_0, Delta=Delta)
    rng = np.random.default_rng(seed)

    # Create core actors (hierarchy level 2)
    core_actors = []
    for i in range(n_core):
        state.add_actor(i, hierarchy_level=2, state=ActorState.ACTIVE)
        core_actors.append(i)

    # Create periphery actors (hierarchy level 1)
    periphery_actors = []
    for i in range(n_core, n_core + n_periphery):
        state.add_actor(i, hierarchy_level=1, state=ActorState.ACTIVE)
        periphery_actors.append(i)

    # Connect core densely
    for i, actor1 in enumerate(core_actors):
        for actor2 in core_actors[i+1:]:
            if rng.random() < core_density:
                trust = rng.uniform(0.75, 0.95)
                state.add_edge(actor1, actor2, initial_trust=trust)
                state.add_edge(actor2, actor1, initial_trust=trust)

    # Connect periphery to core
    connections_per_periphery = int(periphery_to_core_ratio)
    for p_actor in periphery_actors:
        # Select random core actors to connect to
        num_connections = min(connections_per_periphery, len(core_actors))
        core_targets = rng.choice(core_actors, size=num_connections, replace=False)

        for c_actor in core_targets:
            trust_p_to_c = rng.uniform(0.5, 0.7)
            trust_c_to_p = rng.uniform(0.5, 0.7)

            state.add_edge(p_actor, c_actor, initial_trust=trust_p_to_c)
            state.add_edge(c_actor, p_actor, initial_trust=trust_c_to_p)

    return state


def validate_network_structure(state: NetworkState) -> Tuple[bool, List[str]]:
    """
    Validate network structure for simulation readiness.

    Parameters
    ----------
    state : NetworkState
        Network to validate

    Returns
    -------
    is_valid : bool
        True if network is valid for simulation
    errors : List[str]
        List of validation error messages (empty if valid)

    Examples
    --------
    >>> network = create_scale_free_network(50, seed=42)
    >>> is_valid, errors = validate_network_structure(network)
    >>> is_valid
    True
    >>> len(errors)
    0
    """
    errors = []

    # Check minimum actors
    if len(state.V) == 0:
        errors.append("Network must have at least one actor")

    # Check all actors are active
    inactive_count = sum(1 for a in state.A.values() if a != ActorState.ACTIVE)
    if inactive_count > 0:
        errors.append(f"{inactive_count} actors are not in ACTIVE state")

    # Check hierarchy levels assigned
    for actor_id in state.V:
        if actor_id not in state.hierarchy:
            errors.append(f"Actor {actor_id} missing hierarchy level")

    # Check edge validity
    for (i, j) in state.E:
        if i not in state.V:
            errors.append(f"Edge ({i}, {j}) references non-existent source {i}")
        if j not in state.V:
            errors.append(f"Edge ({i}, {j}) references non-existent target {j}")

    # Check connectivity (warn if disconnected)
    if len(state.E) > 0:
        # Build undirected graph for connectivity check
        G = nx.Graph()
        for (i, j) in state.E:
            G.add_edge(i, j)

        if not nx.is_connected(G):
            num_components = nx.number_connected_components(G)
            errors.append(f"Network is disconnected ({num_components} components). May affect dynamics.")

    is_valid = len(errors) == 0
    return is_valid, errors


def get_network_statistics(state: NetworkState) -> dict:
    """
    Compute structural statistics for network.

    Parameters
    ----------
    state : NetworkState
        Network to analyze

    Returns
    -------
    stats : dict
        Dictionary of network statistics

    Examples
    --------
    >>> network = create_hierarchical_network([20, 10, 5], seed=42)
    >>> stats = get_network_statistics(network)
    >>> stats['num_actors']
    35
    >>> stats['num_edges'] > 0
    True
    """
    # Build NetworkX graph for analysis
    G = nx.Graph()
    for (i, j) in state.E:
        G.add_edge(i, j)

    stats = {
        'num_actors': len(state.V),
        'num_edges': len(state.E),
        'avg_degree': np.mean([G.degree(n) for n in G.nodes()]) if len(G) > 0 else 0,
        'max_degree': max([G.degree(n) for n in G.nodes()]) if len(G) > 0 else 0,
        'density': nx.density(G) if len(G) > 0 else 0,
        'is_connected': nx.is_connected(G) if len(G) > 0 else False,
        'num_components': nx.number_connected_components(G) if len(G) > 0 else 0,
        'avg_clustering': nx.average_clustering(G) if len(G) > 0 else 0,
        'hierarchy_distribution': {
            level: sum(1 for a_id in state.V if state.hierarchy.get(a_id) == level)
            for level in set(state.hierarchy.values())
        }
    }

    # Add largest component size
    if len(G) > 0:
        components = list(nx.connected_components(G))
        stats['largest_component_size'] = len(max(components, key=len)) if components else 0
    else:
        stats['largest_component_size'] = 0

    return stats
