"""
Network growth simulation using Trust-Filtered Preferential Attachment (TFPA).

This module implements the TFPA mechanism for adding new edges
based on Jaccard similarity and preferential attachment.

Algorithm: Select source uniformly, target via preferential attachment
with trust filter W^init(i,j) ≥ w_min.

Constitutional Reference: Section II (TFPA Mechanism)
"""

from typing import List, Tuple
import numpy as np


class GrowthProcess:
    """
    Network growth simulator using TFPA mechanism.

    Algorithm: Select source uniformly, target via preferential attachment
    with trust filter W^init(i,j) ≥ w_min.

    Constitutional Reference: Section II (TFPA Mechanism)

    Parameters
    ----------
    w_min : float
        Minimum trust threshold ∈ [0, 1)
    gamma_pa : float
        Preferential attachment strength (≥ 1)
    rng : np.random.Generator
        Random number generator

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> process = GrowthProcess(w_min=0.3, gamma_pa=1.5, rng=rng)
    >>> trust = process.compute_initial_trust(source=0, target=5, network_state)
    >>> 0.0 <= trust <= 1.0
    True
    """

    def __init__(
        self,
        w_min: float,
        gamma_pa: float,
        rng: np.random.Generator
    ):
        """
        Initialize TFPA growth process.

        Parameters
        ----------
        w_min : float
            Minimum trust threshold ∈ [0, 1)
        gamma_pa : float
            Preferential attachment strength (≥ 1)
        rng : np.random.Generator
            Random number generator
        """
        if not (0 <= w_min < 1):
            raise ValueError(f"w_min must be in [0, 1), got {w_min}")
        if gamma_pa < 1:
            raise ValueError(f"gamma_pa must be ≥ 1, got {gamma_pa}")

        self.w_min = w_min
        self.gamma_pa = gamma_pa
        self.rng = rng

    def compute_initial_trust(
        self,
        source: int,
        target: int,
        network_state
    ) -> float:
        """
        Compute Jaccard-based initial trust.

        Parameters
        ----------
        source, target : int
            Actors forming new edge
        network_state : NetworkState
            Current network (for neighborhood computation)

        Returns
        -------
        trust : float
            W^init(source, target) ∈ [0, 1]

        Formula
        -------
        W^init(i,j) = |N_i ∩ N_j| / |N_i ∪ N_j|

        If |N_i ∪ N_j| = 0 (both isolated), return default value 0.5

        Constitutional Reference: Section II, TFPA formula

        Examples
        --------
        >>> trust = process.compute_initial_trust(0, 5, network_state)
        >>> isinstance(trust, float)
        True
        >>> 0.0 <= trust <= 1.0
        True
        """
        # Get neighborhoods (both in and out neighbors)
        N_i = set(network_state.get_all_neighbors(source))
        N_j = set(network_state.get_all_neighbors(target))

        # Jaccard similarity
        intersection = N_i & N_j
        union = N_i | N_j

        if len(union) == 0:
            # Both actors isolated - use default trust
            return 0.5

        trust = len(intersection) / len(union)
        return trust

    def generate_growth_events(
        self,
        network_state,
        growth_rate: float,
        t_start: float,
        t_end: float
    ) -> List[Tuple[float, int, int, float]]:
        """
        Generate new edge additions in time interval.

        Parameters
        ----------
        network_state : NetworkState
            Current network
        growth_rate : float
            Rate of new edge formation (events per time unit)
        t_start, t_end : float
            Time interval

        Returns
        -------
        edges : list of (time, source, target, initial_trust)
            New edges to add

        Algorithm
        ---------
        1. Generate Poisson(growth_rate * dt) new edges
        2. For each edge:
            a. Select source i uniformly from active actors
            b. Compute attachment probabilities:
               π_{i→j} ∝ (k_j^in + 1)^γ_pa · 𝟙{W^init(i,j) ≥ w_min}
            c. Select target j with probability π_{i→j}
            d. Compute Y_ij^init = logit(W^init(i,j))

        Examples
        --------
        >>> edges = process.generate_growth_events(
        ...     network_state=state,
        ...     growth_rate=0.5,
        ...     t_start=0.0,
        ...     t_end=10.0
        ... )
        >>> all(t_start <= t < t_end for t, _, _, _ in edges)
        True
        """
        dt = t_end - t_start
        active_actors = network_state.get_active_actors()

        if len(active_actors) < 2:
            # Need at least 2 actors to form edge
            return []

        # Generate number of new edges
        expected_count = growth_rate * dt
        if expected_count <= 0:
            return []

        num_edges = self.rng.poisson(expected_count)
        if num_edges == 0:
            return []

        # Generate edge times uniformly
        edge_times = sorted(self.rng.uniform(t_start, t_end, size=num_edges))

        new_edges = []

        for t in edge_times:
            # Select source uniformly
            source = self.rng.choice(active_actors)

            # Compute attachment probabilities for all potential targets
            # π_{i→j} ∝ (k_j^in + 1)^γ_pa · 𝟙{W^init(i,j) ≥ w_min}
            candidates = []
            probabilities = []

            for target in active_actors:
                if target == source:
                    continue  # No self-loops

                # Check if edge already exists
                if (source, target) in network_state.E:
                    continue

                # Compute initial trust
                w_init = self.compute_initial_trust(source, target, network_state)

                # Apply trust filter
                if w_init < self.w_min:
                    continue

                # Preferential attachment based on in-degree
                k_in = len(network_state.get_incoming_neighbors(target))
                prob = (k_in + 1) ** self.gamma_pa

                candidates.append(target)
                probabilities.append(prob)

            if not candidates:
                # No valid targets - skip this edge
                continue

            # Normalize probabilities
            probabilities = np.array(probabilities)
            probabilities /= probabilities.sum()

            # Select target
            target = self.rng.choice(candidates, p=probabilities)

            # Compute initial trust (already did above, but recompute for consistency)
            w_init = self.compute_initial_trust(source, target, network_state)

            new_edges.append((t, source, target, w_init))

        return new_edges

    @staticmethod
    def logit(w: float) -> float:
        """
        Compute logit (inverse of expit).

        logit(w) = log(w / (1 - w))

        Parameters
        ----------
        w : float
            Trust value ∈ (0, 1)

        Returns
        -------
        float
            Logit-trust Y ∈ ℝ

        Examples
        --------
        >>> GrowthProcess.logit(0.5)
        0.0
        >>> GrowthProcess.logit(0.7) > 0
        True
        >>> GrowthProcess.logit(0.3) < 0
        True
        """
        # Clamp to avoid numerical issues
        w = np.clip(w, 1e-10, 1 - 1e-10)
        return np.log(w / (1 - w))
