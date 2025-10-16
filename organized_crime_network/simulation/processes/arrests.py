"""
Arrest process simulation using Cox process (non-homogeneous Poisson).

This module implements the time-dependent arrest intensity and
event generation via thinning algorithm.

Intensity: λ_i^Arr(t) = λ_0 · h(H(i)) · v(i,t) · P(t)

Constitutional Reference: Section II (Arrest Intensity)
Research Decision: Decision 2 (Cox Process Simulation)
"""

from dataclasses import dataclass
from typing import List, Tuple, Callable
import numpy as np


@dataclass
class ArrestIntensity:
    """
    Record of computed arrest intensity for an actor.

    Attributes
    ----------
    actor_id : int
        Actor identifier
    base_rate : float
        λ_0
    hierarchy_protection : float
        h(H(i)) = exp(-κ(H(i)-1))
    operational_risk : float
        v(i,t) = 1 + γ Σ_j expit(Y_ij)
    effectiveness : float
        P(t)
    total_intensity : float
        λ_i^Arr(t) = λ_0 · h · v · P
    """
    actor_id: int
    base_rate: float
    hierarchy_protection: float
    operational_risk: float
    effectiveness: float
    total_intensity: float


class ArrestProcess:
    """
    Arrest event simulator using thinning algorithm.

    Intensity: λ_i^Arr(t) = λ_0 · h(H(i)) · v(i,t) · P(t)

    where:
        h(l) = exp(-κ(l-1))  (hierarchical protection)
        v(i,t) = 1 + γ Σ_{j ∈ N_i^out} expit(Y_ij)  (operational risk)

    Constitutional Reference: Section II (Arrest Intensity)
    Research Decision: Decision 2 (Cox Process Simulation)

    Parameters
    ----------
    lambda_0 : float
        Base arrest rate (> 0)
    kappa : float
        Hierarchical protection factor (≥ 0)
    gamma : float
        Operational risk multiplier (≥ 0)
    rng : np.random.Generator
        Random number generator

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> process = ArrestProcess(lambda_0=0.1, kappa=0.5, gamma=0.3, rng=rng)
    >>> intensity = process.compute_intensity(actor_id=0, network_state, effectiveness=1.0)
    >>> intensity > 0
    True
    """

    def __init__(
        self,
        lambda_0: float,
        kappa: float,
        gamma: float,
        rng: np.random.Generator
    ):
        """
        Initialize arrest process simulator.

        Parameters
        ----------
        lambda_0 : float
            Base arrest rate (> 0)
        kappa : float
            Hierarchical protection factor (≥ 0)
        gamma : float
            Operational risk multiplier (≥ 0)
        rng : np.random.Generator
            Random number generator
        """
        if lambda_0 <= 0:
            raise ValueError("lambda_0 must be > 0")
        if kappa < 0:
            raise ValueError("kappa must be ≥ 0")
        if gamma < 0:
            raise ValueError("gamma must be ≥ 0")

        self.lambda_0 = lambda_0
        self.kappa = kappa
        self.gamma = gamma
        self.rng = rng

    def compute_intensity(
        self,
        actor_id: int,
        network_state,
        effectiveness: float
    ) -> float:
        """
        Compute arrest intensity for actor at current time.

        Parameters
        ----------
        actor_id : int
            Actor to compute intensity for
        network_state : NetworkState
            Current network state (for Y_ij, neighbors, hierarchy)
        effectiveness : float
            Current P(t) value

        Returns
        -------
        intensity : float
            λ_i^Arr(t) ≥ 0

        Formula
        -------
        h(l) = exp(-κ(l-1))
        v(i,t) = 1 + γ Σ_{j ∈ N_i^out} expit(Y_ij)
        λ = λ_0 · h · v · P

        Constitutional Reference: Section II, Equation for λ_i^Arr

        Examples
        --------
        >>> intensity = process.compute_intensity(
        ...     actor_id=5,
        ...     network_state=state,
        ...     effectiveness=1.0
        ... )
        >>> isinstance(intensity, float)
        True
        >>> intensity >= 0
        True
        """
        # Hierarchical protection: h(l) = exp(-κ(l-1))
        hierarchy_level = network_state.hierarchy.get(actor_id, 1)
        h = np.exp(-self.kappa * (hierarchy_level - 1))

        # Operational risk: v(i,t) = 1 + γ Σ_j expit(Y_ij)
        out_neighbors = network_state.get_outgoing_neighbors(actor_id)
        if out_neighbors:
            # expit(x) = 1 / (1 + exp(-x))
            trust_sum = sum(
                self._expit(network_state.Y[(actor_id, j)])
                for j in out_neighbors
            )
            v = 1.0 + self.gamma * trust_sum
        else:
            v = 1.0

        # Total intensity
        intensity = self.lambda_0 * h * v * effectiveness

        return max(0.0, intensity)  # Ensure non-negative

    def generate_arrests(
        self,
        active_actors: List[int],
        network_state,
        effectiveness_func: Callable[[float], float],
        t_start: float,
        t_end: float
    ) -> List[Tuple[float, int]]:
        """
        Generate arrest events in time interval using thinning.

        Parameters
        ----------
        active_actors : list of int
            Currently active actors
        network_state : NetworkState
            Current network (for intensity computation)
        effectiveness_func : callable
            Function P(t) for time-dependent effectiveness
        t_start, t_end : float
            Time interval [t_start, t_end)

        Returns
        -------
        arrests : list of (float, int)
            List of (time, actor_id) pairs, sorted by time

        Algorithm
        ---------
        For each actor i:
            1. Estimate λ_max_i = max_{t ∈ [t_start, t_end]} λ_i(t)
            2. Generate Poisson(λ_max_i * (t_end - t_start)) candidates
            3. Accept each with probability λ_i(t) / λ_max_i
            4. Combine all actors' arrests and sort by time

        Performance
        -----------
        O(N × expected_arrests) where N = number of actors

        Examples
        --------
        >>> arrests = process.generate_arrests(
        ...     active_actors=[0, 1, 2],
        ...     network_state=state,
        ...     effectiveness_func=lambda t: 1.0,
        ...     t_start=0.0,
        ...     t_end=10.0
        ... )
        >>> len(arrests) >= 0
        True
        >>> all(t_start <= t < t_end for t, _ in arrests)
        True
        """
        all_arrests = []
        dt = t_end - t_start

        for actor_id in active_actors:
            # Estimate maximum intensity over interval
            # Use conservative upper bound with safety factor
            P_max = max(effectiveness_func(t_start), effectiveness_func(t_end))
            lambda_max = self.compute_intensity(actor_id, network_state, P_max) * 1.2

            if lambda_max <= 0:
                continue

            # Generate candidate events from homogeneous Poisson
            expected_count = lambda_max * dt
            if expected_count <= 0:
                continue

            num_candidates = self.rng.poisson(expected_count)
            if num_candidates == 0:
                continue

            # Uniform candidate times
            candidate_times = self.rng.uniform(t_start, t_end, size=num_candidates)

            # Thinning: accept with probability λ(t) / λ_max
            for t in candidate_times:
                P_t = effectiveness_func(t)
                lambda_t = self.compute_intensity(actor_id, network_state, P_t)
                accept_prob = lambda_t / lambda_max

                if self.rng.uniform() < accept_prob:
                    all_arrests.append((t, actor_id))

        # Sort by time
        all_arrests.sort(key=lambda x: x[0])

        return all_arrests

    @staticmethod
    def _expit(x: float) -> float:
        """
        Numerically stable expit (logistic sigmoid) function.

        expit(x) = 1 / (1 + exp(-x))

        Parameters
        ----------
        x : float
            Input value

        Returns
        -------
        float
            expit(x) ∈ (0, 1)
        """
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)
