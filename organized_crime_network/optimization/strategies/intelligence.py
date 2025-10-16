"""
Intelligence Optimization Strategies.

This module implements algorithms for maximizing expected information gain:
- Lazy Greedy: Provably achieves (1-1/e) ≈ 63% of optimal
- Uses priority queue with marginal gain caching
- Exploits submodularity for efficiency

Theoretical guarantee: F(S_greedy) >= (1 - 1/e) · F(S_optimal)

Mathematical Foundation: gelis.tex §II
Governance: .specify/memory/constitution.md v1.0.0
"""

import time
import heapq
from typing import Dict, FrozenSet, Tuple
from ..base import Optimizer, OptimizationProblem, TargetSet, InformationModel
from ..objectives import (
    compute_intelligence_value,
    compute_marginal_intelligence,
    compute_conversion_probability,
)
from ...core.state import NetworkState, ActorState


class IntelligenceOptimizer(Optimizer):
    """
    Optimizer for maximizing expected intelligence gain from informants.

    Uses lazy greedy algorithm with (1-1/e) approximation guarantee.
    The intelligence objective is provably monotone and submodular,
    which ensures the approximation guarantee.

    Algorithm:
    1. Maintain priority queue of actors by upper bound on marginal gain
    2. Pop actor with highest bound
    3. Re-evaluate marginal gain (lazy evaluation)
    4. If still highest, add to solution; otherwise re-insert
    5. Repeat until K actors selected

    Time complexity: O(K · N · log N) in practice (vs O(K · N²) for standard greedy)

    Constitution Reference: §II - Informant Conversion + Network Structure
    """

    def __init__(self, information_model: InformationModel, config: Dict = None):
        """
        Initialize intelligence optimizer.

        Args:
            information_model: Defines what information actors know
            config: Configuration (e.g., {'use_lazy_greedy': True})

        Raises:
            ValueError: If information_model is invalid
        """
        if information_model is None:
            raise ValueError("information_model cannot be None")

        self.information_model = information_model
        self.config = config or {}

        # Performance tracking
        self._last_metrics = {
            'iterations': 0,
            'evaluations': 0,
            'time': 0.0,
        }

    def optimize(self, problem: OptimizationProblem) -> TargetSet:
        """
        Find optimal targets for intelligence gain.

        Args:
            problem: Optimization problem specification

        Returns:
            TargetSet with selected actors and intelligence metrics

        Raises:
            ValueError: If problem specification is invalid
        """
        start_time = time.time()

        # Validate problem
        if 'intelligence' not in problem.objectives:
            raise ValueError("IntelligenceOptimizer requires 'intelligence' objective")
        if problem.time_horizon is None or problem.time_horizon <= 0:
            raise ValueError("Intelligence optimization requires positive time_horizon")
        if problem.parameters is None:
            raise ValueError("Intelligence optimization requires parameters for conversion calculation")

        # Validate budget
        active_actors = [actor_id for actor_id, status in problem.network_state.A.items()
                        if status == ActorState.ACTIVE]
        if problem.budget > len(active_actors):
            raise ValueError(f"Budget {problem.budget} exceeds {len(active_actors)} active actors")
        if problem.budget <= 0:
            raise ValueError("Budget must be positive")

        # Execute lazy greedy algorithm
        selected, rationale = self._lazy_greedy(
            problem.network_state,
            problem.budget,
            problem.time_horizon,
            problem.parameters
        )

        # Compute metrics
        intelligence = compute_intelligence_value(
            selected, problem.network_state, problem.time_horizon,
            problem.parameters, self.information_model
        )
        metrics = {'intelligence': intelligence}

        # Record performance
        computation_time = time.time() - start_time
        self._last_metrics['time'] = computation_time

        return TargetSet(
            actors=selected,
            metrics=metrics,
            rationale=rationale,
            quality_guarantee="(1-1/e) ~= 63% of optimal",
            computation_time=computation_time
        )

    def evaluate_targets(self, actors: FrozenSet[int],
                        network_state: NetworkState) -> Dict[str, float]:
        """
        Evaluate intelligence for proposed target set.

        Note: Requires time_horizon and parameters from context.
        This method is for interface compliance; actual evaluation
        needs additional parameters.

        Args:
            actors: Set of actors to evaluate
            network_state: Current network state

        Returns:
            Dictionary with 'intelligence' metric (placeholder)
        """
        # Placeholder - full evaluation needs time_horizon and parameters
        return {'intelligence': 0.0}

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Return performance statistics from last optimization.

        Returns:
            Dictionary with iterations, evaluations, time
        """
        return self._last_metrics.copy()

    def _lazy_greedy(self, state: NetworkState, K: int, time_horizon: float,
                     parameters: Dict[str, float]) -> Tuple[FrozenSet[int], Dict[int, str]]:
        """
        Lazy greedy algorithm for submodular maximization.

        Maintains priority queue of actors with upper bounds on marginal gain.
        Re-evaluates marginal gain only when actor reaches top of queue.

        Time complexity: O(K · N · log N) in practice

        Args:
            state: Network state
            K: Budget (number of actors to select)
            time_horizon: Time window for intelligence
            parameters: Simulation parameters

        Returns:
            (selected actors, rationale for each)
        """
        selected = frozenset()
        rationale = {}
        iterations = 0
        evaluations = 0

        # Get active actors
        active_actors = [actor_id for actor_id, status in state.A.items()
                        if status == ActorState.ACTIVE]

        # Priority queue: (-marginal_gain, actor_id, is_stale)
        # Use negative for max heap
        pq = []

        # Initialize: compute initial marginal gains
        for actor_id in active_actors:
            marginal = compute_marginal_intelligence(
                actor_id, selected, state, time_horizon, parameters, self.information_model
            )
            evaluations += 1
            heapq.heappush(pq, (-marginal, actor_id, False))

        # Greedy selection with lazy evaluation
        while len(selected) < K and pq:
            iterations += 1

            # Pop actor with highest bound
            neg_gain, actor_id, is_stale = heapq.heappop(pq)
            current_gain = -neg_gain

            if actor_id in selected:
                continue  # Already selected

            # Re-evaluate if stale or if not confident it's still best
            if is_stale or (pq and -pq[0][0] > current_gain):
                # Re-evaluate marginal gain
                marginal = compute_marginal_intelligence(
                    actor_id, selected, state, time_horizon, parameters, self.information_model
                )
                evaluations += 1

                # Re-insert with updated gain
                heapq.heappush(pq, (-marginal, actor_id, False))
                continue

            # This actor has highest marginal gain - select it
            selected = frozenset(selected | {actor_id})

            # Generate rationale
            conversion = compute_conversion_probability(actor_id, state, time_horizon, parameters)
            info_set = self.information_model.get_information_set(actor_id, state)
            unique_info_count = len(info_set)

            rationale[actor_id] = (
                f"Conversion probability: {conversion.probability:.2%}, "
                f"Unique information: {unique_info_count} edges, "
                f"Marginal intelligence gain: {current_gain:.2f}"
            )

            # Mark all remaining actors as stale (their marginal gains may have changed)
            new_pq = []
            for (neg_g, aid, _) in pq:
                if aid not in selected:
                    new_pq.append((neg_g, aid, True))
            pq = new_pq
            heapq.heapify(pq)

        self._last_metrics['iterations'] = iterations
        self._last_metrics['evaluations'] = evaluations

        return selected, rationale
