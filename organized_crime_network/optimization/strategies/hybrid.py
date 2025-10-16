"""
Hybrid Optimization Strategy for Combined Objectives.

This module implements weighted combination optimization:
- Combines structural damage and intelligence objectives
- Uses λ weight parameter: λ·damage + (1-λ)·intelligence
- Enables flexible trade-off selection

Mathematical Foundation: Multi-objective optimization
Governance: .specify/memory/constitution.md v1.0.0
"""

import time
from typing import Dict, FrozenSet, Tuple
from ..base import Optimizer, OptimizationProblem, TargetSet, InformationModel
from ..objectives import (
    compute_structural_damage,
    compute_intelligence_value,
    compute_marginal_intelligence,
)
from ...core.state import NetworkState, ActorState


class HybridOptimizer(Optimizer):
    """
    Optimizer for combined objective: λ·damage + (1-λ)·intelligence

    Balances structural damage and intelligence gain using weighted combination.
    User controls trade-off via lambda_weight parameter.

    Algorithm:
    - Greedy selection based on weighted marginal gains
    - score(actor) = λ·marginal_damage + (1-λ)·marginal_intelligence
    - Selects actor with highest weighted score at each iteration

    No theoretical guarantee (combined objective is not generally submodular).
    """

    def __init__(self, lambda_weight: float = 0.5, information_model: InformationModel = None,
                 config: Dict = None):
        """
        Initialize hybrid optimizer.

        Args:
            lambda_weight: Weight λ ∈ [0,1] for objective combination
                λ=1: pure damage, λ=0: pure intelligence, λ=0.5: balanced
            information_model: Model for intelligence value computation
            config: Additional configuration

        Raises:
            ValueError: If lambda_weight not in [0,1] or information_model is None
        """
        if not (0 <= lambda_weight <= 1):
            raise ValueError(f"lambda_weight must be in [0,1], got {lambda_weight}")
        if information_model is None:
            raise ValueError("information_model cannot be None")

        self.lambda_weight = lambda_weight
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
        Find optimal targets for combined objective.

        Args:
            problem: Optimization problem specification

        Returns:
            TargetSet with selected actors and combined metrics

        Raises:
            ValueError: If problem specification is invalid
        """
        start_time = time.time()

        # Validate problem
        if 'damage' not in problem.objectives and 'intelligence' not in problem.objectives:
            raise ValueError("HybridOptimizer requires at least one of 'damage' or 'intelligence'")
        if 'intelligence' in problem.objectives:
            if problem.time_horizon is None or problem.time_horizon <= 0:
                raise ValueError("Intelligence objective requires positive time_horizon")
            if problem.parameters is None:
                raise ValueError("Intelligence objective requires parameters")

        # Validate budget
        active_actors = [actor_id for actor_id, status in problem.network_state.A.items()
                        if status == ActorState.ACTIVE]
        if problem.budget > len(active_actors):
            raise ValueError(f"Budget {problem.budget} exceeds {len(active_actors)} active actors")
        if problem.budget <= 0:
            raise ValueError("Budget must be positive")

        # Execute weighted greedy algorithm
        selected, rationale = self._weighted_greedy(
            problem.network_state,
            problem.budget,
            problem.time_horizon,
            problem.parameters
        )

        # Compute metrics
        damage = compute_structural_damage(selected, problem.network_state)
        intelligence = compute_intelligence_value(
            selected, problem.network_state, problem.time_horizon,
            problem.parameters, self.information_model
        )
        combined = self.lambda_weight * damage + (1 - self.lambda_weight) * intelligence

        metrics = {
            'damage': damage,
            'intelligence': intelligence,
            'combined': combined
        }

        # Record performance
        computation_time = time.time() - start_time
        self._last_metrics['time'] = computation_time

        return TargetSet(
            actors=selected,
            metrics=metrics,
            rationale=rationale,
            quality_guarantee=f"Weighted greedy (λ={self.lambda_weight:.2f}), no theoretical guarantee",
            computation_time=computation_time
        )

    def evaluate_targets(self, actors: FrozenSet[int],
                        network_state: NetworkState) -> Dict[str, float]:
        """
        Evaluate combined objective for proposed target set.

        Note: Requires time_horizon and parameters from context.

        Args:
            actors: Set of actors to evaluate
            network_state: Current network state

        Returns:
            Dictionary with metrics (placeholder without full context)
        """
        # Placeholder - full evaluation needs time_horizon and parameters
        damage = compute_structural_damage(actors, network_state)
        return {'damage': damage, 'intelligence': 0.0, 'combined': damage}

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Return performance statistics from last optimization.

        Returns:
            Dictionary with iterations, evaluations, time
        """
        return self._last_metrics.copy()

    def _weighted_greedy(self, state: NetworkState, K: int, time_horizon: float,
                        parameters: Dict[str, float]) -> Tuple[FrozenSet[int], Dict[int, str]]:
        """
        Weighted greedy algorithm for combined objective.

        Iteratively selects actor with highest weighted marginal gain:
        score(i) = λ·marginal_damage(i) + (1-λ)·marginal_intelligence(i)

        Args:
            state: Network state
            K: Budget
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

        # Greedy selection
        for iteration in range(K):
            iterations += 1
            best_actor = None
            best_score = -float('inf')
            best_damage_marginal = 0.0
            best_intel_marginal = 0.0

            for actor_id in active_actors:
                if actor_id in selected:
                    continue

                # Compute marginal damage
                test_set = frozenset(selected | {actor_id})
                current_damage = compute_structural_damage(selected, state)
                new_damage = compute_structural_damage(test_set, state)
                marginal_damage = new_damage - current_damage

                # Compute marginal intelligence
                marginal_intel = compute_marginal_intelligence(
                    actor_id, selected, state, time_horizon, parameters, self.information_model
                )

                evaluations += 2  # One damage eval, one intelligence eval

                # Weighted score
                score = self.lambda_weight * marginal_damage + (1 - self.lambda_weight) * marginal_intel

                if score > best_score:
                    best_score = score
                    best_actor = actor_id
                    best_damage_marginal = marginal_damage
                    best_intel_marginal = marginal_intel

            if best_actor is None:
                break  # No more actors available

            selected = frozenset(selected | {best_actor})
            rationale[best_actor] = (
                f"Weighted score: {best_score:.2f} "
                f"(damage: {best_damage_marginal:.2f}, intel: {best_intel_marginal:.2f}, "
                f"λ={self.lambda_weight:.2f})"
            )

        self._last_metrics['iterations'] = iterations
        self._last_metrics['evaluations'] = evaluations

        return selected, rationale
