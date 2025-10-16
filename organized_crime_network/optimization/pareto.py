"""
Pareto Frontier Computation for Multi-Objective Optimization.

This module provides tools for exploring trade-offs between competing objectives:
- ParetoSolution: Solution with dominance relationships
- ParetoFrontierComputer: Generates Pareto-optimal solutions

Algorithm: Weighted sum scalarization with systematic weight sweep.
Result: 20-50 non-dominated solutions spanning the trade-off space.

Mathematical Foundation: Multi-objective optimization theory
Governance: .specify/memory/constitution.md v1.0.0
"""

import numpy as np
from dataclasses import dataclass
from typing import List, FrozenSet, Dict
from .base import TargetSet, OptimizationProblem
from .strategies import StructuralDamageOptimizer, IntelligenceOptimizer


@dataclass(frozen=True)
class ParetoSolution:
    """
    A solution that is part of the Pareto-optimal set.

    Attributes:
        target_set: The underlying solution with actors and metrics
        dominated_by: Indices of solutions that dominate this one (empty for Pareto-optimal)
        dominates: Indices of solutions this one dominates
        pareto_rank: Rank in Pareto hierarchy (0 = Pareto-optimal)
        crowding_distance: Measure of solution spacing (for diversity)

    Validation:
        - pareto_rank == 0 implies len(dominated_by) == 0
        - crowding_distance >= 0
    """
    target_set: TargetSet
    dominated_by: FrozenSet[int] = frozenset()
    dominates: FrozenSet[int] = frozenset()
    pareto_rank: int = 0
    crowding_distance: float = 0.0

    @property
    def is_pareto_optimal(self) -> bool:
        """Check if solution is Pareto-optimal."""
        return self.pareto_rank == 0

    def __post_init__(self):
        """Validate Pareto solution."""
        if self.pareto_rank == 0 and len(self.dominated_by) > 0:
            raise ValueError("Pareto-optimal solutions cannot be dominated")
        if self.crowding_distance < 0:
            raise ValueError("Crowding distance must be non-negative")


class ParetoFrontierComputer:
    """
    Computes Pareto-optimal solutions for bi-objective optimization.

    Uses weighted sum scalarization: λ·damage + (1-λ)·intelligence
    Systematically sweeps λ ∈ [0, 1] to generate diverse solutions.
    Filters dominated solutions to ensure Pareto optimality.

    Generates 20-50 solutions representing the trade-off space.
    """

    def __init__(self, damage_optimizer: StructuralDamageOptimizer,
                 intelligence_optimizer: IntelligenceOptimizer):
        """
        Initialize Pareto frontier computer.

        Args:
            damage_optimizer: Optimizer for structural damage
            intelligence_optimizer: Optimizer for intelligence

        Raises:
            ValueError: If optimizers are invalid
        """
        if damage_optimizer is None or intelligence_optimizer is None:
            raise ValueError("Both optimizers must be provided")

        self.damage_optimizer = damage_optimizer
        self.intelligence_optimizer = intelligence_optimizer

    def compute_pareto_frontier(self, problem: OptimizationProblem,
                               num_points: int = 30) -> List[ParetoSolution]:
        """
        Generate Pareto-optimal solutions.

        Args:
            problem: Optimization problem with both objectives
            num_points: Desired number of solutions on frontier

        Returns:
            List of Pareto-optimal ParetoSolution objects

        Guarantees:
            - All solutions are Pareto-optimal (no solution dominates another)
            - Solutions span the trade-off space
            - Returns 20-50 diverse solutions (may be less if frontier is small)
            - Sorted by damage objective (descending)

        Algorithm:
            1. Generate solutions using weighted sum with λ ∈ [0, 1]
            2. Filter dominated solutions
            3. Compute crowding distances for diversity
            4. Return Pareto-optimal set

        Raises:
            ValueError: If problem doesn't have both objectives
        """
        # Validate problem
        if 'damage' not in problem.objectives or 'intelligence' not in problem.objectives:
            raise ValueError("Pareto frontier requires both 'damage' and 'intelligence' objectives")
        if num_points < 2:
            raise ValueError("num_points must be at least 2")

        # Generate weight sweep
        weights = self._generate_weight_sweep(num_points)

        # Generate candidate solutions
        candidate_solutions = []
        for lambda_weight in weights:
            solution = self._optimize_for_weight(problem, lambda_weight)
            if solution is not None:
                candidate_solutions.append(solution)

        # Remove duplicates (same actor sets)
        unique_solutions = self._remove_duplicates(candidate_solutions)

        # Filter dominated solutions
        pareto_set = self._filter_dominated(unique_solutions)

        # Compute crowding distances
        pareto_solutions = self._compute_crowding_distances(pareto_set)

        # Sort by damage (descending)
        pareto_solutions.sort(key=lambda x: x.target_set.metrics['damage'], reverse=True)

        return pareto_solutions

    def verify_pareto_optimality(self, solutions: List[ParetoSolution]) -> bool:
        """
        Verify that all solutions are Pareto-optimal.

        Args:
            solutions: List of candidate solutions

        Returns:
            True if no solution strictly dominates any other

        Definition: Solution A dominates B if:
            - A.damage > B.damage AND A.intelligence > B.intelligence
        """
        for i, sol1 in enumerate(solutions):
            for j, sol2 in enumerate(solutions):
                if i != j:
                    if sol1.target_set.dominates(sol2.target_set):
                        return False
        return True

    def _generate_weight_sweep(self, num_points: int) -> List[float]:
        """
        Generate systematic weight sweep.

        Args:
            num_points: Number of weight values

        Returns:
            List of λ values in [0, 1]
        """
        return list(np.linspace(0.0, 1.0, num_points))

    def _optimize_for_weight(self, problem: OptimizationProblem,
                            lambda_weight: float) -> TargetSet:
        """
        Optimize weighted combination: λ·damage + (1-λ)·intelligence

        Args:
            problem: Optimization problem
            lambda_weight: Weight λ ∈ [0, 1] for damage objective

        Returns:
            TargetSet with combined metrics
        """
        # For extreme weights, use single-objective optimizers
        if lambda_weight >= 0.99:
            # Pure damage optimization
            damage_problem = OptimizationProblem(
                network_state=problem.network_state,
                budget=problem.budget,
                objectives=['damage'],
                parameters=problem.parameters
            )
            solution = self.damage_optimizer.optimize(damage_problem)

            # Compute intelligence for this solution
            from .objectives import compute_intelligence_value
            intelligence = compute_intelligence_value(
                solution.actors,
                problem.network_state,
                problem.time_horizon,
                problem.parameters,
                self.intelligence_optimizer.information_model
            )

            # Create combined solution
            combined_metrics = {
                'damage': solution.metrics['damage'],
                'intelligence': intelligence
            }

            return TargetSet(
                actors=solution.actors,
                metrics=combined_metrics,
                rationale=solution.rationale,
                quality_guarantee=f"Optimized for damage (λ={lambda_weight:.2f})",
                computation_time=solution.computation_time
            )

        elif lambda_weight <= 0.01:
            # Pure intelligence optimization
            intel_problem = OptimizationProblem(
                network_state=problem.network_state,
                budget=problem.budget,
                objectives=['intelligence'],
                time_horizon=problem.time_horizon,
                parameters=problem.parameters
            )
            solution = self.intelligence_optimizer.optimize(intel_problem)

            # Compute damage for this solution
            from .objectives import compute_structural_damage
            damage = compute_structural_damage(solution.actors, problem.network_state)

            # Create combined solution
            combined_metrics = {
                'damage': damage,
                'intelligence': solution.metrics['intelligence']
            }

            return TargetSet(
                actors=solution.actors,
                metrics=combined_metrics,
                rationale=solution.rationale,
                quality_guarantee=f"Optimized for intelligence (λ={lambda_weight:.2f})",
                computation_time=solution.computation_time
            )

        else:
            # Hybrid optimization using weighted sum
            # Use a simple greedy approach that considers both objectives
            return self._weighted_greedy(problem, lambda_weight)

    def _weighted_greedy(self, problem: OptimizationProblem,
                        lambda_weight: float) -> TargetSet:
        """
        Greedy algorithm for weighted combination.

        Iteratively selects actor with highest weighted score:
        score(i) = λ·marginal_damage(i) + (1-λ)·marginal_intelligence(i)

        Args:
            problem: Optimization problem
            lambda_weight: Weight for damage objective

        Returns:
            TargetSet with combined metrics
        """
        import time
        from .objectives import (
            compute_structural_damage,
            compute_intelligence_value,
            compute_marginal_intelligence
        )

        start_time = time.time()
        selected = frozenset()
        rationale = {}

        # Get active actors
        active_actors = [aid for aid, status in problem.network_state.A.items()
                        if status == 'Active']

        # Greedy selection
        for iteration in range(problem.budget):
            best_actor = None
            best_score = -float('inf')

            for actor_id in active_actors:
                if actor_id in selected:
                    continue

                # Compute marginal damage
                test_set = frozenset(selected | {actor_id})
                current_damage = compute_structural_damage(selected, problem.network_state)
                new_damage = compute_structural_damage(test_set, problem.network_state)
                marginal_damage = new_damage - current_damage

                # Compute marginal intelligence
                marginal_intel = compute_marginal_intelligence(
                    actor_id, selected, problem.network_state,
                    problem.time_horizon, problem.parameters,
                    self.intelligence_optimizer.information_model
                )

                # Weighted score
                score = lambda_weight * marginal_damage + (1 - lambda_weight) * marginal_intel

                if score > best_score:
                    best_score = score
                    best_actor = actor_id

            if best_actor is None:
                break

            selected = frozenset(selected | {best_actor})
            rationale[best_actor] = f"Weighted score: {best_score:.2f} (λ={lambda_weight:.2f})"

        # Compute final metrics
        damage = compute_structural_damage(selected, problem.network_state)
        intelligence = compute_intelligence_value(
            selected, problem.network_state, problem.time_horizon,
            problem.parameters, self.intelligence_optimizer.information_model
        )

        computation_time = time.time() - start_time

        return TargetSet(
            actors=selected,
            metrics={'damage': damage, 'intelligence': intelligence},
            rationale=rationale,
            quality_guarantee=f"Weighted greedy (λ={lambda_weight:.2f})",
            computation_time=computation_time
        )

    def _remove_duplicates(self, solutions: List[TargetSet]) -> List[TargetSet]:
        """
        Remove duplicate solutions (same actor set).

        Args:
            solutions: List of candidate solutions

        Returns:
            List with duplicates removed
        """
        seen_actors = set()
        unique = []

        for solution in solutions:
            actor_tuple = tuple(sorted(solution.actors))
            if actor_tuple not in seen_actors:
                seen_actors.add(actor_tuple)
                unique.append(solution)

        return unique

    def _filter_dominated(self, solutions: List[TargetSet]) -> List[TargetSet]:
        """
        Filter out dominated solutions.

        Args:
            solutions: List of candidate solutions

        Returns:
            List of non-dominated (Pareto-optimal) solutions
        """
        pareto_set = []

        for candidate in solutions:
            is_dominated = False

            for other in solutions:
                if candidate != other and other.dominates(candidate):
                    is_dominated = True
                    break

            if not is_dominated:
                pareto_set.append(candidate)

        return pareto_set

    def _compute_crowding_distances(self, solutions: List[TargetSet]) -> List[ParetoSolution]:
        """
        Compute crowding distances for diversity measure.

        Crowding distance measures how isolated a solution is in objective space.
        Higher values indicate more isolated (diverse) solutions.

        Args:
            solutions: List of Pareto-optimal solutions

        Returns:
            List of ParetoSolution with crowding distances
        """
        if len(solutions) <= 2:
            # Edge cases: assign infinite distance
            return [ParetoSolution(
                target_set=sol,
                pareto_rank=0,
                crowding_distance=float('inf')
            ) for sol in solutions]

        # Sort by damage for distance calculation
        sorted_solutions = sorted(solutions, key=lambda x: x.metrics['damage'])

        # Compute distances
        pareto_solutions = []

        for i, solution in enumerate(sorted_solutions):
            if i == 0 or i == len(sorted_solutions) - 1:
                # Boundary solutions get infinite distance
                distance = float('inf')
            else:
                # Interior solutions: normalized distance to neighbors
                damage_prev = sorted_solutions[i-1].metrics['damage']
                damage_next = sorted_solutions[i+1].metrics['damage']
                intel_prev = sorted_solutions[i-1].metrics['intelligence']
                intel_next = sorted_solutions[i+1].metrics['intelligence']

                # Normalize by ranges
                damage_range = max(s.metrics['damage'] for s in sorted_solutions) - \
                              min(s.metrics['damage'] for s in sorted_solutions)
                intel_range = max(s.metrics['intelligence'] for s in sorted_solutions) - \
                             min(s.metrics['intelligence'] for s in sorted_solutions)

                # Avoid division by zero
                damage_range = max(damage_range, 1e-10)
                intel_range = max(intel_range, 1e-10)

                # Manhattan distance
                distance = (abs(damage_next - damage_prev) / damage_range +
                           abs(intel_next - intel_prev) / intel_range)

            pareto_solutions.append(ParetoSolution(
                target_set=solution,
                pareto_rank=0,
                crowding_distance=distance
            ))

        return pareto_solutions
