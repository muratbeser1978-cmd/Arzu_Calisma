"""
Structural Damage Optimization Strategies.

This module implements algorithms for maximizing network fragmentation:
- Greedy Degree: Fast, targets high-degree nodes
- Greedy Betweenness: Better quality, uses centrality
- Simulated Annealing: Best quality, slower

No approximation guarantees (NP-hard problem, objective not submodular).

Mathematical Foundation: gelis.tex §II
Governance: .specify/memory/constitution.md v1.0.0
"""

import time
import random
import math
from typing import Dict, FrozenSet, List, Tuple
from ..base import Optimizer, OptimizationProblem, TargetSet
from ..objectives import (
    compute_structural_damage,
    compute_degree_centrality,
    compute_betweenness_centrality,
)
from ...core.state import NetworkState, ActorState


class StructuralDamageOptimizer(Optimizer):
    """
    Optimizer for maximizing network fragmentation (LCC reduction).

    Provides multiple strategies with different time/quality trade-offs:
    - greedy_degree: O(K·N log N) - Fast, targets high-degree nodes
    - greedy_betweenness: O(K·N·M) - Better quality, uses centrality
    - simulated_annealing: O(iterations · M) - Best quality, slower

    No approximation guarantees (NP-hard problem, objective not submodular).

    Constitution Reference: §II - Network Structure
    """

    def __init__(self, strategy: str = 'greedy_degree', config: Dict = None):
        """
        Initialize structural damage optimizer.

        Args:
            strategy: Algorithm choice
                - 'greedy_degree': Fast baseline
                - 'greedy_betweenness': Better quality
                - 'simulated_annealing': Best quality
            config: Strategy-specific configuration
                For SA: {'initial_temp': float, 'cooling_rate': float, 'iterations': int}

        Raises:
            ValueError: If strategy is unknown or config is invalid
        """
        valid_strategies = ['greedy_degree', 'greedy_betweenness', 'simulated_annealing']
        if strategy not in valid_strategies:
            raise ValueError(f"Unknown strategy: {strategy}. Must be one of {valid_strategies}")

        self.strategy = strategy
        self.config = config or {}

        # Performance tracking
        self._last_metrics = {
            'iterations': 0,
            'evaluations': 0,
            'time': 0.0,
        }

    def optimize(self, problem: OptimizationProblem) -> TargetSet:
        """
        Find optimal targets for structural damage.

        Args:
            problem: Optimization problem specification

        Returns:
            TargetSet with selected actors and damage metrics

        Raises:
            ValueError: If problem specification is invalid
        """
        start_time = time.time()

        # Validate problem
        if 'damage' not in problem.objectives:
            raise ValueError("StructuralDamageOptimizer requires 'damage' objective")

        # Validate budget
        active_actors = [actor_id for actor_id, status in problem.network_state.A.items()
                        if status == ActorState.ACTIVE]
        if problem.budget > len(active_actors):
            raise ValueError(f"Budget {problem.budget} exceeds {len(active_actors)} active actors")
        if problem.budget <= 0:
            raise ValueError("Budget must be positive")

        # Execute strategy
        if self.strategy == 'greedy_degree':
            selected, rationale = self._greedy_degree(problem.network_state, problem.budget)
        elif self.strategy == 'greedy_betweenness':
            selected, rationale = self._greedy_betweenness(problem.network_state, problem.budget)
        elif self.strategy == 'simulated_annealing':
            selected, rationale = self._simulated_annealing(problem.network_state, problem.budget)
        else:
            raise ValueError(f"Strategy {self.strategy} not implemented")

        # Compute metrics
        damage = compute_structural_damage(selected, problem.network_state)
        metrics = {'damage': damage}

        # Record performance
        computation_time = time.time() - start_time
        self._last_metrics['time'] = computation_time

        return TargetSet(
            actors=selected,
            metrics=metrics,
            rationale=rationale,
            quality_guarantee=None,  # No theoretical guarantee for NP-hard problem
            computation_time=computation_time
        )

    def evaluate_targets(self, actors: FrozenSet[int],
                        network_state: NetworkState) -> Dict[str, float]:
        """
        Evaluate damage for proposed target set.

        Args:
            actors: Set of actors to evaluate
            network_state: Current network state

        Returns:
            Dictionary with 'damage' metric
        """
        damage = compute_structural_damage(actors, network_state)
        return {'damage': damage}

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Return performance statistics from last optimization.

        Returns:
            Dictionary with iterations, evaluations, time
        """
        return self._last_metrics.copy()

    def _greedy_degree(self, state: NetworkState, K: int) -> Tuple[FrozenSet[int], Dict[int, str]]:
        """
        Greedy strategy: iteratively select highest-degree actors.

        Time complexity: O(K·N log N)

        Args:
            state: Network state
            K: Budget (number of actors to select)

        Returns:
            (selected actors, rationale for each)
        """
        selected = set()
        rationale = {}
        iterations = 0

        for iteration in range(K):
            iterations += 1

            # Compute degrees
            degrees = compute_degree_centrality(state)

            # Remove already selected actors
            for actor_id in selected:
                if actor_id in degrees:
                    del degrees[actor_id]

            if not degrees:
                break  # No more actors available

            # Select actor with highest degree
            best_actor = max(degrees.items(), key=lambda x: x[1])[0]
            best_degree = degrees[best_actor]

            selected.add(best_actor)
            rationale[best_actor] = f"High degree centrality (degree={best_degree})"

        self._last_metrics['iterations'] = iterations
        self._last_metrics['evaluations'] = iterations  # One degree computation per iteration

        return frozenset(selected), rationale

    def _greedy_betweenness(self, state: NetworkState, K: int) -> Tuple[FrozenSet[int], Dict[int, str]]:
        """
        Greedy strategy: iteratively select highest-betweenness actors.

        Time complexity: O(K·N·M) where M = edges

        Args:
            state: Network state
            K: Budget (number of actors to select)

        Returns:
            (selected actors, rationale for each)
        """
        selected = set()
        rationale = {}
        iterations = 0

        for iteration in range(K):
            iterations += 1

            # Compute betweenness centrality
            betweenness = compute_betweenness_centrality(state)

            # Remove already selected actors
            for actor_id in selected:
                if actor_id in betweenness:
                    del betweenness[actor_id]

            if not betweenness:
                break  # No more actors available

            # Select actor with highest betweenness
            best_actor = max(betweenness.items(), key=lambda x: x[1])[0]
            best_betweenness = betweenness[best_actor]

            selected.add(best_actor)
            rationale[best_actor] = f"High betweenness centrality (betweenness={best_betweenness:.4f})"

        self._last_metrics['iterations'] = iterations
        self._last_metrics['evaluations'] = iterations

        return frozenset(selected), rationale

    def _simulated_annealing(self, state: NetworkState, K: int) -> Tuple[FrozenSet[int], Dict[int, str]]:
        """
        Simulated Annealing: meta-heuristic for global optimization.

        Time complexity: O(iterations · M) where M = edges

        Args:
            state: Network state
            K: Budget (number of actors to select)

        Returns:
            (selected actors, rationale for each)
        """
        # Configuration
        initial_temp = self.config.get('initial_temp', 1.0)
        cooling_rate = self.config.get('cooling_rate', 0.95)
        iterations_per_temp = self.config.get('iterations_per_temp', 100 * K)
        min_temp = self.config.get('min_temp', 0.01)

        # Get active actors
        active_actors = [actor_id for actor_id, status in state.A.items()
                        if status == ActorState.ACTIVE]

        if K > len(active_actors):
            K = len(active_actors)

        # Initialize with random solution
        current_solution = frozenset(random.sample(active_actors, K))
        current_damage = compute_structural_damage(current_solution, state)

        best_solution = current_solution
        best_damage = current_damage

        temp = initial_temp
        total_iterations = 0
        evaluations = 1  # Initial evaluation

        while temp > min_temp:
            for _ in range(iterations_per_temp):
                total_iterations += 1

                # Generate neighbor: swap one actor
                current_list = list(current_solution)
                available = [a for a in active_actors if a not in current_solution]

                if not available:
                    break  # Cannot generate neighbors

                # Random swap
                remove_idx = random.randint(0, len(current_list) - 1)
                add_actor = random.choice(available)

                neighbor = set(current_list)
                neighbor.remove(current_list[remove_idx])
                neighbor.add(add_actor)
                neighbor = frozenset(neighbor)

                # Evaluate neighbor
                neighbor_damage = compute_structural_damage(neighbor, state)
                evaluations += 1

                # Accept or reject
                delta = neighbor_damage - current_damage

                if delta > 0 or random.random() < math.exp(delta / temp):
                    # Accept neighbor
                    current_solution = neighbor
                    current_damage = neighbor_damage

                    # Update best
                    if current_damage > best_damage:
                        best_solution = current_solution
                        best_damage = current_damage

            # Cool down
            temp *= cooling_rate

        # Generate rationale for best solution
        rationale = {}
        degrees = compute_degree_centrality(state)
        for actor_id in best_solution:
            degree = degrees.get(actor_id, 0)
            rationale[actor_id] = f"Optimized via SA (degree={degree}, damage contribution estimated)"

        self._last_metrics['iterations'] = total_iterations
        self._last_metrics['evaluations'] = evaluations

        return best_solution, rationale
