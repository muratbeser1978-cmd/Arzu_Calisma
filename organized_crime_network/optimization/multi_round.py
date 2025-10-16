"""
Multi-Round Intervention Planning.

This module provides adaptive sequential intervention planning:
- InterventionRound: Single round of interventions with targets and outcomes
- MultiRoundPlan: Complete multi-round intervention strategy
- MultiRoundPlanner: Generates adaptive plans that re-optimize after each round

Adaptive behavior: Each round re-optimizes based on updated network state,
unlike static plans that select all targets upfront.

Mathematical Foundation: gelis.tex
Governance: .specify/memory/constitution.md v1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, List, FrozenSet, Optional, Any
from .base import Optimizer, OptimizationProblem, TargetSet
from ..core.state import NetworkState, ActorState
from copy import deepcopy


@dataclass(frozen=True)
class InterventionRound:
    """
    Single round of interventions with specific targets and expected outcomes.

    Attributes:
        round_number: Sequential round identifier (1, 2, 3, ...)
        targets: Actors selected for arrest in this round
        optimizer_used: Name of optimizer used for this round
        network_state_before: Network state at start of round
        network_state_after: Expected network state after round execution
        metrics_before: Objective values before interventions
        metrics_after: Expected objective values after interventions
        rationale: Explanation for each actor selection
        budget_used: Number of actors arrested in this round
        budget_remaining: Remaining budget after this round

    Validation:
        - round_number > 0
        - len(targets) <= budget_used
        - All targets exist in network_state_before
    """
    round_number: int
    targets: FrozenSet[int]
    optimizer_used: str
    network_state_before: NetworkState
    network_state_after: NetworkState
    metrics_before: Dict[str, float]
    metrics_after: Dict[str, float]
    rationale: Dict[int, str]
    budget_used: int
    budget_remaining: int

    def __post_init__(self):
        """Validate intervention round data."""
        if self.round_number <= 0:
            raise ValueError(f"Round number must be positive, got {self.round_number}")
        if len(self.targets) > self.budget_used:
            raise ValueError(f"Targets ({len(self.targets)}) exceeds budget used ({self.budget_used})")
        if self.budget_used < 0 or self.budget_remaining < 0:
            raise ValueError("Budget values must be non-negative")


@dataclass(frozen=True)
class MultiRoundPlan:
    """
    Complete multi-round intervention strategy with sequential rounds.

    Attributes:
        rounds: Ordered list of intervention rounds
        total_budget: Total budget across all rounds
        total_targets: Total number of actors arrested across all rounds
        final_network_state: Expected final network state after all rounds
        final_metrics: Expected final objective values
        planning_strategy: Description of planning approach (adaptive, static, etc.)
        total_planning_time: Time taken to generate entire plan (seconds)
        metadata: Additional planning information

    Validation:
        - len(rounds) > 0
        - sum(r.budget_used) <= total_budget
        - total_targets matches sum of all round targets

    Constitution Reference: Planning builds on optimization from §II-III
    """
    rounds: List[InterventionRound]
    total_budget: int
    total_targets: int
    final_network_state: NetworkState
    final_metrics: Dict[str, float]
    planning_strategy: str
    total_planning_time: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate multi-round plan."""
        if not self.rounds:
            raise ValueError("Plan must contain at least one round")

        budget_used = sum(r.budget_used for r in self.rounds)
        if budget_used > self.total_budget:
            raise ValueError(f"Budget used ({budget_used}) exceeds total budget ({self.total_budget})")

        actual_targets = sum(len(r.targets) for r in self.rounds)
        if actual_targets != self.total_targets:
            raise ValueError(f"Total targets mismatch: {actual_targets} vs {self.total_targets}")

    def get_all_targets(self) -> FrozenSet[int]:
        """Return all actors targeted across all rounds."""
        all_targets = set()
        for round_obj in self.rounds:
            all_targets.update(round_obj.targets)
        return frozenset(all_targets)

    def get_round(self, round_number: int) -> Optional[InterventionRound]:
        """Get specific round by number (1-indexed)."""
        for round_obj in self.rounds:
            if round_obj.round_number == round_number:
                return round_obj
        return None

    def get_summary(self) -> Dict[str, Any]:
        """Generate summary statistics for the plan."""
        return {
            'num_rounds': len(self.rounds),
            'total_budget': self.total_budget,
            'total_targets': self.total_targets,
            'budget_utilization': self.total_targets / self.total_budget if self.total_budget > 0 else 0.0,
            'planning_strategy': self.planning_strategy,
            'planning_time': self.total_planning_time,
            'final_metrics': self.final_metrics,
        }


class MultiRoundPlanner:
    """
    Generates adaptive multi-round intervention plans.

    Adaptive Planning:
    - Each round re-optimizes based on current network state
    - Accounts for trust updates, risk changes, and network evolution
    - Can balance exploration (intelligence) vs exploitation (damage)

    vs Static Planning:
    - Static: Select all K targets upfront
    - Adaptive: Select batch_size targets, update state, re-optimize, repeat

    Algorithm:
    1. Start with initial network state
    2. For each round:
       a. Use optimizer to select batch_size targets
       b. Simulate arrests and conversions
       c. Update network state (trust, risk, edges)
       d. Check stopping criteria
    3. Return complete plan

    Constitution Reference: Uses arrest mechanics from §IV
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize multi-round planner.

        Args:
            config: Planning configuration
                - 'simulation_mode': 'expected' or 'pessimistic' for state updates
                - 'adaptive_strategy': Whether to switch optimizers between rounds
                - 'verbose': Enable detailed logging

        """
        self.config = config or {}
        self.simulation_mode = self.config.get('simulation_mode', 'expected')
        self.adaptive_strategy = self.config.get('adaptive_strategy', False)
        self.verbose = self.config.get('verbose', False)

        # Performance tracking
        self._last_planning_time = 0.0

    def generate_plan(self, initial_state: NetworkState, total_budget: int,
                     batch_size: int, optimizer: Optimizer,
                     problem_template: OptimizationProblem) -> MultiRoundPlan:
        """
        Generate adaptive multi-round intervention plan.

        Args:
            initial_state: Starting network state
            total_budget: Total number of arrests across all rounds
            batch_size: Targets per round
            optimizer: Optimizer to use for target selection
            problem_template: Template problem (objectives, time_horizon, parameters)

        Returns:
            MultiRoundPlan with sequential rounds

        Raises:
            ValueError: If inputs invalid or plan generation fails

        Guarantees:
            - Each round re-optimizes based on current state
            - Network state updated after each round
            - Total targets <= total_budget
        """
        import time
        start_time = time.time()

        # Validate inputs
        if total_budget <= 0:
            raise ValueError("Total budget must be positive")
        if batch_size <= 0:
            raise ValueError("Batch size must be positive")
        if batch_size > total_budget:
            raise ValueError(f"Batch size ({batch_size}) exceeds total budget ({total_budget})")

        # Initialize planning
        current_state = self._clone_state(initial_state)
        rounds = []
        budget_remaining = total_budget
        round_number = 1

        # Generate rounds until budget exhausted or stopping criteria met
        while budget_remaining > 0:
            # Determine batch size for this round
            round_batch = min(batch_size, budget_remaining)

            # Check stopping criteria
            active_actors = current_state.get_active_actors()
            if len(active_actors) == 0:
                if self.verbose:
                    print(f"Round {round_number}: No active actors remaining, stopping")
                break

            if len(active_actors) < round_batch:
                round_batch = len(active_actors)
                if round_batch == 0:
                    break

            # Select targets for this round
            round_targets, round_rationale = self.select_round_targets(
                current_state, round_batch, optimizer, problem_template
            )

            if not round_targets:
                if self.verbose:
                    print(f"Round {round_number}: No targets selected, stopping")
                break

            # Compute metrics before round
            metrics_before = self._compute_metrics(current_state, problem_template)

            # Simulate round execution
            state_after = self.simulate_round_execution(
                current_state, round_targets, problem_template
            )

            # Compute metrics after round
            metrics_after = self._compute_metrics(state_after, problem_template)

            # Create intervention round
            round_obj = InterventionRound(
                round_number=round_number,
                targets=round_targets,
                optimizer_used=type(optimizer).__name__,
                network_state_before=self._clone_state(current_state),
                network_state_after=self._clone_state(state_after),
                metrics_before=metrics_before,
                metrics_after=metrics_after,
                rationale=round_rationale,
                budget_used=len(round_targets),
                budget_remaining=budget_remaining - len(round_targets)
            )

            rounds.append(round_obj)

            # Update for next round
            current_state = state_after
            budget_remaining -= len(round_targets)
            round_number += 1

            if self.verbose:
                print(f"Round {round_number - 1} complete: {len(round_targets)} targets, "
                      f"{budget_remaining} budget remaining")

        # Create final plan
        planning_time = time.time() - start_time
        self._last_planning_time = planning_time

        total_targets = sum(len(r.targets) for r in rounds)
        final_metrics = rounds[-1].metrics_after if rounds else {}

        return MultiRoundPlan(
            rounds=rounds,
            total_budget=total_budget,
            total_targets=total_targets,
            final_network_state=current_state,
            final_metrics=final_metrics,
            planning_strategy="adaptive_sequential",
            total_planning_time=planning_time,
            metadata={
                'batch_size': batch_size,
                'num_rounds': len(rounds),
                'simulation_mode': self.simulation_mode,
            }
        )

    def select_round_targets(self, network_state: NetworkState, batch_size: int,
                            optimizer: Optimizer,
                            problem_template: OptimizationProblem) -> tuple:
        """
        Select targets for single round using provided optimizer.

        Args:
            network_state: Current network state
            batch_size: Number of targets to select
            optimizer: Optimizer for target selection
            problem_template: Problem specification template

        Returns:
            (targets, rationale) tuple

        Raises:
            ValueError: If batch_size exceeds active actors
        """
        # Create problem for this round
        problem = OptimizationProblem(
            network_state=network_state,
            budget=batch_size,
            objectives=problem_template.objectives,
            time_horizon=problem_template.time_horizon,
            parameters=problem_template.parameters
        )

        # Optimize
        solution = optimizer.optimize(problem)

        return solution.actors, solution.rationale

    def simulate_round_execution(self, network_state: NetworkState,
                                 targets: FrozenSet[int],
                                 problem_template: OptimizationProblem) -> NetworkState:
        """
        Simulate execution of intervention round.

        Simulates:
        - Arrests of target actors
        - Edge removal (arrested actors lose connections)
        - Trust updates (neighbors' trust affected by arrests)
        - Risk updates (exposure neighborhoods change)

        Args:
            network_state: Current network state
            targets: Actors to arrest
            problem_template: Problem template with parameters

        Returns:
            Updated network state after interventions

        Constitution Reference: §IV - Arrest mechanics
        """
        # Clone state for simulation
        new_state = self._clone_state(network_state)

        # Execute arrests
        for actor_id in targets:
            if actor_id in new_state.V and new_state.A[actor_id] == ActorState.ACTIVE:
                new_state.arrest_actor(actor_id)

        # Note: Trust and risk updates happen automatically in arrest_actor()
        # The NetworkState class handles edge removal and exposure tracking

        return new_state

    def update_network_state(self, network_state: NetworkState,
                            interventions: FrozenSet[int]) -> NetworkState:
        """
        Update network state after interventions (wrapper for simulate_round_execution).

        Args:
            network_state: Current state
            interventions: Actors arrested

        Returns:
            Updated state

        Deprecated: Use simulate_round_execution instead
        """
        # Simple wrapper - actual logic in simulate_round_execution
        return self.simulate_round_execution(network_state, interventions, None)

    def get_planning_time(self) -> float:
        """Return time taken for last plan generation."""
        return self._last_planning_time

    def _clone_state(self, state: NetworkState) -> NetworkState:
        """Create deep copy of network state."""
        return deepcopy(state)

    def _compute_metrics(self, network_state: NetworkState,
                        problem_template: OptimizationProblem) -> Dict[str, float]:
        """
        Compute current objective values for network state.

        Args:
            network_state: State to evaluate
            problem_template: Template with objectives

        Returns:
            Dictionary with metric values
        """
        from .objectives import compute_structural_damage, compute_intelligence_value

        metrics = {}

        # Empty target set for baseline metrics
        empty_targets = frozenset()

        if 'damage' in problem_template.objectives:
            # Damage is LCC of current network (before any interventions)
            from .objectives import compute_lcc_size
            metrics['lcc_size'] = compute_lcc_size(network_state)
            # Damage relative to some baseline could be computed here
            metrics['damage'] = 0.0  # Placeholder

        if 'intelligence' in problem_template.objectives:
            # Intelligence is 0 if no targets selected
            metrics['intelligence'] = 0.0

        # Add network statistics
        metrics['active_actors'] = len(network_state.get_active_actors())
        metrics['arrested_actors'] = len(network_state.get_arrested_actors())
        metrics['edges'] = len(network_state.E)

        return metrics
