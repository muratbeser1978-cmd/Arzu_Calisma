"""
Core data structures and abstract interfaces for optimization module.

This module defines the foundational entities for multi-objective optimization:
- OptimizationProblem: Specification of an optimization task
- TargetSet: A proposed solution with selected actors
- InterventionStrategy: Algorithm configuration
- ConversionProbability: Informant conversion likelihood
- Optimizer: Abstract base class for all optimization strategies
- InformationModel: Abstract interface for intelligence value computation

Mathematical Foundation: gelis.tex
Governance: .specify/memory/constitution.md v1.0.0
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Dict, List, Optional, FrozenSet, Set, Any

# Import NetworkState from existing simulation module
from ..core.state import NetworkState, ActorState


class StrategyType(Enum):
    """Types of optimization strategies."""
    STRUCTURAL_DAMAGE = auto()
    INTELLIGENCE = auto()
    HYBRID = auto()


@dataclass(frozen=True)
class OptimizationProblem:
    """
    Specification of an optimization task.

    Attributes:
        network_state: Current network state (from simulation)
        budget: Number of actors K to select
        objectives: List of objectives to optimize (e.g., ['damage'], ['intelligence'])
        time_horizon: Time window T for intelligence (required if intelligence objective)
        parameters: Simulation parameters (μ_LH, μ_min, etc.)

    Validation:
        - budget > 0
        - budget <= number of active actors
        - time_horizon > 0 if intelligence objective present
    """
    network_state: NetworkState
    budget: int
    objectives: List[str]
    time_horizon: Optional[float] = None
    parameters: Optional[Dict[str, float]] = None

    def __post_init__(self):
        """Validate optimization problem specification."""
        if self.budget <= 0:
            raise ValueError("Budget must be positive")

        # Count active actors
        active_count = sum(1 for status in self.network_state.A.values() if status == ActorState.ACTIVE)
        if self.budget > active_count:
            raise ValueError(f"Budget {self.budget} exceeds {active_count} active actors")

        # Validate time_horizon for intelligence objectives
        if 'intelligence' in self.objectives:
            if self.time_horizon is None or self.time_horizon <= 0:
                raise ValueError("Intelligence objective requires positive time_horizon")


@dataclass(frozen=True)
class TargetSet:
    """
    A proposed solution - collection of K actors selected for intervention.

    Attributes:
        actors: IDs of selected actors (immutable set)
        metrics: Computed objective values (damage, intelligence, etc.)
        rationale: Explanation for why each actor was selected
        quality_guarantee: Theoretical approximation guarantee if applicable
        computation_time: Time taken to compute this solution (seconds)

    Validation:
        - len(actors) matches budget
        - All actors exist in network and are Active
        - All actors have rationale entries
    """
    actors: FrozenSet[int]
    metrics: Dict[str, float]
    rationale: Dict[int, str]
    quality_guarantee: Optional[str] = None
    computation_time: float = 0.0

    def dominates(self, other: 'TargetSet') -> bool:
        """
        Check if this solution strictly dominates another.

        Domination: This solution is better on ALL objectives.
        Used for Pareto frontier computation.

        Args:
            other: Another TargetSet to compare against

        Returns:
            True if this solution strictly dominates other
        """
        if 'damage' not in self.metrics or 'damage' not in other.metrics:
            return False
        if 'intelligence' not in self.metrics or 'intelligence' not in other.metrics:
            return False

        return (
            self.metrics['damage'] > other.metrics['damage'] and
            self.metrics['intelligence'] > other.metrics['intelligence']
        )

    def is_pareto_optimal_in(self, pareto_set: List['TargetSet']) -> bool:
        """
        Check if not dominated by any solution in set.

        Args:
            pareto_set: List of candidate solutions

        Returns:
            True if no solution in pareto_set dominates this one
        """
        return not any(other.dominates(self) for other in pareto_set)


@dataclass(frozen=True)
class InterventionStrategy:
    """
    A specific algorithm/approach for selecting targets.

    Attributes:
        name: Human-readable name (e.g., "Greedy Degree")
        type: Strategy type (STRUCTURAL_DAMAGE, INTELLIGENCE, HYBRID)
        algorithm: Specific algorithm identifier (e.g., "greedy_degree")
        config: Algorithm-specific configuration
        expected_time_complexity: Big-O notation (e.g., "O(K·N·log N)")
        approximation_guarantee: Theoretical guarantee (e.g., "(1-1/e)")

    Validation:
        - name must be non-empty
        - config must match algorithm requirements
    """
    name: str
    type: StrategyType
    algorithm: str
    config: Dict[str, Any] = field(default_factory=dict)
    expected_time_complexity: str = "O(K·N)"
    approximation_guarantee: Optional[str] = None

    def __post_init__(self):
        """Validate strategy configuration."""
        if not self.name:
            raise ValueError("Strategy name cannot be empty")


@dataclass(frozen=True)
class ConversionProbability:
    """
    Likelihood that an arrested actor becomes an informant.

    Attributes:
        actor_id: ID of the actor
        probability: P(becomes informant within time horizon) ∈ [0, 1]
        expected_time: Expected time to conversion (days)
        fragility_rate: μ_HI(i) at time of arrest
        average_trust: Ȳᵢ at time of arrest (logit-scale)
        calculation_method: Method used (e.g., "hypo_exponential_exact")

    Validation:
        - probability ∈ [0, 1]
        - expected_time > 0

    Constitution Reference: §II - Informant Conversion Process
    """
    actor_id: int
    probability: float
    expected_time: float
    fragility_rate: float
    average_trust: float
    calculation_method: str = "hypo_exponential_exact"

    def __post_init__(self):
        """Validate conversion probability data."""
        if not (0 <= self.probability <= 1):
            raise ValueError(f"Probability must be in [0,1], got {self.probability}")
        if self.expected_time <= 0:
            raise ValueError(f"Expected time must be positive, got {self.expected_time}")


class Optimizer(ABC):
    """
    Abstract base class for all optimization strategies.

    Implementations must provide:
    - Objective function evaluation
    - Target selection algorithm
    - Performance metrics

    All optimizers must respect constitution requirements:
    - Use exact formulas from mathematical model
    - Validate inputs against specified domains
    - Handle edge cases (empty networks, isolated actors, etc.)
    """

    @abstractmethod
    def optimize(self, problem: OptimizationProblem) -> TargetSet:
        """
        Find optimal targets for given problem.

        Args:
            problem: Optimization problem specification

        Returns:
            TargetSet with selected actors and metrics

        Raises:
            ValueError: If problem specification is invalid
            RuntimeError: If optimization fails

        Guarantees:
            - Returns exactly K=problem.budget actors
            - All actors are from problem.network_state.V
            - All actors have state 'Active'
            - Metrics dictionary contains all requested objectives
        """
        pass

    @abstractmethod
    def evaluate_targets(self, actors: FrozenSet[int],
                        network_state: NetworkState) -> Dict[str, float]:
        """
        Evaluate objective values for proposed target set.

        Args:
            actors: Set of actor IDs to evaluate
            network_state: Current network state

        Returns:
            Dictionary mapping objective names to values

        Guarantees:
            - Result contains keys for all objectives in self.objectives
            - Values are non-negative floats
            - Computation preserves network_state (read-only)
        """
        pass

    @abstractmethod
    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Return performance statistics from last optimization.

        Returns:
            Dictionary with keys:
                - 'iterations': Number of algorithm iterations
                - 'evaluations': Number of objective evaluations
                - 'time': Computation time (seconds)
                - 'memory': Peak memory usage (MB) [optional]

        Guarantees:
            - All values are non-negative
            - Available immediately after optimize() call
        """
        pass


class InformationModel(ABC):
    """
    Defines what information each actor knows and its value.

    Used by IntelligenceOptimizer to compute information gain.
    Implementations should satisfy submodularity for optimization guarantee.
    """

    @abstractmethod
    def get_information_set(self, actor_id: int, network_state: NetworkState) -> Set[str]:
        """
        Return set of information atoms known by actor.

        Args:
            actor_id: ID of the actor
            network_state: Current network state

        Returns:
            Set of information atom identifiers (strings)

        Examples:
            - Neighborhood model: edges from/to actor
            - Hierarchical model: hierarchy-weighted information

        Constitution Reference: Uses network structure from §II
        """
        pass

    @abstractmethod
    def compute_value(self, info_set: Set[str]) -> float:
        """
        Compute value of information set.

        Args:
            info_set: Union of information atoms

        Returns:
            Non-negative value (higher = more valuable)

        Guarantees:
            - Monotonic: larger sets => at least as much value
            - Preferably submodular for optimization guarantee:
              f(S∪{i}) - f(S) >= f(T∪{i}) - f(T) for S ⊆ T
        """
        pass
