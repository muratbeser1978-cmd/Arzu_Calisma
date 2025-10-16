"""
Performance metrics and strategy comparison utilities.

This module provides data structures and functions for:
- PerformanceMetrics: Measures of optimization quality and efficiency
- StrategyComparison: Results from comparing multiple strategies (added in Phase 7)
- compare_strategies: Function to compare optimization approaches (added in Phase 7)

Mathematical Foundation: gelis.tex
Governance: .specify/memory/constitution.md v1.0.0
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, List, Tuple, Any
import time


@dataclass(frozen=True)
class PerformanceMetrics:
    """
    Measures of optimization quality and efficiency.

    Attributes:
        solution_quality: Objective values achieved
            - 'damage': Expected reduction in largest connected component
            - 'intelligence': Expected information gain
            - 'pareto_coverage': Hypervolume or coverage metric (for multi-objective)
        computation_time: Total time in seconds
        iterations: Number of algorithm iterations
        evaluations: Number of objective function evaluations
        approximation_ratio: Actual / optimal (if known, for validation)
        theoretical_guarantee: Proven bound (e.g., "(1-1/e)")

    Validation:
        - All times >= 0
        - All counts >= 0
        - approximation_ratio ∈ (0, 1] if provided

    Constitution Reference: Performance targets from specification
    """
    solution_quality: Dict[str, float]
    computation_time: float
    iterations: int = 0
    evaluations: int = 0
    approximation_ratio: Optional[float] = None
    theoretical_guarantee: Optional[str] = None

    def __post_init__(self):
        """Validate performance metrics."""
        if self.computation_time < 0:
            raise ValueError("Computation time cannot be negative")
        if self.iterations < 0 or self.evaluations < 0:
            raise ValueError("Counts cannot be negative")
        if self.approximation_ratio is not None:
            if not (0 < self.approximation_ratio <= 1):
                raise ValueError("Approximation ratio must be in (0, 1]")


@dataclass(frozen=True)
class StrategyComparison:
    """
    Results from comparing a single strategy against a problem.

    Attributes:
        strategy_name: Human-readable strategy identifier
        strategy_config: Configuration used for this strategy
        metrics: Objective values achieved (damage, intelligence, etc.)
        computation_time: Time taken to optimize (seconds)
        iterations: Number of algorithm iterations
        evaluations: Number of objective function evaluations
        quality_guarantee: Theoretical approximation guarantee if applicable
        targets: Selected actors (for detailed analysis)
        rationale_sample: Sample explanations (first 3 actors)

    Validation:
        - strategy_name must be non-empty
        - computation_time >= 0
        - iterations, evaluations >= 0

    Constitution Reference: Enables comparison as required by User Story 5
    """
    strategy_name: str
    strategy_config: Dict[str, Any]
    metrics: Dict[str, float]
    computation_time: float
    iterations: int
    evaluations: int
    quality_guarantee: Optional[str] = None
    targets: Optional[frozenset] = None
    rationale_sample: Dict[int, str] = field(default_factory=dict)

    def __post_init__(self):
        """Validate strategy comparison data."""
        if not self.strategy_name:
            raise ValueError("Strategy name cannot be empty")
        if self.computation_time < 0:
            raise ValueError("Computation time cannot be negative")
        if self.iterations < 0 or self.evaluations < 0:
            raise ValueError("Counts cannot be negative")


def compare_strategies(problem, strategies: List[Tuple[str, Any, Dict]]) -> List[StrategyComparison]:
    """
    Compare multiple optimization strategies on the same problem.

    Runs each strategy on the provided problem and collects performance metrics,
    enabling time/quality trade-off analysis.

    Args:
        problem: OptimizationProblem to solve
        strategies: List of (name, optimizer, config) tuples
            - name: Human-readable identifier (e.g., "Greedy Degree")
            - optimizer: Optimizer instance
            - config: Strategy configuration dict

    Returns:
        List of StrategyComparison results, one per strategy

    Raises:
        ValueError: If strategies list is empty or problem is invalid

    Example:
        >>> from organized_crime_network.optimization import *
        >>> damage1 = StructuralDamageOptimizer(strategy='greedy_degree')
        >>> damage2 = StructuralDamageOptimizer(strategy='greedy_betweenness')
        >>> strategies = [
        ...     ("Greedy Degree", damage1, {'strategy': 'greedy_degree'}),
        ...     ("Greedy Betweenness", damage2, {'strategy': 'greedy_betweenness'})
        ... ]
        >>> results = compare_strategies(problem, strategies)
        >>> for r in results:
        ...     print(f"{r.strategy_name}: {r.metrics['damage']:.2f} in {r.computation_time:.3f}s")

    Constitution Reference: No approximations - uses exact optimization
    """
    if not strategies:
        raise ValueError("Strategies list cannot be empty")

    results = []

    for strategy_name, optimizer, config in strategies:
        # Time the optimization
        start_time = time.time()
        solution = optimizer.optimize(problem)
        elapsed = time.time() - start_time

        # Get performance metrics from optimizer
        perf_metrics = optimizer.get_performance_metrics()

        # Extract rationale sample (first 3 actors)
        rationale_sample = {}
        if solution.rationale:
            for i, (actor_id, reason) in enumerate(solution.rationale.items()):
                if i >= 3:
                    break
                rationale_sample[actor_id] = reason

        # Create comparison result
        result = StrategyComparison(
            strategy_name=strategy_name,
            strategy_config=config,
            metrics=solution.metrics,
            computation_time=elapsed,
            iterations=perf_metrics.get('iterations', 0),
            evaluations=perf_metrics.get('evaluations', 0),
            quality_guarantee=solution.quality_guarantee,
            targets=solution.actors,
            rationale_sample=rationale_sample
        )

        results.append(result)

    return results


def format_comparison_table(comparisons: List[StrategyComparison],
                            objectives: List[str] = None) -> str:
    """
    Format strategy comparison results as a readable table.

    Args:
        comparisons: List of StrategyComparison results
        objectives: Objectives to display (default: all in first result)

    Returns:
        Formatted table string

    Example output:
        Strategy              Damage    Time    Iters  Guarantee
        ----------------------------------------------------------
        Greedy Degree          24.00   0.002s      10  None
        Greedy Betweenness     28.00   0.045s      10  None
        Simulated Annealing    32.00   1.234s    5000  None
    """
    if not comparisons:
        return "No strategies to compare"

    # Determine objectives to display
    if objectives is None:
        objectives = list(comparisons[0].metrics.keys())

    # Build header
    header_parts = ["Strategy"]
    for obj in objectives:
        header_parts.append(obj.capitalize()[:12])
    header_parts.extend(["Time", "Iters", "Guarantee"])

    # Column widths
    widths = [20] + [10] * len(objectives) + [10, 8, 25]

    # Format header
    header = "  ".join(f"{h:<{w}}" for h, w in zip(header_parts, widths))
    separator = "-" * len(header)

    lines = [header, separator]

    # Format each strategy
    for comp in comparisons:
        row_parts = [comp.strategy_name[:20]]

        for obj in objectives:
            value = comp.metrics.get(obj, 0.0)
            row_parts.append(f"{value:.2f}")

        row_parts.append(f"{comp.computation_time:.3f}s")
        row_parts.append(f"{comp.iterations}")

        guarantee = comp.quality_guarantee if comp.quality_guarantee else "None"
        row_parts.append(guarantee[:25])

        row = "  ".join(f"{p:<{w}}" for p, w in zip(row_parts, widths))
        lines.append(row)

    return "\n".join(lines)


def generate_strategy_recommendation(comparisons: List[StrategyComparison],
                                     network_size: int,
                                     time_constraint: Optional[float] = None) -> str:
    """
    Generate recommendation for which strategy to use based on results.

    Recommendation logic:
    - Fast strategies (greedy_degree) for large networks (N > 500)
    - Quality strategies (simulated_annealing) for small networks (N < 100)
    - Balanced strategies (greedy_betweenness) for medium networks
    - Guaranteed strategies (lazy_greedy for intelligence) when guarantee needed

    Args:
        comparisons: Strategy comparison results
        network_size: Number of actors in network
        time_constraint: Maximum acceptable time (seconds), optional

    Returns:
        Recommendation string with reasoning

    Example:
        "Recommended: Greedy Betweenness (best balance of quality and speed for N=250)"
    """
    if not comparisons:
        return "No strategies to recommend from"

    # Apply time constraint filter if provided
    viable = comparisons
    if time_constraint is not None:
        viable = [c for c in comparisons if c.computation_time <= time_constraint]
        if not viable:
            return f"No strategies meet time constraint of {time_constraint:.3f}s"

    # Find best quality (highest damage or intelligence)
    # Assume first metric is primary objective
    first_metric = list(viable[0].metrics.keys())[0]
    best_quality = max(viable, key=lambda c: c.metrics.get(first_metric, 0))

    # Find fastest
    fastest = min(viable, key=lambda c: c.computation_time)

    # Generate recommendation based on network size
    if network_size > 500:
        # Large network: prioritize speed
        recommended = fastest
        reason = f"fast execution for large network (N={network_size})"
    elif network_size < 100:
        # Small network: prioritize quality
        recommended = best_quality
        reason = f"best quality for small network (N={network_size})"
    else:
        # Medium network: balance quality and speed
        # Use quality/time ratio
        scored = [(c, c.metrics.get(first_metric, 0) / max(c.computation_time, 0.001))
                  for c in viable]
        recommended, _ = max(scored, key=lambda x: x[1])
        reason = f"best quality/time balance for medium network (N={network_size})"

    rec_str = f"Recommended: {recommended.strategy_name} ({reason})"

    # Add quality guarantee note if applicable
    if recommended.quality_guarantee:
        rec_str += f"\n  Quality guarantee: {recommended.quality_guarantee}"

    # Add performance summary
    rec_str += f"\n  Performance: {recommended.metrics.get(first_metric, 0):.2f} {first_metric} in {recommended.computation_time:.3f}s"

    return rec_str


def get_comparison_summary(comparisons: List[StrategyComparison]) -> Dict[str, Any]:
    """
    Generate summary statistics for strategy comparison.

    Args:
        comparisons: List of comparison results

    Returns:
        Dictionary with summary statistics:
            - num_strategies: Number of strategies compared
            - fastest: Name of fastest strategy
            - best_quality: Name of best quality strategy (by first metric)
            - time_range: (min_time, max_time) tuple
            - quality_range: (min_quality, max_quality) tuple

    Example:
        >>> summary = get_comparison_summary(results)
        >>> print(f"Compared {summary['num_strategies']} strategies")
        >>> print(f"Fastest: {summary['fastest']}")
        >>> print(f"Best quality: {summary['best_quality']}")
    """
    if not comparisons:
        return {'num_strategies': 0}

    # Get first metric as primary quality measure
    first_metric = list(comparisons[0].metrics.keys())[0]

    # Find extremes
    fastest = min(comparisons, key=lambda c: c.computation_time)
    slowest = max(comparisons, key=lambda c: c.computation_time)
    best_quality = max(comparisons, key=lambda c: c.metrics.get(first_metric, 0))
    worst_quality = min(comparisons, key=lambda c: c.metrics.get(first_metric, 0))

    return {
        'num_strategies': len(comparisons),
        'fastest': fastest.strategy_name,
        'slowest': slowest.strategy_name,
        'best_quality': best_quality.strategy_name,
        'worst_quality': worst_quality.strategy_name,
        'time_range': (fastest.computation_time, slowest.computation_time),
        'quality_range': (worst_quality.metrics.get(first_metric, 0),
                         best_quality.metrics.get(first_metric, 0)),
        'primary_metric': first_metric,
    }
