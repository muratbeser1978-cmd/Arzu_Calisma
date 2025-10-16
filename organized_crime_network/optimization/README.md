# Multi-Objective Intervention Optimization Module

**Version**: 1.0.0
**Status**: Production Ready
**Governance**: `.specify/memory/constitution.md`
**Mathematical Foundation**: `gelis.tex`

## Overview

This module provides algorithms for identifying optimal intervention targets in organized crime networks. It supports multiple objectives, including:

- **Structural Damage**: Maximize network fragmentation by selecting high-impact actors
- **Intelligence Gain**: Maximize expected information from potential informants
- **Multi-Objective**: Explore trade-offs between competing objectives
- **Multi-Round Planning**: Generate adaptive sequential intervention strategies
- **Strategy Comparison**: Compare different optimization approaches

## Quick Start

```python
from organized_crime_network.optimization import (
    OptimizationProblem,
    StructuralDamageOptimizer,
    IntelligenceOptimizer,
    NeighborhoodInformationModel,
    ParetoFrontierComputer,
    MultiRoundPlanner,
    compare_strategies,
)
from organized_crime_network.core.state import NetworkState

# Create network state
state = NetworkState(t_current=0.0)
for i in range(50):
    state.add_actor(i, hierarchy_level=1)
# ... add edges ...

# Example 1: Structural Damage Optimization
problem = OptimizationProblem(
    network_state=state,
    budget=10,
    objectives=['damage']
)

optimizer = StructuralDamageOptimizer(strategy='greedy_degree')
solution = optimizer.optimize(problem)

print(f"Selected {len(solution.actors)} actors")
print(f"Expected damage: {solution.metrics['damage']:.2f}")

# Example 2: Intelligence Optimization
info_model = NeighborhoodInformationModel()
intel_optimizer = IntelligenceOptimizer(information_model=info_model)

problem = OptimizationProblem(
    network_state=state,
    budget=10,
    objectives=['intelligence'],
    time_horizon=20.0,
    parameters={'mu_LH': 0.1, 'mu_min': 0.05, 'mu_rng': 0.15, 'theta': 1.0}
)

solution = intel_optimizer.optimize(problem)
print(f"Expected intelligence: {solution.metrics['intelligence']:.2f}")
print(f"Quality guarantee: {solution.quality_guarantee}")

# Example 3: Pareto Frontier Analysis
damage_opt = StructuralDamageOptimizer(strategy='greedy_degree')
pareto = ParetoFrontierComputer(damage_opt, intel_optimizer)

solutions = pareto.compute_pareto_frontier(problem, num_points=20)
print(f"Generated {len(solutions)} Pareto-optimal solutions")

# Example 4: Multi-Round Planning
planner = MultiRoundPlanner(config={'verbose': True})
plan = planner.generate_plan(
    initial_state=state,
    total_budget=20,
    batch_size=5,
    optimizer=damage_opt,
    problem_template=problem
)

print(f"Generated {len(plan.rounds)} rounds")
print(f"Total targets: {plan.total_targets}")

# Example 5: Strategy Comparison
strategies = [
    ("Greedy Degree", StructuralDamageOptimizer(strategy='greedy_degree'),
     {'strategy': 'greedy_degree'}),
    ("Greedy Betweenness", StructuralDamageOptimizer(strategy='greedy_betweenness'),
     {'strategy': 'greedy_betweenness'}),
]

results = compare_strategies(problem, strategies)
for r in results:
    print(f"{r.strategy_name}: {r.metrics['damage']:.2f} in {r.computation_time:.3f}s")
```

## Module Structure

```
organized_crime_network/optimization/
├── __init__.py              # Module exports
├── base.py                  # Core data structures and abstract interfaces
├── metrics.py               # Performance metrics and strategy comparison
├── objectives.py            # Objective function implementations
├── pareto.py                # Pareto frontier computation
├── multi_round.py           # Multi-round intervention planning
├── strategies/              # Optimization strategy implementations
│   ├── __init__.py
│   ├── structural_damage.py # Greedy degree, betweenness, simulated annealing
│   ├── intelligence.py      # Lazy greedy with (1-1/e) guarantee
│   └── hybrid.py            # Weighted combination of objectives
└── README.md                # This file
```

## Core Data Structures

### OptimizationProblem

Specifies what to optimize:

```python
problem = OptimizationProblem(
    network_state=state,        # Current network
    budget=10,                  # Number of actors to select
    objectives=['damage'],      # Objectives to optimize
    time_horizon=20.0,          # Optional: for intelligence
    parameters={}               # Optional: simulation parameters
)
```

### TargetSet

Represents a solution:

```python
solution = optimizer.optimize(problem)
solution.actors            # Selected actor IDs (frozenset)
solution.metrics           # Objective values (dict)
solution.rationale         # Explanation for each selection
solution.quality_guarantee # Theoretical guarantee (if applicable)
solution.computation_time  # Time taken (seconds)
```

## Optimization Strategies

### Structural Damage

Maximizes network fragmentation by removing key actors:

- **greedy_degree**: Fast (O(K·N log N)), targets high-degree nodes
- **greedy_betweenness**: Better quality (O(K·N·M)), uses centrality
- **simulated_annealing**: Best quality, slower, meta-heuristic

```python
optimizer = StructuralDamageOptimizer(strategy='greedy_degree')
solution = optimizer.optimize(problem)
```

### Intelligence

Maximizes expected information gain from informants:

- Uses **lazy greedy** algorithm with priority queue
- **Theoretical guarantee**: (1-1/e) ≈ 63% of optimal
- Time complexity: O(K·N log N) in practice

```python
info_model = NeighborhoodInformationModel()
optimizer = IntelligenceOptimizer(information_model=info_model)
solution = optimizer.optimize(problem)
```

### Hybrid

Combines multiple objectives with weighted sum:

- Score = λ·damage + (1-λ)·intelligence
- λ=1: pure damage, λ=0: pure intelligence, λ=0.5: balanced

```python
optimizer = HybridOptimizer(lambda_weight=0.5, information_model=info_model)
solution = optimizer.optimize(problem)
```

## Advanced Features

### Pareto Frontier Analysis

Explore trade-offs between competing objectives:

```python
pareto = ParetoFrontierComputer(damage_optimizer, intelligence_optimizer)
solutions = pareto.compute_pareto_frontier(problem, num_points=30)

# Verify Pareto optimality
is_valid = pareto.verify_pareto_optimality(solutions)

# Solutions are sorted by damage (descending)
best_damage = solutions[0]
best_intelligence = solutions[-1]
balanced = solutions[len(solutions) // 2]
```

### Multi-Round Planning

Generate adaptive intervention plans:

```python
planner = MultiRoundPlanner(config={'verbose': True})
plan = planner.generate_plan(
    initial_state=state,
    total_budget=50,
    batch_size=10,
    optimizer=optimizer,
    problem_template=problem
)

for round_obj in plan.rounds:
    print(f"Round {round_obj.round_number}: {len(round_obj.targets)} targets")
    print(f"  Metrics before: {round_obj.metrics_before}")
    print(f"  Metrics after: {round_obj.metrics_after}")
```

### Strategy Comparison

Compare time/quality trade-offs:

```python
strategies = [
    ("Strategy 1", optimizer1, config1),
    ("Strategy 2", optimizer2, config2),
]

results = compare_strategies(problem, strategies)

# Display as table
from organized_crime_network.optimization import format_comparison_table
print(format_comparison_table(results))

# Get recommendation
from organized_crime_network.optimization import generate_strategy_recommendation
recommendation = generate_strategy_recommendation(
    results,
    network_size=len(state.V),
    time_constraint=1.0  # Optional: max time in seconds
)
print(recommendation)
```

## Performance

Validated performance targets:

| Network Size | Budget | Strategy | Target | Actual |
|--------------|--------|----------|--------|--------|
| N=100 | K=10 | Greedy Degree | <1s | ~0.01s |
| N=500 | K=20 | Greedy Degree | <30s | ~0.02s |
| N=100 | K=10 | Intelligence | <5s | ~0.03s |

## Constitution Compliance

All implementations follow exact formulas from the mathematical model:

- **Conversion Probability**: Uses two-stage CTMC with μ_LH and μ_HI(i)
- **Fragility Rate**: μ_HI = μ_min + μ_rng · expit(-θ · Ȳᵢ)
- **Expected Time**: E[T] = 1/μ_LH + 1/μ_HI
- **Numerical Stability**: All computations use stable expit/logit from utils.py

## Input Validation

All optimizers validate inputs:

- Budget K > 0 and K ≤ number of active actors
- Time horizon T > 0 (for intelligence objectives)
- Valid parameters (μ_LH, μ_min, μ_rng, θ)
- Network state invariants preserved

## Error Handling

Optimizers raise `ValueError` for invalid inputs:

```python
try:
    solution = optimizer.optimize(problem)
except ValueError as e:
    print(f"Invalid problem: {e}")
```

Common errors:
- Budget exceeds active actors
- Missing time_horizon for intelligence
- Invalid strategy name
- Empty network

## Dependencies

- **numpy** (≥1.20.0): Numerical computations
- **scipy** (≥1.7.0): Statistical functions
- **networkx** (≥2.6.0): Graph algorithms (LCC, betweenness)

## Testing

Comprehensive test suites available:

```bash
python test_pareto_demo.py              # Pareto frontier functionality
python test_multi_round.py              # Multi-round planning
python test_strategy_comparison.py      # Strategy comparison
python test_production_readiness.py     # Production validation
```

## Version History

### 1.0.0 (Current)
- Initial production release
- All user stories implemented (US1-US5)
- Constitution compliance verified
- Performance targets met
- Production-ready quality

## References

- **Mathematical Model**: `gelis.tex` - Complete mathematical formulation
- **Constitution**: `.specify/memory/constitution.md` - Governance and constraints
- **Specification**: `specs/003-ocn-opt-001/spec.md` - Feature specification
- **Design**: `specs/003-ocn-opt-001/plan.md` - Implementation design

## License

See project root LICENSE file.
