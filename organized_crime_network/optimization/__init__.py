"""
Multi-Objective Optimization Module for Organized Crime Networks.

This module provides algorithms for identifying optimal intervention targets:
- Structural Damage: Maximize network fragmentation
- Intelligence Gain: Maximize expected information from informants
- Pareto Analysis: Explore trade-offs between objectives
- Multi-Round Planning: Adaptive sequential interventions

Mathematical Foundation: gelis.tex
Governance: .specify/memory/constitution.md v1.0.0
"""

__version__ = "1.0.0"

# Core data structures
from .base import (
    OptimizationProblem,
    TargetSet,
    InterventionStrategy,
    StrategyType,
    ConversionProbability,
    Optimizer,
    InformationModel,
)

# Metrics and utilities
from .metrics import (
    PerformanceMetrics,
    StrategyComparison,
    compare_strategies,
    format_comparison_table,
    generate_strategy_recommendation,
    get_comparison_summary,
)

# Objective functions
from .objectives import (
    compute_lcc_size,
    compute_structural_damage,
    compute_degree_centrality,
    compute_betweenness_centrality,
    NeighborhoodInformationModel,
)

# Optimization strategies
from .strategies import StructuralDamageOptimizer, IntelligenceOptimizer, HybridOptimizer

# Intelligence objective functions
from .objectives import (
    compute_conversion_probability,
    compute_average_logit_trust,
    compute_fragility_rate,
    compute_intelligence_value,
    compute_marginal_intelligence,
)

# Pareto frontier
from .pareto import ParetoSolution, ParetoFrontierComputer

# Multi-round planning
from .multi_round import InterventionRound, MultiRoundPlan, MultiRoundPlanner

__all__ = [
    # Data structures
    "OptimizationProblem",
    "TargetSet",
    "InterventionStrategy",
    "StrategyType",
    "ConversionProbability",
    "PerformanceMetrics",
    # Pareto
    "ParetoSolution",
    "ParetoFrontierComputer",
    # Multi-round planning
    "InterventionRound",
    "MultiRoundPlan",
    "MultiRoundPlanner",
    # Strategy comparison
    "StrategyComparison",
    "compare_strategies",
    "format_comparison_table",
    "generate_strategy_recommendation",
    "get_comparison_summary",
    # Abstract interfaces
    "Optimizer",
    "InformationModel",
    # Utilities - Structural
    "compute_lcc_size",
    "compute_structural_damage",
    "compute_degree_centrality",
    "compute_betweenness_centrality",
    # Utilities - Intelligence
    "compute_conversion_probability",
    "compute_average_logit_trust",
    "compute_fragility_rate",
    "compute_intelligence_value",
    "compute_marginal_intelligence",
    # Information Models
    "NeighborhoodInformationModel",
    # Optimizers
    "StructuralDamageOptimizer",
    "IntelligenceOptimizer",
    "HybridOptimizer",
]
