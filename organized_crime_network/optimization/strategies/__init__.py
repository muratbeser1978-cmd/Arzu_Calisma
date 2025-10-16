"""
Optimization strategies for organized crime network intervention.

This module provides concrete implementations of optimization algorithms:
- structural_damage: Algorithms for maximizing network fragmentation
- intelligence: Algorithms for maximizing expected information gain (added in Phase 4)
- hybrid: Combined objective optimization (added in Phase 5)

Mathematical Foundation: gelis.tex
Governance: .specify/memory/constitution.md v1.0.0
"""

from .structural_damage import StructuralDamageOptimizer
from .intelligence import IntelligenceOptimizer
from .hybrid import HybridOptimizer

__all__ = [
    "StructuralDamageOptimizer",
    "IntelligenceOptimizer",
    "HybridOptimizer",
]
