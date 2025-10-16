"""
Organized Crime Network Stochastic Model

A rigorous stochastic model for organized crime network dynamics under
law enforcement intervention.

Mathematical Foundation: gelis.tex
Governance: .specify/memory/constitution.md v1.0.0
"""

__version__ = "1.0.0"
__author__ = "OCN Research Team"

from .simulation import SimulationEngine
from .core.parameters import Parameters
from .core.state import NetworkState
from . import optimization

__all__ = ["SimulationEngine", "Parameters", "NetworkState", "optimization"]
