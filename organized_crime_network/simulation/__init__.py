"""
Stochastic simulation engine for organized crime networks.

This module implements a comprehensive simulation framework for studying
the dynamics of criminal networks under law enforcement intervention.

Mathematical Foundation: gelis.tex
Constitution: .specify/memory/constitution.md v1.0.0
Feature: OCN-SIM-001

Main Components:
    - SimulationEngine: Main orchestrator for running simulations
    - SimulationParameters: Validated parameter container
    - SimulationResults: Complete simulation output
    - SimulationEvent: Discrete event records

Usage:
    >>> from organized_crime_network.simulation import SimulationEngine, SimulationParameters
    >>> from organized_crime_network.core.state import NetworkState
    >>>
    >>> # Create parameters with defaults
    >>> params = SimulationParameters.default()
    >>>
    >>> # Initialize network
    >>> state = NetworkState(t_current=0.0)
    >>> # ... add actors and edges ...
    >>>
    >>> # Run simulation
    >>> engine = SimulationEngine(params)
    >>> results = engine.run(state, verbose=True)
    >>>
    >>> print(f"Total arrests: {results.total_arrests}")
    >>> print(f"Final effectiveness: {results.final_effectiveness:.2f}")
"""

from .parameters import SimulationParameters, ParameterValidationError
from .events import SimulationEvent, EventType
from .results import (
    SimulationResults,
    TimeSeries,
    ConvergenceStatus,
    SimulationError,
    NetworkValidationError,
    NumericalInstabilityError,
    EventGenerationError
)
from .engine import SimulationEngine

# Topology generators (US3)
from . import topology

# Analysis utilities (US5)
from . import analysis

# Visualization tools (US6)
from . import visualization
from .visualization import SimulationVisualizer, visualize_results
from . import visualization_advanced
from .visualization_advanced import AdvancedNetworkVisualizer, create_advanced_visualizations

__all__ = [
    'SimulationEngine',
    'SimulationParameters',
    'ParameterValidationError',
    'SimulationEvent',
    'EventType',
    'SimulationResults',
    'TimeSeries',
    'ConvergenceStatus',
    'SimulationError',
    'NetworkValidationError',
    'NumericalInstabilityError',
    'EventGenerationError',
    'topology',
    'analysis',
    'visualization',
    'SimulationVisualizer',
    'visualize_results',
    'visualization_advanced',
    'AdvancedNetworkVisualizer',
    'create_advanced_visualizations',
]

__version__ = "0.1.0"
