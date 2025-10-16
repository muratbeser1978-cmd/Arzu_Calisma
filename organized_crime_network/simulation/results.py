"""
Simulation results data structures.

This module defines the output structures for simulation runs,
including time series trajectories and summary statistics.

Specification Reference: FR-009, FR-010, US-005
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Any
import numpy as np
import uuid

from .events import SimulationEvent
from .parameters import SimulationParameters


class ConvergenceStatus(Enum):
    """
    Status of simulation completion.

    Attributes
    ----------
    COMPLETED : str
        Simulation reached T_max normally
    NETWORK_COLLAPSED : str
        All actors arrested or converted
    NUMERICAL_ERROR : str
        Numerical instability detected
    USER_INTERRUPTED : str
        Manual stop requested
    """
    COMPLETED = "completed"
    NETWORK_COLLAPSED = "collapsed"
    NUMERICAL_ERROR = "error"
    USER_INTERRUPTED = "interrupted"


@dataclass
class TimeSeries:
    """
    Continuous trajectory of aggregate metrics.

    Time series are recorded at regular intervals and at all event times
    to capture both smooth evolution and sudden changes.

    Attributes
    ----------
    times : np.ndarray
        Time points where measurements taken
    network_size : np.ndarray
        |V_active(t)| - number of active actors
    arrested_count : np.ndarray
        Cumulative number of arrests
    informant_count : np.ndarray
        Cumulative number of conversions
    effectiveness : np.ndarray
        P(t) - law enforcement effectiveness
    mean_trust : np.ndarray
        Mean of w_ij(t) = expit(Y_ij) across all edges
    mean_risk : np.ndarray
        Mean of R_i(t) across all actors
    lcc_size : np.ndarray
        Size of largest connected component

    Examples
    --------
    >>> ts = TimeSeries(
    ...     times=np.array([0.0, 1.0, 2.0]),
    ...     network_size=np.array([50, 48, 45]),
    ...     arrested_count=np.array([0, 2, 5]),
    ...     informant_count=np.array([0, 0, 1]),
    ...     effectiveness=np.array([1.0, 1.0, 1.25]),
    ...     mean_trust=np.array([0.5, 0.48, 0.45]),
    ...     mean_risk=np.array([0.0, 0.04, 0.1]),
    ...     lcc_size=np.array([50, 48, 45])
    ... )
    >>> ts.times[-1]
    2.0
    >>> ts.network_size[0]
    50
    """
    times: np.ndarray
    network_size: np.ndarray
    arrested_count: np.ndarray
    informant_count: np.ndarray
    effectiveness: np.ndarray
    mean_trust: np.ndarray
    mean_risk: np.ndarray
    lcc_size: np.ndarray

    def __post_init__(self):
        """Validate that all arrays have same length."""
        lengths = {
            'times': len(self.times),
            'network_size': len(self.network_size),
            'arrested_count': len(self.arrested_count),
            'informant_count': len(self.informant_count),
            'effectiveness': len(self.effectiveness),
            'mean_trust': len(self.mean_trust),
            'mean_risk': len(self.mean_risk),
            'lcc_size': len(self.lcc_size)
        }
        if len(set(lengths.values())) > 1:
            raise ValueError(f"TimeSeries arrays have inconsistent lengths: {lengths}")

    def to_dict(self) -> dict:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        dict
            Dictionary with array values as lists
        """
        return {
            'times': self.times.tolist(),
            'network_size': self.network_size.tolist(),
            'arrested_count': self.arrested_count.tolist(),
            'informant_count': self.informant_count.tolist(),
            'effectiveness': self.effectiveness.tolist(),
            'mean_trust': self.mean_trust.tolist(),
            'mean_risk': self.mean_risk.tolist(),
            'lcc_size': self.lcc_size.tolist()
        }


@dataclass
class SimulationResults:
    """
    Complete output from simulation run.

    Contains event history, time series, final state, and summary statistics.
    Provides export methods for JSON and CSV formats.

    Attributes
    ----------
    parameters : SimulationParameters
        Parameters used for this simulation
    initial_network : Any
        Network state at t=0 (NetworkState object)
    final_network : Any
        Network state at end (NetworkState object)
    run_id : str
        UUID for tracking this simulation
    events : List[SimulationEvent]
        All events chronologically ordered
    time_series : TimeSeries
        Aggregate metrics over time
    total_arrests : int
        Total number of arrest events
    total_conversions : int
        Total number of conversion events
    final_effectiveness : float
        P(T_max) at end of simulation
    simulation_duration : float
        Wall-clock time in seconds
    convergence_status : ConvergenceStatus
        How simulation terminated
    numerical_stability : bool
        True if no NaN/Inf detected
    causality_preserved : bool
        True if all events in valid order

    Examples
    --------
    >>> results = SimulationResults(
    ...     parameters=params,
    ...     initial_network=state0,
    ...     final_network=state_final,
    ...     run_id=str(uuid.uuid4()),
    ...     events=[...],
    ...     time_series=ts,
    ...     total_arrests=15,
    ...     total_conversions=3,
    ...     final_effectiveness=1.5,
    ...     simulation_duration=5.2,
    ...     convergence_status=ConvergenceStatus.COMPLETED,
    ...     numerical_stability=True,
    ...     causality_preserved=True
    ... )
    >>> results.total_arrests
    15
    >>> results.convergence_status
    <ConvergenceStatus.COMPLETED: 'completed'>
    """
    parameters: SimulationParameters
    initial_network: Any
    final_network: Any
    run_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    events: List[SimulationEvent] = field(default_factory=list)
    time_series: TimeSeries = None
    total_arrests: int = 0
    total_conversions: int = 0
    final_effectiveness: float = 0.0
    simulation_duration: float = 0.0
    convergence_status: ConvergenceStatus = ConvergenceStatus.COMPLETED
    numerical_stability: bool = True
    causality_preserved: bool = True

    def export_events(self, filepath: str) -> None:
        """
        Export event history to JSON file.

        Parameters
        ----------
        filepath : str
            Path to output JSON file

        Examples
        --------
        >>> results.export_events("simulation_events.json")
        """
        import json

        events_data = [
            {
                'event_type': event.event_type.value,
                'timestamp': event.timestamp,
                'actor_id': event.actor_id,
                'secondary_actor': event.secondary_actor,
                'details': event.details,
                'state_before': event.state_before,
                'state_after': event.state_after
            }
            for event in self.events
        ]

        output = {
            'run_id': self.run_id,
            'total_events': len(self.events),
            'events': events_data
        }

        with open(filepath, 'w') as f:
            json.dump(output, f, indent=2)

    def export_time_series(self, filepath: str) -> None:
        """
        Export time series to CSV file.

        Parameters
        ----------
        filepath : str
            Path to output CSV file

        Examples
        --------
        >>> results.export_time_series("time_series.csv")
        """
        import csv

        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)

            # Header
            writer.writerow([
                'time',
                'network_size',
                'arrested_count',
                'informant_count',
                'effectiveness',
                'mean_trust',
                'mean_risk',
                'lcc_size'
            ])

            # Data rows
            for i in range(len(self.time_series.times)):
                writer.writerow([
                    self.time_series.times[i],
                    self.time_series.network_size[i],
                    self.time_series.arrested_count[i],
                    self.time_series.informant_count[i],
                    self.time_series.effectiveness[i],
                    self.time_series.mean_trust[i],
                    self.time_series.mean_risk[i],
                    self.time_series.lcc_size[i]
                ])

    def summary(self) -> str:
        """
        Generate human-readable summary of results.

        Returns
        -------
        str
            Formatted summary text

        Examples
        --------
        >>> print(results.summary())
        Simulation Results Summary
        ==========================
        Run ID: ...
        Status: completed
        Total Arrests: 15
        Total Conversions: 3
        Final Effectiveness: 1.50
        ...
        """
        return f"""Simulation Results Summary
==========================
Run ID: {self.run_id}
Status: {self.convergence_status.value}
Duration: {self.simulation_duration:.2f} seconds

Network Evolution:
  Initial Size: {self.initial_network.num_actors if hasattr(self.initial_network, 'num_actors') else 'N/A'}
  Final Active: {self.final_network.num_actors if hasattr(self.final_network, 'num_actors') else 'N/A'}
  Total Arrests: {self.total_arrests}
  Total Conversions: {self.total_conversions}

Effectiveness:
  Initial: {self.parameters.P_0:.2f}
  Final: {self.final_effectiveness:.2f}
  Change: {self.final_effectiveness - self.parameters.P_0:+.2f}

Validation:
  Numerical Stability: {'OK' if self.numerical_stability else 'FAIL'}
  Causality Preserved: {'OK' if self.causality_preserved else 'FAIL'}
"""


class SimulationError(Exception):
    """Base exception for simulation errors."""
    pass


class NetworkValidationError(SimulationError):
    """Raised when initial network is invalid."""
    pass


class NumericalInstabilityError(SimulationError):
    """Raised when SDE integration becomes unstable."""
    pass


class EventGenerationError(SimulationError):
    """Raised when stochastic event simulation fails."""
    pass
