"""
Simulation event recording and management.

This module defines the event types and structures for recording
discrete occurrences during simulation.

Constitutional Reference: Section IV (State Update Rules)
Specification Reference: FR-002 (Event Recording)
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Optional


class EventType(Enum):
    """
    Types of discrete events that can occur during simulation.

    Attributes
    ----------
    ARREST : str
        Actor arrested by law enforcement
    CONVERSION : str
        Arrested actor converted to informant
    GROWTH : str
        New edge added via TFPA mechanism
    TRUST_UPDATE : str
        Significant change in trust value
    """
    ARREST = "arrest"
    CONVERSION = "conversion"
    GROWTH = "growth"
    TRUST_UPDATE = "trust_update"


@dataclass(frozen=True)
class SimulationEvent:
    """
    Record of a discrete event during simulation.

    Events are immutable and contain complete information about
    what happened, when, and the state before/after.

    Attributes
    ----------
    event_type : EventType
        Type of event that occurred
    timestamp : float
        Simulation time when event occurred
    actor_id : int
        Primary actor involved (for ARREST, CONVERSION)
    secondary_actor : Optional[int]
        Secondary actor (for GROWTH - target of new edge)
    details : Dict[str, Any]
        Event-specific data
    state_before : Any
        Snapshot of relevant state before event
    state_after : Any
        Snapshot of relevant state after event

    Examples
    --------
    >>> event = SimulationEvent(
    ...     event_type=EventType.ARREST,
    ...     timestamp=12.5,
    ...     actor_id=7,
    ...     secondary_actor=None,
    ...     details={'hierarchy_level': 2, 'arrest_intensity': 0.15},
    ...     state_before={'num_active': 50},
    ...     state_after={'num_active': 49}
    ... )
    >>> event.event_type
    <EventType.ARREST: 'arrest'>
    >>> event.details['hierarchy_level']
    2
    """
    event_type: EventType
    timestamp: float
    actor_id: Optional[int]
    secondary_actor: Optional[int]
    details: Dict[str, Any]
    state_before: Any
    state_after: Any

    @staticmethod
    def arrest(
        timestamp: float,
        actor_id: int,
        hierarchy_level: int,
        arrest_intensity: float,
        effectiveness_before: float,
        num_active_before: int,
        num_active_after: int
    ) -> 'SimulationEvent':
        """
        Create an ARREST event record.

        Parameters
        ----------
        timestamp : float
            Time of arrest
        actor_id : int
            Actor who was arrested
        hierarchy_level : int
            Hierarchy level of arrested actor
        arrest_intensity : float
            λ_i^Arr(t) at time of arrest
        effectiveness_before : float
            P(t) before arrest
        num_active_before : int
            Number of active actors before
        num_active_after : int
            Number of active actors after

        Returns
        -------
        SimulationEvent
            Arrest event record
        """
        return SimulationEvent(
            event_type=EventType.ARREST,
            timestamp=timestamp,
            actor_id=actor_id,
            secondary_actor=None,
            details={
                'hierarchy_level': hierarchy_level,
                'arrest_intensity': arrest_intensity,
                'effectiveness_before': effectiveness_before
            },
            state_before={'num_active': num_active_before},
            state_after={'num_active': num_active_after}
        )

    @staticmethod
    def conversion(
        timestamp: float,
        actor_id: int,
        loyalty_state_before: str,
        time_in_arrested: float,
        fragility_rate: float,
        avg_trust: float,
        num_informants_before: int,
        num_informants_after: int
    ) -> 'SimulationEvent':
        """
        Create a CONVERSION event record.

        Parameters
        ----------
        timestamp : float
            Time of conversion
        actor_id : int
            Actor who converted to informant
        loyalty_state_before : str
            Loyalty state before conversion ('Loyal' or 'Hesitant')
        time_in_arrested : float
            Duration since arrest
        fragility_rate : float
            μ_HI(i) for this actor
        avg_trust : float
            Ȳ_i at time of arrest
        num_informants_before : int
            Number of informants before
        num_informants_after : int
            Number of informants after

        Returns
        -------
        SimulationEvent
            Conversion event record
        """
        return SimulationEvent(
            event_type=EventType.CONVERSION,
            timestamp=timestamp,
            actor_id=actor_id,
            secondary_actor=None,
            details={
                'loyalty_state_before': loyalty_state_before,
                'time_in_arrested': time_in_arrested,
                'fragility_rate': fragility_rate,
                'avg_trust': avg_trust
            },
            state_before={'num_informants': num_informants_before},
            state_after={'num_informants': num_informants_after}
        )

    @staticmethod
    def growth(
        timestamp: float,
        source_actor: int,
        target_actor: int,
        initial_trust: float,
        attachment_prob: float,
        num_edges_before: int,
        num_edges_after: int
    ) -> 'SimulationEvent':
        """
        Create a GROWTH event record.

        Parameters
        ----------
        timestamp : float
            Time of edge addition
        source_actor : int
            Source node of new edge
        target_actor : int
            Target node of new edge
        initial_trust : float
            W^init(i,j) Jaccard-based trust
        attachment_prob : float
            Preferential attachment probability
        num_edges_before : int
            Number of edges before
        num_edges_after : int
            Number of edges after

        Returns
        -------
        SimulationEvent
            Growth event record
        """
        return SimulationEvent(
            event_type=EventType.GROWTH,
            timestamp=timestamp,
            actor_id=source_actor,
            secondary_actor=target_actor,
            details={
                'initial_trust': initial_trust,
                'attachment_prob': attachment_prob
            },
            state_before={'num_edges': num_edges_before},
            state_after={'num_edges': num_edges_after}
        )

    @staticmethod
    def trust_update(
        timestamp: float,
        edge: tuple,
        old_value: float,
        new_value: float,
        risk_factor: float
    ) -> 'SimulationEvent':
        """
        Create a TRUST_UPDATE event record.

        Only created for significant changes (|ΔY| > threshold).

        Parameters
        ----------
        timestamp : float
            Time of update
        edge : tuple
            (source, target) edge
        old_value : float
            Y_ij before update
        new_value : float
            Y_ij after update
        risk_factor : float
            R_i(t) influencing the change

        Returns
        -------
        SimulationEvent
            Trust update event record
        """
        return SimulationEvent(
            event_type=EventType.TRUST_UPDATE,
            timestamp=timestamp,
            actor_id=edge[0],
            secondary_actor=edge[1],
            details={
                'old_value': old_value,
                'new_value': new_value,
                'risk_factor': risk_factor,
                'change': new_value - old_value
            },
            state_before={'trust': old_value},
            state_after={'trust': new_value}
        )
