"""
Event definitions for discrete state transitions.

All events are {ℱₜ}-stopping times (Constitution III).
"""

from dataclasses import dataclass
from typing import Literal, Dict, Any
from enum import Enum


class EventType(Enum):
    """Types of events in the simulation."""

    ARREST = "arrest"
    CONVERSION = "conversion"
    GROWTH = "growth"


@dataclass
class Event:
    """
    Base class for simulation events.

    All events have:
    - Timestamp: When the event occurs
    - Type: What kind of state transition
    - Metadata: Event-specific information
    """

    time: float
    event_type: EventType

    def to_dict(self) -> Dict[str, Any]:
        """Convert event to dictionary for logging."""
        return {"time": self.time, "type": self.event_type.value}


class ArrestEvent(Event):
    """
    Arrest event: Active actor transitions to Arrested.

    State Update (Constitution IV):
    1. aᵢ(t⁻) = Active → aᵢ(t) = Arrested
    2. Remove all (i,·) and (·,i) from E(t)
    3. Update Rₖ(t) for all k ∈ Nᵢ(t⁻)
    4. Continue trust evolution with updated risk
    5. Initialize CTMC for actor i
    """

    def __init__(self, time: float, actor_id: int):
        super().__init__(time, EventType.ARREST)
        self.actor_id = actor_id

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update({"actor_id": self.actor_id})
        return base


class ConversionEvent(Event):
    """
    Informant conversion: Arrested actor transitions to Informant.

    State Update (Constitution IV):
    1. aᵢ(t⁻) = Arrested → aᵢ(t) = Informant
    2. P(t) = P(t⁻) + η_P (effectiveness jump)
    3. Nᵢ(t) = Nᵢ(t⁻) + 1 (counter update)
    """

    def __init__(self, time: float, actor_id: int, avg_trust_at_arrest: float):
        super().__init__(time, EventType.CONVERSION)
        self.actor_id = actor_id
        self.avg_trust_at_arrest = avg_trust_at_arrest

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {"actor_id": self.actor_id, "avg_trust_at_arrest": self.avg_trust_at_arrest}
        )
        return base


class GrowthEvent(Event):
    """
    TFPA edge addition: New operational edge formed.

    State Update (Constitution IV):
    1. Select source i from V_act(t) uniformly
    2. Select target j with probability πᵢ→ⱼ(t)
    3. Add edge (i,j) to E(t)
    4. Initialize trust: Yᵢⱼ(t) = logit(W^init(i,j))
    5. Trust evolves according to SDE
    """

    def __init__(self, time: float, source_id: int, target_id: int, initial_trust: float):
        super().__init__(time, EventType.GROWTH)
        self.source_id = source_id
        self.target_id = target_id
        self.initial_trust = initial_trust

    def to_dict(self) -> Dict[str, Any]:
        base = super().to_dict()
        base.update(
            {
                "source_id": self.source_id,
                "target_id": self.target_id,
                "initial_trust": self.initial_trust,
            }
        )
        return base
