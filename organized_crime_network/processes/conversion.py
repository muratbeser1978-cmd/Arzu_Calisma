"""
Informant Conversion: Two-stage CTMC with trust-dependent rates.

Mathematical Foundation: Constitution II, Informant Conversion CTMC
"""

from typing import Dict, Optional
from enum import Enum

from ..core.state import NetworkState, ActorState
from ..core.parameters import Parameters
from ..utils.numerical import fragility_rate
from ..utils.random import sample_exponential


class CTMCState(Enum):
    """States in informant conversion CTMC."""

    LOYAL = "L"  # Initial state after arrest
    HESITANT = "H"  # Under external pressure
    INFORMANT = "I"  # Converted (absorbing state)


class InformantConversion:
    """
    Two-stage Continuous-Time Markov Chain for informant conversion.

    States and Transitions (Constitution II):
    - Loyal (L) → Hesitant (H): Rate μ_LH (constant, external pressure)
    - Hesitant (H) → Informant (I): Rate μ_HI(i) (trust-dependent)

    Formula for μ_HI(i):
    μ_HI(i) = μ_min + μ_rng · expit(-θ · Ȳᵢ(t_arrest))

    Expected Conversion Time:
    𝔼[Tᵢᴵ] = 1/μ_LH + 1/μ_HI(i)
    """

    def __init__(self, params: Parameters):
        """
        Initialize conversion process.

        Args:
            params: Model parameters
        """
        self.mu_LH = params.mu_LH  # External pressure rate
        self.mu_min = params.mu_min  # Minimum fragility
        self.mu_rng = params.mu_rng  # Fragility range
        self.theta = params.theta  # Trust sensitivity

        # Track CTMC state for each arrested actor
        self.ctmc_states: Dict[int, CTMCState] = {}

        # Track scheduled transition times
        self.next_transition: Dict[int, float] = {}

    def initialize_actor(self, actor_id: int, arrest_time: float, state: NetworkState) -> None:
        """
        Initialize CTMC for newly arrested actor.

        Args:
            actor_id: Actor ID
            arrest_time: Time of arrest
            state: Network state (for extracting avg trust)

        Effects:
            - Sets CTMC state to LOYAL
            - Samples time to HESITANT transition
        """
        self.ctmc_states[actor_id] = CTMCState.LOYAL

        # Sample waiting time for L → H transition
        waiting_time = sample_exponential(self.mu_LH)
        self.next_transition[actor_id] = arrest_time + waiting_time

    def advance_to_hesitant(self, actor_id: int, current_time: float) -> None:
        """
        Transition actor from LOYAL to HESITANT.

        Args:
            actor_id: Actor ID
            current_time: Current simulation time

        Effects:
            - Updates CTMC state to HESITANT
            - Samples time to INFORMANT transition
        """
        if actor_id not in self.ctmc_states:
            raise ValueError(f"Actor {actor_id} not initialized in CTMC")

        if self.ctmc_states[actor_id] != CTMCState.LOYAL:
            raise ValueError(
                f"Actor {actor_id} not in LOYAL state: {self.ctmc_states[actor_id]}"
            )

        self.ctmc_states[actor_id] = CTMCState.HESITANT

        # Compute trust-dependent fragility rate
        # Need average trust at arrest from state
        # This is stored in state.avg_trust_at_arrest[actor_id]

    def compute_fragility_rate(self, actor_id: int, state: NetworkState) -> float:
        """
        Compute trust-dependent fragility rate.

        Formula (Constitution II):
        μ_HI(i) = μ_min + μ_rng · expit(-θ · Ȳᵢ(t_arrest))

        Args:
            actor_id: Actor ID
            state: Network state (for avg trust at arrest)

        Returns:
            Fragility rate μ_HI ∈ [μ_min, μ_min + μ_rng]
        """
        if actor_id not in state.avg_trust_at_arrest:
            raise ValueError(f"No arrest record for actor {actor_id}")

        avg_trust = state.avg_trust_at_arrest[actor_id]
        return fragility_rate(avg_trust, self.mu_min, self.mu_rng, self.theta)

    def schedule_informant_conversion(
        self, actor_id: int, current_time: float, state: NetworkState
    ) -> None:
        """
        Schedule H → I transition for actor.

        Args:
            actor_id: Actor ID
            current_time: Current simulation time
            state: Network state (for fragility rate computation)

        Effects:
            Updates next_transition[actor_id] with conversion time
        """
        mu_HI = self.compute_fragility_rate(actor_id, state)
        waiting_time = sample_exponential(mu_HI)
        self.next_transition[actor_id] = current_time + waiting_time

    def get_next_conversion(self, state: NetworkState) -> Optional[tuple]:
        """
        Get next informant conversion event.

        Returns:
            (time, actor_id) for nearest conversion, or None
        """
        if not self.next_transition:
            return None

        # Find actor with earliest transition
        next_actor = min(self.next_transition, key=self.next_transition.get)
        next_time = self.next_transition[next_actor]

        return (next_time, next_actor)

    def execute_transition(self, actor_id: int, current_time: float, state: NetworkState) -> None:
        """
        Execute CTMC transition for actor.

        Args:
            actor_id: Actor ID
            current_time: Current simulation time
            state: Network state

        Effects:
            - Advances CTMC state
            - Schedules next transition if needed
        """
        if actor_id not in self.ctmc_states:
            return

        current_state = self.ctmc_states[actor_id]

        if current_state == CTMCState.LOYAL:
            # L → H transition
            self.ctmc_states[actor_id] = CTMCState.HESITANT
            self.schedule_informant_conversion(actor_id, current_time, state)

        elif current_state == CTMCState.HESITANT:
            # H → I transition (absorbing)
            self.ctmc_states[actor_id] = CTMCState.INFORMANT
            self.next_transition.pop(actor_id, None)  # No more transitions

        else:
            # Already informant, no transition
            pass

    def get_expected_conversion_time(self, actor_id: int, state: NetworkState) -> float:
        """
        Compute expected time to informant conversion.

        Formula (Constitution II):
        𝔼[Tᵢᴵ] = 1/μ_LH + 1/μ_HI(i)

        Args:
            actor_id: Actor ID
            state: Network state

        Returns:
            Expected conversion time
        """
        mu_HI = self.compute_fragility_rate(actor_id, state)
        return 1.0 / self.mu_LH + 1.0 / mu_HI
