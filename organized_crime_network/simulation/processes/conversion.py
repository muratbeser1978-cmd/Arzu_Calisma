"""
Informant conversion simulation using CTMC (Gillespie algorithm).

This module implements the two-state continuous-time Markov chain
for arrested actors transitioning to informants.

States: Loyal → Hesitant → Informant
Rates: μ_LH (constant), μ_HI(i) (actor-specific)

Constitutional Reference: Section II (Informant Conversion)
Research Decision: Decision 3 (CTMC Simulation)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Tuple, Dict
import numpy as np


class LoyaltyState(Enum):
    """
    CTMC states for arrested actors.

    Attributes
    ----------
    LOYAL : str
        Initial state after arrest
    HESITANT : str
        Intermediate state under pressure
    INFORMANT : str
        Absorbing state - converted to informant
    """
    LOYAL = "Loyal"
    HESITANT = "Hesitant"
    INFORMANT = "Informant"


@dataclass
class ActorLoyaltyState:
    """
    Track CTMC state for arrested actor.

    Attributes
    ----------
    actor_id : int
        Actor identifier
    current_state : LoyaltyState
        Current CTMC state
    state_history : List[Tuple[float, LoyaltyState]]
        (time, state) transition history
    arrest_time : float
        When actor was arrested
    avg_trust_at_arrest : float
        Ȳ_i(t_arrest) for μ_HI calculation
    fragility_rate : float
        μ_HI(i) = μ_min + μ_rng·expit(-θ·Ȳ_i)
    """
    actor_id: int
    current_state: LoyaltyState
    state_history: List[Tuple[float, LoyaltyState]] = field(default_factory=list)
    arrest_time: float = 0.0
    avg_trust_at_arrest: float = 0.0
    fragility_rate: float = 0.0


class ConversionProcess:
    """
    Informant conversion simulator using Gillespie algorithm.

    States: Loyal → Hesitant → Informant
    Rates: μ_LH (constant), μ_HI(i) (actor-specific)

    Constitutional Reference: Section II (Informant Conversion)
    Research Decision: Decision 3 (CTMC Simulation)

    Parameters
    ----------
    mu_LH : float
        External pressure rate (> 0)
    mu_min : float
        Minimum fragility (> 0)
    mu_rng : float
        Fragility range (> 0)
    theta : float
        Trust sensitivity (> 0)
    rng : np.random.Generator
        Random number generator

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> process = ConversionProcess(
    ...     mu_LH=0.2, mu_min=0.01, mu_rng=0.5, theta=2.0, rng=rng
    ... )
    >>> fragility = process.compute_fragility_rate(avg_trust=0.0)
    >>> fragility > 0
    True
    """

    def __init__(
        self,
        mu_LH: float,
        mu_min: float,
        mu_rng: float,
        theta: float,
        rng: np.random.Generator
    ):
        """
        Initialize conversion process simulator.

        Parameters
        ----------
        mu_LH : float
            External pressure rate (> 0)
        mu_min : float
            Minimum fragility (> 0)
        mu_rng : float
            Fragility range (> 0)
        theta : float
            Trust sensitivity (> 0)
        rng : np.random.Generator
            Random number generator
        """
        if mu_LH <= 0:
            raise ValueError("mu_LH must be > 0")
        if mu_min <= 0:
            raise ValueError("mu_min must be > 0")
        if mu_rng <= 0:
            raise ValueError("mu_rng must be > 0")
        if theta <= 0:
            raise ValueError("theta must be > 0")

        self.mu_LH = mu_LH
        self.mu_min = mu_min
        self.mu_rng = mu_rng
        self.theta = theta
        self.rng = rng

    def compute_fragility_rate(self, avg_trust: float) -> float:
        """
        Compute μ_HI from average trust at arrest.

        Parameters
        ----------
        avg_trust : float
            Ȳ_i = average logit-trust of arrested actor

        Returns
        -------
        mu_HI : float
            Fragility rate ∈ [μ_min, μ_min + μ_rng]

        Formula
        -------
        μ_HI = μ_min + μ_rng · expit(-θ · Ȳ_i)

        Constitutional Reference: Section II, Informant Conversion

        Examples
        --------
        >>> process = ConversionProcess(0.2, 0.01, 0.5, 2.0, rng)
        >>> # High trust (positive) → low fragility
        >>> mu_high = process.compute_fragility_rate(avg_trust=1.0)
        >>> # Low trust (negative) → high fragility
        >>> mu_low = process.compute_fragility_rate(avg_trust=-1.0)
        >>> mu_low > mu_high
        True
        """
        # expit(-θ · Ȳ_i) ∈ (0, 1)
        # High trust → expit(-θ · positive) → small → low fragility
        # Low trust → expit(-θ · negative) → large → high fragility
        expit_term = self._expit(-self.theta * avg_trust)
        mu_HI = self.mu_min + self.mu_rng * expit_term

        return mu_HI

    def simulate_trajectory(
        self,
        actor_id: int,
        avg_trust_at_arrest: float,
        t_start: float,
        t_max: float
    ) -> Tuple[List[float], List[str]]:
        """
        Simulate CTMC trajectory for one arrested actor.

        Parameters
        ----------
        actor_id : int
            Actor identifier
        avg_trust_at_arrest : float
            Ȳ_i(t_arrest) for computing μ_HI
        t_start : float
            Arrest time
        t_max : float
            Simulation horizon

        Returns
        -------
        times : list of float
            Transition times [t_start, t_1, t_2, ...]
        states : list of str
            States at each time ['Loyal', 'Hesitant', 'Informant']

        Algorithm (Gillespie)
        ---------------------
        1. Start in 'Loyal' state
        2. Sample waiting time τ ~ Exp(μ_LH)
        3. Transition to 'Hesitant' at t + τ
        4. Compute μ_HI(actor_id) from avg_trust_at_arrest
        5. Sample waiting time τ' ~ Exp(μ_HI)
        6. Transition to 'Informant' at t + τ'
        7. Stop (absorbing state)

        Expected Time
        -------------
        E[T_convert] = 1/μ_LH + 1/μ_HI  (for validation)

        Examples
        --------
        >>> times, states = process.simulate_trajectory(
        ...     actor_id=5,
        ...     avg_trust_at_arrest=-0.5,
        ...     t_start=10.0,
        ...     t_max=100.0
        ... )
        >>> states[0]
        'Loyal'
        >>> states[-1]
        'Informant'
        >>> len(times) == len(states)
        True
        """
        times = [t_start]
        states = [LoyaltyState.LOYAL.value]
        t_current = t_start

        # Transition 1: Loyal → Hesitant
        tau_LH = self.rng.exponential(1.0 / self.mu_LH)
        t_current += tau_LH

        if t_current < t_max:
            times.append(t_current)
            states.append(LoyaltyState.HESITANT.value)

            # Transition 2: Hesitant → Informant
            mu_HI = self.compute_fragility_rate(avg_trust_at_arrest)
            tau_HI = self.rng.exponential(1.0 / mu_HI)
            t_current += tau_HI

            if t_current < t_max:
                times.append(t_current)
                states.append(LoyaltyState.INFORMANT.value)

        return times, states

    def check_for_conversions(
        self,
        arrested_actors: Dict[int, ActorLoyaltyState],
        t_current: float
    ) -> List[int]:
        """
        Check if any arrested actors converted to informants.

        Parameters
        ----------
        arrested_actors : dict
            Map actor_id → ActorLoyaltyState with CTMC trajectories
        t_current : float
            Current simulation time

        Returns
        -------
        conversions : list of int
            Actor IDs that converted in current time step

        Examples
        --------
        >>> arrested = {
        ...     5: ActorLoyaltyState(
        ...         actor_id=5,
        ...         current_state=LoyaltyState.HESITANT,
        ...         state_history=[(10.0, LoyaltyState.LOYAL), (12.0, LoyaltyState.HESITANT)],
        ...         arrest_time=10.0
        ...     )
        ... }
        >>> conversions = process.check_for_conversions(arrested, t_current=15.0)
        >>> isinstance(conversions, list)
        True
        """
        conversions = []

        for actor_id, loyalty_state in arrested_actors.items():
            # Check if state changed to INFORMANT
            if loyalty_state.current_state == LoyaltyState.INFORMANT:
                # Check if this conversion happened since last check
                if loyalty_state.state_history:
                    last_transition_time = loyalty_state.state_history[-1][0]
                    if last_transition_time <= t_current:
                        conversions.append(actor_id)

        return conversions

    def create_loyalty_state(
        self,
        actor_id: int,
        arrest_time: float,
        avg_trust_at_arrest: float,
        t_max: float
    ) -> ActorLoyaltyState:
        """
        Create and simulate loyalty state for newly arrested actor.

        Parameters
        ----------
        actor_id : int
            Actor identifier
        arrest_time : float
            Time of arrest
        avg_trust_at_arrest : float
            Ȳ_i at arrest time
        t_max : float
            Simulation horizon

        Returns
        -------
        ActorLoyaltyState
            Initialized loyalty state with simulated trajectory

        Examples
        --------
        >>> loyalty = process.create_loyalty_state(
        ...     actor_id=7,
        ...     arrest_time=15.0,
        ...     avg_trust_at_arrest=-0.3,
        ...     t_max=100.0
        ... )
        >>> loyalty.actor_id
        7
        >>> loyalty.arrest_time
        15.0
        """
        # Simulate complete trajectory
        times, state_names = self.simulate_trajectory(
            actor_id, avg_trust_at_arrest, arrest_time, t_max
        )

        # Convert to history format
        state_history = [
            (t, LoyaltyState(s))
            for t, s in zip(times, state_names)
        ]

        # Current state is last in trajectory
        current_state = state_history[-1][1]

        # Compute fragility rate
        fragility_rate = self.compute_fragility_rate(avg_trust_at_arrest)

        return ActorLoyaltyState(
            actor_id=actor_id,
            current_state=current_state,
            state_history=state_history,
            arrest_time=arrest_time,
            avg_trust_at_arrest=avg_trust_at_arrest,
            fragility_rate=fragility_rate
        )

    @staticmethod
    def _expit(x: float) -> float:
        """
        Numerically stable expit (logistic sigmoid) function.

        expit(x) = 1 / (1 + exp(-x))

        Parameters
        ----------
        x : float
            Input value

        Returns
        -------
        float
            expit(x) ∈ (0, 1)
        """
        if x >= 0:
            return 1.0 / (1.0 + np.exp(-x))
        else:
            exp_x = np.exp(x)
            return exp_x / (1.0 + exp_x)
