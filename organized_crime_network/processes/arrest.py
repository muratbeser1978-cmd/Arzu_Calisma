"""
Arrest Process: Cox process with time-dependent intensity.

Mathematical Foundation: Constitution II, Equation for λᵢᴬʳʳ(t)
"""

import numpy as np
from typing import Optional

from ..core.state import NetworkState, ActorState
from ..core.parameters import Parameters
from ..utils.numerical import expit, hierarchical_protection
from ..utils.random import sample_exponential, get_rng


class ArrestProcess:
    """
    Cox process for arrests with state-dependent intensity.

    Intensity Formula (Constitution II):
    λᵢᴬʳʳ(t) = λ₀ · h(H(i)) · v(i,t) · P(t)

    where:
    - λ₀: Base arrest risk
    - h(l) = exp(-κ(l-1)): Hierarchical protection
    - v(i,t) = 1 + γ Σⱼ∈Nᵢᵒᵘᵗ expit(Yᵢⱼ): Operational risk
    - P(t): Law enforcement effectiveness
    """

    def __init__(self, params: Parameters):
        """
        Initialize arrest process.

        Args:
            params: Model parameters
        """
        self.lambda_0 = params.lambda_0
        self.kappa = params.kappa
        self.gamma = params.gamma
        self.rho = params.rho
        self.eta_P = params.eta_P
        self.P_0 = params.P_0

    def compute_intensity(self, actor_id: int, state: NetworkState) -> float:
        """
        Compute arrest intensity for actor.

        Formula (Constitution II):
        λᵢᴬʳʳ(t) = λ₀ · h(H(i)) · v(i,t) · P(t)

        Args:
            actor_id: Actor ID
            state: Current network state

        Returns:
            Arrest intensity λᵢ ≥ 0
        """
        # Only Active actors can be arrested
        if state.A[actor_id] != ActorState.ACTIVE:
            return 0.0

        # Hierarchical protection: h(l) = exp(-κ(l-1))
        level = state.hierarchy[actor_id]
        h_factor = hierarchical_protection(level, self.kappa)

        # Operational risk: v(i,t) = 1 + γ Σⱼ expit(Yᵢⱼ)
        outgoing_neighbors = state.get_outgoing_neighbors(actor_id)
        if len(outgoing_neighbors) == 0:
            v_factor = 1.0
        else:
            trust_sum = sum(
                expit(state.Y[(actor_id, j)]) for j in outgoing_neighbors
            )
            v_factor = 1.0 + self.gamma * trust_sum

        # Law enforcement effectiveness
        P_t = state.P_t

        # Combined intensity
        intensity = self.lambda_0 * h_factor * v_factor * P_t

        return intensity

    def compute_total_intensity(self, state: NetworkState) -> float:
        """
        Compute total arrest intensity across all Active actors.

        Returns:
            Total intensity Λ(t) = Σᵢ λᵢ(t)
        """
        active_actors = state.get_active_actors()
        return sum(self.compute_intensity(i, state) for i in active_actors)

    def sample_next_arrest(self, state: NetworkState) -> Optional[tuple]:
        """
        Sample next arrest event using thinning algorithm.

        Returns:
            (time_to_arrest, actor_id) or None if no Active actors

        Algorithm:
            1. Compute total intensity Λ(t)
            2. Sample waiting time τ ~ Exp(Λ)
            3. Select actor proportional to individual intensities
        """
        active_actors = state.get_active_actors()

        if len(active_actors) == 0:
            return None

        # Compute individual intensities
        intensities = {i: self.compute_intensity(i, state) for i in active_actors}
        total_intensity = sum(intensities.values())

        if total_intensity == 0:
            return None  # No arrest risk

        # Sample waiting time
        waiting_time = sample_exponential(total_intensity)

        # Select actor proportional to intensity
        actor_probs = np.array([intensities[i] for i in active_actors]) / total_intensity
        actor_idx = get_rng().choice(len(active_actors), p=actor_probs)
        actor_id = list(active_actors)[actor_idx]

        return (waiting_time, actor_id)

    def update_effectiveness(
        self, state: NetworkState, dt: float, informant_jump: bool = False
    ) -> None:
        """
        Update law enforcement effectiveness.

        Formula (Constitution II):
        dP(t) = -ρ(P(t) - P₀)dt + η_P dNᵢ(t)

        Args:
            state: Current network state
            dt: Time step
            informant_jump: Whether informant conversion occurred

        Effects:
            - Continuous decay towards P₀
            - Discrete jump +η_P on informant conversion
        """
        # Continuous decay
        dP_decay = -self.rho * (state.P_t - self.P_0) * dt
        state.P_t += dP_decay

        # Discrete jump (handled by state.convert_to_informant)
        # This method only handles continuous part
