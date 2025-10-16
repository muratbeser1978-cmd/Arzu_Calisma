"""
Trust Dynamics: SDE/PDMP with risk-dependent drift.

Mathematical Foundation: Constitution II, Trust Dynamics SDE
"""

import numpy as np
from typing import Set, Tuple

from ..core.state import NetworkState
from ..core.parameters import Parameters
from ..utils.random import sample_normal


class TrustDynamics:
    """
    Stochastic Differential Equation for trust evolution.

    SDE Formula (Constitution II):
    dYᵢⱼ(t) = (-α Yᵢⱼ(t) - β Rᵢ(t))dt + σ dBᵢⱼ(t)

    Properties:
    - Ornstein-Uhlenbeck process with risk-dependent equilibrium
    - Equilibrium mean: μ_∞(r*) = -β r* / α
    - Stationary variance (σ>0): σ² / (2α)
    - PDMP when σ=0 (deterministic between shocks)
    """

    def __init__(self, params: Parameters):
        """
        Initialize trust dynamics.

        Args:
            params: Model parameters
        """
        self.alpha = params.alpha  # Mean reversion rate
        self.beta = params.beta  # Risk sensitivity
        self.sigma = params.sigma  # Volatility (0 for PDMP)
        self.dt = params.dt  # Time step

        # Validate stability (Constitution V)
        if self.sigma > 0:
            max_dt = min(0.01, self.sigma**2 / (10 * self.alpha))
            if self.dt > max_dt:
                raise ValueError(
                    f"Time step dt={self.dt} violates stability criterion. "
                    f"Maximum: {max_dt:.6f}"
                )

    def integrate_step(self, state: NetworkState, dt: float) -> None:
        """
        Integrate trust SDE for one time step using Euler-Maruyama.

        Algorithm:
            For each edge (i,j) ∈ E(t):
                1. Compute current risk Rᵢ(t)
                2. Compute drift: -α Yᵢⱼ - β Rᵢ
                3. Compute diffusion: σ dB ~ N(0, σ√dt)
                4. Update: Yᵢⱼ(t+dt) = Yᵢⱼ(t) + drift·dt + diffusion

        Args:
            state: Current network state
            dt: Time step

        Effects:
            Updates Y values in state for all edges
        """
        # Get all current edges
        edges = list(state.E)

        for (source, target) in edges:
            # Current logit-trust
            Y_current = state.Y[(source, target)]

            # Compute environmental risk for source actor
            R_i = state.get_environmental_risk(source)

            # Drift term: -α Y - β R
            drift = -self.alpha * Y_current - self.beta * R_i

            # Diffusion term: σ dB ~ N(0, σ√dt)
            if self.sigma > 0:
                dB = sample_normal(mean=0.0, std=np.sqrt(dt))
                diffusion = self.sigma * dB
            else:
                # PDMP mode: no stochastic term
                diffusion = 0.0

            # Euler-Maruyama update
            Y_new = Y_current + drift * dt + diffusion

            # Update state
            state.Y[(source, target)] = Y_new

    def compute_equilibrium_mean(self, risk: float) -> float:
        """
        Compute equilibrium mean for given risk level.

        Formula (Constitution II):
        μ_∞(r*) = -β r* / α

        Args:
            risk: Environmental risk level r* ∈ [0, 1]

        Returns:
            Equilibrium logit-trust
        """
        return -self.beta * risk / self.alpha

    def compute_stationary_variance(self) -> float:
        """
        Compute stationary variance of trust process.

        Formula (Constitution II):
        σ² / (2α)  for σ > 0

        Returns:
            Stationary variance (0 if σ=0)
        """
        if self.sigma == 0:
            return 0.0
        return self.sigma**2 / (2 * self.alpha)
