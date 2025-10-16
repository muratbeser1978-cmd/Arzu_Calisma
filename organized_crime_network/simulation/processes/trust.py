"""
Trust dynamics simulation using Ornstein-Uhlenbeck SDE.

This module implements the Switching Ornstein-Uhlenbeck process for
trust evolution with environmental risk feedback.

Process: dY_ij(t) = (-α Y_ij(t) - β R_i(t))dt + σ dB_ij(t)

Constitutional Reference: Section II (Trust Dynamics)
Research Decision: Decision 1 (SDE Numerical Method)
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Dict
import numpy as np


@dataclass
class TrustState:
    """
    Manage trust values and their dynamics.

    Attributes
    ----------
    Y : np.ndarray
        Logit-trust matrix (n_actors × n_actors)
    last_update : float
        Last time Y was updated
    risk_values : np.ndarray
        R_i(t) for each actor
    exposure_history : Dict[int, List[Tuple[float, int]]]
        Memory window tracking: actor_id → [(time, exposed_neighbor_id), ...]
    """
    Y: np.ndarray
    last_update: float
    risk_values: np.ndarray
    exposure_history: Dict[int, List[Tuple[float, int]]] = field(default_factory=dict)


class TrustDynamics:
    """
    Trust dynamics simulator using Euler-Maruyama or exact PDMP.

    Process: dY_ij(t) = (-α Y_ij(t) - β R_i(t))dt + σ dB_ij(t)

    For σ > 0: Uses Euler-Maruyama discretization
    For σ = 0: Uses exact PDMP solution between jumps

    Constitutional Reference: Section II (Trust Dynamics)
    Research Decision: Decision 1 (SDE Numerical Method)

    Parameters
    ----------
    alpha : float
        Mean reversion rate (> 0)
    beta : float
        Risk sensitivity (> 0)
    sigma : float
        Volatility (≥ 0, 0 = PDMP)
    dt : float
        Time step for discretization
    rng : np.random.Generator
        Random number generator for reproducibility

    Examples
    --------
    >>> rng = np.random.default_rng(42)
    >>> dynamics = TrustDynamics(alpha=0.5, beta=1.0, sigma=0.1, dt=0.01, rng=rng)
    >>> Y_current = np.zeros((10, 10))
    >>> risk_values = np.ones(10) * 0.5
    >>> edges = [(i, i+1) for i in range(9)]
    >>> Y_next = dynamics.step(Y_current, risk_values, edges)
    >>> Y_next.shape
    (10, 10)
    """

    def __init__(
        self,
        alpha: float,
        beta: float,
        sigma: float,
        dt: float,
        rng: np.random.Generator
    ):
        """
        Initialize trust dynamics simulator.

        Parameters
        ----------
        alpha : float
            Mean reversion rate (> 0)
        beta : float
            Risk sensitivity (> 0)
        sigma : float
            Volatility (≥ 0, 0 = PDMP)
        dt : float
            Time step for discretization
        rng : np.random.Generator
            Random number generator for reproducibility
        """
        if alpha <= 0:
            raise ValueError("alpha must be > 0")
        if beta <= 0:
            raise ValueError("beta must be > 0")
        if sigma < 0:
            raise ValueError("sigma must be ≥ 0")
        if dt <= 0:
            raise ValueError("dt must be > 0")

        self.alpha = alpha
        self.beta = beta
        self.sigma = sigma
        self.dt = dt
        self.rng = rng
        self.is_pdmp = (sigma == 0.0)

    def step(
        self,
        Y_current: np.ndarray,
        risk_values: np.ndarray,
        active_edges: List[Tuple[int, int]]
    ) -> np.ndarray:
        """
        Perform one Euler-Maruyama step or PDMP update.

        Parameters
        ----------
        Y_current : ndarray, shape (n_actors, n_actors)
            Current logit-trust values
        risk_values : ndarray, shape (n_actors,)
            Environmental risk R_i(t) for each actor
        active_edges : list of (int, int)
            Edges to update (only between active actors)

        Returns
        -------
        Y_next : ndarray
            Updated trust values after dt

        Algorithm
        ---------
        For σ > 0 (Euler-Maruyama):
            drift = (-α * Y - β * R_broadcast) * dt
            diffusion = σ * √dt * dW  (dW ~ N(0,1))
            Y_next = Y + drift + diffusion

        For σ = 0 (PDMP):
            Y_next = Y * exp(-α*dt) - (β*R/α) * (1 - exp(-α*dt))

        Validation
        ----------
        - All Y_ij remain in reasonable range [-10, 10]
        - No NaN/Inf values generated

        Examples
        --------
        >>> dynamics = TrustDynamics(0.5, 1.0, 0.1, 0.01, rng)
        >>> Y = np.zeros((5, 5))
        >>> R = np.array([0.0, 0.1, 0.2, 0.0, 0.1])
        >>> edges = [(0, 1), (1, 2), (2, 3)]
        >>> Y_new = dynamics.step(Y, R, edges)
        """
        Y_next = Y_current.copy()

        if not active_edges:
            return Y_next

        if self.is_pdmp:
            # Exact PDMP solution
            exp_term = np.exp(-self.alpha * self.dt)
            for i, j in active_edges:
                Y_next[i, j] = (
                    Y_current[i, j] * exp_term
                    - (self.beta * risk_values[i] / self.alpha) * (1 - exp_term)
                )
        else:
            # Euler-Maruyama for σ > 0
            sqrt_dt = np.sqrt(self.dt)

            for i, j in active_edges:
                # Drift term
                drift = (-self.alpha * Y_current[i, j] - self.beta * risk_values[i]) * self.dt

                # Diffusion term
                dW = self.rng.standard_normal()
                diffusion = self.sigma * sqrt_dt * dW

                # Update
                Y_next[i, j] = Y_current[i, j] + drift + diffusion

        # Clamp to reasonable range to prevent numerical overflow
        Y_next = np.clip(Y_next, -10.0, 10.0)

        # Validate no NaN/Inf
        if not np.all(np.isfinite(Y_next)):
            raise ValueError("NaN or Inf detected in trust update")

        return Y_next

    def compute_stationary_distribution(self, R: float) -> Tuple[float, float]:
        """
        Compute theoretical stationary distribution for constant R.

        For constant environmental risk R, the OU process has stationary
        distribution N(μ, σ²) where:
            μ = -βR/α
            σ² = σ²/(2α)

        Parameters
        ----------
        R : float
            Constant risk value

        Returns
        -------
        mean : float
            E[Y_∞] = -βR/α
        variance : float
            Var[Y_∞] = σ²/(2α)

        Use Cases
        ---------
        Validation tests to verify SDE implementation

        Examples
        --------
        >>> dynamics = TrustDynamics(alpha=1.0, beta=1.0, sigma=0.5, dt=0.01, rng=rng)
        >>> mean, var = dynamics.compute_stationary_distribution(R=0.5)
        >>> mean
        -0.5
        >>> var
        0.125
        """
        mean = -self.beta * R / self.alpha
        variance = self.sigma**2 / (2 * self.alpha) if not self.is_pdmp else 0.0
        return mean, variance
