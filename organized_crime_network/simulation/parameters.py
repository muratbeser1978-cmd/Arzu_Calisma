"""
Simulation parameter validation and management.

This module provides the SimulationParameters dataclass with comprehensive
validation against constitutional domain constraints.

Constitutional Reference: Section I (Mathematical Fidelity)
"""

from dataclasses import dataclass
from typing import List
import numpy as np


@dataclass(frozen=True)
class SimulationParameters:
    """
    Validated container for all simulation parameters.

    All parameters follow exact specifications from constitution
    (.specify/memory/constitution.md Section I).

    Attributes
    ----------
    Arrest Process:
        lambda_0 : float
            Base arrest rate (> 0)
        kappa : float
            Hierarchical protection factor (≥ 0)
        gamma : float
            Operational risk multiplier (≥ 0)
        P_0 : float
            Baseline effectiveness (> 0)
        rho : float
            Effectiveness decay rate (> 0)
        eta_P : float
            Effectiveness per informant (> 0)

    Trust Dynamics:
        alpha : float
            Mean reversion rate (> 0)
        beta : float
            Risk sensitivity (> 0)
        sigma : float
            Volatility (≥ 0, 0 = PDMP)
        Delta : float
            Memory window (> 0)
        delta_R : float
            Risk decay rate (≈ 1/Delta)

    Informant Conversion:
        mu_LH : float
            External pressure rate (> 0)
        mu_min : float
            Minimum fragility (> 0)
        mu_rng : float
            Fragility range (> 0)
        theta : float
            Trust sensitivity (> 0)
        Y_bar_0 : float
            Default logit-trust for isolated actors

    TFPA Growth:
        w_min : float
            Minimum trust threshold [0, 1)
        gamma_pa : float
            Preferential attachment strength (≥ 1)

    Simulation Control:
        T_max : float
            Time horizon (> 0)
        dt : float
            SDE time step (> 0)
        random_seed : int
            For reproducibility
    """

    # Arrest Process
    lambda_0: float
    kappa: float
    gamma: float
    P_0: float
    rho: float
    eta_P: float

    # Trust Dynamics
    alpha: float
    beta: float
    sigma: float
    Delta: float
    delta_R: float

    # Informant Conversion
    mu_LH: float
    mu_min: float
    mu_rng: float
    theta: float
    Y_bar_0: float

    # TFPA Growth
    w_min: float
    gamma_pa: float

    # Simulation Control
    T_max: float
    dt: float
    random_seed: int

    def __post_init__(self):
        """Validate parameters upon creation."""
        errors = validate_parameters(self)
        if errors:
            raise ParameterValidationError(
                f"Invalid parameters:\n" + "\n".join(f"  - {e}" for e in errors)
            )

    @staticmethod
    def default() -> 'SimulationParameters':
        """
        Create parameters with default values from constitution.

        These defaults are designed for testing and demonstration purposes.
        Real applications should tune parameters based on empirical data.

        Returns
        -------
        SimulationParameters
            Parameters with constitution-specified defaults

        Examples
        --------
        >>> params = SimulationParameters.default()
        >>> params.lambda_0
        0.1
        >>> params.alpha
        0.5
        """
        return SimulationParameters(
            # Arrest Process - moderate enforcement
            lambda_0=0.1,      # Base arrest rate
            kappa=0.5,         # Hierarchical protection
            gamma=0.3,         # Operational risk multiplier
            P_0=1.0,           # Baseline effectiveness
            rho=0.2,           # Effectiveness decay
            eta_P=0.5,         # Effectiveness per informant

            # Trust Dynamics - moderate volatility
            alpha=0.5,         # Mean reversion rate
            beta=1.0,          # Risk sensitivity
            sigma=0.1,         # Volatility
            Delta=10.0,        # Memory window
            delta_R=0.1,       # Risk decay (1/Delta)

            # Informant Conversion - moderate fragility
            mu_LH=0.2,         # External pressure
            mu_min=0.01,       # Minimum fragility
            mu_rng=0.5,        # Fragility range
            theta=2.0,         # Trust sensitivity
            Y_bar_0=0.0,       # Default logit-trust

            # TFPA Growth - moderate trust threshold
            w_min=0.3,         # Minimum trust threshold
            gamma_pa=1.5,      # Preferential attachment

            # Simulation Control
            T_max=100.0,       # Time horizon
            dt=0.01,           # Time step
            random_seed=42     # Default seed
        )

    @staticmethod
    def aggressive_enforcement() -> 'SimulationParameters':
        """
        Aggressive law enforcement strategy preset.

        Characteristics:
        - High arrest rate (3x base)
        - Strong effectiveness boost from informants (2x)
        - Slower effectiveness decay
        - Higher trust sensitivity to risk

        Use Case: Testing impact of intensive enforcement

        Returns
        -------
        SimulationParameters
            Aggressive enforcement parameters

        Examples
        --------
        >>> params = SimulationParameters.aggressive_enforcement()
        >>> params.lambda_0
        0.3
        >>> params.eta_P
        1.0
        """
        return SimulationParameters(
            # Arrest Process - aggressive enforcement
            lambda_0=0.3,      # 3x base arrest rate
            kappa=0.5,         # Same hierarchical protection
            gamma=0.5,         # Higher operational risk impact
            P_0=1.0,           # Baseline effectiveness
            rho=0.1,           # Slower decay (2x slower)
            eta_P=1.0,         # 2x effectiveness boost per informant

            # Trust Dynamics - high sensitivity
            alpha=0.5,         # Mean reversion rate
            beta=1.5,          # Higher risk sensitivity
            sigma=0.1,         # Volatility
            Delta=10.0,        # Memory window
            delta_R=0.1,       # Risk decay

            # Informant Conversion - moderate fragility
            mu_LH=0.3,         # Higher external pressure
            mu_min=0.01,       # Minimum fragility
            mu_rng=0.5,        # Fragility range
            theta=2.0,         # Trust sensitivity
            Y_bar_0=0.0,       # Default logit-trust

            # TFPA Growth
            w_min=0.3,         # Trust threshold
            gamma_pa=1.5,      # Preferential attachment

            # Simulation Control
            T_max=100.0,
            dt=0.01,
            random_seed=42
        )

    @staticmethod
    def conservative_enforcement() -> 'SimulationParameters':
        """
        Conservative law enforcement strategy preset.

        Characteristics:
        - Low arrest rate (0.5x base)
        - Moderate effectiveness boost
        - Faster effectiveness decay
        - Lower trust sensitivity

        Use Case: Testing minimal intervention scenarios

        Returns
        -------
        SimulationParameters
            Conservative enforcement parameters

        Examples
        --------
        >>> params = SimulationParameters.conservative_enforcement()
        >>> params.lambda_0
        0.05
        >>> params.rho
        0.3
        """
        return SimulationParameters(
            # Arrest Process - conservative enforcement
            lambda_0=0.05,     # Half base arrest rate
            kappa=0.5,         # Hierarchical protection
            gamma=0.2,         # Lower operational risk impact
            P_0=1.0,           # Baseline effectiveness
            rho=0.3,           # Faster decay
            eta_P=0.3,         # Lower effectiveness boost

            # Trust Dynamics - low sensitivity
            alpha=0.5,         # Mean reversion rate
            beta=0.7,          # Lower risk sensitivity
            sigma=0.1,         # Volatility
            Delta=10.0,        # Memory window
            delta_R=0.1,       # Risk decay

            # Informant Conversion - lower pressure
            mu_LH=0.1,         # Lower external pressure
            mu_min=0.01,       # Minimum fragility
            mu_rng=0.5,        # Fragility range
            theta=2.0,         # Trust sensitivity
            Y_bar_0=0.0,       # Default logit-trust

            # TFPA Growth
            w_min=0.3,         # Trust threshold
            gamma_pa=1.5,      # Preferential attachment

            # Simulation Control
            T_max=100.0,
            dt=0.01,
            random_seed=42
        )

    @staticmethod
    def balanced_strategy() -> 'SimulationParameters':
        """
        Balanced enforcement strategy preset.

        Characteristics:
        - Moderate arrest rate
        - Balanced effectiveness dynamics
        - Standard trust sensitivity
        - Focus on sustainable long-term impact

        Use Case: Realistic baseline for comparative analysis

        Returns
        -------
        SimulationParameters
            Balanced strategy parameters

        Examples
        --------
        >>> params = SimulationParameters.balanced_strategy()
        >>> params.lambda_0
        0.15
        """
        return SimulationParameters(
            # Arrest Process - balanced enforcement
            lambda_0=0.15,     # Moderate arrest rate
            kappa=0.5,         # Hierarchical protection
            gamma=0.3,         # Operational risk multiplier
            P_0=1.0,           # Baseline effectiveness
            rho=0.15,          # Balanced decay
            eta_P=0.6,         # Moderate effectiveness boost

            # Trust Dynamics - balanced sensitivity
            alpha=0.5,         # Mean reversion rate
            beta=1.0,          # Standard risk sensitivity
            sigma=0.1,         # Volatility
            Delta=10.0,        # Memory window
            delta_R=0.1,       # Risk decay

            # Informant Conversion - balanced pressure
            mu_LH=0.25,        # Moderate external pressure
            mu_min=0.01,       # Minimum fragility
            mu_rng=0.5,        # Fragility range
            theta=2.0,         # Trust sensitivity
            Y_bar_0=0.0,       # Default logit-trust

            # TFPA Growth
            w_min=0.3,         # Trust threshold
            gamma_pa=1.5,      # Preferential attachment

            # Simulation Control
            T_max=100.0,
            dt=0.01,
            random_seed=42
        )

    @staticmethod
    def high_resilience_network() -> 'SimulationParameters':
        """
        Network with high resilience to enforcement preset.

        Characteristics:
        - Strong hierarchical protection
        - High trust baseline
        - Low fragility to conversion
        - Slow trust decay

        Use Case: Testing enforcement against resilient networks

        Returns
        -------
        SimulationParameters
            High resilience network parameters

        Examples
        --------
        >>> params = SimulationParameters.high_resilience_network()
        >>> params.kappa
        1.0
        >>> params.theta
        3.0
        """
        return SimulationParameters(
            # Arrest Process - strong protection
            lambda_0=0.1,      # Base arrest rate
            kappa=1.0,         # Strong hierarchical protection (2x)
            gamma=0.2,         # Lower operational risk
            P_0=1.0,           # Baseline effectiveness
            rho=0.2,           # Effectiveness decay
            eta_P=0.3,         # Lower informant impact

            # Trust Dynamics - high resilience
            alpha=0.3,         # Slower mean reversion
            beta=0.7,          # Lower risk sensitivity
            sigma=0.1,         # Volatility
            Delta=15.0,        # Longer memory window
            delta_R=0.067,     # Risk decay (1/Delta)

            # Informant Conversion - high loyalty
            mu_LH=0.1,         # Lower external pressure
            mu_min=0.005,      # Lower minimum fragility
            mu_rng=0.3,        # Smaller fragility range
            theta=3.0,         # Higher trust sensitivity (harder to convert high-trust)
            Y_bar_0=0.0,       # Default logit-trust

            # TFPA Growth
            w_min=0.4,         # Higher trust threshold
            gamma_pa=1.5,      # Preferential attachment

            # Simulation Control
            T_max=100.0,
            dt=0.01,
            random_seed=42
        )

    def get_parameter_description(self) -> str:
        """
        Get human-readable description of parameter configuration.

        Returns
        -------
        str
            Formatted parameter description

        Examples
        --------
        >>> params = SimulationParameters.default()
        >>> print(params.get_parameter_description())
        """
        return f"""Simulation Parameters Configuration
{'=' * 50}

ARREST PROCESS:
  Base arrest rate (lambda_0):        {self.lambda_0:.3f}
  Hierarchical protection (kappa):    {self.kappa:.3f}
  Operational risk multiplier (gamma):{self.gamma:.3f}
  Baseline effectiveness (P_0):       {self.P_0:.3f}
  Effectiveness decay rate (rho):     {self.rho:.3f}
  Effectiveness per informant (eta_P):{self.eta_P:.3f}

TRUST DYNAMICS:
  Mean reversion rate (alpha):        {self.alpha:.3f}
  Risk sensitivity (beta):            {self.beta:.3f}
  Volatility (sigma):                 {self.sigma:.3f} {'(PDMP)' if self.sigma == 0 else '(SDE)'}
  Memory window (Delta):              {self.Delta:.1f}
  Risk decay rate (delta_R):          {self.delta_R:.3f}

INFORMANT CONVERSION:
  External pressure rate (mu_LH):     {self.mu_LH:.3f}
  Minimum fragility (mu_min):         {self.mu_min:.3f}
  Fragility range (mu_rng):           {self.mu_rng:.3f}
  Trust sensitivity (theta):          {self.theta:.3f}
  Default logit-trust (Y_bar_0):      {self.Y_bar_0:.3f}

NETWORK GROWTH:
  Minimum trust threshold (w_min):    {self.w_min:.3f}
  Preferential attachment (gamma_pa): {self.gamma_pa:.3f}

SIMULATION CONTROL:
  Time horizon (T_max):               {self.T_max:.1f}
  Time step (dt):                     {self.dt:.4f}
  Random seed:                        {self.random_seed}
  Total steps:                        {int(self.T_max / self.dt):,}

STABILITY:
  SDE stability limit:                {2 * self.alpha / (self.alpha**2 + self.sigma**2):.4f}
  Safety margin:                      {2 * self.alpha / (self.alpha**2 + self.sigma**2) / self.dt:.1f}x
"""


def validate_parameters(params: SimulationParameters) -> List[str]:
    """
    Validate all parameters against constitutional domain constraints.

    Checks all positivity, non-negativity, range, and stability conditions
    specified in constitution Section I.

    Parameters
    ----------
    params : SimulationParameters
        Parameters to validate

    Returns
    -------
    errors : List[str]
        Empty if valid, otherwise list of error messages

    Constitutional Reference
    ------------------------
    Section I: Parameter Domains
    Section V: Numerical Accuracy Standards

    Examples
    --------
    >>> params = SimulationParameters.default()
    >>> errors = validate_parameters(params)
    >>> len(errors)
    0

    >>> bad_params = SimulationParameters(lambda_0=-0.1, ...)
    >>> errors = validate_parameters(bad_params)
    >>> 'lambda_0 must be > 0' in errors
    True
    """
    errors = []

    # Arrest Process - positivity constraints
    if params.lambda_0 <= 0:
        errors.append("lambda_0 (base arrest rate) must be > 0")
    if params.P_0 <= 0:
        errors.append("P_0 (baseline effectiveness) must be > 0")
    if params.rho <= 0:
        errors.append("rho (effectiveness decay) must be > 0")
    if params.eta_P <= 0:
        errors.append("eta_P (effectiveness per informant) must be > 0")

    # Arrest Process - non-negativity
    if params.kappa < 0:
        errors.append("kappa (hierarchical protection) must be ≥ 0")
    if params.gamma < 0:
        errors.append("gamma (operational risk multiplier) must be ≥ 0")

    # Trust Dynamics - positivity constraints
    if params.alpha <= 0:
        errors.append("alpha (mean reversion rate) must be > 0")
    if params.beta <= 0:
        errors.append("beta (risk sensitivity) must be > 0")
    if params.Delta <= 0:
        errors.append("Delta (memory window) must be > 0")
    if params.delta_R <= 0:
        errors.append("delta_R (risk decay rate) must be > 0")

    # Trust Dynamics - non-negativity
    if params.sigma < 0:
        errors.append("sigma (volatility) must be ≥ 0")

    # Trust Dynamics - consistency check
    expected_delta_R = 1.0 / params.Delta
    if abs(params.delta_R - expected_delta_R) > 0.01:
        errors.append(
            f"delta_R should be approximately 1/Delta: "
            f"expected {expected_delta_R:.3f}, got {params.delta_R:.3f}"
        )

    # Trust Dynamics - numerical stability criterion
    # From research.md Decision 1: dt < 2α/(α² + σ²)
    if params.sigma > 0:
        dt_stable = 2 * params.alpha / (params.alpha**2 + params.sigma**2)
        # Use safety factor of 0.5
        if params.dt >= 0.5 * dt_stable:
            errors.append(
                f"dt too large for numerical stability: "
                f"dt = {params.dt:.4f} >= 0.5 * {dt_stable:.4f} (stability limit)"
            )

    # Informant Conversion - positivity constraints
    if params.mu_LH <= 0:
        errors.append("mu_LH (external pressure rate) must be > 0")
    if params.mu_min <= 0:
        errors.append("mu_min (minimum fragility) must be > 0")
    if params.mu_rng <= 0:
        errors.append("mu_rng (fragility range) must be > 0")
    if params.theta <= 0:
        errors.append("theta (trust sensitivity) must be > 0")

    # TFPA Growth - range constraints
    if not (0 <= params.w_min < 1):
        errors.append(f"w_min (trust threshold) must be in [0, 1), got {params.w_min}")
    if params.gamma_pa < 1:
        errors.append(f"gamma_pa (preferential attachment) must be ≥ 1, got {params.gamma_pa}")

    # Simulation Control - positivity
    if params.T_max <= 0:
        errors.append("T_max (time horizon) must be > 0")
    if params.dt <= 0:
        errors.append("dt (time step) must be > 0")

    # Simulation Control - practical limits
    num_steps = int(params.T_max / params.dt)
    if num_steps > 1_000_000:
        errors.append(
            f"Too many time steps: T_max/dt = {num_steps:,} > 1,000,000 "
            f"(increase dt or decrease T_max)"
        )

    return errors


class ParameterValidationError(ValueError):
    """Raised when simulation parameters violate constitutional constraints."""
    pass
