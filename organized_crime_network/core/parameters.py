"""
Parameter validation and management.

All parameters must satisfy constitutional domains (Constitution I).
"""

from dataclasses import dataclass, asdict
from typing import Dict, Any
import json


@dataclass
class Parameters:
    """
    Complete parameter set for OCN stochastic model.

    All parameters validated against constitutional domains.
    See .specify/memory/constitution.md Section I for definitions.
    """

    # Arrest Process Parameters (Constitution I)
    lambda_0: float = 0.1  # Base arrest risk, λ₀ > 0
    kappa: float = 0.5  # Hierarchical protection, κ ≥ 0
    gamma: float = 0.3  # Operational risk multiplier, γ ≥ 0
    P_0: float = 1.0  # Baseline law enforcement effectiveness, P₀ > 0
    rho: float = 0.2  # Effectiveness decay rate, ρ > 0
    eta_P: float = 0.5  # Effectiveness per informant, η_P > 0

    # Trust Dynamics Parameters (Constitution I)
    alpha: float = 0.5  # Mean reversion rate, α > 0
    beta: float = 1.0  # Risk sensitivity, β > 0
    sigma: float = 0.1  # Volatility, σ ≥ 0 (σ=0 gives PDMP)
    Delta: float = 10.0  # Memory window, Δ > 0

    # Informant Conversion Parameters (Constitution I)
    mu_LH: float = 0.2  # External pressure rate, μ_LH > 0
    mu_min: float = 0.01  # Minimum fragility, μ_min > 0
    mu_rng: float = 0.5  # Fragility range, μ_rng > 0
    theta: float = 2.0  # Trust sensitivity, θ > 0
    Y_bar_0: float = 0.0  # Default logit-trust for isolated nodes

    # TFPA Mechanism Parameters (Constitution I)
    w_min: float = 0.1  # Minimum trust threshold, w_min ∈ [0,1)
    gamma_pa: float = 1.5  # Preferential attachment strength, γ_pa ≥ 1

    # Simulation Parameters
    T_max: float = 10.0  # Simulation horizon (reduced for testing)
    dt: float = 0.002  # Time step for SDE integration (satisfies stability)
    random_seed: int = 42  # For reproducibility

    def __post_init__(self):
        """Validate all parameters against constitutional domains."""
        self.validate()

    def validate(self) -> None:
        """
        Validate all parameters against constitutional domains.

        Raises:
            ValueError: If any parameter violates domain constraints

        Constitutional Reference: Section I - Mathematical Fidelity
        """
        errors = []

        # Arrest process validation
        if self.lambda_0 <= 0:
            errors.append(
                f"lambda_0 must be > 0 (base arrest risk), got {self.lambda_0}"
            )
        if self.kappa < 0:
            errors.append(
                f"kappa must be ≥ 0 (hierarchical protection), got {self.kappa}"
            )
        if self.gamma < 0:
            errors.append(
                f"gamma must be ≥ 0 (operational risk multiplier), got {self.gamma}"
            )
        if self.P_0 <= 0:
            errors.append(
                f"P_0 must be > 0 (baseline effectiveness), got {self.P_0}"
            )
        if self.rho <= 0:
            errors.append(
                f"rho must be > 0 (effectiveness decay rate), got {self.rho}"
            )
        if self.eta_P <= 0:
            errors.append(
                f"eta_P must be > 0 (effectiveness per informant), got {self.eta_P}"
            )

        # Trust dynamics validation
        if self.alpha <= 0:
            errors.append(
                f"alpha must be > 0 (mean reversion rate), got {self.alpha}"
            )
        if self.beta <= 0:
            errors.append(f"beta must be > 0 (risk sensitivity), got {self.beta}")
        if self.sigma < 0:
            errors.append(
                f"sigma must be ≥ 0 (volatility, σ=0 for PDMP), got {self.sigma}"
            )
        if self.Delta <= 0:
            errors.append(f"Delta must be > 0 (memory window), got {self.Delta}")

        # Informant conversion validation
        if self.mu_LH <= 0:
            errors.append(
                f"mu_LH must be > 0 (external pressure rate), got {self.mu_LH}"
            )
        if self.mu_min <= 0:
            errors.append(
                f"mu_min must be > 0 (minimum fragility), got {self.mu_min}"
            )
        if self.mu_rng <= 0:
            errors.append(
                f"mu_rng must be > 0 (fragility range), got {self.mu_rng}"
            )
        if self.theta <= 0:
            errors.append(
                f"theta must be > 0 (trust sensitivity), got {self.theta}"
            )

        # TFPA validation
        if not (0 <= self.w_min < 1):
            errors.append(
                f"w_min must be in [0,1) (trust threshold), got {self.w_min}"
            )
        if self.gamma_pa < 1:
            errors.append(
                f"gamma_pa must be ≥ 1 (preferential attachment), got {self.gamma_pa}"
            )

        # Simulation parameters validation
        if self.T_max <= 0:
            errors.append(f"T_max must be > 0 (simulation horizon), got {self.T_max}")
        if self.dt <= 0:
            errors.append(f"dt must be > 0 (time step), got {self.dt}")

        # Time step stability check (Constitution V)
        if self.sigma > 0:
            max_dt = min(0.01, self.sigma**2 / (10 * self.alpha))
            if self.dt > max_dt:
                errors.append(
                    f"dt={self.dt} violates stability criterion. "
                    f"Must satisfy dt ≤ min(0.01, σ²/(10α)) = {max_dt:.6f} "
                    f"with σ={self.sigma}, α={self.alpha}. "
                    f"See constitution Section V."
                )

        if errors:
            error_msg = "Parameter validation failed:\n" + "\n".join(
                f"  - {err}" for err in errors
            )
            error_msg += "\n\nSee .specify/memory/constitution.md Section I for parameter domains."
            raise ValueError(error_msg)

    def to_dict(self) -> Dict[str, Any]:
        """Convert parameters to dictionary."""
        return asdict(self)

    def to_json(self, filepath: str) -> None:
        """Save parameters to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Parameters":
        """Create parameters from dictionary."""
        return cls(**data)

    @classmethod
    def from_json(cls, filepath: str) -> "Parameters":
        """Load parameters from JSON file."""
        with open(filepath, "r") as f:
            data = json.load(f)
        return cls.from_dict(data)

    def __repr__(self) -> str:
        """String representation with parameter categories."""
        return (
            f"Parameters(\n"
            f"  Arrest: λ₀={self.lambda_0}, κ={self.kappa}, γ={self.gamma}, "
            f"P₀={self.P_0}, ρ={self.rho}, η_P={self.eta_P}\n"
            f"  Trust: α={self.alpha}, β={self.beta}, σ={self.sigma}, Δ={self.Delta}\n"
            f"  Conversion: μ_LH={self.mu_LH}, μ_min={self.mu_min}, μ_rng={self.mu_rng}, "
            f"θ={self.theta}\n"
            f"  TFPA: w_min={self.w_min}, γ_pa={self.gamma_pa}\n"
            f"  Simulation: T_max={self.T_max}, dt={self.dt}, seed={self.random_seed}\n"
            f")"
        )


def get_default_parameters() -> Parameters:
    """
    Get default parameter set from constitution.

    Returns validated Parameters object with values from
    Constitution "Testing Parameters" section.
    """
    return Parameters()  # Uses dataclass defaults
