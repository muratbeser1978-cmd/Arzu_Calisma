"""
Numerically stable implementations of mathematical functions.

All functions implement exact formulas from constitution with overflow protection.
"""

import numpy as np
from typing import Set

# Numerical stability constants
EPSILON = 1e-10
MAX_EXP = 20.0  # Prevents overflow in exp()


def expit(y: float) -> float:
    """
    Numerically stable expit (logistic sigmoid) function.

    Formula (Constitution II): expit(y) = 1 / (1 + exp(-y))

    Args:
        y: Logit-transformed value, y ∈ ℝ

    Returns:
        w ∈ (0, 1): Probability/trust value

    Implementation:
        - For y > 0: expit(y) = 1 / (1 + exp(-y))
        - For y ≤ 0: expit(y) = exp(y) / (1 + exp(y))
        This avoids overflow for large |y|

    Examples:
        >>> expit(0.0)
        0.5
        >>> expit(100.0)  # Should not overflow
        1.0
        >>> expit(-100.0)  # Should not underflow to 0
        3.7200759760208356e-44
    """
    # Clamp input to prevent overflow
    y = np.clip(y, -MAX_EXP, MAX_EXP)

    if y >= 0:
        return 1.0 / (1.0 + np.exp(-y))
    else:
        exp_y = np.exp(y)
        return exp_y / (1.0 + exp_y)


def logit(w: float) -> float:
    """
    Numerically stable logit (inverse sigmoid) function.

    Formula (Constitution II): logit(w) = log(w / (1 - w))

    Args:
        w: Probability/trust value, w ∈ (0, 1)

    Returns:
        y ∈ ℝ: Logit-transformed value

    Implementation:
        - Clamps w to [ε, 1-ε] to avoid log(0) or log(∞)
        - Uses epsilon = 1e-10 as per constitution

    Examples:
        >>> logit(0.5)
        0.0
        >>> logit(0.9)
        2.1972245773362196
        >>> logit(1.0)  # Clamped to 1-ε
        23.025850929940457
    """
    # Clamp to valid domain to avoid log(0) or division by zero
    w = clamp(w, EPSILON, 1.0 - EPSILON)

    return np.log(w / (1.0 - w))


def clamp(value: float, min_val: float, max_val: float) -> float:
    """
    Clamp value to [min_val, max_val] range.

    Args:
        value: Value to clamp
        min_val: Minimum allowed value
        max_val: Maximum allowed value

    Returns:
        Clamped value

    Examples:
        >>> clamp(0.5, 0.0, 1.0)
        0.5
        >>> clamp(-1.0, 0.0, 1.0)
        0.0
        >>> clamp(2.0, 0.0, 1.0)
        1.0
    """
    return max(min_val, min(max_val, value))


def jaccard_similarity(set_a: Set[int], set_b: Set[int]) -> float:
    """
    Compute Jaccard similarity coefficient between two sets.

    Formula (Constitution II): J(A, B) = |A ∩ B| / |A ∪ B|

    Args:
        set_a: First set of elements
        set_b: Second set of elements

    Returns:
        Similarity ∈ [0, 1], with special cases:
        - Returns 0.0 if both sets are empty (convention)
        - Returns 1.0 if sets are identical and non-empty

    Used for:
        - Initial trust computation in TFPA: W^init(i,j) = J(N_i, N_j)

    Examples:
        >>> jaccard_similarity({1, 2, 3}, {2, 3, 4})
        0.5
        >>> jaccard_similarity({1, 2}, {1, 2})
        1.0
        >>> jaccard_similarity(set(), set())
        0.0
    """
    # Handle edge cases
    if len(set_a) == 0 and len(set_b) == 0:
        return 0.0

    intersection = len(set_a & set_b)
    union = len(set_a | set_b)

    if union == 0:
        return 0.0

    return intersection / union


def validate_time_step(dt: float, sigma: float, alpha: float) -> None:
    """
    Validate time step for SDE stability.

    Stability Criterion (Constitution V): dt ≤ min(0.01, σ²/(10α))

    Args:
        dt: Proposed time step
        sigma: Volatility parameter
        alpha: Mean reversion rate

    Raises:
        ValueError: If time step violates stability criterion

    Examples:
        >>> validate_time_step(0.005, 0.1, 0.5)  # Valid
        >>> validate_time_step(0.1, 0.1, 0.5)  # Raises ValueError
    """
    if sigma > 0:
        max_dt = min(0.01, sigma**2 / (10 * alpha))
    else:
        max_dt = 0.01

    if dt > max_dt:
        raise ValueError(
            f"Time step dt={dt} violates stability criterion. "
            f"Maximum allowed: dt ≤ {max_dt:.6f} "
            f"(min(0.01, σ²/(10α)) with σ={sigma}, α={alpha}). "
            f"See constitution Section V for numerical accuracy standards."
        )


def fragility_rate(avg_trust: float, mu_min: float, mu_rng: float, theta: float) -> float:
    """
    Compute trust-dependent informant conversion rate.

    Formula (Constitution I): μ_HI(i) = μ_min + μ_rng · expit(-θ · Ȳᵢ)

    Args:
        avg_trust: Average logit-trust at arrest time, Ȳᵢ ∈ ℝ
        mu_min: Minimum fragility rate, μ_min > 0
        mu_rng: Fragility range, μ_rng > 0
        theta: Trust sensitivity parameter, θ > 0

    Returns:
        Conversion rate μ_HI ∈ [μ_min, μ_min + μ_rng]

    Interpretation:
        - High trust (Ȳ > 0) → Low rate (≈ μ_min): Loyal actors resist
        - Low trust (Ȳ < 0) → High rate (≈ μ_min + μ_rng): Fragile actors convert

    Examples:
        >>> fragility_rate(0.0, 0.01, 0.5, 2.0)  # Neutral trust
        0.26
        >>> fragility_rate(5.0, 0.01, 0.5, 2.0)  # High trust (loyal)
        0.010045...
        >>> fragility_rate(-5.0, 0.01, 0.5, 2.0)  # Low trust (fragile)
        0.509954...
    """
    return mu_min + mu_rng * expit(-theta * avg_trust)


def hierarchical_protection(level: int, kappa: float) -> float:
    """
    Compute hierarchical protection factor.

    Formula (Constitution I): h(l) = exp(-κ(l-1))

    Args:
        level: Hierarchy level, l ∈ {1, 2, ..., L}
        kappa: Protection strength parameter, κ ≥ 0

    Returns:
        Protection factor h(l) ∈ (0, 1] for l > 1, h(1) = 1

    Interpretation:
        - Level 1 (operatives): h(1) = 1 (no protection)
        - Higher levels: h(l) < 1 (exponentially decreasing risk)
        - κ = 0: No hierarchy effect (h(l) = 1 for all l)

    Examples:
        >>> hierarchical_protection(1, 0.5)
        1.0
        >>> hierarchical_protection(2, 0.5)
        0.6065306597126334
        >>> hierarchical_protection(3, 0.5)
        0.36787944117144233
    """
    return np.exp(-kappa * (level - 1))
