"""
Random number generation management for reproducibility.

Ensures consistent random state across all stochastic processes.
"""

import numpy as np
from typing import Optional

# Global random number generator
_rng: Optional[np.random.Generator] = None


def set_random_seed(seed: int) -> None:
    """
    Set random seed for reproducibility.

    Args:
        seed: Integer seed for RNG initialization

    Effects:
        - Initializes global numpy Generator with specified seed
        - Ensures bit-identical results across runs with same seed
        - Required for FR-009: reproducibility guarantee

    Examples:
        >>> set_random_seed(42)
        >>> get_rng().random()  # Deterministic output
        0.7739560485559633
    """
    global _rng
    _rng = np.random.Generator(np.random.PCG64(seed))


def get_rng() -> np.random.Generator:
    """
    Get global random number generator.

    Returns:
        Initialized numpy Generator

    Raises:
        RuntimeError: If RNG not initialized (seed not set)

    Usage:
        All stochastic processes must use this RNG, not np.random directly,
        to ensure reproducibility.

    Examples:
        >>> set_random_seed(42)
        >>> rng = get_rng()
        >>> rng.exponential(scale=1.0)
        0.38350864613166064
    """
    if _rng is None:
        raise RuntimeError(
            "Random number generator not initialized. "
            "Call set_random_seed(seed) before using stochastic processes."
        )
    return _rng


def sample_exponential(rate: float) -> float:
    """
    Sample from exponential distribution.

    Args:
        rate: Rate parameter λ > 0

    Returns:
        Sample from Exp(λ) distribution

    Used for:
        - Waiting times in Cox processes (arrests)
        - CTMC transition times (informant conversion)

    Examples:
        >>> set_random_seed(42)
        >>> sample_exponential(1.0)
        0.38350864613166064
    """
    if rate <= 0:
        raise ValueError(f"Rate must be positive, got {rate}")
    return get_rng().exponential(scale=1.0 / rate)


def sample_normal(mean: float = 0.0, std: float = 1.0) -> float:
    """
    Sample from normal distribution.

    Args:
        mean: Mean of distribution
        std: Standard deviation (must be non-negative)

    Returns:
        Sample from N(mean, std²)

    Used for:
        - Wiener process increments dB = N(0, √dt)
        - Trust SDE stochastic term

    Examples:
        >>> set_random_seed(42)
        >>> sample_normal(0.0, 1.0)
        0.30471707975443135
    """
    if std < 0:
        raise ValueError(f"Standard deviation must be non-negative, got {std}")
    return get_rng().normal(loc=mean, scale=std)


def sample_categorical(probabilities: np.ndarray) -> int:
    """
    Sample from categorical distribution.

    Args:
        probabilities: Array of probabilities (must sum to 1)

    Returns:
        Index of selected category

    Used for:
        - TFPA target selection with π_i→j probabilities
        - Discrete choice among alternatives

    Examples:
        >>> set_random_seed(42)
        >>> sample_categorical(np.array([0.2, 0.3, 0.5]))
        2
    """
    # Validate probabilities
    if not np.isclose(probabilities.sum(), 1.0, atol=1e-6):
        raise ValueError(
            f"Probabilities must sum to 1, got sum={probabilities.sum():.6f}. "
            "See constitution Section V for numerical precision requirements."
        )

    if np.any(probabilities < 0):
        raise ValueError("Probabilities must be non-negative")

    return get_rng().choice(len(probabilities), p=probabilities)
