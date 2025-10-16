"""
Stochastic process modules for simulation engine.

This package contains implementations of the four coupled stochastic processes:
    - TrustDynamics: Ornstein-Uhlenbeck SDE for trust evolution
    - ArrestProcess: Cox process for arrest events
    - ConversionProcess: CTMC for informant conversion
    - GrowthProcess: TFPA for network growth

Constitutional Reference: Section II (Stochastic Processes)
"""

from .trust import TrustDynamics, TrustState
from .arrests import ArrestProcess, ArrestIntensity
from .conversion import ConversionProcess, ActorLoyaltyState, LoyaltyState
from .growth import GrowthProcess

__all__ = [
    'TrustDynamics',
    'TrustState',
    'ArrestProcess',
    'ArrestIntensity',
    'ConversionProcess',
    'ActorLoyaltyState',
    'LoyaltyState',
    'GrowthProcess',
]
