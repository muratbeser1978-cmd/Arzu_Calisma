"""
TFPA: Trust-Filtered Preferential Attachment growth mechanism.

Mathematical Foundation: Constitution II, TFPA Growth Mechanism
"""

import numpy as np
from typing import Optional

from ..core.state import NetworkState
from ..core.parameters import Parameters
from ..utils.numerical import jaccard_similarity, logit, clamp, EPSILON
from ..utils.random import get_rng, sample_categorical


class TFPAMechanism:
    """
    Trust-Filtered Preferential Attachment for network growth.

    Edge Formation Probability (Constitution II):
    πᵢ→ⱼ(t) ∝ (kⱼⁱⁿ(t) + 1)^γ_pa · 𝟙{W^init(i,j) ≥ w_min}

    Initial Trust (Jaccard Similarity):
    W^init(i,j) = |Nᵢ ∩ Nⱼ| / |Nᵢ ∪ Nⱼ|
    Yᵢⱼ^init = logit(W^init)

    Properties:
    - Preferential attachment: Favor high in-degree nodes
    - Trust filter: Require minimum common neighbors
    - Homophily: Similar actors more likely to connect
    """

    def __init__(self, params: Parameters):
        """
        Initialize TFPA mechanism.

        Args:
            params: Model parameters
        """
        self.w_min = params.w_min  # Minimum trust threshold
        self.gamma_pa = params.gamma_pa  # Preferential attachment strength
        self.Y_bar_0 = params.Y_bar_0  # Default logit-trust

    def compute_attachment_probabilities(
        self, source: int, state: NetworkState
    ) -> Optional[np.ndarray]:
        """
        Compute probabilities for target selection.

        Formula (Constitution II):
        πᵢ→ⱼ ∝ (kⱼⁱⁿ + 1)^γ_pa · 𝟙{W^init ≥ w_min}

        Args:
            source: Source actor ID
            state: Current network state

        Returns:
            Probability array for valid targets, or None if no valid targets
        """
        active_actors = state.get_active_actors()

        # Remove source and existing neighbors
        existing_neighbors = state.get_outgoing_neighbors(source)
        candidates = [j for j in active_actors if j != source and j not in existing_neighbors]

        if len(candidates) == 0:
            return None, None

        # Compute unnormalized probabilities
        unnormalized_probs = []

        for target in candidates:
            # Preferential attachment component: (k_in + 1)^γ
            k_in = len(state.get_incoming_neighbors(target))
            pa_component = (k_in + 1) ** self.gamma_pa

            # Trust filter: Check if W^init ≥ w_min
            source_neighbors = state.get_all_neighbors(source)
            target_neighbors = state.get_all_neighbors(target)
            W_init = jaccard_similarity(source_neighbors, target_neighbors)

            if W_init >= self.w_min:
                trust_indicator = 1.0
            else:
                trust_indicator = 0.0

            # Combined probability
            prob = pa_component * trust_indicator
            unnormalized_probs.append(prob)

        unnormalized_probs = np.array(unnormalized_probs)

        # Check if any valid targets
        if unnormalized_probs.sum() == 0:
            return None, None

        # Normalize
        probabilities = unnormalized_probs / unnormalized_probs.sum()

        return candidates, probabilities

    def sample_new_edge(self, state: NetworkState) -> Optional[tuple]:
        """
        Sample new edge formation event.

        Algorithm:
            1. Select source i uniformly from V_act(t)
            2. Compute πᵢ→ⱼ for all valid targets j
            3. Sample target j ~ Categorical(π)
            4. Compute initial trust W^init(i,j)

        Returns:
            (source, target, W_init) or None if no valid edges
        """
        active_actors = list(state.get_active_actors())

        if len(active_actors) < 2:
            return None  # Need at least 2 actors

        # Select source uniformly
        source = get_rng().choice(active_actors)

        # Compute target probabilities
        candidates, probabilities = self.compute_attachment_probabilities(source, state)

        if candidates is None:
            return None  # No valid targets

        # Sample target
        target_idx = sample_categorical(probabilities)
        target = candidates[target_idx]

        # Compute initial trust
        source_neighbors = state.get_all_neighbors(source)
        target_neighbors = state.get_all_neighbors(target)
        W_init = jaccard_similarity(source_neighbors, target_neighbors)

        # Handle edge case: no common neighbors
        if W_init == 0.0:
            W_init = EPSILON  # Use small value to avoid log(0)

        return (source, target, W_init)

    def get_initial_logit_trust(
        self, source: int, target: int, state: NetworkState
    ) -> float:
        """
        Compute initial logit-trust for new edge.

        Formula (Constitution II):
        W^init(i,j) = |Nᵢ ∩ Nⱼ| / |Nᵢ ∪ Nⱼ|
        Yᵢⱼ^init = logit(W^init)

        Args:
            source: Source actor ID
            target: Target actor ID
            state: Current network state

        Returns:
            Initial logit-trust Y_ij^init
        """
        source_neighbors = state.get_all_neighbors(source)
        target_neighbors = state.get_all_neighbors(target)
        W_init = jaccard_similarity(source_neighbors, target_neighbors)

        if W_init == 0.0:
            return self.Y_bar_0  # Default for no common neighbors

        # Clamp and convert to logit
        W_init = clamp(W_init, EPSILON, 1.0 - EPSILON)
        return logit(W_init)
