"""
Network state management with invariant preservation.

State Space Definition (Constitution II):
- V(t) ⊆ V: Current members
- E(t) ⊆ V(t) × V(t): Operational edges
- A(t): Actor states ∈ {Active, Arrested, Informant}
- Y(t): Logit-trust matrix, Yᵢⱼ ∈ ℝ
"""

from typing import Set, Dict, Tuple, Optional, List
from enum import Enum
from collections import deque
import numpy as np

from ..utils.numerical import expit, logit, jaccard_similarity, clamp, EPSILON


class ActorState(Enum):
    """Actor states as defined in Constitution II."""

    ACTIVE = "Active"
    ARRESTED = "Arrested"
    INFORMANT = "Informant"


class NetworkState:
    """
    Complete network state at time t with invariant checking.

    Maintains:
    - V(t): Set of current members
    - E(t): Directed operational edges
    - A(t): Actor states
    - Y(t): Logit-trust matrix
    - P(t): Law enforcement effectiveness
    - Exposure history for risk computation

    Invariants (Constitution II):
    - E ⊆ V × V
    - Y keys = E
    - Edges only between Active actors
    - All trust values in valid domain
    """

    def __init__(self, t_current: float = 0.0, P_0: float = 1.0, Delta: float = 10.0):
        """
        Initialize empty network state.

        Args:
            t_current: Current simulation time
            P_0: Initial law enforcement effectiveness
            Delta: Memory window for exposure neighborhood
        """
        self.t_current = t_current

        # State components (Constitution II)
        self.V: Set[int] = set()  # Current members
        self.E: Set[Tuple[int, int]] = set()  # Directed edges
        self.A: Dict[int, ActorState] = {}  # Actor states
        self.Y: Dict[Tuple[int, int], float] = {}  # Logit-trust matrix
        self.hierarchy: Dict[int, int] = {}  # Actor → hierarchy level

        # Process state
        self.P_t = P_0  # Law enforcement effectiveness
        self.informant_count = 0  # Total informants converted

        # Exposure history for risk computation (Constitution II)
        self.Delta = Delta
        self.exposure_history: Dict[int, deque] = {}  # actor → [(neighbor, time)]

        # Arrest tracking for CTMC
        self.arrest_times: Dict[int, float] = {}  # actor → arrest time
        self.avg_trust_at_arrest: Dict[int, float] = {}  # actor → Ȳᵢ(t_arrest)

    def add_actor(
        self, actor_id: int, hierarchy_level: int, state: ActorState = ActorState.ACTIVE
    ) -> None:
        """
        Add new actor to network.

        Args:
            actor_id: Unique actor identifier
            hierarchy_level: Hierarchy level l ∈ {1, 2, ..., L}
            state: Initial state (default Active)
        """
        if actor_id in self.V:
            raise ValueError(f"Actor {actor_id} already exists in network")

        self.V.add(actor_id)
        self.A[actor_id] = state
        self.hierarchy[actor_id] = hierarchy_level
        self.exposure_history[actor_id] = deque()

    def add_edge(
        self,
        source: int,
        target: int,
        initial_trust: Optional[float] = None,
        Y_bar_0: float = 0.0,
    ) -> None:
        """
        Add directed edge with initial trust.

        Args:
            source: Source actor ID
            target: Target actor ID
            initial_trust: Initial trust w ∈ (0,1), or None for Jaccard
            Y_bar_0: Default logit-trust if Jaccard undefined

        Raises:
            ValueError: If actors not Active or edge already exists
        """
        # Validate actors exist and are Active
        if source not in self.V or target not in self.V:
            raise ValueError(f"Both actors must exist in V. Got source={source}, target={target}")

        if self.A[source] != ActorState.ACTIVE or self.A[target] != ActorState.ACTIVE:
            raise ValueError(
                f"Edges only allowed between Active actors. "
                f"source state={self.A[source]}, target state={self.A[target]}"
            )

        if (source, target) in self.E:
            raise ValueError(f"Edge ({source}, {target}) already exists")

        # Compute initial trust
        if initial_trust is None:
            # Use Jaccard similarity (Constitution II: TFPA)
            neighbors_source = self.get_all_neighbors(source)
            neighbors_target = self.get_all_neighbors(target)
            w_init = jaccard_similarity(neighbors_source, neighbors_target)

            if w_init == 0.0:
                # No common neighbors, use default
                Y_ij = Y_bar_0
            else:
                # Clamp and convert to logit
                w_init = clamp(w_init, EPSILON, 1.0 - EPSILON)
                Y_ij = logit(w_init)
        else:
            # Validate and convert provided trust
            if not (0 < initial_trust < 1):
                raise ValueError(
                    f"Initial trust must be in (0,1), got {initial_trust}"
                )
            w_init = clamp(initial_trust, EPSILON, 1.0 - EPSILON)
            Y_ij = logit(w_init)

        # Add edge and initialize trust
        self.E.add((source, target))
        self.Y[(source, target)] = Y_ij

        # Update exposure history
        self._record_exposure(source, target, self.t_current)
        self._record_exposure(target, source, self.t_current)

    def remove_edge(self, source: int, target: int) -> None:
        """
        Remove directed edge.

        Args:
            source: Source actor ID
            target: Target actor ID
        """
        if (source, target) not in self.E:
            return  # Edge doesn't exist, no-op

        self.E.discard((source, target))
        self.Y.pop((source, target), None)

    def arrest_actor(self, actor_id: int) -> None:
        """
        Execute arrest state transition (Constitution IV).

        Steps:
        1. Update state: Active → Arrested
        2. Remove all edges involving actor
        3. Risk updates handled by caller
        4. Record arrest time and average trust

        Args:
            actor_id: Actor to arrest

        Raises:
            ValueError: If actor not Active
        """
        if actor_id not in self.V:
            raise ValueError(f"Actor {actor_id} not in network")

        if self.A[actor_id] != ActorState.ACTIVE:
            raise ValueError(
                f"Can only arrest Active actors. Actor {actor_id} is {self.A[actor_id]}"
            )

        # Compute average trust before arrest
        avg_trust = self.get_average_outgoing_trust(actor_id)

        # State update
        self.A[actor_id] = ActorState.ARRESTED
        self.arrest_times[actor_id] = self.t_current
        self.avg_trust_at_arrest[actor_id] = avg_trust

        # Remove all operational edges
        edges_to_remove = [
            (s, t) for (s, t) in self.E if s == actor_id or t == actor_id
        ]
        for edge in edges_to_remove:
            self.remove_edge(*edge)

    def convert_to_informant(self, actor_id: int, eta_P: float) -> None:
        """
        Execute informant conversion state transition (Constitution IV).

        Steps:
        1. Update state: Arrested → Informant
        2. Jump effectiveness: P(t) = P(t⁻) + η_P
        3. Increment counter

        Args:
            actor_id: Actor to convert
            eta_P: Effectiveness increase per informant

        Raises:
            ValueError: If actor not Arrested
        """
        if actor_id not in self.V:
            raise ValueError(f"Actor {actor_id} not in network")

        if self.A[actor_id] != ActorState.ARRESTED:
            raise ValueError(
                f"Can only convert Arrested actors. Actor {actor_id} is {self.A[actor_id]}"
            )

        # State update
        self.A[actor_id] = ActorState.INFORMANT
        self.P_t += eta_P
        self.informant_count += 1

    def get_active_actors(self) -> Set[int]:
        """Get set of currently Active actors."""
        return {i for i in self.V if self.A[i] == ActorState.ACTIVE}

    def get_arrested_actors(self) -> Set[int]:
        """Get set of currently Arrested actors."""
        return {i for i in self.V if self.A[i] == ActorState.ARRESTED}

    def get_informants(self) -> Set[int]:
        """Get set of Informants."""
        return {i for i in self.V if self.A[i] == ActorState.INFORMANT}

    def get_outgoing_neighbors(self, actor_id: int) -> Set[int]:
        """
        Get outgoing neighbors (Constitution II: Nᵢᵒᵘᵗ(t)).

        Args:
            actor_id: Actor ID

        Returns:
            Set of target actors j where (i,j) ∈ E(t)
        """
        return {t for (s, t) in self.E if s == actor_id}

    def get_incoming_neighbors(self, actor_id: int) -> Set[int]:
        """
        Get incoming neighbors (Constitution II: Nᵢⁱⁿ(t)).

        Args:
            actor_id: Actor ID

        Returns:
            Set of source actors j where (j,i) ∈ E(t)
        """
        return {s for (s, t) in self.E if t == actor_id}

    def get_all_neighbors(self, actor_id: int) -> Set[int]:
        """
        Get all neighbors (Constitution II: Nᵢ(t) = Nᵢᵒᵘᵗ ∪ Nᵢⁱⁿ).

        Args:
            actor_id: Actor ID

        Returns:
            Set of all neighbors (undirected)
        """
        return self.get_outgoing_neighbors(actor_id) | self.get_incoming_neighbors(
            actor_id
        )

    def get_exposure_neighborhood(self, actor_id: int) -> Set[int]:
        """
        Get exposure neighborhood (Constitution II: Nᵢᵉˣᵖ(t)).

        Nᵢᵉˣᵖ(t) = {k ∈ V(t) : ∃s ∈ [t-Δ, t] s.t. (i,k) or (k,i) ∈ E(s)}

        Args:
            actor_id: Actor ID

        Returns:
            Set of actors in exposure neighborhood
        """
        if actor_id not in self.exposure_history:
            return set()

        cutoff_time = self.t_current - self.Delta
        exposure_set = set()

        for neighbor, time in self.exposure_history[actor_id]:
            if time >= cutoff_time:
                exposure_set.add(neighbor)

        return exposure_set

    def get_environmental_risk(self, actor_id: int) -> float:
        """
        Compute environmental risk (Constitution II).

        Rᵢ(t) = (1/|Nᵢᵉˣᵖ|) Σₖ∈Nᵢᵉˣᵖ 𝟙{aₖ ∈ {Arrested, Informant}}

        Args:
            actor_id: Actor ID

        Returns:
            Risk ∈ [0, 1]
        """
        exposure = self.get_exposure_neighborhood(actor_id)

        if len(exposure) == 0:
            return 0.0

        compromised_count = sum(
            1
            for k in exposure
            if self.A[k] in {ActorState.ARRESTED, ActorState.INFORMANT}
        )

        return compromised_count / len(exposure)

    def get_trust(self, source: int, target: int) -> float:
        """
        Get trust value w ∈ (0,1) for edge.

        Args:
            source: Source actor ID
            target: Target actor ID

        Returns:
            Trust value w = expit(Y)

        Raises:
            KeyError: If edge doesn't exist
        """
        if (source, target) not in self.Y:
            raise KeyError(f"Edge ({source}, {target}) does not exist")

        return expit(self.Y[(source, target)])

    def get_logit_trust(self, source: int, target: int) -> float:
        """
        Get logit-trust value Y ∈ ℝ for edge.

        Args:
            source: Source actor ID
            target: Target actor ID

        Returns:
            Logit-trust Y

        Raises:
            KeyError: If edge doesn't exist
        """
        if (source, target) not in self.Y:
            raise KeyError(f"Edge ({source}, {target}) does not exist")

        return self.Y[(source, target)]

    def set_logit_trust(self, source: int, target: int, Y_new: float) -> None:
        """
        Update logit-trust value for edge.

        Args:
            source: Source actor ID
            target: Target actor ID
            Y_new: New logit-trust value

        Raises:
            KeyError: If edge doesn't exist
        """
        if (source, target) not in self.E:
            raise KeyError(f"Edge ({source}, {target}) does not exist")

        self.Y[(source, target)] = Y_new

    def get_average_outgoing_trust(self, actor_id: int, Y_bar_0: float = 0.0) -> float:
        """
        Compute average outgoing logit-trust (Constitution II).

        Ȳᵢ(t) = (1/|Nᵢᵒᵘᵗ|) Σⱼ∈Nᵢᵒᵘᵗ Yᵢⱼ(t) if |Nᵢᵒᵘᵗ| > 0, else Ȳ₀

        Args:
            actor_id: Actor ID
            Y_bar_0: Default value for isolated actors

        Returns:
            Average logit-trust
        """
        outgoing = self.get_outgoing_neighbors(actor_id)

        if len(outgoing) == 0:
            return Y_bar_0

        total = sum(self.Y[(actor_id, j)] for j in outgoing)
        return total / len(outgoing)

    def validate_invariants(self) -> None:
        """
        Validate all state invariants (Constitution II).

        Invariants:
        - E ⊆ V × V
        - Y keys = E
        - Edges only between Active actors
        - All actors have valid states

        Raises:
            AssertionError: If any invariant violated
        """
        # Invariant 1: E ⊆ V × V
        for (s, t) in self.E:
            assert s in self.V, f"Edge source {s} not in V"
            assert t in self.V, f"Edge target {t} not in V"

        # Invariant 2: Y keys = E
        assert set(self.Y.keys()) == self.E, "Trust matrix keys must equal edge set"

        # Invariant 3: Edges only between Active actors
        for (s, t) in self.E:
            assert (
                self.A[s] == ActorState.ACTIVE
            ), f"Edge source {s} not Active: {self.A[s]}"
            assert (
                self.A[t] == ActorState.ACTIVE
            ), f"Edge target {t} not Active: {self.A[t]}"

        # Invariant 4: All actors have valid states
        for actor_id in self.V:
            assert actor_id in self.A, f"Actor {actor_id} missing state"
            assert isinstance(
                self.A[actor_id], ActorState
            ), f"Invalid state for {actor_id}"

    def _record_exposure(self, observer: int, observed: int, time: float) -> None:
        """
        Record exposure event for risk computation.

        Args:
            observer: Actor who is exposed
            observed: Actor to whom observer is exposed
            time: Time of exposure
        """
        if observer not in self.exposure_history:
            self.exposure_history[observer] = deque()

        # Add new exposure
        self.exposure_history[observer].append((observed, time))

        # Remove old exposures outside memory window
        cutoff_time = time - self.Delta
        while (
            self.exposure_history[observer]
            and self.exposure_history[observer][0][1] < cutoff_time
        ):
            self.exposure_history[observer].popleft()

    def get_state_summary(self) -> Dict[str, any]:
        """Get summary statistics of current state."""
        return {
            "time": self.t_current,
            "total_actors": len(self.V),
            "active": len(self.get_active_actors()),
            "arrested": len(self.get_arrested_actors()),
            "informants": len(self.get_informants()),
            "edges": len(self.E),
            "effectiveness_P": self.P_t,
            "avg_risk": (
                np.mean([self.get_environmental_risk(i) for i in self.get_active_actors()])
                if len(self.get_active_actors()) > 0
                else 0.0
            ),
            "avg_trust": (
                np.mean([expit(Y) for Y in self.Y.values()]) if len(self.Y) > 0 else 0.0
            ),
        }
