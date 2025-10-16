"""
Main simulation engine orchestrating all stochastic processes.

This module provides the SimulationEngine class that couples the four
stochastic processes: trust dynamics (SDE), arrests (Cox), conversion (CTMC),
and growth (TFPA).

Constitutional Reference: Section IV (State Update Rules)
Research Decision: Decision 4 (Process Coupling)
"""

import time
from typing import List, Dict, Optional, Callable
import numpy as np
import networkx as nx

from ..core.state import NetworkState, ActorState
from .parameters import SimulationParameters
from .results import (
    SimulationResults,
    TimeSeries,
    ConvergenceStatus,
    NumericalInstabilityError,
    NetworkValidationError
)
from .events import SimulationEvent, EventType
from .processes import (
    TrustDynamics,
    TrustState,
    ArrestProcess,
    ConversionProcess,
    ActorLoyaltyState,
    LoyaltyState,
    GrowthProcess
)


class SimulationEngine:
    """
    Event-driven simulation engine for organized crime network dynamics.

    Couples four stochastic processes:
    - Trust Dynamics: Switching OU SDE
    - Arrest Process: Cox process
    - Conversion Process: CTMC
    - Growth Process: TFPA

    Mathematical Foundation: gelis.tex
    Constitution: .specify/memory/constitution.md v1.0.0

    Parameters
    ----------
    parameters : SimulationParameters
        Validated simulation parameters

    Examples
    --------
    >>> params = SimulationParameters.default()
    >>> engine = SimulationEngine(params)
    >>> state = NetworkState(t_current=0.0)
    >>> # ... add actors and edges ...
    >>> results = engine.run(state, verbose=True)
    """

    def __init__(self, parameters: SimulationParameters):
        """
        Initialize simulation engine with validated parameters.

        Parameters
        ----------
        parameters : SimulationParameters
            Validated simulation parameters (raises ParameterValidationError if invalid)
        """
        self.params = parameters

        # Initialize RNG for reproducibility
        self.rng = np.random.default_rng(self.params.random_seed)

        # Initialize stochastic processes
        self.trust_dynamics = TrustDynamics(
            alpha=self.params.alpha,
            beta=self.params.beta,
            sigma=self.params.sigma,
            dt=self.params.dt,
            rng=self.rng
        )

        self.arrest_process = ArrestProcess(
            lambda_0=self.params.lambda_0,
            kappa=self.params.kappa,
            gamma=self.params.gamma,
            rng=self.rng
        )

        self.conversion_process = ConversionProcess(
            mu_LH=self.params.mu_LH,
            mu_min=self.params.mu_min,
            mu_rng=self.params.mu_rng,
            theta=self.params.theta,
            rng=self.rng
        )

        self.growth_process = GrowthProcess(
            w_min=self.params.w_min,
            gamma_pa=self.params.gamma_pa,
            rng=self.rng
        )

        # Performance tracking
        self._wall_time = 0.0
        self._steps_executed = 0
        self._events_generated = 0

    def run(
        self,
        initial_network: NetworkState,
        verbose: bool = False,
        progress_callback: Optional[Callable[[float], None]] = None
    ) -> SimulationResults:
        """
        Run complete simulation from t=0 to T_max.

        Algorithm
        ---------
        1. Initialize all stochastic processes
        2. For each time step dt:
            a. Update trust via Euler-Maruyama
            b. Update environmental risk R_i(t)
            c. Check for arrest events (Cox process)
            d. Check for conversion events (CTMC)
            e. Check for growth events (TFPA)
            f. Record time series at intervals
        3. Return complete results

        Parameters
        ----------
        initial_network : NetworkState
            Network state at t=0
        verbose : bool, default=False
            Enable progress logging to stdout
        progress_callback : callable, optional
            Function called with progress fraction ∈ [0, 1]

        Returns
        -------
        results : SimulationResults
            Complete simulation output

        Raises
        ------
        NetworkValidationError
            If initial_network has invalid structure
        NumericalInstabilityError
            If instability detected (NaN/Inf in computations)
        """
        start_time = time.time()

        # Validate initial network
        errors = self.validate_initial_network(initial_network)
        if errors:
            raise NetworkValidationError(
                f"Invalid initial network:\n" + "\n".join(f"  - {e}" for e in errors)
            )

        # Copy state to avoid modifying original
        state = self._copy_network_state(initial_network)
        state.t_current = 0.0

        # Initialize tracking
        events: List[SimulationEvent] = []
        arrested_actors: Dict[int, ActorLoyaltyState] = {}

        # Time series recording
        ts_times = []
        ts_network_size = []
        ts_arrested = []
        ts_informants = []
        ts_effectiveness = []
        ts_mean_trust = []
        ts_mean_risk = []
        ts_lcc_size = []

        # Initial recording
        self._record_time_series(
            state, ts_times, ts_network_size, ts_arrested, ts_informants,
            ts_effectiveness, ts_mean_trust, ts_mean_risk, ts_lcc_size
        )

        # Main simulation loop
        num_steps = int(self.params.T_max / self.params.dt)
        record_interval = max(1, num_steps // 1000)  # Record ~1000 points

        if verbose:
            print(f"Starting simulation: T_max={self.params.T_max}, dt={self.params.dt}, steps={num_steps}")

        for step in range(num_steps):
            t_current = step * self.params.dt
            t_next = (step + 1) * self.params.dt
            state.t_current = t_current

            # Progress reporting
            if verbose and step % (num_steps // 10) == 0:
                progress = step / num_steps
                active = len(state.get_active_actors())
                print(f"  Progress: {progress*100:.0f}% | t={t_current:.1f} | Active={active} | P={state.P_t:.2f}")

            if progress_callback and step % record_interval == 0:
                progress_callback(step / num_steps)

            # Check for network collapse
            active_actors = list(state.get_active_actors())
            if len(active_actors) == 0:
                if verbose:
                    print("Network collapsed - no active actors remaining")
                convergence_status = ConvergenceStatus.NETWORK_COLLAPSED
                break

            # Step 1: Update trust via SDE
            try:
                self._update_trust(state, active_actors)
            except Exception as e:
                raise NumericalInstabilityError(f"Trust update failed at t={t_current}: {e}")

            # Step 2: Update environmental risk
            self._update_effectiveness(state)

            # Step 3: Generate arrest events in [t_current, t_next)
            arrests = self.arrest_process.generate_arrests(
                active_actors=active_actors,
                network_state=state,
                effectiveness_func=lambda t: state.P_t,  # Piecewise constant
                t_start=t_current,
                t_end=t_next
            )

            for arrest_time, actor_id in arrests:
                # Skip if already arrested in this time step
                if state.A.get(actor_id) != ActorState.ACTIVE:
                    continue

                # Execute arrest
                num_active_before = len(state.get_active_actors())
                hierarchy = state.hierarchy.get(actor_id, 1)
                intensity = self.arrest_process.compute_intensity(actor_id, state, state.P_t)

                state.arrest_actor(actor_id)
                num_active_after = len(state.get_active_actors())

                # Create CTMC trajectory for this actor
                avg_trust = state.avg_trust_at_arrest.get(actor_id, self.params.Y_bar_0)
                loyalty_state = self.conversion_process.create_loyalty_state(
                    actor_id=actor_id,
                    arrest_time=arrest_time,
                    avg_trust_at_arrest=avg_trust,
                    t_max=self.params.T_max
                )
                arrested_actors[actor_id] = loyalty_state

                # Record event
                event = SimulationEvent.arrest(
                    timestamp=arrest_time,
                    actor_id=actor_id,
                    hierarchy_level=hierarchy,
                    arrest_intensity=intensity,
                    effectiveness_before=state.P_t,
                    num_active_before=num_active_before,
                    num_active_after=num_active_after
                )
                events.append(event)
                self._events_generated += 1

            # Step 4: Check for conversion events
            conversions = self.conversion_process.check_for_conversions(arrested_actors, t_next)

            for actor_id in conversions:
                loyalty_state = arrested_actors[actor_id]

                # Only convert if state indicates informant AND not already converted
                if (loyalty_state.current_state == LoyaltyState.INFORMANT and
                    state.A.get(actor_id) == ActorState.ARRESTED):
                    num_inf_before = state.informant_count

                    state.convert_to_informant(actor_id, self.params.eta_P)
                    num_inf_after = state.informant_count

                    # Get transition time from history
                    conversion_time = t_next
                    for t, s in loyalty_state.state_history:
                        if s == LoyaltyState.INFORMANT:
                            conversion_time = t
                            break

                    # Record event
                    event = SimulationEvent.conversion(
                        timestamp=conversion_time,
                        actor_id=actor_id,
                        loyalty_state_before="Hesitant",
                        time_in_arrested=conversion_time - loyalty_state.arrest_time,
                        fragility_rate=loyalty_state.fragility_rate,
                        avg_trust=loyalty_state.avg_trust_at_arrest,
                        num_informants_before=num_inf_before,
                        num_informants_after=num_inf_after
                    )
                    events.append(event)
                    self._events_generated += 1

            # Step 5: Generate growth events (TFPA)
            # For now, skip growth to match MVP scope
            # growth_events = self.growth_process.generate_growth_events(...)

            # Record time series at intervals
            if step % record_interval == 0 or step == num_steps - 1:
                self._record_time_series(
                    state, ts_times, ts_network_size, ts_arrested, ts_informants,
                    ts_effectiveness, ts_mean_trust, ts_mean_risk, ts_lcc_size
                )

            self._steps_executed += 1

        else:
            # Completed normally
            convergence_status = ConvergenceStatus.COMPLETED

        # Final time series record
        state.t_current = self.params.T_max
        self._record_time_series(
            state, ts_times, ts_network_size, ts_arrested, ts_informants,
            ts_effectiveness, ts_mean_trust, ts_mean_risk, ts_lcc_size
        )

        # Compute wall time
        self._wall_time = time.time() - start_time

        if verbose:
            print(f"Simulation complete in {self._wall_time:.2f}s")
            print(f"  Total arrests: {len([e for e in events if e.event_type == EventType.ARREST])}")
            print(f"  Total conversions: {len([e for e in events if e.event_type == EventType.CONVERSION])}")

        # Build results
        time_series = TimeSeries(
            times=np.array(ts_times),
            network_size=np.array(ts_network_size),
            arrested_count=np.array(ts_arrested),
            informant_count=np.array(ts_informants),
            effectiveness=np.array(ts_effectiveness),
            mean_trust=np.array(ts_mean_trust),
            mean_risk=np.array(ts_mean_risk),
            lcc_size=np.array(ts_lcc_size)
        )

        # Validate numerical stability
        numerical_stability = self._check_numerical_stability(time_series)

        # Validate causality
        causality_preserved = self._check_causality(events)

        results = SimulationResults(
            parameters=self.params,
            initial_network=initial_network,
            final_network=state,
            events=sorted(events, key=lambda e: e.timestamp),
            time_series=time_series,
            total_arrests=len([e for e in events if e.event_type == EventType.ARREST]),
            total_conversions=len([e for e in events if e.event_type == EventType.CONVERSION]),
            final_effectiveness=state.P_t,
            simulation_duration=self._wall_time,
            convergence_status=convergence_status,
            numerical_stability=numerical_stability,
            causality_preserved=causality_preserved
        )

        return results

    def _update_trust(self, state: NetworkState, active_actors: List[int]) -> None:
        """Update trust values via SDE step."""
        # Get all active edges
        active_edges = [(i, j) for (i, j) in state.E]

        if not active_edges:
            return

        # Compute risk values for all actors
        risk_values = {i: state.get_environmental_risk(i) for i in active_actors}

        # Convert Y dict to matrix format for vectorized update
        # (For now, update edge-by-edge for simplicity)
        for (i, j) in active_edges:
            Y_current = state.get_logit_trust(i, j)
            R_i = risk_values.get(i, 0.0)

            # Single-edge SDE step
            Y_new_array = self.trust_dynamics.step(
                Y_current=np.array([[Y_current]]),
                risk_values=np.array([R_i]),
                active_edges=[(0, 0)]  # Single edge in local coords
            )

            Y_new = Y_new_array[0, 0]
            state.set_logit_trust(i, j, Y_new)

    def _update_effectiveness(self, state: NetworkState) -> None:
        """Update law enforcement effectiveness with decay."""
        # P(t) decays exponentially: dP/dt = -ρ P
        # Exact solution: P(t+dt) = P(t) * exp(-ρ * dt)
        state.P_t *= np.exp(-self.params.rho * self.params.dt)

        # Ensure doesn't drop below baseline
        state.P_t = max(state.P_t, self.params.P_0)

    def _record_time_series(
        self, state, times, network_size, arrested, informants,
        effectiveness, mean_trust, mean_risk, lcc_size
    ) -> None:
        """Record current state in time series."""
        times.append(state.t_current)
        network_size.append(len(state.get_active_actors()))
        arrested.append(len(state.get_arrested_actors()))
        informants.append(len(state.get_informants()))
        effectiveness.append(state.P_t)

        # Compute mean trust
        if len(state.Y) > 0:
            from ..utils.numerical import expit
            mean_trust.append(np.mean([expit(y) for y in state.Y.values()]))
        else:
            mean_trust.append(0.0)

        # Compute mean risk
        active = state.get_active_actors()
        if len(active) > 0:
            mean_risk.append(np.mean([state.get_environmental_risk(i) for i in active]))
        else:
            mean_risk.append(0.0)

        # Compute LCC size
        lcc_size.append(self._compute_lcc_size(state))

    def _compute_lcc_size(self, state: NetworkState) -> int:
        """Compute size of largest connected component."""
        if len(state.E) == 0:
            return 0

        # Build undirected graph from directed edges
        G = nx.Graph()
        for (i, j) in state.E:
            G.add_edge(i, j)

        if len(G) == 0:
            return 0

        # Find largest connected component
        components = list(nx.connected_components(G))
        if components:
            return len(max(components, key=len))
        return 0

    def _check_numerical_stability(self, time_series: TimeSeries) -> bool:
        """Check for NaN/Inf in time series."""
        return (
            np.all(np.isfinite(time_series.effectiveness)) and
            np.all(np.isfinite(time_series.mean_trust)) and
            np.all(np.isfinite(time_series.mean_risk))
        )

    def _check_causality(self, events: List[SimulationEvent]) -> bool:
        """Check that events are in chronological order."""
        if len(events) <= 1:
            return True

        for i in range(len(events) - 1):
            if events[i].timestamp > events[i+1].timestamp:
                return False
        return True

    def _copy_network_state(self, state: NetworkState) -> NetworkState:
        """Create deep copy of network state."""
        new_state = NetworkState(
            t_current=state.t_current,
            P_0=self.params.P_0,
            Delta=self.params.Delta
        )

        # Copy actors
        for actor_id in state.V:
            new_state.add_actor(
                actor_id=actor_id,
                hierarchy_level=state.hierarchy[actor_id],
                state=state.A[actor_id]
            )

        # Copy edges and trust
        for (i, j) in state.E:
            Y_ij = state.get_logit_trust(i, j)
            from ..utils.numerical import expit
            w_ij = expit(Y_ij)
            new_state.add_edge(i, j, initial_trust=w_ij)

        new_state.P_t = state.P_t
        new_state.informant_count = state.informant_count

        return new_state

    def get_performance_metrics(self) -> Dict[str, float]:
        """
        Get performance statistics from last run.

        Returns
        -------
        metrics : dict
            Performance metrics
        """
        return {
            'wall_time': self._wall_time,
            'steps_per_second': self._steps_executed / self._wall_time if self._wall_time > 0 else 0,
            'events_generated': self._events_generated,
            'memory_peak_mb': 0.0  # TODO: Implement memory tracking
        }

    @staticmethod
    def validate_initial_network(network: NetworkState) -> List[str]:
        """
        Check if network meets simulation requirements.

        Parameters
        ----------
        network : NetworkState
            Network to validate

        Returns
        -------
        errors : list of str
            Empty if valid, otherwise list of error messages
        """
        errors = []

        # Must have at least one active actor
        if len(network.get_active_actors()) == 0:
            errors.append("Network must have at least one active actor")

        # All edges must involve active actors
        for (i, j) in network.E:
            if network.A[i] != ActorState.ACTIVE:
                errors.append(f"Edge ({i}, {j}) source not active: {network.A[i]}")
            if network.A[j] != ActorState.ACTIVE:
                errors.append(f"Edge ({i}, {j}) target not active: {network.A[j]}")

        # No self-loops
        for (i, j) in network.E:
            if i == j:
                errors.append(f"Self-loop detected: ({i}, {i})")

        # Trust values in reasonable range
        for (i, j), Y_ij in network.Y.items():
            if not (-10 <= Y_ij <= 10):
                errors.append(f"Trust Y[{i},{j}]={Y_ij} outside reasonable range [-10, 10]")

        # All actors have hierarchy levels
        for actor_id in network.V:
            if actor_id not in network.hierarchy:
                errors.append(f"Actor {actor_id} missing hierarchy level")

        return errors
