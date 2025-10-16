"""
Simulation Engine: Event-driven coordinator for all stochastic processes.

Integrates:
- ArrestProcess (Cox)
- TrustDynamics (SDE/PDMP)
- InformantConversion (CTMC)
- TFPAMechanism (network growth)
"""

from typing import Dict, List, Any, Optional
import json
from pathlib import Path

from .core.state import NetworkState, ActorState
from .core.parameters import Parameters
from .core.events import Event, ArrestEvent, ConversionEvent, GrowthEvent
from .processes.arrest import ArrestProcess
from .processes.trust import TrustDynamics
from .processes.conversion import InformantConversion
from .processes.tfpa import TFPAMechanism
from .utils.random import set_random_seed


class SimulationEngine:
    """
    Event-driven simulation engine for OCN dynamics.

    Architecture:
        1. Initialize all processes and state
        2. Main loop: Sample competing exponentials for next event
        3. Integrate continuous processes (trust, effectiveness)
        4. Execute next event
        5. Record event and update state
        6. Repeat until T_max or network collapse
    """

    def __init__(self, params: Parameters, initial_state: Optional[NetworkState] = None):
        """
        Initialize simulation engine.

        Args:
            params: Model parameters
            initial_state: Initial network state (or None for empty)
        """
        self.params = params
        set_random_seed(params.random_seed)

        # Initialize state
        if initial_state is None:
            self.state = NetworkState(t_current=0.0, P_0=params.P_0, Delta=params.Delta)
        else:
            self.state = initial_state

        # Initialize processes
        self.arrest_process = ArrestProcess(params)
        self.trust_dynamics = TrustDynamics(params)
        self.conversion_process = InformantConversion(params)
        self.tfpa_mechanism = TFPAMechanism(params)

        # Event log
        self.events: List[Event] = []

        # Trajectory storage (sampled at intervals)
        self.trajectory_interval = max(0.1, params.T_max / 1000)  # ~1000 points
        self.next_sample_time = self.trajectory_interval
        self.trajectories: Dict[str, List] = {
            "time": [],
            "P_t": [],
            "mean_risk": [],
            "mean_trust": [],
            "active_count": [],
            "arrested_count": [],
            "informant_count": [],
        }

    def run(self, verbose: bool = False) -> Dict[str, Any]:
        """
        Run complete simulation.

        Args:
            verbose: Print progress messages

        Returns:
            Complete simulation results including events and trajectories
        """
        if verbose:
            print(f"Starting simulation: T_max={self.params.T_max}, dt={self.params.dt}")
            print(f"Initial state: {len(self.state.V)} actors, {len(self.state.E)} edges")

        while self.state.t_current < self.params.T_max:
            # Check for network collapse
            if len(self.state.get_active_actors()) == 0:
                if verbose:
                    print(f"\nNetwork collapsed at t={self.state.t_current:.2f}")
                break

            # Sample next events from competing processes
            next_event_time, event = self._get_next_event()

            if next_event_time is None:
                if verbose:
                    print("\nNo more events possible")
                break

            # Don't exceed T_max
            next_event_time = min(next_event_time, self.params.T_max)

            # Integrate continuous processes up to next event
            self._integrate_continuous(next_event_time)

            # Update current time
            self.state.t_current = next_event_time

            # Execute event if within time limit
            if event is not None and self.state.t_current < self.params.T_max:
                self._execute_event(event)
                self.events.append(event)

                if verbose and len(self.events) % 10 == 0:
                    print(
                        f"t={self.state.t_current:.2f}: {len(self.events)} events, "
                        f"{len(self.state.get_active_actors())} active"
                    )

            # Sample trajectory
            if self.state.t_current >= self.next_sample_time:
                self._sample_trajectory()
                self.next_sample_time += self.trajectory_interval

        # Final trajectory sample
        self._sample_trajectory()

        if verbose:
            print(f"\nSimulation complete: {len(self.events)} total events")
            print(f"Final state: {self.state.get_state_summary()}")

        return self._compile_results()

    def _get_next_event(self) -> tuple:
        """
        Sample next event from competing processes.

        Returns:
            (time_to_event, event) or (None, None) if no events possible
        """
        candidates = []

        # 1. Sample next arrest
        arrest_result = self.arrest_process.sample_next_arrest(self.state)
        if arrest_result is not None:
            wait_time, actor_id = arrest_result
            event_time = self.state.t_current + wait_time
            event = ArrestEvent(time=event_time, actor_id=actor_id)
            candidates.append((event_time, event))

        # 2. Sample next conversion
        conversion_result = self.conversion_process.get_next_conversion(self.state)
        if conversion_result is not None:
            event_time, actor_id = conversion_result
            avg_trust = self.state.avg_trust_at_arrest.get(actor_id, 0.0)
            event = ConversionEvent(
                time=event_time, actor_id=actor_id, avg_trust_at_arrest=avg_trust
            )
            candidates.append((event_time, event))

        # 3. Sample next TFPA growth (for simplicity, use constant rate)
        # In full implementation, this could be Poisson with network-size dependent rate
        # For now, skip TFPA to focus on core dynamics

        if len(candidates) == 0:
            return (None, None)

        # Select earliest event
        candidates.sort(key=lambda x: x[0])
        return candidates[0]

    def _integrate_continuous(self, target_time: float) -> None:
        """
        Integrate continuous processes from current time to target time.

        Processes:
        - Trust dynamics (SDE)
        - Effectiveness decay (ODE)

        Args:
            target_time: Target simulation time
        """
        current_time = self.state.t_current
        dt = self.params.dt

        while current_time < target_time:
            # Don't overshoot
            step = min(dt, target_time - current_time)

            # Integrate trust SDE
            self.trust_dynamics.integrate_step(self.state, step)

            # Update effectiveness (continuous decay)
            self.arrest_process.update_effectiveness(self.state, step)

            current_time += step

    def _execute_event(self, event: Event) -> None:
        """
        Execute discrete event and update state.

        Args:
            event: Event to execute
        """
        if isinstance(event, ArrestEvent):
            # Execute arrest (Constitution IV)
            self.state.arrest_actor(event.actor_id)

            # Initialize CTMC for conversion
            self.conversion_process.initialize_actor(
                event.actor_id, event.time, self.state
            )

            # Risk updates are automatic via exposure history

        elif isinstance(event, ConversionEvent):
            # Only convert if still arrested (not already converted)
            if self.state.A[event.actor_id] == ActorState.ARRESTED:
                # Execute conversion (Constitution IV)
                self.state.convert_to_informant(event.actor_id, self.params.eta_P)

                # CTMC advances automatically via next_transition tracking
                self.conversion_process.execute_transition(
                    event.actor_id, event.time, self.state
                )

        elif isinstance(event, GrowthEvent):
            # Execute TFPA edge addition (Constitution IV)
            self.state.add_edge(
                event.source_id,
                event.target_id,
                initial_trust=event.initial_trust,
                Y_bar_0=self.params.Y_bar_0,
            )

        # Validate invariants (can be disabled for performance)
        # self.state.validate_invariants()

    def _sample_trajectory(self) -> None:
        """Sample current state for trajectory storage."""
        summary = self.state.get_state_summary()

        self.trajectories["time"].append(self.state.t_current)
        self.trajectories["P_t"].append(summary["effectiveness_P"])
        self.trajectories["mean_risk"].append(summary["avg_risk"])
        self.trajectories["mean_trust"].append(summary["avg_trust"])
        self.trajectories["active_count"].append(summary["active"])
        self.trajectories["arrested_count"].append(summary["arrested"])
        self.trajectories["informant_count"].append(summary["informants"])

    def _compile_results(self) -> Dict[str, Any]:
        """Compile complete simulation results."""
        return {
            "parameters": self.params.to_dict(),
            "events": [e.to_dict() for e in self.events],
            "trajectories": self.trajectories,
            "final_state": self.state.get_state_summary(),
            "statistics": self._compute_statistics(),
        }

    def _compute_statistics(self) -> Dict[str, Any]:
        """Compute summary statistics."""
        return {
            "total_events": len(self.events),
            "total_arrests": sum(1 for e in self.events if isinstance(e, ArrestEvent)),
            "total_conversions": sum(
                1 for e in self.events if isinstance(e, ConversionEvent)
            ),
            "final_time": self.state.t_current,
            "network_collapsed": len(self.state.get_active_actors()) == 0,
        }

    def save_results(self, filepath: str) -> None:
        """Save simulation results to JSON file."""
        results = self._compile_results()
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)

        with open(filepath, "w") as f:
            json.dump(results, f, indent=2)
