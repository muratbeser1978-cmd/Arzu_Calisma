"""
CLI entry point for running simulations.

Usage:
    python main.py

This runs a complete simulation with default parameters and saves results.
"""

import numpy as np
from organized_crime_network.simulation import (
    SimulationEngine,
    SimulationParameters,
    visualize_results
)
from organized_crime_network.simulation.visualization_advanced import create_advanced_visualizations
from organized_crime_network.core.state import NetworkState, ActorState


def create_initial_network(params: SimulationParameters) -> NetworkState:
    """
    Create initial network with hierarchical structure.

    Network Structure:
    - 3 hierarchy levels
    - Level 1 (operatives): 20 actors
    - Level 2 (mid-level): 10 actors
    - Level 3 (leaders): 5 actors
    - Initial edges based on hierarchy (operatives connected to mid-level)

    Args:
        params: Model parameters

    Returns:
        Initialized NetworkState
    """
    state = NetworkState(t_current=0.0, P_0=params.P_0, Delta=params.Delta)

    # Add actors with hierarchy
    actor_id = 0

    # Level 1: 20 operatives
    level1_actors = []
    for i in range(20):
        state.add_actor(actor_id, hierarchy_level=1, state=ActorState.ACTIVE)
        level1_actors.append(actor_id)
        actor_id += 1

    # Level 2: 10 mid-level
    level2_actors = []
    for i in range(10):
        state.add_actor(actor_id, hierarchy_level=2, state=ActorState.ACTIVE)
        level2_actors.append(actor_id)
        actor_id += 1

    # Level 3: 5 leaders
    level3_actors = []
    for i in range(5):
        state.add_actor(actor_id, hierarchy_level=3, state=ActorState.ACTIVE)
        level3_actors.append(actor_id)
        actor_id += 1

    # Create initial edges (hierarchy-based connections)
    # Each level 1 actor connects to 2 level 2 actors
    for i, l1_actor in enumerate(level1_actors):
        # Connect to mid-level actors
        mid1 = level2_actors[i % len(level2_actors)]
        mid2 = level2_actors[(i + 1) % len(level2_actors)]

        state.add_edge(l1_actor, mid1, initial_trust=0.7, Y_bar_0=params.Y_bar_0)
        state.add_edge(mid1, l1_actor, initial_trust=0.7, Y_bar_0=params.Y_bar_0)

        if mid1 != mid2:
            state.add_edge(l1_actor, mid2, initial_trust=0.6, Y_bar_0=params.Y_bar_0)

    # Each level 2 actor connects to 2 level 3 actors
    for i, l2_actor in enumerate(level2_actors):
        lead1 = level3_actors[i % len(level3_actors)]
        lead2 = level3_actors[(i + 1) % len(level3_actors)]

        state.add_edge(l2_actor, lead1, initial_trust=0.8, Y_bar_0=params.Y_bar_0)
        state.add_edge(lead1, l2_actor, initial_trust=0.8, Y_bar_0=params.Y_bar_0)

        if lead1 != lead2:
            state.add_edge(l2_actor, lead2, initial_trust=0.7, Y_bar_0=params.Y_bar_0)

    # Leaders connected to each other (high trust)
    for i, lead1 in enumerate(level3_actors):
        for lead2 in level3_actors[i + 1 :]:
            state.add_edge(lead1, lead2, initial_trust=0.9, Y_bar_0=params.Y_bar_0)
            state.add_edge(lead2, lead1, initial_trust=0.9, Y_bar_0=params.Y_bar_0)

    print(f"Initialized network:")
    print(f"  Total actors: {len(state.V)}")
    print(f"  Level 1 (operatives): {len(level1_actors)}")
    print(f"  Level 2 (mid-level): {len(level2_actors)}")
    print(f"  Level 3 (leaders): {len(level3_actors)}")
    print(f"  Total edges: {len(state.E)}")
    print()

    return state


def main():
    """Run simulation with default parameters."""
    print("=" * 70)
    print("Organized Crime Network Stochastic Simulation")
    print("Mathematical Foundation: gelis.tex")
    print("Constitution: .specify/memory/constitution.md v1.0.0")
    print("=" * 70)
    print()

    # Load parameters with EXTENDED simulation time
    print("Loading parameters...")
    params = SimulationParameters.default()
    # Create extended simulation parameters (3x longer)
    params_dict = params.__dict__.copy()
    params_dict['T_max'] = 300.0  # Extended from 100 to 300
    params = SimulationParameters(**params_dict)

    print(f"  Simulation horizon: T_max = {params.T_max} (EXTENDED)")
    print(f"  Time step: dt = {params.dt}")
    print(f"  Random seed: {params.random_seed}")
    print(f"  Volatility: sigma = {params.sigma} {'(PDMP mode)' if params.sigma == 0 else '(SDE mode)'}")
    print(f"  Expected steps: ~{int(params.T_max / params.dt):,}")
    print()

    # Create initial network
    print("Creating initial network...")
    initial_state = create_initial_network(params)

    # Initialize simulation engine
    print("Initializing simulation engine...")
    engine = SimulationEngine(params)
    print()

    # Run simulation
    print("Running simulation...")
    print("-" * 70)
    results = engine.run(initial_state, verbose=True)
    print("-" * 70)
    print()

    # Display results summary
    print(results.summary())

    # Save results
    print("Saving results...")
    results.export_time_series("time_series.csv")
    results.export_events("simulation_events.json")
    print("  [OK] time_series.csv")
    print("  [OK] simulation_events.json")
    print()

    # Generate standard visualizations
    print("Generating standard visualizations...")
    print("-" * 70)
    visualize_results(results, output_dir="visualization_output", show=False)
    print("-" * 70)
    print()

    # Generate ADVANCED visualizations
    print("Generating ADVANCED visualizations...")
    print("-" * 70)
    create_advanced_visualizations(results, output_dir="visualization_output/advanced", show=True)
    print("-" * 70)
    print()

    print("=" * 70)
    print("Simulation complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
