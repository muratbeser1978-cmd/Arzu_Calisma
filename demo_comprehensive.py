"""
Comprehensive demonstration of all simulation features.

This script demonstrates:
- Multiple parameter presets (US2)
- Different network topologies (US3)
- Progress monitoring (US4)
- Analysis utilities (US5)
"""

from organized_crime_network.simulation import (
    SimulationEngine,
    SimulationParameters,
    topology,
    analysis,
    visualization,
    create_advanced_visualizations
)


def main():
    print("="*80)
    print("COMPREHENSIVE SIMULATION DEMONSTRATION")
    print("Stochastic Organized Crime Network Model")
    print("="*80)
    print()

    # Part 1: Parameter Presets (US2)
    print("PART 1: Parameter Presets")
    print("-"*80)

    presets = {
        'Default': SimulationParameters.default(),
        'Aggressive': SimulationParameters.aggressive_enforcement(),
        'Conservative': SimulationParameters.conservative_enforcement(),
        'Balanced': SimulationParameters.balanced_strategy(),
        'High Resilience': SimulationParameters.high_resilience_network()
    }

    print(f"Available parameter presets: {', '.join(presets.keys())}")
    print()

    # Show one preset in detail
    print("Aggressive Enforcement Preset:")
    print(presets['Aggressive'].get_parameter_description())
    print()

    # Part 2: Network Topology Generators (US3)
    print("PART 2: Network Topology Generators")
    print("-"*80)

    print("Creating scale-free network...")
    network_sf = topology.create_scale_free_network(n_actors=50, m=2, seed=42)
    stats_sf = topology.get_network_statistics(network_sf)
    print(f"  Actors: {stats_sf['num_actors']}, Edges: {stats_sf['num_edges']}")
    print(f"  Avg Degree: {stats_sf['avg_degree']:.2f}, Clustering: {stats_sf['avg_clustering']:.3f}")

    print("\nCreating hierarchical network...")
    network_hier = topology.create_hierarchical_network([20, 10, 5], seed=42)
    stats_hier = topology.get_network_statistics(network_hier)
    print(f"  Actors: {stats_hier['num_actors']}, Edges: {stats_hier['num_edges']}")
    print(f"  Hierarchy: {stats_hier['hierarchy_distribution']}")

    print("\nCreating small-world network...")
    network_sw = topology.create_small_world_network(n_actors=40, k=4, p=0.1, seed=42)
    stats_sw = topology.get_network_statistics(network_sw)
    print(f"  Actors: {stats_sw['num_actors']}, Edges: {stats_sw['num_edges']}")
    print(f"  Clustering: {stats_sw['avg_clustering']:.3f}")

    print("\nCreating core-periphery network...")
    network_cp = topology.create_core_periphery_network(n_core=10, n_periphery=30, seed=42)
    stats_cp = topology.get_network_statistics(network_cp)
    print(f"  Actors: {stats_cp['num_actors']}, Edges: {stats_cp['num_edges']}")
    print(f"  Hierarchy: {stats_cp['hierarchy_distribution']}")
    print()

    # Part 3: Run Simulations with Progress Monitoring (US4)
    print("PART 3: Running Simulations with Different Strategies")
    print("-"*80)

    # Use hierarchical network for comparison
    network = topology.create_hierarchical_network([30, 15, 8], seed=42)  # LARGER NETWORK

    # EXTENDED simulation time for more detailed evolution
    params_agg = SimulationParameters.aggressive_enforcement()
    params_agg = SimulationParameters(**{**params_agg.__dict__, 'T_max': 200.0})  # Extended

    params_con = SimulationParameters.conservative_enforcement()
    params_con = SimulationParameters(**{**params_con.__dict__, 'T_max': 200.0})  # Extended

    print("\nRunning AGGRESSIVE enforcement simulation...")
    engine_agg = SimulationEngine(params_agg)
    results_agg = engine_agg.run(network, verbose=True)

    print("\nRunning CONSERVATIVE enforcement simulation...")
    engine_con = SimulationEngine(params_con)
    results_con = engine_con.run(network, verbose=True)

    print()

    # Part 4: Analysis Utilities (US5)
    print("PART 4: Comparative Analysis")
    print("-"*80)

    # Compare strategies
    comparison = analysis.compare_strategies(
        [results_agg, results_con],
        ['Aggressive', 'Conservative']
    )

    print("\nStrategy Comparison:")
    print(f"{'Metric':<30} {'Aggressive':>15} {'Conservative':>15}")
    print("-"*62)
    print(f"{'Total Arrests':<30} {comparison['total_arrests']['Aggressive']:>15} {comparison['total_arrests']['Conservative']:>15}")
    print(f"{'Total Conversions':<30} {comparison['total_conversions']['Aggressive']:>15} {comparison['total_conversions']['Conservative']:>15}")
    print(f"{'Final Effectiveness':<30} {comparison['final_effectiveness']['Aggressive']:>15.2f} {comparison['final_effectiveness']['Conservative']:>15.2f}")
    print(f"{'Arrest Rate (per time)':<30} {comparison['arrest_rate']['Aggressive']:>15.3f} {comparison['arrest_rate']['Conservative']:>15.3f}")
    print(f"{'Conversion Rate':<30} {comparison['conversion_rate']['Aggressive']:>15.1%} {comparison['conversion_rate']['Conservative']:>15.1%}")
    print()

    # Detailed analysis of one strategy
    print("\nDetailed Analysis: Aggressive Enforcement")
    print("-"*80)

    arrest_stats = analysis.compute_arrest_statistics(results_agg)
    print(f"\nArrest Statistics:")
    print(f"  Total: {arrest_stats['total_arrests']}")
    print(f"  Rate: {arrest_stats['arrests_per_time_unit']:.3f} per time unit")
    print(f"  Hierarchy Distribution: {arrest_stats['hierarchy_distribution']}")

    conv_stats = analysis.compute_conversion_statistics(results_agg)
    print(f"\nConversion Statistics:")
    print(f"  Total: {conv_stats['total_conversions']}")
    print(f"  Rate: {conv_stats['conversion_rate']:.1%}")
    if not np.isnan(conv_stats['mean_time_to_conversion']):
        print(f"  Mean Time: {conv_stats['mean_time_to_conversion']:.2f}")

    net_metrics = analysis.compute_network_evolution_metrics(results_agg)
    print(f"\nNetwork Evolution:")
    print(f"  Initial Size: {net_metrics['initial_size']}")
    print(f"  Final Size: {net_metrics['final_size']}")
    print(f"  Collapsed: {'Yes' if net_metrics['collapsed'] else 'No'}")
    if net_metrics['collapsed']:
        print(f"  Collapse Time: {net_metrics['collapse_time']:.2f}")

    # Fragility cycle analysis
    cycle_corr = analysis.compute_fragility_cycle_correlation(results_agg)
    print(f"\nFragility Cycle Correlations:")
    if not np.isnan(cycle_corr['arrests_vs_risk']):
        print(f"  Arrests -> Risk: {cycle_corr['arrests_vs_risk']:+.3f}")
    if not np.isnan(cycle_corr['risk_vs_trust']):
        print(f"  Risk -> Trust: {cycle_corr['risk_vs_trust']:+.3f}")
    if not np.isnan(cycle_corr['conversions_vs_effectiveness']):
        print(f"  Conversions -> Effectiveness: {cycle_corr['conversions_vs_effectiveness']:+.3f}")
    print()

    # Generate comprehensive report
    print("\nGenerating Comprehensive Report...")
    print("-"*80)
    report = analysis.generate_summary_report(results_agg)
    print(report)

    # Part 5: Export Results
    print("PART 5: Exporting Results")
    print("-"*80)

    results_agg.export_time_series("demo_aggressive_timeseries.csv")
    results_agg.export_events("demo_aggressive_events.json")
    results_con.export_time_series("demo_conservative_timeseries.csv")
    results_con.export_events("demo_conservative_events.json")

    print("Exported files:")
    print("  - demo_aggressive_timeseries.csv")
    print("  - demo_aggressive_events.json")
    print("  - demo_conservative_timeseries.csv")
    print("  - demo_conservative_events.json")
    print()

    # Part 6: Generate Visualizations
    print("PART 6: Generating Visualizations")
    print("-"*80)

    # Individual strategy visualizations
    print("\nGenerating aggressive strategy standard visualizations...")
    visualization.visualize_results(
        results_agg,
        output_dir="demo_visualizations/aggressive",
        show=False
    )

    print("\nGenerating aggressive strategy ADVANCED visualizations...")
    create_advanced_visualizations(
        results_agg,
        output_dir="demo_visualizations/aggressive/advanced",
        show=False
    )

    print("\nGenerating conservative strategy standard visualizations...")
    visualization.visualize_results(
        results_con,
        output_dir="demo_visualizations/conservative",
        show=False
    )

    print("\nGenerating conservative strategy ADVANCED visualizations...")
    create_advanced_visualizations(
        results_con,
        output_dir="demo_visualizations/conservative/advanced",
        show=False
    )

    # Strategy comparison visualization
    print("\nGenerating strategy comparison...")
    visualizer = visualization.SimulationVisualizer()
    visualizer.plot_strategy_comparison(
        [results_agg, results_con],
        ['Aggressive', 'Conservative'],
        save_path="demo_visualizations/strategy_comparison.png",
        show=True
    )

    print("\nAll visualizations saved to demo_visualizations/")
    print("  - Standard: network topology, time series, events")
    print("  - Advanced: network evolution (6 snapshots), centrality metrics, community detection")
    print()

    # Summary
    print("="*80)
    print("DEMONSTRATION COMPLETE")
    print("="*80)
    print()
    print("Key Features Demonstrated:")
    print("  [OK] US2: Parameter presets (5 strategies)")
    print("  [OK] US3: Network topology generators (4 types)")
    print("  [OK] US4: Progress monitoring during simulation")
    print("  [OK] US5: Comprehensive analysis utilities")
    print("  [OK] US6: Standard visualization suite")
    print("  [OK] US6+: ADVANCED visualizations")
    print("      - Multi-layout network topology (hierarchical/spring/kamada-kawai)")
    print("      - Network evolution (6-snapshot timeline)")
    print("      - Centrality analysis (degree/betweenness/PageRank)")
    print("      - Community detection")
    print("      - Network fragmentation metrics")
    print("  [OK] Data export (CSV & JSON formats)")
    print("  [OK] EXTENDED simulations (T_max = 200, larger networks)")
    print()
    print("The simulation engine is production-ready with advanced visualization!")
    print()


if __name__ == '__main__':
    import numpy as np
    main()
