"""
Comprehensive visualization tools for simulation results.

This module provides detailed plotting capabilities for:
- Network topology visualization
- Time series analysis
- Strategy comparison
- Event distribution analysis

Specification Reference: US-006 (Visualization)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import List, Optional, Tuple, Dict
import networkx as nx
from pathlib import Path

from .results import SimulationResults
from ..core.state import ActorState


class SimulationVisualizer:
    """
    Comprehensive visualization suite for simulation results.

    This class provides static methods for creating various plots and charts
    to analyze simulation outcomes.

    Examples
    --------
    >>> visualizer = SimulationVisualizer()
    >>> visualizer.plot_full_analysis(results, "output_dir")
    >>> visualizer.plot_network_evolution(results, save_path="network.png")
    """

    @staticmethod
    def plot_network_topology(
        network_state,
        title: str = "Network Topology",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 10),
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize network structure with hierarchical layout.

        Parameters
        ----------
        network_state : NetworkState
            Network to visualize
        title : str
            Plot title
        save_path : Optional[str]
            Path to save figure
        figsize : Tuple[int, int]
            Figure dimensions
        show : bool
            Whether to display the plot

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Build networkx graph
        G = nx.DiGraph()

        # Add nodes with hierarchy info
        for actor_id in network_state.V:
            G.add_node(
                actor_id,
                hierarchy=network_state.hierarchy[actor_id],
                state=network_state.A[actor_id].value
            )

        # Add edges with trust weights
        for (i, j) in network_state.E:
            trust = network_state.get_trust(i, j)
            G.add_edge(i, j, weight=trust)

        # Create hierarchical layout
        pos = nx.spring_layout(G, k=2, iterations=50, seed=42)

        # Separate nodes by state
        active_nodes = [n for n in G.nodes() if G.nodes[n]['state'] == 'active']
        arrested_nodes = [n for n in G.nodes() if G.nodes[n]['state'] == 'arrested']
        informant_nodes = [n for n in G.nodes() if G.nodes[n]['state'] == 'informant']

        # Separate nodes by hierarchy
        hierarchy_colors = {1: '#e74c3c', 2: '#3498db', 3: '#2ecc71'}

        # Draw nodes by hierarchy and state
        for hierarchy_level in [1, 2, 3]:
            level_nodes = [n for n in G.nodes() if G.nodes[n]['hierarchy'] == hierarchy_level]

            # Active nodes
            active_level = [n for n in level_nodes if n in active_nodes]
            if active_level:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=active_level,
                    node_color=hierarchy_colors[hierarchy_level],
                    node_size=500, alpha=0.9, ax=ax
                )

            # Arrested nodes (darker)
            arrested_level = [n for n in level_nodes if n in arrested_nodes]
            if arrested_level:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=arrested_level,
                    node_color=hierarchy_colors[hierarchy_level],
                    node_size=500, alpha=0.3, ax=ax,
                    node_shape='s'  # Square for arrested
                )

            # Informant nodes (with border)
            informant_level = [n for n in level_nodes if n in informant_nodes]
            if informant_level:
                nx.draw_networkx_nodes(
                    G, pos, nodelist=informant_level,
                    node_color='white',
                    edgecolors=hierarchy_colors[hierarchy_level],
                    linewidths=3,
                    node_size=500, alpha=0.9, ax=ax
                )

        # Draw edges with varying thickness based on trust
        edges = G.edges()
        weights = [G[u][v]['weight'] for u, v in edges]
        nx.draw_networkx_edges(
            G, pos, edgelist=edges,
            width=[w * 2 for w in weights],
            alpha=0.4, arrows=True,
            arrowsize=15, ax=ax
        )

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=8, ax=ax)

        # Create legend
        legend_elements = [
            mpatches.Patch(color='#e74c3c', label='Level 1 (Operatives)'),
            mpatches.Patch(color='#3498db', label='Level 2 (Mid-level)'),
            mpatches.Patch(color='#2ecc71', label='Level 3 (Leaders)'),
            mpatches.Patch(facecolor='gray', alpha=0.9, label='Active'),
            mpatches.Patch(facecolor='gray', alpha=0.3, label='Arrested'),
            mpatches.Patch(facecolor='white', edgecolor='black', label='Informant'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=10)

        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)
        ax.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_time_series(
        results: SimulationResults,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 10),
        show: bool = True
    ) -> plt.Figure:
        """
        Plot comprehensive time series analysis.

        Parameters
        ----------
        results : SimulationResults
            Simulation results to visualize
        save_path : Optional[str]
            Path to save figure
        figsize : Tuple[int, int]
            Figure dimensions
        show : bool
            Whether to display the plot

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(3, 2, figure=fig, hspace=0.3, wspace=0.3)

        ts = results.time_series

        # Plot 1: Network Size
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.plot(ts.times, ts.network_size, 'b-', linewidth=2, label='Active Actors')
        ax1.fill_between(ts.times, 0, ts.network_size, alpha=0.3)
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Network Size', fontsize=11)
        ax1.set_title('Network Size Evolution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Arrests & Conversions
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.plot(ts.times, ts.arrested_count, 'r-', linewidth=2, label='Arrests')
        ax2.plot(ts.times, ts.informant_count, 'g-', linewidth=2, label='Conversions')
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('Count', fontsize=11)
        ax2.set_title('Cumulative Arrests & Conversions', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: Law Enforcement Effectiveness
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.plot(ts.times, ts.effectiveness, 'purple', linewidth=2)
        ax3.axhline(y=results.parameters.P_0, color='k', linestyle='--',
                   alpha=0.5, label='Initial P₀')
        ax3.set_xlabel('Time', fontsize=11)
        ax3.set_ylabel('Effectiveness P(t)', fontsize=11)
        ax3.set_title('Law Enforcement Effectiveness', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Plot 4: Mean Trust Level
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.plot(ts.times, ts.mean_trust, 'orange', linewidth=2)
        ax4.set_xlabel('Time', fontsize=11)
        ax4.set_ylabel('Mean Trust w(t)', fontsize=11)
        ax4.set_title('Average Trust Between Actors', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        ax4.set_ylim([0, 1])

        # Plot 5: Mean Risk Perception
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.plot(ts.times, ts.mean_risk, 'brown', linewidth=2)
        ax5.set_xlabel('Time', fontsize=11)
        ax5.set_ylabel('Mean Risk R(t)', fontsize=11)
        ax5.set_title('Average Risk Perception', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)

        # Plot 6: Largest Connected Component
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.plot(ts.times, ts.lcc_size, 'teal', linewidth=2)
        ax6.fill_between(ts.times, 0, ts.lcc_size, alpha=0.3, color='teal')
        ax6.set_xlabel('Time', fontsize=11)
        ax6.set_ylabel('LCC Size', fontsize=11)
        ax6.set_title('Largest Connected Component', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)

        fig.suptitle(
            f'Simulation Time Series Analysis (Run: {results.run_id[:8]})',
            fontsize=16, fontweight='bold', y=0.995
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_strategy_comparison(
        results_list: List[SimulationResults],
        strategy_names: List[str],
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10),
        show: bool = True
    ) -> plt.Figure:
        """
        Compare multiple strategies side-by-side.

        Parameters
        ----------
        results_list : List[SimulationResults]
            List of simulation results to compare
        strategy_names : List[str]
            Names for each strategy
        save_path : Optional[str]
            Path to save figure
        figsize : Tuple[int, int]
            Figure dimensions
        show : bool
            Whether to display the plot

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig = plt.figure(figsize=figsize)
        gs = GridSpec(2, 3, figure=fig, hspace=0.3, wspace=0.3)

        colors = ['#e74c3c', '#3498db', '#2ecc71', '#f39c12', '#9b59b6']

        # Plot 1: Network Size Comparison
        ax1 = fig.add_subplot(gs[0, 0])
        for i, (results, name) in enumerate(zip(results_list, strategy_names)):
            ts = results.time_series
            ax1.plot(ts.times, ts.network_size, color=colors[i % len(colors)],
                    linewidth=2, label=name)
        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Network Size', fontsize=11)
        ax1.set_title('Network Size Evolution', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend()

        # Plot 2: Arrests Comparison
        ax2 = fig.add_subplot(gs[0, 1])
        for i, (results, name) in enumerate(zip(results_list, strategy_names)):
            ts = results.time_series
            ax2.plot(ts.times, ts.arrested_count, color=colors[i % len(colors)],
                    linewidth=2, label=name)
        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('Cumulative Arrests', fontsize=11)
        ax2.set_title('Arrest Rate Comparison', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend()

        # Plot 3: Effectiveness Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        for i, (results, name) in enumerate(zip(results_list, strategy_names)):
            ts = results.time_series
            ax3.plot(ts.times, ts.effectiveness, color=colors[i % len(colors)],
                    linewidth=2, label=name)
        ax3.set_xlabel('Time', fontsize=11)
        ax3.set_ylabel('Effectiveness P(t)', fontsize=11)
        ax3.set_title('Law Enforcement Effectiveness', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        ax3.legend()

        # Plot 4: Final Metrics Bar Chart
        ax4 = fig.add_subplot(gs[1, 0])
        x = np.arange(len(strategy_names))
        width = 0.35
        arrests = [r.total_arrests for r in results_list]
        conversions = [r.total_conversions for r in results_list]

        ax4.bar(x - width/2, arrests, width, label='Arrests', color='#e74c3c', alpha=0.8)
        ax4.bar(x + width/2, conversions, width, label='Conversions', color='#2ecc71', alpha=0.8)
        ax4.set_xlabel('Strategy', fontsize=11)
        ax4.set_ylabel('Count', fontsize=11)
        ax4.set_title('Total Arrests & Conversions', fontsize=12, fontweight='bold')
        ax4.set_xticks(x)
        ax4.set_xticklabels(strategy_names, rotation=45, ha='right')
        ax4.legend()
        ax4.grid(True, alpha=0.3, axis='y')

        # Plot 5: Trust Evolution
        ax5 = fig.add_subplot(gs[1, 1])
        for i, (results, name) in enumerate(zip(results_list, strategy_names)):
            ts = results.time_series
            ax5.plot(ts.times, ts.mean_trust, color=colors[i % len(colors)],
                    linewidth=2, label=name)
        ax5.set_xlabel('Time', fontsize=11)
        ax5.set_ylabel('Mean Trust', fontsize=11)
        ax5.set_title('Trust Evolution Comparison', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        ax5.legend()

        # Plot 6: Final Effectiveness Bar Chart
        ax6 = fig.add_subplot(gs[1, 2])
        final_eff = [r.final_effectiveness for r in results_list]
        initial_eff = [r.parameters.P_0 for r in results_list]
        change = [f - i for f, i in zip(final_eff, initial_eff)]

        bars = ax6.bar(strategy_names, change,
                      color=['#2ecc71' if c > 0 else '#e74c3c' for c in change],
                      alpha=0.8)
        ax6.axhline(y=0, color='k', linestyle='-', linewidth=0.5)
        ax6.set_xlabel('Strategy', fontsize=11)
        ax6.set_ylabel('Change in Effectiveness', fontsize=11)
        ax6.set_title('Effectiveness Change (Final - Initial)', fontsize=12, fontweight='bold')
        ax6.tick_params(axis='x', rotation=45)
        plt.setp(ax6.xaxis.get_majorticklabels(), rotation=45, ha='right')
        ax6.grid(True, alpha=0.3, axis='y')

        fig.suptitle(
            'Strategy Comparison Analysis',
            fontsize=16, fontweight='bold', y=0.995
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_event_distribution(
        results: SimulationResults,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (14, 6),
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize distribution of events over time.

        Parameters
        ----------
        results : SimulationResults
            Simulation results to visualize
        save_path : Optional[str]
            Path to save figure
        figsize : Tuple[int, int]
            Figure dimensions
        show : bool
            Whether to display the plot

        Returns
        -------
        plt.Figure
            The created figure
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Extract event data
        arrest_times = [e.timestamp for e in results.events if e.event_type.value == 'arrest']
        conversion_times = [e.timestamp for e in results.events if e.event_type.value == 'conversion']

        # Event timeline
        if arrest_times:
            ax1.scatter(arrest_times, [1]*len(arrest_times),
                       c='red', s=100, alpha=0.6, label='Arrests', marker='x')
        if conversion_times:
            ax1.scatter(conversion_times, [0]*len(conversion_times),
                       c='green', s=100, alpha=0.6, label='Conversions', marker='o')

        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Event Type', fontsize=11)
        ax1.set_yticks([0, 1])
        ax1.set_yticklabels(['Conversion', 'Arrest'])
        ax1.set_title('Event Timeline', fontsize=12, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3, axis='x')

        # Event histogram
        all_times = arrest_times + conversion_times
        if all_times:
            bins = np.linspace(0, results.parameters.T_max, 30)
            ax2.hist(arrest_times, bins=bins, alpha=0.6, color='red',
                    label=f'Arrests (n={len(arrest_times)})')
            ax2.hist(conversion_times, bins=bins, alpha=0.6, color='green',
                    label=f'Conversions (n={len(conversion_times)})')
            ax2.set_xlabel('Time', fontsize=11)
            ax2.set_ylabel('Event Count', fontsize=11)
            ax2.set_title('Event Frequency Distribution', fontsize=12, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3, axis='y')

        fig.suptitle(
            f'Event Distribution Analysis (Run: {results.run_id[:8]})',
            fontsize=16, fontweight='bold', y=1.02
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()

        return fig

    @staticmethod
    def plot_full_analysis(
        results: SimulationResults,
        output_dir: str = ".",
        prefix: str = "simulation",
        show: bool = True
    ) -> Dict[str, plt.Figure]:
        """
        Generate complete visualization suite and save all plots.

        Parameters
        ----------
        results : SimulationResults
            Simulation results to visualize
        output_dir : str
            Directory to save figures
        prefix : str
            Prefix for output filenames
        show : bool
            Whether to display plots

        Returns
        -------
        Dict[str, plt.Figure]
            Dictionary of plot names to figures
        """
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        figures = {}

        # Network topology
        print("Generating network topology visualization...")
        fig_network = SimulationVisualizer.plot_network_topology(
            results.final_network,
            title=f"Final Network State (t={results.time_series.times[-1]:.1f})",
            save_path=str(output_path / f"{prefix}_network_topology.png"),
            show=show
        )
        figures['network_topology'] = fig_network

        # Time series
        print("Generating time series plots...")
        fig_ts = SimulationVisualizer.plot_time_series(
            results,
            save_path=str(output_path / f"{prefix}_time_series.png"),
            show=show
        )
        figures['time_series'] = fig_ts

        # Event distribution
        print("Generating event distribution plots...")
        fig_events = SimulationVisualizer.plot_event_distribution(
            results,
            save_path=str(output_path / f"{prefix}_event_distribution.png"),
            show=show
        )
        figures['event_distribution'] = fig_events

        print(f"\nAll visualizations saved to: {output_path}")
        print(f"  - {prefix}_network_topology.png")
        print(f"  - {prefix}_time_series.png")
        print(f"  - {prefix}_event_distribution.png")

        return figures


def visualize_results(
    results: SimulationResults,
    output_dir: str = ".",
    show: bool = True
) -> None:
    """
    Convenience function to generate all visualizations.

    Parameters
    ----------
    results : SimulationResults
        Simulation results to visualize
    output_dir : str
        Directory to save figures
    show : bool
        Whether to display plots

    Examples
    --------
    >>> from organized_crime_network.simulation import visualization
    >>> visualization.visualize_results(results, output_dir="plots")
    """
    visualizer = SimulationVisualizer()
    visualizer.plot_full_analysis(results, output_dir=output_dir, show=show)
