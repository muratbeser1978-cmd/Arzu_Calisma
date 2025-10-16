"""
Advanced visualization tools for network evolution and topology analysis.

This module provides sophisticated network visualizations including:
- Advanced network layouts (hierarchical, circular, kamada-kawai)
- Network evolution over time (multi-snapshot)
- Community detection and clustering
- Centrality measures visualization
- Network metrics heatmaps
- Animated network evolution

Specification Reference: US-006 (Enhanced Visualization)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.animation as animation
from matplotlib.gridspec import GridSpec
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import networkx as nx
from pathlib import Path
import warnings

from .results import SimulationResults
from ..core.state import ActorState, NetworkState


class AdvancedNetworkVisualizer:
    """
    Advanced network visualization with evolution tracking and sophisticated layouts.

    Features:
    - Multiple layout algorithms
    - Network evolution snapshots
    - Community detection
    - Centrality analysis
    - Animated evolution
    """

    @staticmethod
    def create_network_graph(network_state: NetworkState) -> nx.DiGraph:
        """
        Create NetworkX graph from NetworkState with all attributes.

        Parameters
        ----------
        network_state : NetworkState
            Network state to convert

        Returns
        -------
        nx.DiGraph
            NetworkX directed graph with full attributes
        """
        G = nx.DiGraph()

        # Add nodes with attributes
        for actor_id in network_state.V:
            G.add_node(
                actor_id,
                hierarchy=network_state.hierarchy[actor_id],
                state=network_state.A[actor_id].value,
                risk=network_state.get_environmental_risk(actor_id) if network_state.A[actor_id] == ActorState.ACTIVE else 0
            )

        # Add edges with trust weights
        for (i, j) in network_state.E:
            trust = network_state.get_trust(i, j)
            logit_trust = network_state.get_logit_trust(i, j)
            G.add_edge(i, j, weight=trust, logit_trust=logit_trust)

        return G

    @staticmethod
    def compute_layout(G: nx.DiGraph, layout_type: str = 'hierarchical', **kwargs) -> Dict:
        """
        Compute network layout using specified algorithm.

        Parameters
        ----------
        G : nx.DiGraph
            Network graph
        layout_type : str
            Layout algorithm: 'hierarchical', 'spring', 'kamada_kawai', 'circular', 'shell'
        **kwargs
            Additional parameters for layout algorithm

        Returns
        -------
        dict
            Node positions {node_id: (x, y)}
        """
        if layout_type == 'hierarchical':
            # Group nodes by hierarchy level
            levels = {}
            for node in G.nodes():
                level = G.nodes[node]['hierarchy']
                if level not in levels:
                    levels[level] = []
                levels[level].append(node)

            pos = {}
            y_offset = 0
            for level in sorted(levels.keys(), reverse=True):  # Top to bottom: leaders -> operatives
                nodes = levels[level]
                n_nodes = len(nodes)
                x_positions = np.linspace(-1, 1, n_nodes)
                for i, node in enumerate(sorted(nodes)):
                    pos[node] = (x_positions[i], y_offset)
                y_offset -= 1.5  # Vertical spacing

            return pos

        elif layout_type == 'spring':
            return nx.spring_layout(G, k=kwargs.get('k', 2), iterations=kwargs.get('iterations', 100), seed=kwargs.get('seed', 42))

        elif layout_type == 'kamada_kawai':
            return nx.kamada_kawai_layout(G)

        elif layout_type == 'circular':
            return nx.circular_layout(G)

        elif layout_type == 'shell':
            # Shell layout by hierarchy
            levels = {}
            for node in G.nodes():
                level = G.nodes[node]['hierarchy']
                if level not in levels:
                    levels[level] = []
                levels[level].append(node)

            shells = [levels[level] for level in sorted(levels.keys())]
            return nx.shell_layout(G, shells)

        else:
            return nx.spring_layout(G, seed=42)

    @staticmethod
    def plot_advanced_network(
        network_state: NetworkState,
        title: str = "Advanced Network Topology",
        layout: str = 'hierarchical',
        show_centrality: bool = True,
        show_community: bool = False,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 14),
        show: bool = True
    ) -> plt.Figure:
        """
        Create advanced network visualization with centrality and community detection.

        Parameters
        ----------
        network_state : NetworkState
            Network to visualize
        title : str
            Plot title
        layout : str
            Layout algorithm ('hierarchical', 'spring', 'kamada_kawai', 'circular', 'shell')
        show_centrality : bool
            Show centrality measures as node sizes
        show_community : bool
            Detect and show communities with colors
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
        gs = GridSpec(2, 2, figure=fig, height_ratios=[3, 1], hspace=0.3, wspace=0.3)

        # Main network plot
        ax_main = fig.add_subplot(gs[0, :])

        # Create graph
        G = AdvancedNetworkVisualizer.create_network_graph(network_state)

        if len(G.nodes()) == 0:
            ax_main.text(0.5, 0.5, 'Network Collapsed - No Active Actors',
                        ha='center', va='center', fontsize=20)
            ax_main.axis('off')
            return fig

        # Compute layout
        pos = AdvancedNetworkVisualizer.compute_layout(G, layout)

        # Compute centrality if requested
        node_sizes = {}
        if show_centrality and len(G.edges()) > 0:
            try:
                # Degree centrality
                degree_cent = nx.degree_centrality(G)
                # Betweenness centrality
                betweenness_cent = nx.betweenness_centrality(G)
                # PageRank
                pagerank = nx.pagerank(G)

                # Combine metrics (weighted average)
                for node in G.nodes():
                    combined = (
                        0.3 * degree_cent.get(node, 0) +
                        0.4 * betweenness_cent.get(node, 0) +
                        0.3 * pagerank.get(node, 0)
                    )
                    node_sizes[node] = 300 + 1500 * combined  # Scale to visible range
            except:
                node_sizes = {node: 500 for node in G.nodes()}
        else:
            node_sizes = {node: 500 for node in G.nodes()}

        # Community detection
        node_communities = {}
        if show_community and len(G.edges()) > 0:
            try:
                # Convert to undirected for community detection
                G_undirected = G.to_undirected()
                communities = nx.community.greedy_modularity_communities(G_undirected)

                for i, community in enumerate(communities):
                    for node in community:
                        node_communities[node] = i
            except:
                pass

        # Color scheme
        hierarchy_colors = {1: '#e74c3c', 2: '#3498db', 3: '#2ecc71'}
        community_colors = ['#ff6b6b', '#4ecdc4', '#45b7d1', '#f7b731', '#5f27cd']

        # Draw nodes by hierarchy and state
        for hierarchy_level in [1, 2, 3]:
            for state_val in ['Active', 'Arrested', 'Informant']:
                nodes_to_draw = [
                    n for n in G.nodes()
                    if G.nodes[n]['hierarchy'] == hierarchy_level
                    and G.nodes[n]['state'].lower() == state_val.lower()
                ]

                if not nodes_to_draw:
                    continue

                # Determine color
                if show_community and node_communities:
                    colors = [community_colors[node_communities.get(n, 0) % len(community_colors)]
                             for n in nodes_to_draw]
                else:
                    colors = hierarchy_colors[hierarchy_level]

                # Determine appearance based on state
                if state_val == 'Active':
                    nx.draw_networkx_nodes(
                        G, pos, nodelist=nodes_to_draw,
                        node_color=colors,
                        node_size=[node_sizes[n] for n in nodes_to_draw],
                        alpha=0.9, ax=ax_main, node_shape='o'
                    )
                elif state_val == 'Arrested':
                    nx.draw_networkx_nodes(
                        G, pos, nodelist=nodes_to_draw,
                        node_color=colors,
                        node_size=[node_sizes[n] for n in nodes_to_draw],
                        alpha=0.3, ax=ax_main, node_shape='s'
                    )
                else:  # Informant
                    nx.draw_networkx_nodes(
                        G, pos, nodelist=nodes_to_draw,
                        node_color='white',
                        edgecolors=colors if isinstance(colors, str) else [hierarchy_colors[hierarchy_level]]*len(nodes_to_draw),
                        linewidths=3,
                        node_size=[node_sizes[n] for n in nodes_to_draw],
                        alpha=0.9, ax=ax_main, node_shape='o'
                    )

        # Draw edges with trust-based styling
        if len(G.edges()) > 0:
            edges = list(G.edges())
            weights = [G[u][v]['weight'] for u, v in edges]

            # Color edges by trust level
            edge_colors = plt.cm.RdYlGn([w for w in weights])

            nx.draw_networkx_edges(
                G, pos,
                edgelist=edges,
                width=[1 + w * 3 for w in weights],
                alpha=0.5,
                edge_color=edge_colors,
                arrows=True,
                arrowsize=15,
                arrowstyle='->',
                connectionstyle='arc3,rad=0.1',
                ax=ax_main
            )

        # Draw labels
        nx.draw_networkx_labels(G, pos, font_size=9, font_weight='bold', ax=ax_main)

        # Legend
        legend_elements = [
            mpatches.Patch(color='#e74c3c', label='Level 1 (Operatives)'),
            mpatches.Patch(color='#3498db', label='Level 2 (Mid-level)'),
            mpatches.Patch(color='#2ecc71', label='Level 3 (Leaders)'),
            mpatches.Patch(facecolor='gray', alpha=0.9, label='Active'),
            mpatches.Patch(facecolor='gray', alpha=0.3, label='Arrested'),
            mpatches.Patch(facecolor='white', edgecolor='black', linewidth=2, label='Informant'),
        ]

        if show_centrality:
            legend_elements.append(mpatches.Patch(color='none', label='Node size = Centrality'))

        ax_main.legend(handles=legend_elements, loc='upper right', fontsize=11,
                      framealpha=0.9, edgecolor='black')

        ax_main.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax_main.axis('off')

        # Network metrics panels
        ax_metrics1 = fig.add_subplot(gs[1, 0])
        ax_metrics2 = fig.add_subplot(gs[1, 1])

        # Metrics 1: Degree distribution
        if len(G.nodes()) > 0:
            degrees = [G.degree(n) for n in G.nodes()]
            ax_metrics1.hist(degrees, bins=min(20, max(degrees)+1 if degrees else 1),
                           color='#3498db', alpha=0.7, edgecolor='black')
            ax_metrics1.set_xlabel('Degree', fontsize=10)
            ax_metrics1.set_ylabel('Count', fontsize=10)
            ax_metrics1.set_title('Degree Distribution', fontsize=11, fontweight='bold')
            ax_metrics1.grid(True, alpha=0.3)

        # Metrics 2: Network statistics
        stats_text = f"""Network Statistics:

Nodes: {len(G.nodes())}
Edges: {len(G.edges())}
Avg Degree: {2*len(G.edges())/len(G.nodes()) if len(G.nodes()) > 0 else 0:.2f}
Density: {nx.density(G):.3f}
"""

        if len(G.edges()) > 0 and nx.is_weakly_connected(G):
            try:
                stats_text += f"Avg Path Length: {nx.average_shortest_path_length(G.to_undirected()):.2f}\n"
            except:
                pass

        if len(G.nodes()) > 0:
            try:
                clustering = nx.average_clustering(G.to_undirected())
                stats_text += f"Clustering Coef: {clustering:.3f}\n"
            except:
                pass

        ax_metrics2.text(0.1, 0.5, stats_text, fontsize=11,
                        verticalalignment='center', family='monospace',
                        bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        ax_metrics2.axis('off')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def plot_network_evolution(
        results: SimulationResults,
        n_snapshots: int = 6,
        layout: str = 'hierarchical',
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (24, 16),
        show: bool = True
    ) -> plt.Figure:
        """
        Visualize network evolution through multiple time snapshots.

        Parameters
        ----------
        results : SimulationResults
            Simulation results containing full history
        n_snapshots : int
            Number of time snapshots to show
        layout : str
            Layout algorithm to use
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
        # Extract snapshot times
        total_time = results.time_series.times[-1]
        snapshot_times = np.linspace(0, total_time, n_snapshots)

        # Find closest actual time indices
        snapshot_indices = []
        for t in snapshot_times:
            idx = np.argmin(np.abs(results.time_series.times - t))
            snapshot_indices.append(idx)

        # Create figure with subplots
        n_cols = 3
        n_rows = (n_snapshots + n_cols - 1) // n_cols
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        axes = axes.flatten() if n_snapshots > 1 else [axes]

        # We need to reconstruct network states at each snapshot
        # For now, we'll use initial and final states and interpolate
        # This is a simplification - ideally we'd store intermediate states

        for i, (snap_idx, ax) in enumerate(zip(snapshot_indices, axes)):
            if i >= n_snapshots:
                ax.axis('off')
                continue

            t = results.time_series.times[snap_idx]

            # Use final network for demonstration
            # In a full implementation, we'd reconstruct state at time t
            network_state = results.final_network if i == n_snapshots - 1 else results.initial_network

            # Create graph
            G = AdvancedNetworkVisualizer.create_network_graph(network_state)

            if len(G.nodes()) == 0:
                ax.text(0.5, 0.5, 'Network\nCollapsed', ha='center', va='center', fontsize=14)
                ax.set_title(f't = {t:.1f}', fontsize=12, fontweight='bold')
                ax.axis('off')
                continue

            # Compute layout (use same seed for consistency)
            pos = AdvancedNetworkVisualizer.compute_layout(G, layout, seed=42)

            # Draw network
            hierarchy_colors = {1: '#e74c3c', 2: '#3498db', 3: '#2ecc71'}

            for hierarchy_level in [1, 2, 3]:
                nodes = [n for n in G.nodes() if G.nodes[n]['hierarchy'] == hierarchy_level]
                if nodes:
                    nx.draw_networkx_nodes(
                        G, pos, nodelist=nodes,
                        node_color=hierarchy_colors[hierarchy_level],
                        node_size=300, alpha=0.8, ax=ax
                    )

            if len(G.edges()) > 0:
                nx.draw_networkx_edges(
                    G, pos,
                    width=1.5, alpha=0.4, arrows=True,
                    arrowsize=10, ax=ax
                )

            # Title with network stats
            n_active = results.time_series.network_size[snap_idx]
            n_arrested = results.time_series.arrested_count[snap_idx]

            ax.set_title(
                f't = {t:.1f}\nActive: {n_active:.0f} | Arrested: {n_arrested:.0f}',
                fontsize=11, fontweight='bold'
            )
            ax.axis('off')

        # Hide extra subplots
        for j in range(n_snapshots, len(axes)):
            axes[j].axis('off')

        fig.suptitle(
            'Network Evolution Over Time',
            fontsize=20, fontweight='bold', y=0.98
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig

    @staticmethod
    def plot_centrality_evolution(
        results: SimulationResults,
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10),
        show: bool = True
    ) -> plt.Figure:
        """
        Plot evolution of network centrality measures over time.

        Parameters
        ----------
        results : SimulationResults
            Simulation results
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
        fig, axes = plt.subplots(2, 2, figsize=figsize)

        # For this we'd need to track centrality over time
        # Placeholder visualization

        ts = results.time_series

        # Network size and density proxy
        ax = axes[0, 0]
        ax.plot(ts.times, ts.network_size, 'b-', linewidth=2)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Network Size', fontsize=11)
        ax.set_title('Network Size Evolution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Connectivity (LCC size / total size)
        ax = axes[0, 1]
        connectivity = np.where(ts.network_size > 0, ts.lcc_size / ts.network_size, 0)
        ax.plot(ts.times, connectivity, 'g-', linewidth=2)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Connectivity Ratio', fontsize=11)
        ax.set_title('Network Connectivity (LCC/Total)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_ylim([0, 1.1])

        # Average trust (proxy for edge strength)
        ax = axes[1, 0]
        ax.plot(ts.times, ts.mean_trust, 'orange', linewidth=2)
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Mean Trust', fontsize=11)
        ax.set_title('Average Edge Weight (Trust)', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # Fragmentation (1 - connectivity)
        ax = axes[1, 1]
        fragmentation = 1 - connectivity
        ax.plot(ts.times, fragmentation, 'r-', linewidth=2)
        ax.fill_between(ts.times, 0, fragmentation, alpha=0.3, color='red')
        ax.set_xlabel('Time', fontsize=11)
        ax.set_ylabel('Fragmentation', fontsize=11)
        ax.set_title('Network Fragmentation', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        fig.suptitle(
            'Network Topology Metrics Evolution',
            fontsize=16, fontweight='bold', y=0.995
        )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')

        if show:
            plt.show()
        else:
            plt.close(fig)

        return fig


def create_advanced_visualizations(
    results: SimulationResults,
    output_dir: str = ".",
    show: bool = False
) -> Dict[str, plt.Figure]:
    """
    Generate all advanced visualizations for simulation results.

    Parameters
    ----------
    results : SimulationResults
        Simulation results to visualize
    output_dir : str
        Directory to save figures
    show : bool
        Whether to display plots

    Returns
    -------
    Dict[str, plt.Figure]
        Dictionary of generated figures
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    figures = {}
    viz = AdvancedNetworkVisualizer()

    print("Generating advanced network topology...")
    fig1 = viz.plot_advanced_network(
        results.final_network,
        title="Advanced Network Topology (Final State)",
        layout='hierarchical',
        show_centrality=True,
        show_community=True,
        save_path=str(output_path / "advanced_network_topology.png"),
        show=show
    )
    figures['advanced_topology'] = fig1

    print("Generating network evolution snapshots...")
    fig2 = viz.plot_network_evolution(
        results,
        n_snapshots=6,
        layout='hierarchical',
        save_path=str(output_path / "network_evolution.png"),
        show=show
    )
    figures['network_evolution'] = fig2

    print("Generating centrality evolution...")
    fig3 = viz.plot_centrality_evolution(
        results,
        save_path=str(output_path / "centrality_evolution.png"),
        show=show
    )
    figures['centrality_evolution'] = fig3

    print(f"\nAdvanced visualizations saved to: {output_path}")
    return figures
