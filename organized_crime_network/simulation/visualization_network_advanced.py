"""
Advanced network visualization system for organized crime networks.

This module provides publication-quality network visualizations with:
- Interactive layouts
- Temporal evolution animations
- Community structure analysis
- Centrality heatmaps
- 3D network visualization
- Force-directed layouts
- Hierarchical clustering views
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation
import json

from ..core.state import NetworkState, ActorState
from .results import SimulationResults


class AdvancedNetworkVisualizer:
    """
    Advanced network visualization with multiple representation modes.
    """

    def __init__(self):
        self.color_schemes = {
            'hierarchy': {
                1: '#e74c3c',  # Red - Operatives
                2: '#3498db',  # Blue - Mid-level
                3: '#2ecc71',  # Green - Leaders
            },
            'state': {
                ActorState.ACTIVE: '#2ecc71',      # Green
                ActorState.ARRESTED: '#e74c3c',    # Red
                ActorState.INFORMANT: '#f39c12',   # Orange
            }
        }

    def plot_circular_hierarchy_network(
        self,
        network_state: NetworkState,
        title: str = "Circular Hierarchy Network",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (20, 20)
    ):
        """
        Create circular hierarchical network visualization.

        Leaders at center, mid-level in middle ring, operatives in outer ring.
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_facecolor('#f8f9fa')

        # Build graph
        G = nx.DiGraph()
        for actor_id in network_state.V:
            G.add_node(actor_id,
                      hierarchy=network_state.hierarchy[actor_id],
                      state=network_state.A[actor_id])

        for (i, j) in network_state.E:
            trust = network_state.get_trust(i, j)
            G.add_edge(i, j, weight=trust)

        # Group by hierarchy
        hierarchy_groups = {}
        for node in G.nodes():
            h = G.nodes[node]['hierarchy']
            if h not in hierarchy_groups:
                hierarchy_groups[h] = []
            hierarchy_groups[h].append(node)

        # Circular layout by hierarchy
        pos = {}
        max_level = max(hierarchy_groups.keys())

        for level, nodes in hierarchy_groups.items():
            # Radius increases with level (leaders at center)
            radius = (max_level - level + 1) * 2.0
            n = len(nodes)

            # Distribute evenly on circle
            angles = np.linspace(0, 2 * np.pi, n, endpoint=False)

            for i, node in enumerate(nodes):
                pos[node] = (radius * np.cos(angles[i]), radius * np.sin(angles[i]))

        # Draw edges with varying opacity based on trust
        for (i, j) in G.edges():
            trust = G[i][j]['weight']
            x = [pos[i][0], pos[j][0]]
            y = [pos[i][1], pos[j][1]]

            # Color based on trust level
            if trust > 0.7:
                color = '#2ecc71'  # High trust - green
                alpha = 0.6
            elif trust > 0.4:
                color = '#f39c12'  # Medium trust - orange
                alpha = 0.4
            else:
                color = '#e74c3c'  # Low trust - red
                alpha = 0.2

            ax.plot(x, y, color=color, alpha=alpha, linewidth=trust*2, zorder=1)

        # Draw nodes
        for node in G.nodes():
            x, y = pos[node]
            hierarchy = G.nodes[node]['hierarchy']
            state = G.nodes[node]['state']

            # Size based on hierarchy
            size = 300 + (hierarchy - 1) * 200

            # Color based on state
            if state == ActorState.ACTIVE:
                color = self.color_schemes['hierarchy'][hierarchy]
                edgecolor = 'black'
                linewidth = 2
            elif state == ActorState.ARRESTED:
                color = '#95a5a6'  # Gray
                edgecolor = 'red'
                linewidth = 3
            else:  # INFORMANT
                color = '#f39c12'  # Orange
                edgecolor = 'black'
                linewidth = 3

            # Shape based on hierarchy
            if hierarchy == 3:
                marker = '*'  # Star for leaders
            elif hierarchy == 2:
                marker = 'D'  # Diamond for mid-level
            else:
                marker = 'o'  # Circle for operatives

            ax.scatter(x, y, s=size, c=color, marker=marker,
                      edgecolors=edgecolor, linewidth=linewidth, zorder=2)

            # Label for leaders
            if hierarchy == 3:
                ax.text(x, y, str(node), ha='center', va='center',
                       fontsize=10, fontweight='bold', zorder=3)

        # Add legend
        legend_elements = [
            mpatches.Patch(color=self.color_schemes['hierarchy'][1], label='Operatives (L1)'),
            mpatches.Patch(color=self.color_schemes['hierarchy'][2], label='Mid-level (L2)'),
            mpatches.Patch(color=self.color_schemes['hierarchy'][3], label='Leaders (L3)'),
            mpatches.Patch(color='#2ecc71', alpha=0.6, label='High Trust (>0.7)'),
            mpatches.Patch(color='#f39c12', alpha=0.4, label='Medium Trust (0.4-0.7)'),
            mpatches.Patch(color='#e74c3c', alpha=0.2, label='Low Trust (<0.4)'),
        ]
        ax.legend(handles=legend_elements, loc='upper right', fontsize=12, framealpha=0.9)

        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)
        ax.axis('off')
        ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_force_directed_network(
        self,
        network_state: NetworkState,
        title: str = "Force-Directed Network Layout",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (20, 16)
    ):
        """
        Force-directed layout with community detection and centrality.
        """
        fig, axes = plt.subplots(2, 2, figsize=figsize, facecolor='white')
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)

        # Build graph
        G = nx.DiGraph()
        for actor_id in network_state.V:
            G.add_node(actor_id,
                      hierarchy=network_state.hierarchy[actor_id],
                      state=network_state.A[actor_id])

        for (i, j) in network_state.E:
            trust = network_state.get_trust(i, j)
            G.add_edge(i, j, weight=trust)

        # Convert to undirected for layout
        G_undirected = G.to_undirected()

        # Force-directed layout
        pos = nx.spring_layout(G_undirected, k=2, iterations=100, seed=42)

        # 1. Hierarchy-based coloring
        ax = axes[0, 0]
        ax.set_facecolor('#f8f9fa')
        self._draw_network_on_axis(ax, G, pos, color_by='hierarchy', title='Hierarchy Structure')

        # 2. Community detection
        ax = axes[0, 1]
        ax.set_facecolor('#f8f9fa')
        communities = list(nx.community.greedy_modularity_communities(G_undirected))
        self._draw_network_on_axis(ax, G, pos, color_by='community',
                                   communities=communities, title='Community Structure')

        # 3. Degree centrality
        ax = axes[1, 0]
        ax.set_facecolor('#f8f9fa')
        degree_cent = nx.degree_centrality(G)
        self._draw_network_on_axis(ax, G, pos, color_by='centrality',
                                   centrality=degree_cent, title='Degree Centrality')

        # 4. Betweenness centrality
        ax = axes[1, 1]
        ax.set_facecolor('#f8f9fa')
        betweenness_cent = nx.betweenness_centrality(G)
        self._draw_network_on_axis(ax, G, pos, color_by='centrality',
                                   centrality=betweenness_cent, title='Betweenness Centrality')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()
        else:
            plt.close()

    def _draw_network_on_axis(
        self,
        ax,
        G,
        pos,
        color_by='hierarchy',
        communities=None,
        centrality=None,
        title=''
    ):
        """Helper to draw network on axis with different coloring schemes."""

        # Draw edges
        for (i, j) in G.edges():
            trust = G[i][j]['weight']
            x = [pos[i][0], pos[j][0]]
            y = [pos[i][1], pos[j][1]]
            ax.plot(x, y, color='gray', alpha=0.3, linewidth=trust*2, zorder=1)

        # Draw nodes
        for node in G.nodes():
            x, y = pos[node]
            hierarchy = G.nodes[node]['hierarchy']
            state = G.nodes[node]['state']

            # Determine color
            if color_by == 'hierarchy':
                color = self.color_schemes['hierarchy'][hierarchy]
            elif color_by == 'community' and communities:
                # Find which community this node belongs to
                comm_idx = 0
                for idx, comm in enumerate(communities):
                    if node in comm:
                        comm_idx = idx
                        break
                colors_palette = sns.color_palette('husl', len(communities))
                color = colors_palette[comm_idx]
            elif color_by == 'centrality' and centrality:
                cent_value = centrality[node]
                color = plt.cm.RdYlGn(cent_value)
            else:
                color = 'gray'

            # Size based on hierarchy
            size = 200 + (hierarchy - 1) * 150

            # Border based on state
            if state == ActorState.ARRESTED:
                edgecolor = 'red'
                linewidth = 3
            elif state == ActorState.INFORMANT:
                edgecolor = 'orange'
                linewidth = 3
            else:
                edgecolor = 'black'
                linewidth = 1.5

            ax.scatter(x, y, s=size, c=[color], edgecolors=edgecolor,
                      linewidth=linewidth, zorder=2)

        ax.set_title(title, fontsize=14, fontweight='bold')
        ax.axis('off')
        ax.set_aspect('equal')

    def plot_adjacency_matrix_heatmap(
        self,
        network_state: NetworkState,
        title: str = "Network Adjacency Matrix",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (16, 14)
    ):
        """
        Adjacency matrix heatmap with hierarchical clustering.
        """
        fig, axes = plt.subplots(1, 2, figsize=figsize, facecolor='white')
        fig.suptitle(title, fontsize=18, fontweight='bold')

        # Build adjacency matrix
        actors = sorted(network_state.V)
        n = len(actors)
        adj_matrix = np.zeros((n, n))

        actor_to_idx = {actor: idx for idx, actor in enumerate(actors)}

        for (i, j) in network_state.E:
            if i in actor_to_idx and j in actor_to_idx:
                trust = network_state.get_trust(i, j)
                adj_matrix[actor_to_idx[i], actor_to_idx[j]] = trust

        # 1. Raw adjacency matrix
        ax = axes[0]
        im = ax.imshow(adj_matrix, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        ax.set_title('Trust Adjacency Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Actor', fontsize=12)
        ax.set_ylabel('Source Actor', fontsize=12)

        # Add colorbar
        cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.set_label('Trust Level', fontsize=12)

        # Add hierarchy dividers
        hierarchy_counts = {}
        for actor in actors:
            h = network_state.hierarchy[actor]
            hierarchy_counts[h] = hierarchy_counts.get(h, 0) + 1

        cumsum = 0
        for h in sorted(hierarchy_counts.keys()):
            cumsum += hierarchy_counts[h]
            ax.axhline(cumsum - 0.5, color='white', linewidth=2)
            ax.axvline(cumsum - 0.5, color='white', linewidth=2)

        # 2. Hierarchically clustered
        ax = axes[1]

        # Perform hierarchical clustering
        from scipy.cluster.hierarchy import dendrogram, linkage

        # Make symmetric for clustering (combine with transpose)
        adj_symmetric = (adj_matrix + adj_matrix.T) / 2.0

        # Convert to distance matrix
        dist_matrix = 1 - adj_symmetric
        dist_matrix[dist_matrix < 0] = 0

        # Ensure diagonal is zero
        np.fill_diagonal(dist_matrix, 0)

        # Linkage using condensed form
        from scipy.spatial.distance import squareform
        condensed_dist = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='average')
        dendro = dendrogram(linkage_matrix, no_plot=True)

        # Reorder matrix (use original directed matrix)
        idx = dendro['leaves']
        adj_matrix_clustered = adj_matrix[idx, :][:, idx]

        im2 = ax.imshow(adj_matrix_clustered, cmap='RdYlGn', aspect='auto', interpolation='nearest')
        ax.set_title('Hierarchically Clustered Matrix', fontsize=14, fontweight='bold')
        ax.set_xlabel('Target Actor (clustered)', fontsize=12)
        ax.set_ylabel('Source Actor (clustered)', fontsize=12)

        cbar2 = plt.colorbar(im2, ax=ax, fraction=0.046, pad=0.04)
        cbar2.set_label('Trust Level', fontsize=12)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_network_3d(
        self,
        network_state: NetworkState,
        title: str = "3D Network Visualization",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (16, 12)
    ):
        """
        3D network visualization with z-axis as hierarchy level.
        """
        fig = plt.figure(figsize=figsize, facecolor='white')
        ax = fig.add_subplot(111, projection='3d', facecolor='#f8f9fa')

        # Build graph
        G = nx.DiGraph()
        for actor_id in network_state.V:
            G.add_node(actor_id,
                      hierarchy=network_state.hierarchy[actor_id],
                      state=network_state.A[actor_id])

        for (i, j) in network_state.E:
            trust = network_state.get_trust(i, j)
            G.add_edge(i, j, weight=trust)

        # 2D layout (x, y)
        G_undirected = G.to_undirected()
        pos_2d = nx.spring_layout(G_undirected, seed=42)

        # 3D positions: z = hierarchy level
        pos_3d = {}
        for node in G.nodes():
            x, y = pos_2d[node]
            z = G.nodes[node]['hierarchy'] * 2.0  # Spread levels vertically
            pos_3d[node] = (x, y, z)

        # Draw edges
        for (i, j) in G.edges():
            trust = G[i][j]['weight']
            x = [pos_3d[i][0], pos_3d[j][0]]
            y = [pos_3d[i][1], pos_3d[j][1]]
            z = [pos_3d[i][2], pos_3d[j][2]]

            color = plt.cm.RdYlGn(trust)
            ax.plot(x, y, z, color=color, alpha=0.4, linewidth=trust*2)

        # Draw nodes
        for node in G.nodes():
            x, y, z = pos_3d[node]
            hierarchy = G.nodes[node]['hierarchy']
            state = G.nodes[node]['state']

            color = self.color_schemes['hierarchy'][hierarchy]
            size = 100 + (hierarchy - 1) * 100

            if state == ActorState.ARRESTED:
                marker = 'x'
                edgecolor = 'red'
            elif state == ActorState.INFORMANT:
                marker = 's'
                edgecolor = 'orange'
            else:
                marker = 'o'
                edgecolor = 'black'

            ax.scatter(x, y, z, s=size, c=[color], marker=marker,
                      edgecolors=edgecolor, linewidths=2, depthshade=True)

        # Labels
        ax.set_xlabel('X', fontsize=12)
        ax.set_ylabel('Y', fontsize=12)
        ax.set_zlabel('Hierarchy Level', fontsize=12)
        ax.set_title(title, fontsize=16, fontweight='bold', pad=20)

        # Set z-axis ticks
        ax.set_zticks([2, 4, 6])
        ax.set_zticklabels(['L1 (Operatives)', 'L2 (Mid-level)', 'L3 (Leaders)'])

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_temporal_network_evolution_grid(
        self,
        results: SimulationResults,
        n_snapshots: int = 9,
        title: str = "Network Evolution Over Time",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (24, 18)
    ):
        """
        Grid of network snapshots showing temporal evolution.
        """
        fig, axes = plt.subplots(3, 3, figsize=figsize, facecolor='white')
        fig.suptitle(title, fontsize=22, fontweight='bold', y=0.98)
        axes = axes.flatten()

        # Get snapshot times
        total_time = results.time_series.times[-1]
        snapshot_times = np.linspace(0, total_time, n_snapshots)

        # Build initial graph for consistent layout
        G_layout = nx.DiGraph()
        for actor_id in results.initial_network.V:
            G_layout.add_node(actor_id)
        for (i, j) in results.initial_network.E:
            G_layout.add_edge(i, j)

        # Consistent layout
        pos = nx.spring_layout(G_layout.to_undirected(), k=2, iterations=100, seed=42)

        for idx, t in enumerate(snapshot_times):
            ax = axes[idx]
            ax.set_facecolor('#f8f9fa')

            # Find closest time in time series
            time_idx = np.argmin(np.abs(results.time_series.times - t))

            # Get network state at this time
            # (simplified - just show active count and arrested count)
            active_count = results.time_series.network_size[time_idx]
            arrested_count = results.time_series.arrested_count[time_idx]
            informant_count = results.time_series.informant_count[time_idx]

            # Draw simplified network
            for (i, j) in G_layout.edges():
                x = [pos[i][0], pos[j][0]]
                y = [pos[i][1], pos[j][1]]
                ax.plot(x, y, color='gray', alpha=0.2, linewidth=1, zorder=1)

            # Draw nodes (simplified state representation)
            for node in G_layout.nodes():
                x, y = pos[node]

                # Estimate if arrested based on counts
                if node < arrested_count:
                    if node < informant_count:
                        color = '#f39c12'  # Informant
                    else:
                        color = '#e74c3c'  # Arrested
                else:
                    color = '#2ecc71'  # Active

                ax.scatter(x, y, s=100, c=color, edgecolors='black',
                          linewidth=1, zorder=2)

            # Stats box
            stats_text = f't = {t:.1f}\n'
            stats_text += f'Active: {active_count}\n'
            stats_text += f'Arrested: {arrested_count}\n'
            stats_text += f'Informants: {informant_count}'

            ax.text(0.02, 0.98, stats_text, transform=ax.transAxes,
                   fontsize=10, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

            ax.set_title(f'Snapshot {idx+1}: t={t:.1f}', fontsize=12, fontweight='bold')
            ax.axis('off')
            ax.set_aspect('equal')

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()
        else:
            plt.close()


def create_advanced_network_visualizations(
    results: SimulationResults,
    output_dir: str = "advanced_network_viz",
    show: bool = False
):
    """
    Create complete suite of advanced network visualizations.

    Parameters
    ----------
    results : SimulationResults
        Simulation results
    output_dir : str
        Output directory
    show : bool
        Whether to display plots
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    viz = AdvancedNetworkVisualizer()

    print("Creating advanced network visualizations...")

    # 1. Circular hierarchy
    print("  [1/5] Circular hierarchy layout...")
    viz.plot_circular_hierarchy_network(
        results.initial_network,
        title="Initial Network - Circular Hierarchy",
        save_path=str(output_path / "network_circular_hierarchy.png"),
        show=show
    )

    # 2. Force-directed with analysis
    print("  [2/5] Force-directed multi-view...")
    viz.plot_force_directed_network(
        results.initial_network,
        title="Network Analysis - Force-Directed Layout",
        save_path=str(output_path / "network_force_directed_analysis.png"),
        show=show
    )

    # 3. Adjacency matrix heatmap
    print("  [3/5] Adjacency matrix heatmap...")
    viz.plot_adjacency_matrix_heatmap(
        results.initial_network,
        title="Trust Adjacency Matrix",
        save_path=str(output_path / "network_adjacency_heatmap.png"),
        show=show
    )

    # 4. 3D visualization
    print("  [4/5] 3D network visualization...")
    viz.plot_network_3d(
        results.initial_network,
        title="3D Network Visualization",
        save_path=str(output_path / "network_3d_visualization.png"),
        show=show
    )

    # 5. Temporal evolution grid
    print("  [5/5] Temporal evolution grid...")
    viz.plot_temporal_network_evolution_grid(
        results,
        n_snapshots=9,
        title="Network Evolution Over Time (9 Snapshots)",
        save_path=str(output_path / "network_temporal_evolution_grid.png"),
        show=show
    )

    print(f"\nAll advanced network visualizations saved to: {output_dir}/")
    print("  - network_circular_hierarchy.png")
    print("  - network_force_directed_analysis.png")
    print("  - network_adjacency_heatmap.png")
    print("  - network_3d_visualization.png")
    print("  - network_temporal_evolution_grid.png")
