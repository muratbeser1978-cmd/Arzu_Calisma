"""
Ultra-advanced static visualizations.

This module provides:
- Sankey diagrams for flow analysis
- Radial dendrogram clustering
- Chord diagrams for connections
- Network metrics dashboard
- Multi-layer network views
- Publication-quality figures
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Wedge
import seaborn as sns
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform

from ..core.state import NetworkState, ActorState
from .results import SimulationResults


class UltraAdvancedVisualizer:
    """
    Ultra-advanced static visualizations for network analysis.
    """

    def __init__(self):
        sns.set_style("white")
        self.color_schemes = {
            'hierarchy': {1: '#e74c3c', 2: '#3498db', 3: '#2ecc71'},
            'state': {
                ActorState.ACTIVE: '#2ecc71',
                ActorState.ARRESTED: '#e74c3c',
                ActorState.INFORMANT: '#f39c12',
            }
        }

    def plot_radial_dendrogram(
        self,
        network_state: NetworkState,
        title: str = "Radial Hierarchical Clustering",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (16, 16)
    ):
        """
        Create radial dendrogram showing network clustering.
        """
        fig, ax = plt.subplots(figsize=figsize, subplot_kw=dict(polar=True), facecolor='white')

        # Build adjacency matrix
        actors = sorted(network_state.V)
        n = len(actors)
        adj_matrix = np.zeros((n, n))
        actor_to_idx = {actor: idx for idx, actor in enumerate(actors)}

        for (i, j) in network_state.E:
            if i in actor_to_idx and j in actor_to_idx:
                trust = network_state.get_trust(i, j)
                adj_matrix[actor_to_idx[i], actor_to_idx[j]] = trust

        # Symmetrize
        adj_symmetric = (adj_matrix + adj_matrix.T) / 2.0

        # Distance matrix
        dist_matrix = 1 - adj_symmetric
        dist_matrix[dist_matrix < 0] = 0
        np.fill_diagonal(dist_matrix, 0)

        # Hierarchical clustering
        condensed_dist = squareform(dist_matrix, checks=False)
        linkage_matrix = linkage(condensed_dist, method='ward')

        # Convert to radial coordinates
        dendro = dendrogram(linkage_matrix, no_plot=True)
        leaves_order = dendro['leaves']

        # Plot leaves in circle
        n_leaves = len(leaves_order)
        angles = np.linspace(0, 2 * np.pi, n_leaves, endpoint=False)

        for i, (leaf_idx, angle) in enumerate(zip(leaves_order, angles)):
            actor_id = actors[leaf_idx]
            hierarchy = network_state.hierarchy[actor_id]
            state = network_state.A[actor_id]

            # Color
            if state == ActorState.ACTIVE:
                color = self.color_schemes['hierarchy'][hierarchy]
            elif state == ActorState.ARRESTED:
                color = '#95a5a6'
            else:
                color = '#f39c12'

            # Plot point
            ax.scatter(angle, 1.0, s=100, c=color, edgecolors='black', linewidth=1, zorder=3)

            # Label
            if hierarchy == 3:  # Leaders
                ax.text(angle, 1.05, str(actor_id), ha='center', va='bottom',
                       fontsize=8, fontweight='bold')

        # Draw clustering connections
        icoord = np.array(dendro['icoord'])
        dcoord = np.array(dendro['dcoord'])

        # Normalize distances
        max_dist = np.max(dcoord)
        dcoord_norm = 1.0 - (dcoord / max_dist * 0.3)  # Scale to [0.7, 1.0]

        for xs, ys in zip(icoord, dcoord_norm):
            # Convert x coordinates to angles
            angles_line = []
            for x in xs:
                leaf_idx = int((x - 5) / 10)  # Dendrogram x spacing
                if 0 <= leaf_idx < n_leaves:
                    angles_line.append(angles[leaf_idx])
                else:
                    angles_line.append(angles[min(leaf_idx, n_leaves-1)])

            ax.plot(angles_line, ys, 'k-', alpha=0.3, linewidth=1)

        ax.set_ylim(0, 1.1)
        ax.set_theta_zero_location('N')
        ax.set_theta_direction(-1)
        ax.set_rticks([])
        ax.grid(False)
        ax.spines['polar'].set_visible(False)

        ax.set_title(title, fontsize=18, fontweight='bold', pad=30)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_network_metrics_dashboard(
        self,
        network_state: NetworkState,
        title: str = "Network Metrics Dashboard",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (20, 12)
    ):
        """
        Comprehensive network metrics visualization.
        """
        fig = plt.figure(figsize=figsize, facecolor='white')
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)

        # Build graph
        G = nx.DiGraph()
        for actor_id in network_state.V:
            G.add_node(actor_id, hierarchy=network_state.hierarchy[actor_id])
        for (i, j) in network_state.E:
            trust = network_state.get_trust(i, j)
            G.add_edge(i, j, weight=trust)

        G_undirected = G.to_undirected()

        # 1. Degree distribution
        ax = fig.add_subplot(gs[0, 0])
        degrees = [G.degree(n) for n in G.nodes()]
        ax.hist(degrees, bins=20, color='steelblue', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Degree', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Degree Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 2. Degree centrality
        ax = fig.add_subplot(gs[0, 1])
        degree_cent = nx.degree_centrality(G)
        values = list(degree_cent.values())
        ax.hist(values, bins=20, color='coral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Degree Centrality', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Degree Centrality Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 3. Betweenness centrality
        ax = fig.add_subplot(gs[0, 2])
        betweenness = nx.betweenness_centrality(G)
        values = list(betweenness.values())
        ax.hist(values, bins=20, color='seagreen', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Betweenness Centrality', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Betweenness Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 4. Closeness centrality
        ax = fig.add_subplot(gs[0, 3])
        if nx.is_connected(G_undirected):
            closeness = nx.closeness_centrality(G)
            values = list(closeness.values())
            ax.hist(values, bins=20, color='mediumpurple', edgecolor='black', alpha=0.7)
        else:
            ax.text(0.5, 0.5, 'Network\nDisconnected', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='red')
        ax.set_xlabel('Closeness Centrality', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Closeness Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 5. Clustering coefficient
        ax = fig.add_subplot(gs[1, 0])
        clustering = nx.clustering(G_undirected)
        values = list(clustering.values())
        ax.hist(values, bins=20, color='gold', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Clustering Coefficient', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Local Clustering', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 6. PageRank
        ax = fig.add_subplot(gs[1, 1])
        pagerank = nx.pagerank(G)
        values = list(pagerank.values())
        ax.hist(values, bins=20, color='crimson', edgecolor='black', alpha=0.7)
        ax.set_xlabel('PageRank', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('PageRank Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 7. Eigenvector centrality
        ax = fig.add_subplot(gs[1, 2])
        try:
            eigenvector = nx.eigenvector_centrality(G, max_iter=1000)
            values = list(eigenvector.values())
            ax.hist(values, bins=20, color='teal', edgecolor='black', alpha=0.7)
        except:
            ax.text(0.5, 0.5, 'Cannot\nCompute', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14, color='red')
        ax.set_xlabel('Eigenvector Centrality', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Eigenvector Centrality', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 8. Trust distribution
        ax = fig.add_subplot(gs[1, 3])
        trust_values = [G[i][j]['weight'] for (i, j) in G.edges()]
        ax.hist(trust_values, bins=20, color='lightcoral', edgecolor='black', alpha=0.7)
        ax.set_xlabel('Trust Level', fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.set_title('Edge Trust Distribution', fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.3)

        # 9. Network statistics table
        ax = fig.add_subplot(gs[2, :2])
        ax.axis('off')

        stats = {
            'Nodes': len(G.nodes()),
            'Edges': len(G.edges()),
            'Density': f"{nx.density(G):.3f}",
            'Avg Degree': f"{np.mean(degrees):.2f}",
            'Avg Clustering': f"{nx.average_clustering(G_undirected):.3f}",
            'Transitivity': f"{nx.transitivity(G_undirected):.3f}",
            'Components': nx.number_connected_components(G_undirected),
        }

        table_data = [[key, value] for key, value in stats.items()]
        table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                        loc='center', cellLoc='left',
                        colWidths=[0.5, 0.5])
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1, 2)

        for (i, j), cell in table.get_celld().items():
            if i == 0:
                cell.set_facecolor('#3498db')
                cell.set_text_props(weight='bold', color='white')
            else:
                cell.set_facecolor('#ecf0f1' if i % 2 == 0 else 'white')

        ax.set_title('Network Statistics', fontsize=12, fontweight='bold', pad=10)

        # 10. Hierarchy composition
        ax = fig.add_subplot(gs[2, 2:])
        hierarchy_counts = {}
        for node in G.nodes():
            h = G.nodes[node]['hierarchy']
            hierarchy_counts[h] = hierarchy_counts.get(h, 0) + 1

        levels = sorted(hierarchy_counts.keys())
        counts = [hierarchy_counts[l] for l in levels]
        colors = [self.color_schemes['hierarchy'][l] for l in levels]

        ax.bar(levels, counts, color=colors, edgecolor='black', alpha=0.7)
        ax.set_xlabel('Hierarchy Level', fontsize=10)
        ax.set_ylabel('Count', fontsize=10)
        ax.set_title('Hierarchy Composition', fontsize=12, fontweight='bold')
        ax.set_xticks(levels)
        ax.set_xticklabels([f'L{l}' for l in levels])
        ax.grid(True, alpha=0.3, axis='y')

        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.98)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()
        else:
            plt.close()

    def plot_chord_diagram(
        self,
        network_state: NetworkState,
        title: str = "Network Chord Diagram",
        save_path: Optional[str] = None,
        show: bool = True,
        figsize: Tuple[int, int] = (16, 16)
    ):
        """
        Create chord diagram showing inter-hierarchy connections.
        """
        fig, ax = plt.subplots(figsize=figsize, facecolor='white')
        ax.set_aspect('equal')

        # Group actors by hierarchy
        hierarchy_groups = {}
        for actor_id in network_state.V:
            h = network_state.hierarchy[actor_id]
            if h not in hierarchy_groups:
                hierarchy_groups[h] = []
            hierarchy_groups[h].append(actor_id)

        # Count connections between hierarchies
        connections = {}
        for (i, j) in network_state.E:
            h_i = network_state.hierarchy[i]
            h_j = network_state.hierarchy[j]
            key = tuple(sorted([h_i, h_j]))
            connections[key] = connections.get(key, 0) + 1

        # Draw outer circle for each hierarchy
        levels = sorted(hierarchy_groups.keys())
        n_levels = len(levels)
        arc_size = 360 / n_levels
        gap = 5  # degrees

        level_positions = {}
        for idx, level in enumerate(levels):
            start_angle = idx * arc_size + gap/2
            end_angle = (idx + 1) * arc_size - gap/2
            level_positions[level] = (start_angle, end_angle)

            # Draw arc
            wedge = Wedge(center=(0, 0), r=1.0, theta1=start_angle, theta2=end_angle,
                         width=0.1, facecolor=self.color_schemes['hierarchy'][level],
                         edgecolor='black', linewidth=2)
            ax.add_patch(wedge)

            # Label
            mid_angle = (start_angle + end_angle) / 2
            label_angle_rad = np.deg2rad(mid_angle)
            x = 1.15 * np.cos(label_angle_rad)
            y = 1.15 * np.sin(label_angle_rad)
            ax.text(x, y, f'Level {level}\n({len(hierarchy_groups[level])} actors)',
                   ha='center', va='center', fontsize=12, fontweight='bold')

        # Draw connection ribbons
        for (h1, h2), count in connections.items():
            if count < 5:  # Skip weak connections for clarity
                continue

            start1, end1 = level_positions[h1]
            start2, end2 = level_positions[h2]

            # Connection points
            mid1 = (start1 + end1) / 2
            mid2 = (start2 + end2) / 2

            angle1_rad = np.deg2rad(mid1)
            angle2_rad = np.deg2rad(mid2)

            x1, y1 = 0.95 * np.cos(angle1_rad), 0.95 * np.sin(angle1_rad)
            x2, y2 = 0.95 * np.cos(angle2_rad), 0.95 * np.sin(angle2_rad)

            # Bezier curve
            from matplotlib.patches import FancyArrowPatch
            arrow = FancyArrowPatch((x1, y1), (x2, y2),
                                   connectionstyle=f"arc3,rad=0.3",
                                   arrowstyle='-',
                                   linewidth=count/5,
                                   alpha=0.3,
                                   color='gray')
            ax.add_patch(arrow)

        ax.set_xlim(-1.3, 1.3)
        ax.set_ylim(-1.3, 1.3)
        ax.axis('off')
        ax.set_title(title, fontsize=18, fontweight='bold', pad=20)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight', facecolor='white')

        if show:
            plt.show()
        else:
            plt.close()


def create_ultra_advanced_visualizations(
    results: SimulationResults,
    output_dir: str = "ultra_advanced_viz",
    show: bool = False
):
    """
    Create ultra-advanced static visualizations.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    viz = UltraAdvancedVisualizer()

    print("Creating ultra-advanced visualizations...")

    # 1. Radial dendrogram
    print("  [1/3] Radial hierarchical clustering...")
    viz.plot_radial_dendrogram(
        results.initial_network,
        title="Radial Hierarchical Clustering",
        save_path=str(output_path / "radial_dendrogram.png"),
        show=show
    )

    # 2. Network metrics dashboard
    print("  [2/3] Network metrics dashboard...")
    viz.plot_network_metrics_dashboard(
        results.initial_network,
        title="Comprehensive Network Metrics Dashboard",
        save_path=str(output_path / "network_metrics_dashboard.png"),
        show=show
    )

    # 3. Chord diagram
    print("  [3/3] Chord diagram...")
    viz.plot_chord_diagram(
        results.initial_network,
        title="Inter-Hierarchy Connection Patterns",
        save_path=str(output_path / "chord_diagram.png"),
        show=show
    )

    print(f"\nAll ultra-advanced visualizations saved to: {output_dir}/")
    print("  - radial_dendrogram.png")
    print("  - network_metrics_dashboard.png")
    print("  - chord_diagram.png")
