"""
Expert-level network visualizations:
1. Sankey diagrams for state transitions
2. Network motif analysis
3. Community evolution tracking
4. Network resilience analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, Circle, Rectangle
from matplotlib.sankey import Sankey
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import seaborn as sns
from collections import defaultdict, Counter
from itertools import combinations

from ..core.state import NetworkState, ActorState
from .results import SimulationResults


class ExpertNetworkVisualizer:
    """
    Expert-level network visualizations for advanced analysis.
    """

    def __init__(self):
        self.color_schemes = {
            'state': {
                'Active': '#2ecc71',      # Green
                'Arrested': '#e74c3c',    # Red
                'Informant': '#f39c12',   # Orange
            },
            'hierarchy': {
                1: '#e74c3c',  # Red - Leaders
                2: '#3498db',  # Blue - Mid-level
                3: '#2ecc71',  # Green - Operatives
            }
        }

    def plot_sankey_state_transitions(
        self,
        results: SimulationResults,
        title: str = "Actor State Transitions (Sankey Diagram)",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10),
        show: bool = True
    ):
        """
        Create Sankey diagram showing state transitions over time.

        Shows flows:
        - Active → Arrested
        - Active → Informant
        - Arrested → Informant (if conversion after arrest)
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        # Panel 1: Overall state transition flow
        ax1.set_title("State Transition Flow", fontsize=16, fontweight='bold')

        # Count transitions from events
        active_to_arrested = 0
        arrested_to_informant = 0
        active_to_informant = 0

        # Track actor states over time
        actor_states = {}  # actor_id -> last_known_state

        for event in results.events:
            if event.event_type.value == 'arrest':
                actor_id = event.actor_id
                active_to_arrested += 1
                actor_states[actor_id] = 'Arrested'

            elif event.event_type.value == 'conversion':
                actor_id = event.actor_id
                # Check if previously arrested
                if actor_states.get(actor_id) == 'Arrested':
                    arrested_to_informant += 1
                else:
                    active_to_informant += 1
                actor_states[actor_id] = 'Informant'

        # Calculate final states
        n_actors = len(results.initial_network.V)
        final_active = n_actors - active_to_arrested

        # Draw flow diagram manually (Sankey in matplotlib is limited)
        # Three columns: Initial, Transition, Final

        # Initial state (left)
        y_init = 0.5
        rect_init = FancyBboxPatch((0.05, 0.3), 0.15, 0.4,
                                   boxstyle="round,pad=0.02",
                                   facecolor=self.color_schemes['state']['Active'],
                                   edgecolor='black', linewidth=2)
        ax1.add_patch(rect_init)
        ax1.text(0.125, 0.5, f'Active\n({n_actors})',
                ha='center', va='center', fontsize=12, fontweight='bold')

        # Arrested state (middle-top)
        rect_arrested = FancyBboxPatch((0.425, 0.55), 0.15, 0.25,
                                       boxstyle="round,pad=0.02",
                                       facecolor=self.color_schemes['state']['Arrested'],
                                       edgecolor='black', linewidth=2)
        ax1.add_patch(rect_arrested)
        ax1.text(0.5, 0.675, f'Arrested\n({active_to_arrested})',
                ha='center', va='center', fontsize=12, fontweight='bold')

        # Informant state (middle-bottom)
        rect_informant = FancyBboxPatch((0.425, 0.2), 0.15, 0.25,
                                        boxstyle="round,pad=0.02",
                                        facecolor=self.color_schemes['state']['Informant'],
                                        edgecolor='black', linewidth=2)
        ax1.add_patch(rect_informant)
        informant_total = arrested_to_informant + active_to_informant
        ax1.text(0.5, 0.325, f'Informant\n({informant_total})',
                ha='center', va='center', fontsize=12, fontweight='bold')

        # Final active (right)
        rect_final = FancyBboxPatch((0.8, 0.3), 0.15, 0.4,
                                    boxstyle="round,pad=0.02",
                                    facecolor='lightgray',
                                    edgecolor='black', linewidth=2)
        ax1.add_patch(rect_final)
        ax1.text(0.875, 0.5, f'Final\nActive\n({final_active})',
                ha='center', va='center', fontsize=12, fontweight='bold')

        # Draw arrows with widths proportional to flow
        # Active → Arrested
        if active_to_arrested > 0:
            width1 = min(0.3, active_to_arrested / n_actors * 0.5)
            ax1.arrow(0.21, 0.58, 0.2, 0.08, head_width=0.05, head_length=0.03,
                     fc=self.color_schemes['state']['Arrested'], ec='black',
                     linewidth=2, alpha=0.7, length_includes_head=True)
            ax1.text(0.31, 0.64, f'{active_to_arrested}', fontsize=10, fontweight='bold')

        # Active → Informant (direct)
        if active_to_informant > 0:
            ax1.arrow(0.21, 0.42, 0.2, -0.08, head_width=0.05, head_length=0.03,
                     fc=self.color_schemes['state']['Informant'], ec='black',
                     linewidth=2, alpha=0.7, length_includes_head=True)
            ax1.text(0.31, 0.34, f'{active_to_informant}', fontsize=10, fontweight='bold')

        # Arrested → Informant
        if arrested_to_informant > 0:
            ax1.arrow(0.5, 0.545, 0, -0.11, head_width=0.05, head_length=0.03,
                     fc=self.color_schemes['state']['Informant'], ec='black',
                     linewidth=2, alpha=0.7, length_includes_head=True)
            ax1.text(0.52, 0.48, f'{arrested_to_informant}', fontsize=10, fontweight='bold')

        # Informant → Final (stays informant)
        if informant_total > 0:
            ax1.arrow(0.58, 0.325, 0.21, 0.15, head_width=0.04, head_length=0.03,
                     fc='gray', ec='black', linewidth=1.5, alpha=0.5,
                     length_includes_head=True, linestyle='--')

        # Arrested → Final (stays arrested)
        remaining_arrested = active_to_arrested - arrested_to_informant
        if remaining_arrested > 0:
            ax1.arrow(0.58, 0.675, 0.21, -0.15, head_width=0.04, head_length=0.03,
                     fc='gray', ec='black', linewidth=1.5, alpha=0.5,
                     length_includes_head=True, linestyle='--')

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)
        ax1.axis('off')

        # Panel 2: Transition statistics by hierarchy
        ax2.set_title("State Transitions by Hierarchy Level", fontsize=16, fontweight='bold')

        # Count transitions by hierarchy
        hierarchy_transitions = {1: {'arrest': 0, 'conversion': 0},
                                2: {'arrest': 0, 'conversion': 0},
                                3: {'arrest': 0, 'conversion': 0}}

        for event in results.events:
            hierarchy = event.details.get('hierarchy_level', 1)
            if event.event_type.value == 'arrest':
                hierarchy_transitions[hierarchy]['arrest'] += 1
            elif event.event_type.value == 'conversion':
                hierarchy_transitions[hierarchy]['conversion'] += 1

        levels = [1, 2, 3]
        arrests = [hierarchy_transitions[h]['arrest'] for h in levels]
        conversions = [hierarchy_transitions[h]['conversion'] for h in levels]

        x = np.arange(len(levels))
        width = 0.35

        bars1 = ax2.bar(x - width/2, arrests, width, label='Arrests',
                       color=self.color_schemes['state']['Arrested'], edgecolor='black')
        bars2 = ax2.bar(x + width/2, conversions, width, label='Conversions',
                       color=self.color_schemes['state']['Informant'], edgecolor='black')

        ax2.set_xlabel('Hierarchy Level', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Count', fontsize=12, fontweight='bold')
        ax2.set_xticks(x)
        ax2.set_xticklabels(['L1 (Leaders)', 'L2 (Mid-level)', 'L3 (Operatives)'])
        ax2.legend(fontsize=11)
        ax2.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                if height > 0:
                    ax2.text(bar.get_x() + bar.get_width()/2., height,
                            f'{int(height)}',
                            ha='center', va='bottom', fontsize=10, fontweight='bold')

        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Sankey diagram saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_network_motif_analysis(
        self,
        network_state: NetworkState,
        title: str = "Network Motif Analysis",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 10),
        show: bool = True
    ):
        """
        Analyze and visualize network motifs (3-node patterns).

        Common motifs:
        - Feed-forward loop
        - Bi-fan
        - Triangle (cycle)
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 4, hspace=0.3, wspace=0.3)

        # Build graph
        G = nx.DiGraph()
        for actor_id in network_state.V:
            G.add_node(actor_id)
        for (i, j) in network_state.E:
            G.add_edge(i, j)

        # Convert to undirected for some analyses
        G_undirected = G.to_undirected()

        # Panel 1: Triangle count (3-node cycles)
        ax1 = fig.add_subplot(gs[0, 0])
        triangles = nx.triangles(G_undirected)
        triangle_count = sum(triangles.values()) // 3  # Each triangle counted 3 times

        # Plot triangle distribution by hierarchy
        hierarchy_triangles = defaultdict(int)
        for node, count in triangles.items():
            h = network_state.hierarchy[node]
            hierarchy_triangles[h] += count

        levels = sorted(hierarchy_triangles.keys())
        counts = [hierarchy_triangles[h] / 3 for h in levels]  # Normalize
        colors_h = [self.color_schemes['hierarchy'][h] for h in levels]

        ax1.bar(levels, counts, color=colors_h, edgecolor='black', linewidth=1.5)
        ax1.set_xlabel('Hierarchy Level', fontsize=11, fontweight='bold')
        ax1.set_ylabel('Triangle Count', fontsize=11, fontweight='bold')
        ax1.set_title(f'Triangles by Hierarchy\n(Total: {triangle_count})',
                     fontsize=12, fontweight='bold')
        ax1.set_xticks(levels)
        ax1.grid(axis='y', alpha=0.3)

        # Panel 2: Clustering coefficient distribution
        ax2 = fig.add_subplot(gs[0, 1])
        clustering_coeffs = nx.clustering(G_undirected)

        # By hierarchy
        hierarchy_clustering = {1: [], 2: [], 3: []}
        for node, coeff in clustering_coeffs.items():
            h = network_state.hierarchy[node]
            hierarchy_clustering[h].append(coeff)

        data_to_plot = [hierarchy_clustering[h] for h in [1, 2, 3]]
        bp = ax2.boxplot(data_to_plot, labels=['L1', 'L2', 'L3'],
                         patch_artist=True, showmeans=True)

        for patch, h in zip(bp['boxes'], [1, 2, 3]):
            patch.set_facecolor(self.color_schemes['hierarchy'][h])
            patch.set_alpha(0.7)

        ax2.set_xlabel('Hierarchy Level', fontsize=11, fontweight='bold')
        ax2.set_ylabel('Clustering Coefficient', fontsize=11, fontweight='bold')
        ax2.set_title('Clustering by Hierarchy', fontsize=12, fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        # Panel 3: Reciprocity (bidirectional edges)
        ax3 = fig.add_subplot(gs[0, 2])

        reciprocal_edges = 0
        total_edges = G.number_of_edges()

        for (i, j) in G.edges():
            if G.has_edge(j, i):
                reciprocal_edges += 1

        reciprocal_edges = reciprocal_edges // 2  # Each counted twice
        reciprocity_ratio = reciprocal_edges / total_edges if total_edges > 0 else 0

        ax3.bar(['Reciprocal', 'One-way'],
               [reciprocal_edges, total_edges - reciprocal_edges * 2],
               color=['#3498db', '#95a5a6'], edgecolor='black', linewidth=1.5)
        ax3.set_ylabel('Edge Count', fontsize=11, fontweight='bold')
        ax3.set_title(f'Edge Reciprocity\n({reciprocity_ratio:.1%} reciprocal)',
                     fontsize=12, fontweight='bold')
        ax3.grid(axis='y', alpha=0.3)

        # Panel 4: Feed-forward loops (A→B, A→C, B→C)
        ax4 = fig.add_subplot(gs[0, 3])

        ffl_count = 0
        for node_a in G.nodes():
            out_neighbors = list(G.successors(node_a))
            for node_b, node_c in combinations(out_neighbors, 2):
                if G.has_edge(node_b, node_c) or G.has_edge(node_c, node_b):
                    ffl_count += 1

        ax4.bar(['Feed-Forward\nLoops', 'Triangles'],
               [ffl_count, triangle_count],
               color=['#e74c3c', '#2ecc71'], edgecolor='black', linewidth=1.5)
        ax4.set_ylabel('Count', fontsize=11, fontweight='bold')
        ax4.set_title('3-Node Motifs', fontsize=12, fontweight='bold')
        ax4.grid(axis='y', alpha=0.3)

        # Panel 5: Example motif visualizations (bottom left)
        ax5 = fig.add_subplot(gs[1, 0])
        ax5.set_title('Triangle Motif', fontsize=11, fontweight='bold')

        # Draw triangle motif schematic
        motif_g1 = nx.DiGraph()
        motif_g1.add_edges_from([(0, 1), (1, 2), (2, 0)])
        pos1 = {0: (0, 1), 1: (1, 0), 2: (-1, 0)}
        nx.draw_networkx(motif_g1, pos1, ax=ax5, with_labels=True,
                        node_color='#2ecc71', node_size=800,
                        font_size=12, font_weight='bold',
                        arrows=True, arrowsize=20, edge_color='black', width=2)
        ax5.axis('off')

        # Panel 6: Feed-forward loop (bottom second)
        ax6 = fig.add_subplot(gs[1, 1])
        ax6.set_title('Feed-Forward Loop', fontsize=11, fontweight='bold')

        motif_g2 = nx.DiGraph()
        motif_g2.add_edges_from([(0, 1), (0, 2), (1, 2)])
        pos2 = {0: (0, 1), 1: (-0.5, 0), 2: (0.5, 0)}
        nx.draw_networkx(motif_g2, pos2, ax=ax6, with_labels=True,
                        node_color='#e74c3c', node_size=800,
                        font_size=12, font_weight='bold',
                        arrows=True, arrowsize=20, edge_color='black', width=2)
        ax6.axis('off')

        # Panel 7: Bi-directional edge (bottom third)
        ax7 = fig.add_subplot(gs[1, 2])
        ax7.set_title('Reciprocal Edge', fontsize=11, fontweight='bold')

        motif_g3 = nx.DiGraph()
        motif_g3.add_edges_from([(0, 1), (1, 0)])
        pos3 = {0: (-0.5, 0), 1: (0.5, 0)}
        nx.draw_networkx(motif_g3, pos3, ax=ax7, with_labels=True,
                        node_color='#3498db', node_size=800,
                        font_size=12, font_weight='bold',
                        arrows=True, arrowsize=20, edge_color='black', width=2)
        ax7.axis('off')

        # Panel 8: Summary statistics table
        ax8 = fig.add_subplot(gs[1, 3])
        ax8.axis('off')

        # Calculate additional statistics
        global_clustering = nx.transitivity(G_undirected)
        avg_clustering = np.mean(list(clustering_coeffs.values()))

        stats_text = f"""
        Motif Statistics
        {'=' * 25}

        Triangles: {triangle_count}
        Feed-Forward Loops: {ffl_count}

        Reciprocal Edges: {reciprocal_edges}
        Reciprocity Ratio: {reciprocity_ratio:.2%}

        Global Clustering: {global_clustering:.3f}
        Avg Clustering: {avg_clustering:.3f}

        Total Nodes: {G.number_of_nodes()}
        Total Edges: {G.number_of_edges()}
        """

        ax8.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
                verticalalignment='center')

        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.98)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Motif analysis saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_community_evolution(
        self,
        results: SimulationResults,
        n_snapshots: int = 6,
        title: str = "Community Evolution Over Time",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (20, 12),
        show: bool = True
    ):
        """
        Track community structure evolution over time.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, n_snapshots, hspace=0.35, wspace=0.25)

        # Get time points
        total_time = results.time_series.times[-1]
        snapshot_times = np.linspace(0, total_time, n_snapshots)

        # Build initial layout (keep consistent across time)
        G_layout = nx.DiGraph()
        for actor_id in results.initial_network.V:
            G_layout.add_node(actor_id,
                            hierarchy=results.initial_network.hierarchy[actor_id])
        for (i, j) in results.initial_network.E:
            G_layout.add_edge(i, j)

        pos = nx.spring_layout(G_layout.to_undirected(), k=1, iterations=50, seed=42)

        # Track communities over time
        community_history = []
        modularity_history = []

        # Top row: Community detection at each snapshot
        for idx, t in enumerate(snapshot_times):
            ax = fig.add_subplot(gs[0, idx])

            # Find time index
            time_idx = np.argmin(np.abs(results.time_series.times - t))
            active_count = results.time_series.network_size[time_idx]

            # Build graph at this time (only active nodes)
            G_t = nx.Graph()
            active_nodes = list(range(active_count))
            G_t.add_nodes_from(active_nodes)

            for (i, j) in results.initial_network.E:
                if i in active_nodes and j in active_nodes:
                    G_t.add_edge(i, j)

            if G_t.number_of_nodes() > 0:
                # Detect communities
                try:
                    communities = nx.community.greedy_modularity_communities(G_t)
                    modularity = nx.community.modularity(G_t, communities)
                    modularity_history.append(modularity)
                    community_history.append(communities)

                    # Assign colors
                    node_colors = {}
                    for comm_idx, comm in enumerate(communities):
                        color = plt.cm.Set3(comm_idx % 12)
                        for node in comm:
                            node_colors[node] = color

                    # Draw network
                    colors = [node_colors.get(n, 'lightgray') for n in G_t.nodes()]

                    nx.draw_networkx_nodes(G_t, pos, ax=ax, node_color=colors,
                                          node_size=100, alpha=0.8, edgecolors='black')
                    nx.draw_networkx_edges(G_t, pos, ax=ax, alpha=0.3, width=0.5)

                    ax.set_title(f't={t:.1f}\n{len(communities)} communities\nQ={modularity:.3f}',
                                fontsize=10, fontweight='bold')
                except:
                    modularity_history.append(0)
                    community_history.append([])
                    ax.set_title(f't={t:.1f}\nNo communities', fontsize=10)
            else:
                modularity_history.append(0)
                community_history.append([])
                ax.set_title(f't={t:.1f}\nNetwork collapsed', fontsize=10)

            ax.axis('off')

        # Middle row: Community size distribution
        for idx, communities in enumerate(community_history):
            ax = fig.add_subplot(gs[1, idx])

            if communities:
                sizes = [len(c) for c in communities]
                colors = [plt.cm.Set3(i % 12) for i in range(len(sizes))]

                ax.bar(range(1, len(sizes) + 1), sizes, color=colors,
                      edgecolor='black', linewidth=1)
                ax.set_xlabel('Community', fontsize=9)
                ax.set_ylabel('Size', fontsize=9)
                ax.set_title('Community Sizes', fontsize=10, fontweight='bold')
                ax.grid(axis='y', alpha=0.3)
            else:
                ax.text(0.5, 0.5, 'No data', ha='center', va='center',
                       transform=ax.transAxes, fontsize=10)
                ax.axis('off')

        # Bottom row: Alluvial-style community tracking
        ax_bottom = fig.add_subplot(gs[2, :])
        ax_bottom.set_title('Community Size Evolution', fontsize=14, fontweight='bold')

        # Plot stacked area chart of community sizes
        if len(community_history) > 0:
            max_communities = max([len(c) for c in community_history if c] or [0])

            if max_communities > 0:
                # Create matrix: time x community
                size_matrix = np.zeros((len(snapshot_times), max_communities))

                for t_idx, communities in enumerate(community_history):
                    for c_idx, comm in enumerate(communities):
                        if c_idx < max_communities:
                            size_matrix[t_idx, c_idx] = len(comm)

                # Stacked area plot
                colors = [plt.cm.Set3(i % 12) for i in range(max_communities)]
                ax_bottom.stackplot(snapshot_times, size_matrix.T,
                                   colors=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
                ax_bottom.set_xlabel('Time', fontsize=12, fontweight='bold')
                ax_bottom.set_ylabel('Community Size', fontsize=12, fontweight='bold')
                ax_bottom.grid(alpha=0.3)
            else:
                ax_bottom.text(0.5, 0.5, 'Insufficient community data',
                             ha='center', va='center', transform=ax_bottom.transAxes)

        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Community evolution saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_network_resilience_analysis(
        self,
        network_state: NetworkState,
        title: str = "Network Resilience Analysis",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 12),
        show: bool = True
    ):
        """
        Analyze network resilience through targeted vs random node removal.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        # Build graph
        G = nx.Graph()  # Use undirected for resilience
        for actor_id in network_state.V:
            G.add_node(actor_id, hierarchy=network_state.hierarchy[actor_id])
        for (i, j) in network_state.E:
            G.add_edge(i, j)

        n_nodes = G.number_of_nodes()

        # Panel 1: Degree-based attack (targeted)
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('Targeted Attack\n(Remove High-Degree Nodes)',
                     fontsize=12, fontweight='bold')

        G_attack = G.copy()
        nodes_by_degree = sorted(G_attack.degree(), key=lambda x: x[1], reverse=True)

        attack_fractions = []
        attack_largest_component = []
        attack_avg_path = []

        for i in range(0, n_nodes, max(1, n_nodes // 50)):
            frac = i / n_nodes
            attack_fractions.append(frac)

            # Largest component size
            if G_attack.number_of_nodes() > 0:
                largest_cc = max(nx.connected_components(G_attack), key=len)
                attack_largest_component.append(len(largest_cc) / n_nodes)

                # Average path length
                if len(largest_cc) > 1:
                    subgraph = G_attack.subgraph(largest_cc)
                    try:
                        avg_path = nx.average_shortest_path_length(subgraph)
                        attack_avg_path.append(avg_path)
                    except:
                        attack_avg_path.append(0)
                else:
                    attack_avg_path.append(0)
            else:
                attack_largest_component.append(0)
                attack_avg_path.append(0)

            # Remove next highest degree node
            if i < len(nodes_by_degree):
                node_to_remove = nodes_by_degree[i][0]
                if G_attack.has_node(node_to_remove):
                    G_attack.remove_node(node_to_remove)

        ax1.plot(attack_fractions, attack_largest_component, 'r-', linewidth=2,
                label='Largest Component')
        ax1.set_xlabel('Fraction Removed', fontsize=11)
        ax1.set_ylabel('Relative Size', fontsize=11)
        ax1.grid(alpha=0.3)
        ax1.legend()

        # Panel 2: Random failure
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('Random Failure\n(Remove Random Nodes)',
                     fontsize=12, fontweight='bold')

        # Average over multiple random trials
        n_trials = 10
        random_fractions = []
        random_largest_component = []

        for trial in range(n_trials):
            G_random = G.copy()
            nodes_list = list(G_random.nodes())
            np.random.shuffle(nodes_list)

            trial_fracs = []
            trial_sizes = []

            for i in range(0, n_nodes, max(1, n_nodes // 50)):
                frac = i / n_nodes
                trial_fracs.append(frac)

                if G_random.number_of_nodes() > 0:
                    largest_cc = max(nx.connected_components(G_random), key=len)
                    trial_sizes.append(len(largest_cc) / n_nodes)
                else:
                    trial_sizes.append(0)

                if i < len(nodes_list):
                    node_to_remove = nodes_list[i]
                    if G_random.has_node(node_to_remove):
                        G_random.remove_node(node_to_remove)

            ax2.plot(trial_fracs, trial_sizes, 'b-', alpha=0.3, linewidth=1)

        ax2.set_xlabel('Fraction Removed', fontsize=11)
        ax2.set_ylabel('Relative Size', fontsize=11)
        ax2.grid(alpha=0.3)

        # Panel 3: Comparison
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title('Attack vs. Random\nComparison',
                     fontsize=12, fontweight='bold')

        ax3.plot(attack_fractions, attack_largest_component, 'r-',
                linewidth=3, label='Targeted Attack')

        # Average random
        G_random_avg = G.copy()
        nodes_list = list(G_random_avg.nodes())
        np.random.shuffle(nodes_list)

        random_avg_fracs = []
        random_avg_sizes = []

        for i in range(0, n_nodes, max(1, n_nodes // 50)):
            frac = i / n_nodes
            random_avg_fracs.append(frac)

            if G_random_avg.number_of_nodes() > 0:
                largest_cc = max(nx.connected_components(G_random_avg), key=len)
                random_avg_sizes.append(len(largest_cc) / n_nodes)
            else:
                random_avg_sizes.append(0)

            if i < len(nodes_list):
                node_to_remove = nodes_list[i]
                if G_random_avg.has_node(node_to_remove):
                    G_random_avg.remove_node(node_to_remove)

        ax3.plot(random_avg_fracs, random_avg_sizes, 'b-',
                linewidth=3, label='Random Failure')

        ax3.set_xlabel('Fraction Removed', fontsize=11)
        ax3.set_ylabel('Relative Size', fontsize=11)
        ax3.legend(fontsize=10)
        ax3.grid(alpha=0.3)

        # Panel 4: Critical nodes heatmap
        ax4 = fig.add_subplot(gs[1, :])
        ax4.set_title('Node Criticality Analysis (Top 30 Critical Nodes)',
                     fontsize=14, fontweight='bold')

        # Compute multiple centrality measures
        degree_cent = nx.degree_centrality(G)
        betweenness_cent = nx.betweenness_centrality(G)
        closeness_cent = nx.closeness_centrality(G)
        eigenvector_cent = nx.eigenvector_centrality(G, max_iter=1000)

        # Combined criticality score
        criticality = {}
        for node in G.nodes():
            score = (0.3 * degree_cent[node] +
                    0.4 * betweenness_cent[node] +
                    0.2 * closeness_cent[node] +
                    0.1 * eigenvector_cent[node])
            criticality[node] = score

        # Top 30 critical nodes
        top_critical = sorted(criticality.items(), key=lambda x: x[1], reverse=True)[:30]

        # Create matrix for heatmap
        nodes_top = [n for n, _ in top_critical]
        metrics = ['Degree', 'Betweenness', 'Closeness', 'Eigenvector', 'Combined']

        matrix = np.zeros((len(nodes_top), len(metrics)))
        for i, node in enumerate(nodes_top):
            matrix[i, 0] = degree_cent[node]
            matrix[i, 1] = betweenness_cent[node]
            matrix[i, 2] = closeness_cent[node]
            matrix[i, 3] = eigenvector_cent[node]
            matrix[i, 4] = criticality[node]

        im = ax4.imshow(matrix.T, cmap='YlOrRd', aspect='auto')
        ax4.set_yticks(range(len(metrics)))
        ax4.set_yticklabels(metrics, fontsize=10)
        ax4.set_xlabel('Node ID (Top 30 Critical)', fontsize=11)
        ax4.set_xticks(range(0, len(nodes_top), 5))
        ax4.set_xticklabels([nodes_top[i] for i in range(0, len(nodes_top), 5)], fontsize=8)

        plt.colorbar(im, ax=ax4, label='Centrality Score')

        # Panel 5: Hierarchy vulnerability
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.set_title('Vulnerability by Hierarchy',
                     fontsize=12, fontweight='bold')

        hierarchy_criticality = {1: [], 2: [], 3: []}
        for node, score in criticality.items():
            h = network_state.hierarchy[node]
            hierarchy_criticality[h].append(score)

        data_to_plot = [hierarchy_criticality[h] for h in [1, 2, 3]]
        bp = ax5.boxplot(data_to_plot, labels=['L1\n(Leaders)', 'L2\n(Mid)', 'L3\n(Ops)'],
                        patch_artist=True, showmeans=True)

        for patch, h in zip(bp['boxes'], [1, 2, 3]):
            patch.set_facecolor(self.color_schemes['hierarchy'][h])
            patch.set_alpha(0.7)

        ax5.set_ylabel('Criticality Score', fontsize=11, fontweight='bold')
        ax5.grid(axis='y', alpha=0.3)

        # Panel 6: Network robustness metrics
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.axis('off')

        # Calculate robustness metrics
        avg_degree = np.mean([d for n, d in G.degree()])
        density = nx.density(G)
        algebraic_connectivity = 0
        try:
            algebraic_connectivity = nx.algebraic_connectivity(G)
        except:
            pass

        # Find critical threshold (when network fragments)
        critical_threshold = 0
        for i, size in enumerate(attack_largest_component):
            if size < 0.5:
                critical_threshold = attack_fractions[i]
                break

        metrics_text = f"""
        Robustness Metrics
        {'=' * 30}

        Average Degree: {avg_degree:.2f}
        Density: {density:.3f}
        Algebraic Connectivity: {algebraic_connectivity:.3f}

        Critical Threshold: {critical_threshold:.2%}
        (Targeted attack breaks network)

        Most Critical Node: {top_critical[0][0]}
        (Criticality: {top_critical[0][1]:.3f})

        Top 5 Critical Nodes:
        {', '.join([str(n) for n, _ in top_critical[:5]])}
        """

        ax6.text(0.1, 0.5, metrics_text, fontsize=10, family='monospace',
                verticalalignment='center')

        # Panel 7: Degree distribution with critical nodes highlighted
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.set_title('Degree Distribution\n(Critical Nodes in Red)',
                     fontsize=12, fontweight='bold')

        degrees = [d for n, d in G.degree()]
        critical_nodes_set = set([n for n, _ in top_critical[:10]])
        critical_degrees = [d for n, d in G.degree() if n in critical_nodes_set]

        ax7.hist(degrees, bins=20, alpha=0.7, color='gray', edgecolor='black', label='All')
        ax7.hist(critical_degrees, bins=20, alpha=0.8, color='red', edgecolor='black', label='Critical (Top 10)')
        ax7.set_xlabel('Degree', fontsize=11)
        ax7.set_ylabel('Count', fontsize=11)
        ax7.legend(fontsize=9)
        ax7.grid(axis='y', alpha=0.3)

        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Resilience analysis saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig


def create_expert_visualizations(
    results: SimulationResults,
    output_dir: str = "expert_viz",
    show: bool = False
):
    """
    Create complete suite of expert-level visualizations.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    viz = ExpertNetworkVisualizer()

    print("Creating expert-level visualizations...")

    # 1. Sankey diagram
    print("  [1/4] Sankey state transitions...")
    viz.plot_sankey_state_transitions(
        results,
        save_path=str(output_path / "sankey_state_transitions.png"),
        show=show
    )

    # 2. Motif analysis
    print("  [2/4] Network motif analysis...")
    viz.plot_network_motif_analysis(
        results.initial_network,
        save_path=str(output_path / "network_motif_analysis.png"),
        show=show
    )

    # 3. Community evolution
    print("  [3/4] Community evolution...")
    viz.plot_community_evolution(
        results,
        n_snapshots=6,
        save_path=str(output_path / "community_evolution.png"),
        show=show
    )

    # 4. Resilience analysis
    print("  [4/4] Network resilience analysis...")
    viz.plot_network_resilience_analysis(
        results.initial_network,
        save_path=str(output_path / "network_resilience_analysis.png"),
        show=show
    )

    print(f"\nAll expert visualizations saved to: {output_dir}/")
    print("  - sankey_state_transitions.png")
    print("  - network_motif_analysis.png")
    print("  - community_evolution.png")
    print("  - network_resilience_analysis.png")
