"""
Interactive network visualizations using Plotly.

This module provides:
- Interactive 3D networks
- Animated temporal evolution
- Hoverable nodes with detailed info
- Zoomable, rotatable views
- Export to HTML for web viewing
"""

import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import networkx as nx
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json

from ..core.state import NetworkState, ActorState
from .results import SimulationResults


class InteractiveNetworkVisualizer:
    """
    Interactive network visualizations using Plotly.
    """

    def __init__(self):
        self.color_schemes = {
            'hierarchy': {
                1: '#e74c3c',  # Red
                2: '#3498db',  # Blue
                3: '#2ecc71',  # Green
            },
            'state': {
                ActorState.ACTIVE: '#2ecc71',
                ActorState.ARRESTED: '#e74c3c',
                ActorState.INFORMANT: '#f39c12',
            }
        }

    def create_interactive_3d_network(
        self,
        network_state: NetworkState,
        title: str = "Interactive 3D Network",
        save_path: Optional[str] = None,
        auto_open: bool = False
    ):
        """
        Create interactive 3D network visualization with Plotly.

        Features:
        - Rotatable 3D view
        - Hoverable nodes with info
        - Zoomable
        - Color by hierarchy
        - Size by degree
        """
        # Build graph
        G = nx.DiGraph()
        for actor_id in network_state.V:
            G.add_node(actor_id,
                      hierarchy=network_state.hierarchy[actor_id],
                      state=network_state.A[actor_id])

        for (i, j) in network_state.E:
            trust = network_state.get_trust(i, j)
            G.add_edge(i, j, weight=trust)

        # 3D layout
        G_undirected = G.to_undirected()
        pos_2d = nx.spring_layout(G_undirected, dim=2, seed=42)

        # Add z-dimension (hierarchy)
        pos_3d = {}
        for node in G.nodes():
            x, y = pos_2d[node]
            z = G.nodes[node]['hierarchy'] * 3.0
            pos_3d[node] = (x, y, z)

        # Edge traces
        edge_traces = []
        for (i, j) in G.edges():
            trust = G[i][j]['weight']
            x_coords = [pos_3d[i][0], pos_3d[j][0], None]
            y_coords = [pos_3d[i][1], pos_3d[j][1], None]
            z_coords = [pos_3d[i][2], pos_3d[j][2], None]

            color = f'rgba(150, 150, 150, {trust * 0.6})'

            edge_trace = go.Scatter3d(
                x=x_coords,
                y=y_coords,
                z=z_coords,
                mode='lines',
                line=dict(color=color, width=trust * 3),
                hoverinfo='none',
                showlegend=False
            )
            edge_traces.append(edge_trace)

        # Node traces (by hierarchy)
        node_traces = []
        for h_level in [1, 2, 3]:
            nodes_h = [n for n in G.nodes() if G.nodes[n]['hierarchy'] == h_level]

            x = [pos_3d[n][0] for n in nodes_h]
            y = [pos_3d[n][1] for n in nodes_h]
            z = [pos_3d[n][2] for n in nodes_h]

            # Hover text
            hover_text = []
            for n in nodes_h:
                state = G.nodes[n]['state']
                degree = G.degree(n)
                text = f"Actor {n}<br>"
                text += f"Hierarchy: L{h_level}<br>"
                text += f"State: {state.value}<br>"
                text += f"Degree: {degree}"
                hover_text.append(text)

            # Size by degree
            sizes = [10 + G.degree(n) * 2 for n in nodes_h]

            node_trace = go.Scatter3d(
                x=x, y=y, z=z,
                mode='markers',
                name=f'Level {h_level}',
                marker=dict(
                    size=sizes,
                    color=self.color_schemes['hierarchy'][h_level],
                    line=dict(color='black', width=1),
                    symbol='circle'
                ),
                text=hover_text,
                hoverinfo='text'
            )
            node_traces.append(node_trace)

        # Create figure
        fig = go.Figure(data=edge_traces + node_traces)

        # Layout
        fig.update_layout(
            title=dict(text=title, x=0.5, xanchor='center', font=dict(size=20)),
            scene=dict(
                xaxis=dict(showbackground=False, showticklabels=False, title=''),
                yaxis=dict(showbackground=False, showticklabels=False, title=''),
                zaxis=dict(showbackground=False, showticklabels=True, title='Hierarchy'),
                camera=dict(eye=dict(x=1.5, y=1.5, z=1.2))
            ),
            showlegend=True,
            hovermode='closest',
            width=1200,
            height=900,
            paper_bgcolor='white'
        )

        if save_path:
            fig.write_html(save_path, auto_open=auto_open)
            print(f"Interactive 3D network saved to: {save_path}")

        return fig

    def create_animated_network_evolution(
        self,
        results: SimulationResults,
        n_frames: int = 20,
        title: str = "Network Evolution Animation",
        save_path: Optional[str] = None,
        auto_open: bool = False
    ):
        """
        Create animated network showing temporal evolution.
        """
        # Build initial graph for layout
        G_layout = nx.DiGraph()
        for actor_id in results.initial_network.V:
            G_layout.add_node(actor_id)
        for (i, j) in results.initial_network.E:
            G_layout.add_edge(i, j)

        pos = nx.spring_layout(G_layout.to_undirected(), seed=42)

        # Time points
        total_time = results.time_series.times[-1]
        frame_times = np.linspace(0, total_time, n_frames)

        # Create frames
        frames = []
        for t_idx, t in enumerate(frame_times):
            # Find closest time in results
            time_idx = np.argmin(np.abs(results.time_series.times - t))

            active_count = results.time_series.network_size[time_idx]
            arrested_count = results.time_series.arrested_count[time_idx]
            informant_count = results.time_series.informant_count[time_idx]

            # Create node trace for this frame
            x, y, colors, hover_texts = [], [], [], []
            for node in G_layout.nodes():
                x.append(pos[node][0])
                y.append(pos[node][1])

                # Estimate state
                if node < arrested_count:
                    if node < informant_count:
                        color = self.color_schemes['state'][ActorState.INFORMANT]
                        state_text = "Informant"
                    else:
                        color = self.color_schemes['state'][ActorState.ARRESTED]
                        state_text = "Arrested"
                else:
                    color = self.color_schemes['state'][ActorState.ACTIVE]
                    state_text = "Active"

                colors.append(color)
                hover_texts.append(f"Actor {node}<br>State: {state_text}")

            # Edge trace
            edge_x, edge_y = [], []
            for (i, j) in G_layout.edges():
                edge_x.extend([pos[i][0], pos[j][0], None])
                edge_y.extend([pos[i][1], pos[j][1], None])

            frame_data = [
                go.Scatter(
                    x=edge_x, y=edge_y,
                    mode='lines',
                    line=dict(color='gray', width=0.5),
                    hoverinfo='none',
                    showlegend=False
                ),
                go.Scatter(
                    x=x, y=y,
                    mode='markers',
                    marker=dict(size=10, color=colors, line=dict(color='black', width=1)),
                    text=hover_texts,
                    hoverinfo='text',
                    showlegend=False
                )
            ]

            frame = go.Frame(
                data=frame_data,
                name=str(t_idx),
                layout=go.Layout(
                    title=f"t = {t:.1f}<br>Active: {active_count}, Arrested: {arrested_count}, Informants: {informant_count}"
                )
            )
            frames.append(frame)

        # Initial frame
        fig = go.Figure(data=frames[0].data, frames=frames)

        # Add play/pause buttons
        fig.update_layout(
            title=title,
            xaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            yaxis=dict(showgrid=False, zeroline=False, showticklabels=False),
            updatemenus=[
                dict(
                    type='buttons',
                    showactive=False,
                    buttons=[
                        dict(label='Play',
                             method='animate',
                             args=[None, dict(frame=dict(duration=500, redraw=True),
                                            fromcurrent=True, mode='immediate')]),
                        dict(label='Pause',
                             method='animate',
                             args=[[None], dict(frame=dict(duration=0, redraw=False),
                                              mode='immediate')])
                    ],
                    x=0.1, y=1.15
                )
            ],
            sliders=[
                dict(
                    active=0,
                    steps=[dict(args=[[f.name], dict(frame=dict(duration=0, redraw=True),
                                                     mode='immediate')],
                               method='animate',
                               label=f"{frame_times[int(f.name)]:.1f}")
                          for f in frames],
                    x=0.1, y=0, len=0.9
                )
            ],
            width=1200,
            height=800,
            paper_bgcolor='white'
        )

        if save_path:
            fig.write_html(save_path, auto_open=auto_open)
            print(f"Animated network saved to: {save_path}")

        return fig

    def create_interactive_dashboard(
        self,
        results: SimulationResults,
        save_path: Optional[str] = None,
        auto_open: bool = False
    ):
        """
        Create comprehensive interactive dashboard.
        """
        # Create subplots
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Network Topology', 'Time Series Metrics',
                          'Arrest Distribution', 'Trust Evolution'),
            specs=[[{'type': 'scatter'}, {'type': 'scatter'}],
                   [{'type': 'bar'}, {'type': 'scatter'}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # 1. Network topology (simplified 2D)
        G = nx.DiGraph()
        for actor_id in results.initial_network.V:
            G.add_node(actor_id, hierarchy=results.initial_network.hierarchy[actor_id])
        for (i, j) in results.initial_network.E:
            G.add_edge(i, j)

        pos = nx.spring_layout(G.to_undirected(), seed=42)

        # Edges
        edge_x, edge_y = [], []
        for (i, j) in G.edges():
            edge_x.extend([pos[i][0], pos[j][0], None])
            edge_y.extend([pos[i][1], pos[j][1], None])

        fig.add_trace(
            go.Scatter(x=edge_x, y=edge_y, mode='lines',
                      line=dict(color='gray', width=0.5),
                      hoverinfo='none', showlegend=False),
            row=1, col=1
        )

        # Nodes by hierarchy
        for h in [1, 2, 3]:
            nodes_h = [n for n in G.nodes() if G.nodes[n]['hierarchy'] == h]
            x = [pos[n][0] for n in nodes_h]
            y = [pos[n][1] for n in nodes_h]

            fig.add_trace(
                go.Scatter(x=x, y=y, mode='markers',
                          marker=dict(size=8, color=self.color_schemes['hierarchy'][h]),
                          name=f'Level {h}', showlegend=True),
                row=1, col=1
            )

        # 2. Time series metrics
        fig.add_trace(
            go.Scatter(x=results.time_series.times, y=results.time_series.network_size,
                      mode='lines', name='Active Actors',
                      line=dict(color='green', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=results.time_series.times, y=results.time_series.arrested_count,
                      mode='lines', name='Arrested',
                      line=dict(color='red', width=2)),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=results.time_series.times, y=results.time_series.informant_count,
                      mode='lines', name='Informants',
                      line=dict(color='orange', width=2)),
            row=1, col=2
        )

        # 3. Arrest distribution (hierarchy)
        arrest_events = [e for e in results.events if e.event_type.value == 'arrest']
        hierarchy_counts = {}
        for event in arrest_events:
            h = event.details.get('hierarchy_level', 1)
            hierarchy_counts[h] = hierarchy_counts.get(h, 0) + 1

        fig.add_trace(
            go.Bar(x=list(hierarchy_counts.keys()),
                  y=list(hierarchy_counts.values()),
                  marker=dict(color=['#e74c3c', '#3498db', '#2ecc71']),
                  showlegend=False),
            row=2, col=1
        )

        # 4. Trust evolution
        fig.add_trace(
            go.Scatter(x=results.time_series.times, y=results.time_series.mean_trust,
                      mode='lines', name='Mean Trust',
                      line=dict(color='purple', width=2),
                      fill='tozeroy', fillcolor='rgba(128, 0, 128, 0.2)'),
            row=2, col=2
        )

        # Update layout
        fig.update_xaxes(showgrid=False, row=1, col=1)
        fig.update_yaxes(showgrid=False, row=1, col=1)

        fig.update_xaxes(title_text="Time", row=1, col=2)
        fig.update_yaxes(title_text="Count", row=1, col=2)

        fig.update_xaxes(title_text="Hierarchy Level", row=2, col=1)
        fig.update_yaxes(title_text="Arrests", row=2, col=1)

        fig.update_xaxes(title_text="Time", row=2, col=2)
        fig.update_yaxes(title_text="Mean Trust", row=2, col=2)

        fig.update_layout(
            title_text="Interactive Simulation Dashboard",
            showlegend=True,
            height=900,
            width=1400,
            paper_bgcolor='white'
        )

        if save_path:
            fig.write_html(save_path, auto_open=auto_open)
            print(f"Interactive dashboard saved to: {save_path}")

        return fig


def create_interactive_visualizations(
    results: SimulationResults,
    output_dir: str = "interactive_viz",
    auto_open: bool = False
):
    """
    Create complete suite of interactive visualizations.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    viz = InteractiveNetworkVisualizer()

    print("Creating interactive visualizations...")

    # 1. Interactive 3D network
    print("  [1/3] Interactive 3D network...")
    viz.create_interactive_3d_network(
        results.initial_network,
        title="Interactive 3D Network (Click and Drag to Rotate)",
        save_path=str(output_path / "interactive_3d_network.html"),
        auto_open=auto_open
    )

    # 2. Animated evolution
    print("  [2/3] Animated network evolution...")
    viz.create_animated_network_evolution(
        results,
        n_frames=30,
        title="Animated Network Evolution (Click Play)",
        save_path=str(output_path / "animated_network_evolution.html"),
        auto_open=auto_open
    )

    # 3. Interactive dashboard
    print("  [3/3] Interactive dashboard...")
    viz.create_interactive_dashboard(
        results,
        save_path=str(output_path / "interactive_dashboard.html"),
        auto_open=auto_open
    )

    print(f"\nAll interactive visualizations saved to: {output_dir}/")
    print("  - interactive_3d_network.html     (Rotatable 3D view)")
    print("  - animated_network_evolution.html (Play/pause animation)")
    print("  - interactive_dashboard.html      (Comprehensive dashboard)")
    print("\nOpen these HTML files in your web browser!")
