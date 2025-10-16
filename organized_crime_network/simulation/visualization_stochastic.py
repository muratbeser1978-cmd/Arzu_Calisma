"""
Advanced stochastic process visualizations:
1. Stochastic path analysis with confidence bands
2. Cox process intensity visualization
3. Trust SDE dynamics
4. CTMC transition analysis
5. Parameter sensitivity analysis
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.collections import LineCollection
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from scipy import stats
from scipy.interpolate import interp1d
import warnings

from ..core.state import NetworkState, ActorState
from .results import SimulationResults


class StochasticProcessVisualizer:
    """
    Advanced visualizations for stochastic process analysis.
    """

    def __init__(self):
        self.color_schemes = {
            'quantiles': ['#d62728', '#ff7f0e', '#2ca02c', '#ff7f0e', '#d62728'],
            'intensity': 'YlOrRd',
            'trust': 'RdYlGn',
        }

    def plot_stochastic_path_analysis(
        self,
        mc_results_list: List[SimulationResults],
        title: str = "Stochastic Path Analysis with Confidence Bands",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 12),
        show: bool = True
    ):
        """
        Plot multiple simulation trajectories with confidence bands.

        Shows:
        - Mean trajectory
        - Quantile bands (5%, 25%, 75%, 95%)
        - Sample trajectories
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.3, wspace=0.3)

        n_sims = len(mc_results_list)
        print(f"  Analyzing {n_sims} stochastic trajectories...")

        # Find common time grid
        max_time = max([r.time_series.times[-1] for r in mc_results_list])
        time_grid = np.linspace(0, max_time, 200)

        # Interpolate all trajectories to common grid
        def interpolate_trajectory(times, values, time_grid):
            if len(times) > 1:
                f = interp1d(times, values, kind='linear',
                           bounds_error=False, fill_value=(values[0], values[-1]))
                return f(time_grid)
            else:
                return np.full_like(time_grid, values[0])

        # Panel 1: Active actors
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('Active Actors Over Time', fontsize=12, fontweight='bold')

        trajectories_active = []
        for result in mc_results_list[:min(50, n_sims)]:  # Sample for speed
            traj = interpolate_trajectory(result.time_series.times,
                                         result.time_series.network_size,
                                         time_grid)
            trajectories_active.append(traj)
            if len(trajectories_active) <= 20:  # Plot first 20
                ax1.plot(time_grid, traj, 'gray', alpha=0.1, linewidth=0.5)

        trajectories_active = np.array(trajectories_active)

        # Compute quantiles
        mean_traj = np.mean(trajectories_active, axis=0)
        q05 = np.percentile(trajectories_active, 5, axis=0)
        q25 = np.percentile(trajectories_active, 25, axis=0)
        q75 = np.percentile(trajectories_active, 75, axis=0)
        q95 = np.percentile(trajectories_active, 95, axis=0)

        # Plot confidence bands
        ax1.fill_between(time_grid, q05, q95, alpha=0.2, color='blue', label='90% CI')
        ax1.fill_between(time_grid, q25, q75, alpha=0.3, color='blue', label='50% CI')
        ax1.plot(time_grid, mean_traj, 'b-', linewidth=3, label='Mean')

        ax1.set_xlabel('Time', fontsize=11)
        ax1.set_ylabel('Active Actors', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # Panel 2: Arrested count
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('Arrested Actors Over Time', fontsize=12, fontweight='bold')

        trajectories_arrested = []
        for result in mc_results_list[:min(50, n_sims)]:
            traj = interpolate_trajectory(result.time_series.times,
                                         result.time_series.arrested_count,
                                         time_grid)
            trajectories_arrested.append(traj)
            if len(trajectories_arrested) <= 20:
                ax2.plot(time_grid, traj, 'gray', alpha=0.1, linewidth=0.5)

        trajectories_arrested = np.array(trajectories_arrested)

        mean_arrested = np.mean(trajectories_arrested, axis=0)
        q05_arr = np.percentile(trajectories_arrested, 5, axis=0)
        q25_arr = np.percentile(trajectories_arrested, 25, axis=0)
        q75_arr = np.percentile(trajectories_arrested, 75, axis=0)
        q95_arr = np.percentile(trajectories_arrested, 95, axis=0)

        ax2.fill_between(time_grid, q05_arr, q95_arr, alpha=0.2, color='red', label='90% CI')
        ax2.fill_between(time_grid, q25_arr, q75_arr, alpha=0.3, color='red', label='50% CI')
        ax2.plot(time_grid, mean_arrested, 'r-', linewidth=3, label='Mean')

        ax2.set_xlabel('Time', fontsize=11)
        ax2.set_ylabel('Arrested Actors', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        # Panel 3: Informant count
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title('Informant Count Over Time', fontsize=12, fontweight='bold')

        trajectories_informant = []
        for result in mc_results_list[:min(50, n_sims)]:
            traj = interpolate_trajectory(result.time_series.times,
                                         result.time_series.informant_count,
                                         time_grid)
            trajectories_informant.append(traj)
            if len(trajectories_informant) <= 20:
                ax3.plot(time_grid, traj, 'gray', alpha=0.1, linewidth=0.5)

        trajectories_informant = np.array(trajectories_informant)

        mean_informant = np.mean(trajectories_informant, axis=0)
        q05_inf = np.percentile(trajectories_informant, 5, axis=0)
        q25_inf = np.percentile(trajectories_informant, 25, axis=0)
        q75_inf = np.percentile(trajectories_informant, 75, axis=0)
        q95_inf = np.percentile(trajectories_informant, 95, axis=0)

        ax3.fill_between(time_grid, q05_inf, q95_inf, alpha=0.2, color='orange', label='90% CI')
        ax3.fill_between(time_grid, q25_inf, q75_inf, alpha=0.3, color='orange', label='50% CI')
        ax3.plot(time_grid, mean_informant, color='orange', linewidth=3, label='Mean')

        ax3.set_xlabel('Time', fontsize=11)
        ax3.set_ylabel('Informants', fontsize=11)
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)

        # Panel 4: Mean trust
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_title('Mean Trust Over Time', fontsize=12, fontweight='bold')

        trajectories_trust = []
        for result in mc_results_list[:min(50, n_sims)]:
            traj = interpolate_trajectory(result.time_series.times,
                                         result.time_series.mean_trust,
                                         time_grid)
            trajectories_trust.append(traj)
            if len(trajectories_trust) <= 20:
                ax4.plot(time_grid, traj, 'gray', alpha=0.1, linewidth=0.5)

        trajectories_trust = np.array(trajectories_trust)

        mean_trust = np.mean(trajectories_trust, axis=0)
        q05_trust = np.percentile(trajectories_trust, 5, axis=0)
        q25_trust = np.percentile(trajectories_trust, 25, axis=0)
        q75_trust = np.percentile(trajectories_trust, 75, axis=0)
        q95_trust = np.percentile(trajectories_trust, 95, axis=0)

        ax4.fill_between(time_grid, q05_trust, q95_trust, alpha=0.2, color='purple', label='90% CI')
        ax4.fill_between(time_grid, q25_trust, q75_trust, alpha=0.3, color='purple', label='50% CI')
        ax4.plot(time_grid, mean_trust, 'purple', linewidth=3, label='Mean')

        ax4.set_xlabel('Time', fontsize=11)
        ax4.set_ylabel('Mean Trust', fontsize=11)
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)

        # Panel 5: Trajectory clustering (final outcomes)
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_title('Final Outcome Distribution', fontsize=12, fontweight='bold')

        final_active = [r.time_series.network_size[-1] for r in mc_results_list]
        final_arrested = [r.time_series.arrested_count[-1] for r in mc_results_list]
        final_informant = [r.time_series.informant_count[-1] for r in mc_results_list]

        outcomes = np.array([final_active, final_arrested, final_informant]).T

        ax5.hist(final_active, bins=20, alpha=0.5, label='Active', color='green', edgecolor='black')
        ax5.hist(final_arrested, bins=20, alpha=0.5, label='Arrested', color='red', edgecolor='black')
        ax5.hist(final_informant, bins=20, alpha=0.5, label='Informant', color='orange', edgecolor='black')

        ax5.set_xlabel('Final Count', fontsize=11)
        ax5.set_ylabel('Frequency', fontsize=11)
        ax5.legend(fontsize=9)
        ax5.grid(alpha=0.3)

        # Panel 6: Collapse time distribution
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_title('Network Collapse Time Distribution', fontsize=12, fontweight='bold')

        collapse_times = []
        for result in mc_results_list:
            # Find when network size drops below threshold (e.g., 10% of initial)
            initial_size = result.time_series.network_size[0]
            threshold = initial_size * 0.1

            collapse_idx = np.where(result.time_series.network_size < threshold)[0]
            if len(collapse_idx) > 0:
                collapse_time = result.time_series.times[collapse_idx[0]]
                collapse_times.append(collapse_time)

        if len(collapse_times) > 0:
            ax6.hist(collapse_times, bins=30, color='darkred', alpha=0.7, edgecolor='black')
            ax6.axvline(np.mean(collapse_times), color='red', linestyle='--',
                       linewidth=2, label=f'Mean: {np.mean(collapse_times):.1f}')
            ax6.axvline(np.median(collapse_times), color='blue', linestyle='--',
                       linewidth=2, label=f'Median: {np.median(collapse_times):.1f}')
            ax6.set_xlabel('Collapse Time', fontsize=11)
            ax6.set_ylabel('Frequency', fontsize=11)
            ax6.legend(fontsize=9)
            ax6.grid(alpha=0.3)
        else:
            ax6.text(0.5, 0.5, 'No collapses detected', ha='center', va='center',
                    transform=ax6.transAxes, fontsize=12)

        # Panel 7: Variance over time
        ax7 = fig.add_subplot(gs[2, 0])
        ax7.set_title('Trajectory Variance Over Time', fontsize=12, fontweight='bold')

        var_active = np.var(trajectories_active, axis=0)
        var_arrested = np.var(trajectories_arrested, axis=0)

        ax7.plot(time_grid, var_active, 'b-', linewidth=2, label='Active')
        ax7.plot(time_grid, var_arrested, 'r-', linewidth=2, label='Arrested')

        ax7.set_xlabel('Time', fontsize=11)
        ax7.set_ylabel('Variance', fontsize=11)
        ax7.legend(fontsize=9)
        ax7.grid(alpha=0.3)

        # Panel 8: Coefficient of variation
        ax8 = fig.add_subplot(gs[2, 1])
        ax8.set_title('Coefficient of Variation (CV)', fontsize=12, fontweight='bold')

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cv_active = np.std(trajectories_active, axis=0) / (mean_traj + 1e-10)
            cv_arrested = np.std(trajectories_arrested, axis=0) / (mean_arrested + 1e-10)

        ax8.plot(time_grid, cv_active, 'b-', linewidth=2, label='Active')
        ax8.plot(time_grid, cv_arrested, 'r-', linewidth=2, label='Arrested')

        ax8.set_xlabel('Time', fontsize=11)
        ax8.set_ylabel('CV (σ/μ)', fontsize=11)
        ax8.legend(fontsize=9)
        ax8.grid(alpha=0.3)

        # Panel 9: Summary statistics table
        ax9 = fig.add_subplot(gs[2, 2])
        ax9.axis('off')

        summary_text = f"""
        Stochastic Summary
        {'=' * 35}

        Simulations Analyzed: {n_sims}

        Final Active (mean ± std):
          {np.mean(final_active):.1f} ± {np.std(final_active):.1f}

        Final Arrested (mean ± std):
          {np.mean(final_arrested):.1f} ± {np.std(final_arrested):.1f}

        Final Informants (mean ± std):
          {np.mean(final_informant):.1f} ± {np.std(final_informant):.1f}

        Collapse Rate:
          {len(collapse_times) / n_sims * 100:.1f}%

        Mean Collapse Time:
          {np.mean(collapse_times) if collapse_times else 0:.1f}
        """

        ax9.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                verticalalignment='center')

        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Stochastic path analysis saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_cox_process_intensity(
        self,
        results: SimulationResults,
        title: str = "Cox Process Intensity Evolution (Arrest Rate λ(t))",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 10),
        show: bool = True
    ):
        """
        Visualize Cox process intensity evolution over time.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Extract arrest events
        arrest_events = [e for e in results.events if e.event_type.value == 'arrest']
        arrest_times = [e.timestamp for e in arrest_events]
        arrest_actors = [e.actor_id for e in arrest_events]

        # Panel 1: Arrest event timeline
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_title('Arrest Event Timeline (Rug Plot)', fontsize=14, fontweight='bold')

        # Rug plot
        ax1.eventplot(arrest_times, lineoffsets=0, linelengths=0.8,
                     colors='red', linewidths=1.5)

        # Add network size as background
        ax1_twin = ax1.twinx()
        ax1_twin.plot(results.time_series.times, results.time_series.network_size,
                     'b-', alpha=0.3, linewidth=2, label='Network Size')
        ax1_twin.set_ylabel('Network Size', fontsize=11, color='blue')
        ax1_twin.tick_params(axis='y', labelcolor='blue')

        ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Arrest Events', fontsize=12, fontweight='bold')
        ax1.set_ylim(-1, 1)
        ax1.set_xlim(0, results.time_series.times[-1])
        ax1.grid(alpha=0.3, axis='x')

        # Panel 2: Inter-arrival time distribution
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_title('Inter-Arrival Time Distribution', fontsize=12, fontweight='bold')

        if len(arrest_times) > 1:
            inter_arrival = np.diff(sorted(arrest_times))

            ax2.hist(inter_arrival, bins=30, color='darkred', alpha=0.7,
                    edgecolor='black', density=True, label='Observed')

            # Fit exponential (if Poisson process)
            rate_est = 1 / np.mean(inter_arrival)
            x_exp = np.linspace(0, max(inter_arrival), 100)
            y_exp = rate_est * np.exp(-rate_est * x_exp)
            ax2.plot(x_exp, y_exp, 'b--', linewidth=2,
                    label=f'Exponential (λ={rate_est:.2f})')

            ax2.set_xlabel('Inter-Arrival Time', fontsize=11)
            ax2.set_ylabel('Density', fontsize=11)
            ax2.legend(fontsize=9)
            ax2.grid(alpha=0.3)

        # Panel 3: Intensity estimation (smoothed rate)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_title('Estimated Intensity λ(t)', fontsize=12, fontweight='bold')

        # Kernel density estimation of intensity
        if len(arrest_times) > 2:
            from scipy.stats import gaussian_kde

            # Create time bins
            time_bins = np.linspace(0, results.time_series.times[-1], 50)
            counts, _ = np.histogram(arrest_times, bins=time_bins)
            bin_centers = (time_bins[:-1] + time_bins[1:]) / 2
            dt = time_bins[1] - time_bins[0]

            # Intensity = counts / dt
            intensity = counts / dt

            ax3.plot(bin_centers, intensity, 'r-', linewidth=2, marker='o',
                    markersize=4, label='Intensity λ(t)')
            ax3.fill_between(bin_centers, 0, intensity, alpha=0.3, color='red')

            ax3.set_xlabel('Time', fontsize=11)
            ax3.set_ylabel('Intensity λ(t)', fontsize=11)
            ax3.legend(fontsize=9)
            ax3.grid(alpha=0.3)

        # Panel 4: Cumulative arrests
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_title('Cumulative Arrests', fontsize=12, fontweight='bold')

        sorted_times = sorted(arrest_times)
        cumulative = np.arange(1, len(sorted_times) + 1)

        ax4.plot(sorted_times, cumulative, 'r-', linewidth=2, label='Observed')

        # Expected under constant rate
        if len(sorted_times) > 0:
            expected_rate = len(sorted_times) / results.time_series.times[-1]
            time_range = np.linspace(0, results.time_series.times[-1], 100)
            expected_cumulative = expected_rate * time_range
            ax4.plot(time_range, expected_cumulative, 'b--', linewidth=2,
                    label=f'Constant Rate (λ={expected_rate:.2f})')

        ax4.set_xlabel('Time', fontsize=11)
        ax4.set_ylabel('Cumulative Arrests', fontsize=11)
        ax4.legend(fontsize=9)
        ax4.grid(alpha=0.3)

        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Cox process visualization saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_trust_sde_dynamics(
        self,
        results: SimulationResults,
        sample_actors: Optional[List[int]] = None,
        n_samples: int = 10,
        title: str = "Trust SDE Dynamics (Ornstein-Uhlenbeck Process)",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 12),
        show: bool = True
    ):
        """
        Visualize trust dynamics as SDE/PDMP process.
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        # Note: We don't have individual trust trajectories stored
        # We'll work with aggregate trust data and infer properties

        times = results.time_series.times
        mean_trust = results.time_series.mean_trust

        # Panel 1: Mean trust trajectory
        ax1 = fig.add_subplot(gs[0, :])
        ax1.set_title('Mean Trust Evolution (Population Average)',
                     fontsize=14, fontweight='bold')

        ax1.plot(times, mean_trust, 'b-', linewidth=2.5, label='Mean Trust')

        # Add arrest events as vertical lines
        arrest_events = [e for e in results.events if e.event_type.value == 'arrest']
        for event in arrest_events[:50]:  # First 50 for visibility
            ax1.axvline(event.timestamp, color='red', alpha=0.1, linewidth=0.5)

        ax1.set_xlabel('Time', fontsize=12, fontweight='bold')
        ax1.set_ylabel('Mean Trust', fontsize=12, fontweight='bold')
        ax1.legend(fontsize=10)
        ax1.grid(alpha=0.3)

        # Panel 2: Trust change rate (numerical derivative)
        ax2 = fig.add_subplot(gs[1, 0])
        ax2.set_title('Trust Drift dY/dt', fontsize=12, fontweight='bold')

        if len(times) > 1:
            dt_vals = np.diff(times)
            dY_dt = np.diff(mean_trust) / dt_vals

            # Plot drift
            ax2.plot(times[:-1], dY_dt, 'g-', linewidth=1.5, alpha=0.7)

            # Add smoothed version
            from scipy.ndimage import gaussian_filter1d
            dY_dt_smooth = gaussian_filter1d(dY_dt, sigma=5)
            ax2.plot(times[:-1], dY_dt_smooth, 'b-', linewidth=2, label='Smoothed')

            ax2.axhline(0, color='black', linestyle='--', linewidth=1)
            ax2.set_xlabel('Time', fontsize=11)
            ax2.set_ylabel('dY/dt', fontsize=11)
            ax2.legend(fontsize=9)
            ax2.grid(alpha=0.3)

        # Panel 3: Phase space (Trust vs. dTrust/dt)
        ax3 = fig.add_subplot(gs[1, 1])
        ax3.set_title('Phase Space (Y vs dY/dt)', fontsize=12, fontweight='bold')

        if len(times) > 1:
            # Color by time
            points = np.array([mean_trust[:-1], dY_dt]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            lc = LineCollection(segments, cmap='viridis', linewidth=2)
            lc.set_array(times[:-1])
            ax3.add_collection(lc)

            ax3.set_xlim(mean_trust.min() * 0.95, mean_trust.max() * 1.05)

            # Handle NaN/Inf in dY_dt
            valid_dy = dY_dt[np.isfinite(dY_dt)]
            if len(valid_dy) > 0:
                ax3.set_ylim(valid_dy.min() * 1.1, valid_dy.max() * 1.1)
            else:
                ax3.set_ylim(-1, 1)
            ax3.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
            ax3.axvline(mean_trust[0], color='red', linestyle='--',
                       linewidth=1, alpha=0.5, label='Initial Trust')

            ax3.set_xlabel('Trust (Y)', fontsize=11)
            ax3.set_ylabel('dY/dt', fontsize=11)
            ax3.legend(fontsize=9)
            ax3.grid(alpha=0.3)

            cbar = plt.colorbar(lc, ax=ax3)
            cbar.set_label('Time', fontsize=10)

        # Panel 4: Trust distribution over time (histogram evolution)
        ax4 = fig.add_subplot(gs[1, 2])
        ax4.set_title('Trust Distribution Evolution', fontsize=12, fontweight='bold')

        # We'll use mean trust as proxy, show its trajectory
        ax4.hist(mean_trust, bins=30, color='skyblue', alpha=0.7,
                edgecolor='black', orientation='horizontal')
        ax4.set_ylabel('Trust Level', fontsize=11)
        ax4.set_xlabel('Frequency', fontsize=11)
        ax4.grid(alpha=0.3)

        # Panel 5: Mean reversion analysis (if OU process)
        ax5 = fig.add_subplot(gs[2, 0])
        ax5.set_title('Mean Reversion (OU Process)', fontsize=12, fontweight='bold')

        if len(times) > 1:
            # For OU process: dY = -α(Y - Y_bar)dt + σdW
            # Estimate α from drift vs. deviation

            Y_bar = np.mean(mean_trust)  # Long-run mean
            deviations = mean_trust[:-1] - Y_bar

            # Avoid division by zero
            mask = np.abs(deviations) > 1e-6
            if np.any(mask):
                drift_normalized = dY_dt[mask] / deviations[mask]

                ax5.scatter(deviations[mask], dY_dt[mask], alpha=0.5, s=10)

                # Linear fit
                from scipy.stats import linregress
                slope, intercept, r_value, _, _ = linregress(deviations[mask],
                                                             dY_dt[mask])

                x_fit = np.linspace(deviations.min(), deviations.max(), 100)
                y_fit = slope * x_fit + intercept
                ax5.plot(x_fit, y_fit, 'r--', linewidth=2,
                        label=f'α ≈ {-slope:.3f}, R²={r_value**2:.3f}')

                ax5.axhline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                ax5.axvline(0, color='black', linestyle='--', linewidth=1, alpha=0.5)
                ax5.set_xlabel('Deviation (Y - Y̅)', fontsize=11)
                ax5.set_ylabel('Drift dY/dt', fontsize=11)
                ax5.legend(fontsize=9)
                ax5.grid(alpha=0.3)

        # Panel 6: Autocorrelation of trust
        ax6 = fig.add_subplot(gs[2, 1])
        ax6.set_title('Trust Autocorrelation', fontsize=12, fontweight='bold')

        if len(mean_trust) > 10:
            from statsmodels.tsa.stattools import acf

            max_lag = min(50, len(mean_trust) // 4)
            try:
                autocorr = acf(mean_trust, nlags=max_lag, fft=True)
                lags = np.arange(max_lag + 1)

                ax6.stem(lags, autocorr, basefmt=' ', linefmt='b-', markerfmt='bo')
                ax6.axhline(0, color='black', linestyle='-', linewidth=0.5)
                ax6.axhline(0.05, color='red', linestyle='--', linewidth=1, alpha=0.5)
                ax6.axhline(-0.05, color='red', linestyle='--', linewidth=1, alpha=0.5)

                ax6.set_xlabel('Lag', fontsize=11)
                ax6.set_ylabel('Autocorrelation', fontsize=11)
                ax6.grid(alpha=0.3)
            except:
                ax6.text(0.5, 0.5, 'ACF calculation failed',
                        ha='center', va='center', transform=ax6.transAxes)

        # Panel 7: Volatility estimation (rolling std)
        ax7 = fig.add_subplot(gs[2, 2])
        ax7.set_title('Trust Volatility (Rolling σ)', fontsize=12, fontweight='bold')

        if len(mean_trust) > 20:
            window = min(20, len(mean_trust) // 5)
            rolling_std = np.array([np.std(mean_trust[max(0, i-window):i+1])
                                   for i in range(len(mean_trust))])

            ax7.plot(times, rolling_std, 'purple', linewidth=2)
            ax7.fill_between(times, 0, rolling_std, alpha=0.3, color='purple')

            ax7.set_xlabel('Time', fontsize=11)
            ax7.set_ylabel('Rolling Std Dev', fontsize=11)
            ax7.grid(alpha=0.3)

        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Trust SDE dynamics saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_ctmc_transition_analysis(
        self,
        results: SimulationResults,
        title: str = "CTMC State Transition Analysis (Conversion Process)",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (16, 10),
        show: bool = True
    ):
        """
        Analyze CTMC transitions (Arrested → Informant conversions).
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)

        # Extract conversion events
        conversion_events = [e for e in results.events if e.event_type.value == 'conversion']
        arrest_events = [e for e in results.events if e.event_type.value == 'arrest']

        # Panel 1: State transition diagram
        ax1 = fig.add_subplot(gs[0, :2])
        ax1.set_title('State Transition Diagram (CTMC)', fontsize=14, fontweight='bold')
        ax1.axis('off')

        # Draw states
        from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

        # Active state
        active_box = FancyBboxPatch((0.1, 0.4), 0.15, 0.2,
                                   boxstyle="round,pad=0.02",
                                   facecolor='#2ecc71', edgecolor='black', linewidth=2)
        ax1.add_patch(active_box)
        ax1.text(0.175, 0.5, 'Active', ha='center', va='center',
                fontsize=14, fontweight='bold')

        # Arrested state
        arrested_box = FancyBboxPatch((0.45, 0.4), 0.15, 0.2,
                                     boxstyle="round,pad=0.02",
                                     facecolor='#e74c3c', edgecolor='black', linewidth=2)
        ax1.add_patch(arrested_box)
        ax1.text(0.525, 0.5, 'Arrested', ha='center', va='center',
                fontsize=14, fontweight='bold')

        # Informant state
        informant_box = FancyBboxPatch((0.75, 0.4), 0.15, 0.2,
                                      boxstyle="round,pad=0.02",
                                      facecolor='#f39c12', edgecolor='black', linewidth=2)
        ax1.add_patch(informant_box)
        ax1.text(0.825, 0.5, 'Informant', ha='center', va='center',
                fontsize=14, fontweight='bold')

        # Arrows with rates
        n_arrests = len(arrest_events)
        n_conversions = len(conversion_events)
        total_time = results.time_series.times[-1] if len(results.time_series.times) > 0 else 1.0

        arrest_rate = n_arrests / total_time if total_time > 0 else 0
        conversion_rate = n_conversions / total_time if total_time > 0 else 0

        # Active → Arrested
        arrow1 = FancyArrowPatch((0.26, 0.5), (0.44, 0.5),
                                arrowstyle='->', mutation_scale=30,
                                linewidth=3, color='red')
        ax1.add_patch(arrow1)
        ax1.text(0.35, 0.55, f'λ={arrest_rate:.2f}', fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        # Arrested → Informant
        arrow2 = FancyArrowPatch((0.61, 0.5), (0.74, 0.5),
                                arrowstyle='->', mutation_scale=30,
                                linewidth=3, color='orange')
        ax1.add_patch(arrow2)
        ax1.text(0.675, 0.55, f'μ={conversion_rate:.2f}', fontsize=11, ha='center',
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

        ax1.set_xlim(0, 1)
        ax1.set_ylim(0, 1)

        # Panel 2: Conversion rate estimation
        ax2 = fig.add_subplot(gs[0, 2])
        ax2.set_title('Conversion Rate μ(t)', fontsize=12, fontweight='bold')

        if len(conversion_events) > 1:
            conversion_times = [e.timestamp for e in conversion_events]

            # Cumulative conversions
            sorted_times = sorted(conversion_times)
            cumulative = np.arange(1, len(sorted_times) + 1)

            ax2.plot(sorted_times, cumulative, 'o-', color='orange',
                    linewidth=2, markersize=4, label='Cumulative')

            # Fit rate
            from scipy.stats import linregress
            slope, _, _, _, _ = linregress(sorted_times, cumulative)

            time_range = np.linspace(0, max(sorted_times), 100)
            ax2.plot(time_range, slope * time_range, 'r--', linewidth=2,
                    label=f'Rate μ={slope:.3f}')

            ax2.set_xlabel('Time', fontsize=11)
            ax2.set_ylabel('Cumulative Conversions', fontsize=11)
            ax2.legend(fontsize=9)
            ax2.grid(alpha=0.3)

        # Panel 3: Time to conversion distribution
        ax3 = fig.add_subplot(gs[1, 0])
        ax3.set_title('Time-to-Conversion Distribution', fontsize=12, fontweight='bold')

        # Calculate time between arrest and conversion for same actors
        arrest_dict = {e.actor_id: e.timestamp for e in arrest_events}
        conversion_dict = {e.actor_id: e.timestamp for e in conversion_events}

        time_to_conversion = []
        for actor_id, conv_time in conversion_dict.items():
            if actor_id in arrest_dict:
                waiting_time = conv_time - arrest_dict[actor_id]
                if waiting_time > 0:
                    time_to_conversion.append(waiting_time)

        if len(time_to_conversion) > 0:
            ax3.hist(time_to_conversion, bins=25, color='darkorange',
                    alpha=0.7, edgecolor='black', density=True, label='Observed')

            # Fit exponential
            mean_time = np.mean(time_to_conversion)
            mu_est = 1 / mean_time
            x_exp = np.linspace(0, max(time_to_conversion), 100)
            y_exp = mu_est * np.exp(-mu_est * x_exp)
            ax3.plot(x_exp, y_exp, 'b--', linewidth=2,
                    label=f'Exponential (μ={mu_est:.3f})')

            ax3.set_xlabel('Time to Conversion', fontsize=11)
            ax3.set_ylabel('Density', fontsize=11)
            ax3.legend(fontsize=9)
            ax3.grid(alpha=0.3)
        else:
            ax3.text(0.5, 0.5, 'Insufficient data', ha='center', va='center',
                    transform=ax3.transAxes, fontsize=12)

        # Panel 4: Conversion probability vs time
        ax4 = fig.add_subplot(gs[1, 1])
        ax4.set_title('Conversion Probability Over Time', fontsize=12, fontweight='bold')

        if len(time_to_conversion) > 0:
            # Kaplan-Meier style survival curve
            sorted_times = np.sort(time_to_conversion)
            survival = 1 - np.arange(1, len(sorted_times) + 1) / len(sorted_times)

            ax4.step(sorted_times, survival, where='post', linewidth=2,
                    color='darkgreen', label='Empirical')

            # Theoretical exponential survival
            t_range = np.linspace(0, max(sorted_times), 100)
            survival_exp = np.exp(-mu_est * t_range)
            ax4.plot(t_range, survival_exp, 'r--', linewidth=2,
                    label='Exponential Model')

            ax4.set_xlabel('Time Since Arrest', fontsize=11)
            ax4.set_ylabel('P(Not Yet Converted)', fontsize=11)
            ax4.legend(fontsize=9)
            ax4.grid(alpha=0.3)

        # Panel 5: Hierarchy effect on conversion rate
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.set_title('Conversion Rate by Hierarchy', fontsize=12, fontweight='bold')

        # Count conversions by hierarchy
        hierarchy_conversions = {1: 0, 2: 0, 3: 0}
        hierarchy_arrests = {1: 0, 2: 0, 3: 0}

        for event in arrest_events:
            h = event.details.get('hierarchy_level', 1)
            hierarchy_arrests[h] += 1

        for event in conversion_events:
            h = event.details.get('hierarchy_level', 1)
            hierarchy_conversions[h] += 1

        # Conversion probabilities
        levels = [1, 2, 3]
        conv_probs = [hierarchy_conversions[h] / max(1, hierarchy_arrests[h])
                     for h in levels]

        colors_h = ['#e74c3c', '#3498db', '#2ecc71']
        bars = ax5.bar(levels, conv_probs, color=colors_h, alpha=0.7,
                      edgecolor='black', linewidth=2)

        # Add labels
        for i, (bar, prob) in enumerate(zip(bars, conv_probs)):
            height = bar.get_height()
            ax5.text(bar.get_x() + bar.get_width()/2., height,
                    f'{prob:.2%}',
                    ha='center', va='bottom', fontsize=10, fontweight='bold')

        ax5.set_xlabel('Hierarchy Level', fontsize=11)
        ax5.set_ylabel('Conversion Probability', fontsize=11)
        ax5.set_xticks(levels)
        ax5.set_xticklabels(['L1\n(Leaders)', 'L2\n(Mid)', 'L3\n(Ops)'])
        ax5.set_ylim(0, 1.1)
        ax5.grid(axis='y', alpha=0.3)

        plt.suptitle(title, fontsize=18, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"CTMC transition analysis saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig

    def plot_parameter_sensitivity(
        self,
        mc_results_dict: Dict[str, List[SimulationResults]],
        param_name: str,
        param_values: List[float],
        title: str = "Parameter Sensitivity Analysis",
        save_path: Optional[str] = None,
        figsize: Tuple[int, int] = (18, 12),
        show: bool = True
    ):
        """
        Analyze sensitivity to parameter variations.

        mc_results_dict: {param_value: [SimulationResults]}
        """
        fig = plt.figure(figsize=figsize)
        gs = fig.add_gridspec(3, 3, hspace=0.35, wspace=0.3)

        print(f"  Analyzing sensitivity to {param_name}...")

        # Aggregate metrics for each parameter value
        metrics = {
            'param_values': [],
            'mean_final_active': [],
            'std_final_active': [],
            'mean_arrests': [],
            'std_arrests': [],
            'mean_conversions': [],
            'std_conversions': [],
            'collapse_rate': [],
            'mean_collapse_time': [],
        }

        for pval in param_values:
            results_list = mc_results_dict.get(str(pval), [])

            if len(results_list) == 0:
                continue

            metrics['param_values'].append(pval)

            # Final active
            final_active = [r.time_series.network_size[-1] for r in results_list]
            metrics['mean_final_active'].append(np.mean(final_active))
            metrics['std_final_active'].append(np.std(final_active))

            # Total arrests
            total_arrests = [r.time_series.arrested_count[-1] for r in results_list]
            metrics['mean_arrests'].append(np.mean(total_arrests))
            metrics['std_arrests'].append(np.std(total_arrests))

            # Total conversions
            total_conversions = [r.time_series.informant_count[-1] for r in results_list]
            metrics['mean_conversions'].append(np.mean(total_conversions))
            metrics['std_conversions'].append(np.std(total_conversions))

            # Collapse analysis
            collapse_count = sum([1 for r in results_list
                                 if r.time_series.network_size[-1] <
                                 r.time_series.network_size[0] * 0.1])
            metrics['collapse_rate'].append(collapse_count / len(results_list))

            collapse_times = []
            for r in results_list:
                threshold = r.time_series.network_size[0] * 0.1
                collapse_idx = np.where(r.time_series.network_size < threshold)[0]
                if len(collapse_idx) > 0:
                    collapse_times.append(r.time_series.times[collapse_idx[0]])

            metrics['mean_collapse_time'].append(np.mean(collapse_times) if collapse_times else 0)

        # Convert to arrays
        pvals = np.array(metrics['param_values'])

        # Panel 1: Final active actors
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.set_title('Final Active Actors', fontsize=12, fontweight='bold')

        mean_active = np.array(metrics['mean_final_active'])
        std_active = np.array(metrics['std_final_active'])

        ax1.plot(pvals, mean_active, 'bo-', linewidth=2, markersize=8, label='Mean')
        ax1.fill_between(pvals, mean_active - std_active, mean_active + std_active,
                        alpha=0.3, color='blue', label='±1σ')

        ax1.set_xlabel(param_name, fontsize=11, fontweight='bold')
        ax1.set_ylabel('Final Active', fontsize=11)
        ax1.legend(fontsize=9)
        ax1.grid(alpha=0.3)

        # Panel 2: Total arrests
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.set_title('Total Arrests', fontsize=12, fontweight='bold')

        mean_arrests = np.array(metrics['mean_arrests'])
        std_arrests = np.array(metrics['std_arrests'])

        ax2.plot(pvals, mean_arrests, 'ro-', linewidth=2, markersize=8, label='Mean')
        ax2.fill_between(pvals, mean_arrests - std_arrests, mean_arrests + std_arrests,
                        alpha=0.3, color='red', label='±1σ')

        ax2.set_xlabel(param_name, fontsize=11, fontweight='bold')
        ax2.set_ylabel('Total Arrests', fontsize=11)
        ax2.legend(fontsize=9)
        ax2.grid(alpha=0.3)

        # Panel 3: Total conversions
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.set_title('Total Conversions', fontsize=12, fontweight='bold')

        mean_conv = np.array(metrics['mean_conversions'])
        std_conv = np.array(metrics['std_conversions'])

        ax3.plot(pvals, mean_conv, color='orange', marker='o', linewidth=2,
                markersize=8, label='Mean')
        ax3.fill_between(pvals, mean_conv - std_conv, mean_conv + std_conv,
                        alpha=0.3, color='orange', label='±1σ')

        ax3.set_xlabel(param_name, fontsize=11, fontweight='bold')
        ax3.set_ylabel('Total Conversions', fontsize=11)
        ax3.legend(fontsize=9)
        ax3.grid(alpha=0.3)

        # Panel 4: Collapse rate
        ax4 = fig.add_subplot(gs[1, 0])
        ax4.set_title('Network Collapse Rate', fontsize=12, fontweight='bold')

        collapse_rate = np.array(metrics['collapse_rate'])

        ax4.plot(pvals, collapse_rate, 'go-', linewidth=2, markersize=8)
        ax4.fill_between(pvals, 0, collapse_rate, alpha=0.3, color='green')

        ax4.set_xlabel(param_name, fontsize=11, fontweight='bold')
        ax4.set_ylabel('Collapse Probability', fontsize=11)
        ax4.set_ylim(0, 1.1)
        ax4.grid(alpha=0.3)

        # Panel 5: Mean collapse time
        ax5 = fig.add_subplot(gs[1, 1])
        ax5.set_title('Mean Collapse Time', fontsize=12, fontweight='bold')

        mean_collapse_time = np.array(metrics['mean_collapse_time'])

        ax5.plot(pvals, mean_collapse_time, 'mo-', linewidth=2, markersize=8)

        ax5.set_xlabel(param_name, fontsize=11, fontweight='bold')
        ax5.set_ylabel('Time', fontsize=11)
        ax5.grid(alpha=0.3)

        # Panel 6: Sensitivity indices
        ax6 = fig.add_subplot(gs[1, 2])
        ax6.set_title('Normalized Sensitivity', fontsize=12, fontweight='bold')

        # Compute normalized sensitivity (relative change)
        if len(pvals) > 1:
            def normalize_sensitivity(values):
                if np.max(values) > 0:
                    return (values - values[0]) / (np.max(values) - np.min(values) + 1e-10)
                return values * 0

            sens_active = normalize_sensitivity(mean_active)
            sens_arrests = normalize_sensitivity(mean_arrests)
            sens_conv = normalize_sensitivity(mean_conv)

            ax6.plot(pvals, sens_active, 'b-o', linewidth=2, label='Active', markersize=6)
            ax6.plot(pvals, sens_arrests, 'r-o', linewidth=2, label='Arrests', markersize=6)
            ax6.plot(pvals, sens_conv, color='orange', marker='o', linewidth=2,
                    label='Conversions', markersize=6)

            ax6.axhline(0, color='black', linestyle='--', linewidth=1)
            ax6.set_xlabel(param_name, fontsize=11, fontweight='bold')
            ax6.set_ylabel('Normalized Change', fontsize=11)
            ax6.legend(fontsize=9)
            ax6.grid(alpha=0.3)

        # Panel 7-9: Outcome distributions for selected parameter values
        selected_indices = [0, len(pvals) // 2, -1]
        selected_pvals = [pvals[i] for i in selected_indices if i < len(pvals)]

        for idx, (subplot_idx, pval) in enumerate(zip([gs[2, 0], gs[2, 1], gs[2, 2]],
                                                       selected_pvals)):
            ax = fig.add_subplot(subplot_idx)
            ax.set_title(f'{param_name}={pval:.3f}', fontsize=12, fontweight='bold')

            results_list = mc_results_dict.get(str(pval), [])

            if len(results_list) > 0:
                final_active = [r.time_series.network_size[-1] for r in results_list]
                final_arrested = [r.time_series.arrested_count[-1] for r in results_list]
                final_informant = [r.time_series.informant_count[-1] for r in results_list]

                ax.hist(final_active, bins=15, alpha=0.5, label='Active',
                       color='green', edgecolor='black')
                ax.hist(final_arrested, bins=15, alpha=0.5, label='Arrested',
                       color='red', edgecolor='black')
                ax.hist(final_informant, bins=15, alpha=0.5, label='Informant',
                       color='orange', edgecolor='black')

                ax.set_xlabel('Final Count', fontsize=10)
                ax.set_ylabel('Frequency', fontsize=10)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.3)

        plt.suptitle(title + f' ({param_name})', fontsize=18, fontweight='bold', y=0.995)

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Parameter sensitivity analysis saved to: {save_path}")

        if show:
            plt.show()
        else:
            plt.close()

        return fig


def create_stochastic_visualizations(
    mc_results_list: List[SimulationResults],
    single_result: SimulationResults,
    output_dir: str = "stochastic_viz",
    show: bool = False
):
    """
    Create complete suite of stochastic analysis visualizations.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    viz = StochasticProcessVisualizer()

    print("Creating stochastic analysis visualizations...")

    # 1. Stochastic path analysis
    print("  [1/4] Stochastic path analysis...")
    viz.plot_stochastic_path_analysis(
        mc_results_list,
        save_path=str(output_path / "stochastic_path_analysis.png"),
        show=show
    )

    # 2. Cox process intensity
    print("  [2/4] Cox process intensity...")
    viz.plot_cox_process_intensity(
        single_result,
        save_path=str(output_path / "cox_process_intensity.png"),
        show=show
    )

    # 3. Trust SDE dynamics
    print("  [3/4] Trust SDE dynamics...")
    viz.plot_trust_sde_dynamics(
        single_result,
        save_path=str(output_path / "trust_sde_dynamics.png"),
        show=show
    )

    # 4. CTMC transition analysis
    print("  [4/4] CTMC transition analysis...")
    viz.plot_ctmc_transition_analysis(
        single_result,
        save_path=str(output_path / "ctmc_transition_analysis.png"),
        show=show
    )

    print(f"\nAll stochastic visualizations saved to: {output_dir}/")
    print("  - stochastic_path_analysis.png")
    print("  - cox_process_intensity.png")
    print("  - trust_sde_dynamics.png")
    print("  - ctmc_transition_analysis.png")
