"""
Analysis utilities for simulation results.

This module provides statistical analysis functions for comparing
theoretical predictions with simulation outputs and computing
aggregate metrics across multiple runs.

Specification Reference: US-005 (Extract and Analyze Event Histories)
"""

from typing import List, Dict, Tuple
import numpy as np
from scipy import stats

from .results import SimulationResults
from .events import EventType


def compute_conversion_statistics(results: SimulationResults) -> Dict[str, float]:
    """
    Compute statistics for informant conversion events.

    Parameters
    ----------
    results : SimulationResults
        Simulation results to analyze

    Returns
    -------
    stats : dict
        Dictionary of conversion statistics:
        - total_conversions: Total number of conversions
        - mean_time_to_conversion: Average time from arrest to conversion
        - std_time_to_conversion: Std dev of conversion times
        - conversion_rate: Fraction of arrests that converted
        - theoretical_mean_time: Expected conversion time from parameters

    Examples
    --------
    >>> stats = compute_conversion_statistics(results)
    >>> stats['conversion_rate']
    0.85
    """
    # Filter conversion events
    conversions = [e for e in results.events if e.event_type == EventType.CONVERSION]

    if len(conversions) == 0:
        return {
            'total_conversions': 0,
            'mean_time_to_conversion': np.nan,
            'std_time_to_conversion': np.nan,
            'conversion_rate': 0.0,
            'theoretical_mean_time': np.nan
        }

    # Extract time to conversion
    times_to_conversion = [e.details['time_in_arrested'] for e in conversions]

    # Compute statistics
    mean_time = np.mean(times_to_conversion)
    std_time = np.std(times_to_conversion)

    # Conversion rate
    total_arrests = results.total_arrests
    conversion_rate = len(conversions) / total_arrests if total_arrests > 0 else 0.0

    # Theoretical expected time: E[T] = 1/μ_LH + 1/μ_HI
    # Use mean fragility rate across conversions
    if conversions and 'fragility_rate' in conversions[0].details:
        mean_fragility = np.mean([e.details['fragility_rate'] for e in conversions])
        theoretical_mean = (1.0 / results.parameters.mu_LH) + (1.0 / mean_fragility)
    else:
        theoretical_mean = np.nan

    return {
        'total_conversions': len(conversions),
        'mean_time_to_conversion': mean_time,
        'std_time_to_conversion': std_time,
        'conversion_rate': conversion_rate,
        'theoretical_mean_time': theoretical_mean,
        'min_time': np.min(times_to_conversion),
        'max_time': np.max(times_to_conversion),
        'median_time': np.median(times_to_conversion)
    }


def compute_arrest_statistics(results: SimulationResults) -> Dict[str, float]:
    """
    Compute statistics for arrest events.

    Parameters
    ----------
    results : SimulationResults
        Simulation results to analyze

    Returns
    -------
    stats : dict
        Dictionary of arrest statistics

    Examples
    --------
    >>> stats = compute_arrest_statistics(results)
    >>> stats['total_arrests']
    35
    """
    arrests = [e for e in results.events if e.event_type == EventType.ARREST]

    if len(arrests) == 0:
        return {
            'total_arrests': 0,
            'mean_arrest_time': np.nan,
            'arrests_per_time_unit': 0.0,
            'mean_intensity': np.nan,
            'hierarchy_distribution': {}
        }

    # Arrest times
    arrest_times = [e.timestamp for e in arrests]

    # Arrest intensities
    intensities = [e.details.get('arrest_intensity', np.nan) for e in arrests]

    # Hierarchy distribution
    hierarchies = [e.details.get('hierarchy_level', 0) for e in arrests]
    hierarchy_dist = {}
    for h in set(hierarchies):
        hierarchy_dist[h] = sum(1 for x in hierarchies if x == h)

    # Compute statistics
    mean_time = np.mean(arrest_times)
    arrests_per_time = len(arrests) / results.parameters.T_max if results.parameters.T_max > 0 else 0

    return {
        'total_arrests': len(arrests),
        'mean_arrest_time': mean_time,
        'std_arrest_time': np.std(arrest_times),
        'first_arrest_time': np.min(arrest_times) if arrest_times else np.nan,
        'last_arrest_time': np.max(arrest_times) if arrest_times else np.nan,
        'arrests_per_time_unit': arrests_per_time,
        'mean_intensity': np.mean([x for x in intensities if not np.isnan(x)]),
        'hierarchy_distribution': hierarchy_dist
    }


def compute_network_evolution_metrics(results: SimulationResults) -> Dict[str, float]:
    """
    Compute metrics describing network evolution over time.

    Parameters
    ----------
    results : SimulationResults
        Simulation results to analyze

    Returns
    -------
    metrics : dict
        Network evolution metrics

    Examples
    --------
    >>> metrics = compute_network_evolution_metrics(results)
    >>> metrics['collapse_time']
    52.3
    """
    ts = results.time_series

    # Find collapse time (when network size drops to 0 or 1)
    collapse_indices = np.where(ts.network_size <= 1)[0]
    if len(collapse_indices) > 0:
        collapse_time = ts.times[collapse_indices[0]]
        collapsed = True
    else:
        collapse_time = np.nan
        collapsed = False

    # Compute rates of change
    if len(ts.times) > 1:
        dt_series = np.diff(ts.times)

        # Arrest rate over time
        arrests_diff = np.diff(ts.arrested_count)
        arrest_rate = np.mean(arrests_diff / dt_series) if len(dt_series) > 0 else 0

        # Effectiveness change
        effectiveness_change = ts.effectiveness[-1] - ts.effectiveness[0]

        # Trust decay
        trust_change = ts.mean_trust[-1] - ts.mean_trust[0]
    else:
        arrest_rate = 0
        effectiveness_change = 0
        trust_change = 0

    # Peak effectiveness
    peak_effectiveness = np.max(ts.effectiveness)
    peak_effectiveness_time = ts.times[np.argmax(ts.effectiveness)]

    return {
        'collapsed': collapsed,
        'collapse_time': collapse_time,
        'initial_size': int(ts.network_size[0]),
        'final_size': int(ts.network_size[-1]),
        'size_reduction': int(ts.network_size[0] - ts.network_size[-1]),
        'arrest_rate': arrest_rate,
        'initial_effectiveness': ts.effectiveness[0],
        'final_effectiveness': ts.effectiveness[-1],
        'peak_effectiveness': peak_effectiveness,
        'peak_effectiveness_time': peak_effectiveness_time,
        'effectiveness_change': effectiveness_change,
        'initial_mean_trust': ts.mean_trust[0],
        'final_mean_trust': ts.mean_trust[-1],
        'trust_change': trust_change,
        'max_risk_observed': np.max(ts.mean_risk),
        'mean_risk_overall': np.mean(ts.mean_risk)
    }


def compare_strategies(results_list: List[SimulationResults], strategy_names: List[str]) -> Dict:
    """
    Compare multiple simulation results from different strategies.

    Parameters
    ----------
    results_list : List[SimulationResults]
        List of simulation results to compare
    strategy_names : List[str]
        Names for each strategy

    Returns
    -------
    comparison : dict
        Comparative statistics across strategies

    Examples
    --------
    >>> results_agg = run_simulation(aggressive_params, network)
    >>> results_con = run_simulation(conservative_params, network)
    >>> comparison = compare_strategies(
    ...     [results_agg, results_con],
    ...     ['Aggressive', 'Conservative']
    ... )
    >>> comparison['total_arrests']['Aggressive']
    45
    """
    if len(results_list) != len(strategy_names):
        raise ValueError("Length of results_list must match strategy_names")

    comparison = {
        'total_arrests': {},
        'total_conversions': {},
        'final_effectiveness': {},
        'collapse_time': {},
        'arrest_rate': {},
        'conversion_rate': {},
        'simulation_duration': {}
    }

    for results, name in zip(results_list, strategy_names):
        # Basic metrics
        comparison['total_arrests'][name] = results.total_arrests
        comparison['total_conversions'][name] = results.total_conversions
        comparison['final_effectiveness'][name] = results.final_effectiveness
        comparison['simulation_duration'][name] = results.simulation_duration

        # Network evolution
        net_metrics = compute_network_evolution_metrics(results)
        comparison['collapse_time'][name] = net_metrics['collapse_time']
        comparison['arrest_rate'][name] = net_metrics['arrest_rate']

        # Conversion statistics
        conv_stats = compute_conversion_statistics(results)
        comparison['conversion_rate'][name] = conv_stats['conversion_rate']

    return comparison


def compute_fragility_cycle_correlation(results: SimulationResults) -> Dict[str, float]:
    """
    Analyze the fragility cycle: arrests → risk ↑ → trust ↓ → conversions ↑ → effectiveness ↑

    Parameters
    ----------
    results : SimulationResults
        Simulation results to analyze

    Returns
    -------
    correlations : dict
        Correlation coefficients between key variables

    Examples
    --------
    >>> corr = compute_fragility_cycle_correlation(results)
    >>> corr['arrests_vs_risk']
    0.85
    """
    ts = results.time_series

    # Compute correlations between key variables
    # Note: Use lagged correlations where causal relationship expected

    correlations = {}

    # Arrests should increase risk (lagged)
    if len(ts.times) > 10:
        # Use difference series
        arrests_diff = np.diff(ts.arrested_count)
        risk_series = ts.mean_risk[1:]  # Align with diff

        if np.std(arrests_diff) > 0 and np.std(risk_series) > 0:
            corr_arrests_risk = np.corrcoef(arrests_diff, risk_series)[0, 1]
        else:
            corr_arrests_risk = np.nan

        # Risk should decrease trust
        risk_series_full = ts.mean_risk[:-1]
        trust_series = ts.mean_trust[1:]

        if np.std(risk_series_full) > 0 and np.std(trust_series) > 0:
            corr_risk_trust = np.corrcoef(risk_series_full, trust_series)[0, 1]
        else:
            corr_risk_trust = np.nan

        # Conversions should increase effectiveness
        conv_diff = np.diff(ts.informant_count)
        effect_series = ts.effectiveness[1:]

        if np.std(conv_diff) > 0 and np.std(effect_series) > 0:
            corr_conv_effect = np.corrcoef(conv_diff, effect_series)[0, 1]
        else:
            corr_conv_effect = np.nan

        correlations = {
            'arrests_vs_risk': corr_arrests_risk,
            'risk_vs_trust': corr_risk_trust,
            'conversions_vs_effectiveness': corr_conv_effect,
            'arrests_vs_trust': np.corrcoef(arrests_diff, ts.mean_trust[1:])[0, 1] if np.std(ts.mean_trust[1:]) > 0 else np.nan
        }
    else:
        correlations = {
            'arrests_vs_risk': np.nan,
            'risk_vs_trust': np.nan,
            'conversions_vs_effectiveness': np.nan,
            'arrests_vs_trust': np.nan
        }

    return correlations


def validate_stochastic_processes(results: SimulationResults) -> Dict[str, Dict]:
    """
    Validate stochastic processes against theoretical distributions.

    Parameters
    ----------
    results : SimulationResults
        Simulation results to validate

    Returns
    -------
    validation : dict
        Validation results for each process

    Examples
    --------
    >>> validation = validate_stochastic_processes(results)
    >>> validation['conversion_times']['ks_statistic']
    0.12
    """
    validation = {}

    # Validate conversion times (exponential distribution check)
    conversions = [e for e in results.events if e.event_type == EventType.CONVERSION]
    if len(conversions) > 5:
        times = [e.details['time_in_arrested'] for e in conversions]

        # Fit exponential distribution
        mean_time = np.mean(times)
        rate_param = 1.0 / mean_time if mean_time > 0 else 0

        # Kolmogorov-Smirnov test
        ks_stat, ks_pval = stats.kstest(times, 'expon', args=(0, mean_time))

        validation['conversion_times'] = {
            'n_samples': len(times),
            'mean_observed': mean_time,
            'ks_statistic': ks_stat,
            'ks_pvalue': ks_pval,
            'passes_ks_test': ks_pval > 0.05
        }
    else:
        validation['conversion_times'] = {
            'n_samples': len(conversions),
            'insufficient_data': True
        }

    # Validate arrest event intervals
    arrests = [e for e in results.events if e.event_type == EventType.ARREST]
    if len(arrests) > 5:
        arrest_times = sorted([e.timestamp for e in arrests])
        intervals = np.diff(arrest_times)

        # Mean interval
        mean_interval = np.mean(intervals)

        validation['arrest_intervals'] = {
            'n_intervals': len(intervals),
            'mean_interval': mean_interval,
            'std_interval': np.std(intervals),
            'min_interval': np.min(intervals),
            'max_interval': np.max(intervals)
        }
    else:
        validation['arrest_intervals'] = {
            'n_intervals': len(arrests) - 1 if len(arrests) > 1 else 0,
            'insufficient_data': True
        }

    return validation


def generate_summary_report(results: SimulationResults) -> str:
    """
    Generate comprehensive text summary of simulation results.

    Parameters
    ----------
    results : SimulationResults
        Simulation results to summarize

    Returns
    -------
    report : str
        Formatted summary report

    Examples
    --------
    >>> report = generate_summary_report(results)
    >>> print(report)
    """
    # Compute all statistics
    arrest_stats = compute_arrest_statistics(results)
    conv_stats = compute_conversion_statistics(results)
    net_metrics = compute_network_evolution_metrics(results)
    cycle_corr = compute_fragility_cycle_correlation(results)

    report = f"""
{'='*70}
SIMULATION ANALYSIS REPORT
{'='*70}

RUN INFORMATION:
  Run ID:                    {results.run_id}
  Convergence Status:        {results.convergence_status.value}
  Numerical Stability:       {'OK' if results.numerical_stability else 'FAILED'}
  Causality Preserved:       {'OK' if results.causality_preserved else 'FAILED'}
  Simulation Duration:       {results.simulation_duration:.2f} seconds

NETWORK EVOLUTION:
  Initial Size:              {net_metrics['initial_size']}
  Final Size:                {net_metrics['final_size']}
  Size Reduction:            {net_metrics['size_reduction']} ({net_metrics['size_reduction']/net_metrics['initial_size']*100:.1f}%)
  Network Collapsed:         {'Yes' if net_metrics['collapsed'] else 'No'}
  Collapse Time:             {'N/A' if np.isnan(net_metrics['collapse_time']) else f"{net_metrics['collapse_time']:.2f}"}

ARREST STATISTICS:
  Total Arrests:             {arrest_stats['total_arrests']}
  Arrests per Time Unit:     {arrest_stats['arrests_per_time_unit']:.3f}
  First Arrest:              {'N/A' if np.isnan(arrest_stats['first_arrest_time']) else f"{arrest_stats['first_arrest_time']:.2f}"}
  Last Arrest:               {'N/A' if np.isnan(arrest_stats['last_arrest_time']) else f"{arrest_stats['last_arrest_time']:.2f}"}
  Mean Intensity:            {'N/A' if np.isnan(arrest_stats['mean_intensity']) else f"{arrest_stats['mean_intensity']:.4f}"}

CONVERSION STATISTICS:
  Total Conversions:         {conv_stats['total_conversions']}
  Conversion Rate:           {conv_stats['conversion_rate']:.1%}
  Mean Time to Convert:      {'N/A' if np.isnan(conv_stats['mean_time_to_conversion']) else f"{conv_stats['mean_time_to_conversion']:.2f}"}
  Std Time to Convert:       {'N/A' if np.isnan(conv_stats['std_time_to_conversion']) else f"{conv_stats['std_time_to_conversion']:.2f}"}
  Theoretical Mean Time:     {'N/A' if np.isnan(conv_stats['theoretical_mean_time']) else f"{conv_stats['theoretical_mean_time']:.2f}"}

EFFECTIVENESS DYNAMICS:
  Initial:                   {net_metrics['initial_effectiveness']:.3f}
  Final:                     {net_metrics['final_effectiveness']:.3f}
  Peak:                      {net_metrics['peak_effectiveness']:.3f} (at t={net_metrics['peak_effectiveness_time']:.2f})
  Total Change:              {net_metrics['effectiveness_change']:+.3f}

TRUST AND RISK:
  Initial Mean Trust:        {net_metrics['initial_mean_trust']:.3f}
  Final Mean Trust:          {net_metrics['final_mean_trust']:.3f}
  Trust Change:              {net_metrics['trust_change']:+.3f}
  Max Risk Observed:         {net_metrics['max_risk_observed']:.3f}
  Mean Risk Overall:         {net_metrics['mean_risk_overall']:.3f}

FRAGILITY CYCLE CORRELATIONS:
  Arrests vs Risk:           {'N/A' if np.isnan(cycle_corr['arrests_vs_risk']) else f"{cycle_corr['arrests_vs_risk']:.3f}"}
  Risk vs Trust:             {'N/A' if np.isnan(cycle_corr['risk_vs_trust']) else f"{cycle_corr['risk_vs_trust']:.3f}"}
  Conversions vs Effect.:    {'N/A' if np.isnan(cycle_corr['conversions_vs_effectiveness']) else f"{cycle_corr['conversions_vs_effectiveness']:.3f}"}

{'='*70}
"""
    return report
