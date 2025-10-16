"""
Basic integration tests for simulation engine.

Tests core functionality end-to-end without extensive validation.
"""

import numpy as np
import pytest

from organized_crime_network.simulation import (
    SimulationEngine,
    SimulationParameters,
    topology,
    analysis
)
from organized_crime_network.core.state import NetworkState, ActorState


class TestParameterPresets:
    """Test parameter preset functionality (US2)."""

    def test_default_parameters(self):
        """Test default parameters are valid."""
        params = SimulationParameters.default()
        assert params.lambda_0 == 0.1
        assert params.alpha > 0
        assert params.T_max > 0

    def test_aggressive_preset(self):
        """Test aggressive enforcement preset."""
        params = SimulationParameters.aggressive_enforcement()
        assert params.lambda_0 == 0.3  # 3x base
        assert params.eta_P == 1.0     # 2x boost

    def test_conservative_preset(self):
        """Test conservative enforcement preset."""
        params = SimulationParameters.conservative_enforcement()
        assert params.lambda_0 == 0.05  # 0.5x base
        assert params.rho == 0.3        # Faster decay

    def test_balanced_preset(self):
        """Test balanced strategy preset."""
        params = SimulationParameters.balanced_strategy()
        assert 0.1 < params.lambda_0 < 0.2

    def test_high_resilience_preset(self):
        """Test high resilience network preset."""
        params = SimulationParameters.high_resilience_network()
        assert params.kappa == 1.0  # Strong protection
        assert params.theta == 3.0  # High loyalty

    def test_parameter_description(self):
        """Test parameter description generation."""
        params = SimulationParameters.default()
        desc = params.get_parameter_description()
        assert 'ARREST PROCESS' in desc
        assert 'TRUST DYNAMICS' in desc
        assert 'SIMULATION CONTROL' in desc


class TestTopologyGenerators:
    """Test network topology generators (US3)."""

    def test_scale_free_network(self):
        """Test scale-free network generation."""
        network = topology.create_scale_free_network(n_actors=50, m=2, seed=42)

        assert len(network.V) == 50
        assert len(network.E) > 0
        assert len(network.get_active_actors()) == 50

        # Verify all actors have hierarchy
        for actor_id in network.V:
            assert actor_id in network.hierarchy

    def test_hierarchical_network(self):
        """Test hierarchical network generation."""
        network = topology.create_hierarchical_network(
            level_sizes=[20, 10, 5],
            connections_per_level=2,
            seed=42
        )

        assert len(network.V) == 35
        assert len(network.get_active_actors()) == 35

        # Check hierarchy distribution
        level_1 = sum(1 for a in network.V if network.hierarchy[a] == 1)
        level_2 = sum(1 for a in network.V if network.hierarchy[a] == 2)
        level_3 = sum(1 for a in network.V if network.hierarchy[a] == 3)

        assert level_1 == 20
        assert level_2 == 10
        assert level_3 == 5

    def test_random_network(self):
        """Test random network generation."""
        network = topology.create_random_network(
            n_actors=30,
            edge_probability=0.15,
            seed=42
        )

        assert len(network.V) == 30
        assert len(network.get_active_actors()) == 30

    def test_small_world_network(self):
        """Test small-world network generation."""
        network = topology.create_small_world_network(
            n_actors=40,
            k=4,
            p=0.1,
            seed=42
        )

        assert len(network.V) == 40
        assert len(network.E) > 0

    def test_core_periphery_network(self):
        """Test core-periphery network generation."""
        network = topology.create_core_periphery_network(
            n_core=10,
            n_periphery=30,
            core_density=0.8,
            seed=42
        )

        assert len(network.V) == 40

        # Check hierarchy levels
        core_count = sum(1 for a in network.V if network.hierarchy[a] == 2)
        periphery_count = sum(1 for a in network.V if network.hierarchy[a] == 1)

        assert core_count == 10
        assert periphery_count == 30

    def test_network_validation(self):
        """Test network structure validation."""
        network = topology.create_scale_free_network(50, seed=42)
        is_valid, errors = topology.validate_network_structure(network)

        assert is_valid
        assert len(errors) == 0

    def test_network_statistics(self):
        """Test network statistics computation."""
        network = topology.create_hierarchical_network([20, 10, 5], seed=42)
        stats = topology.get_network_statistics(network)

        assert stats['num_actors'] == 35
        assert stats['num_edges'] > 0
        assert stats['avg_degree'] > 0
        assert stats['is_connected'] or stats['num_components'] > 0


class TestSimulationEngine:
    """Test simulation engine core functionality."""

    def test_basic_simulation_runs(self):
        """Test that basic simulation completes without errors."""
        params = SimulationParameters.default()
        params = SimulationParameters(
            **{**params.__dict__, 'T_max': 10.0}  # Short simulation
        )

        network = topology.create_hierarchical_network([10, 5, 3], seed=42)
        engine = SimulationEngine(params)

        results = engine.run(network, verbose=False)

        assert results is not None
        assert results.numerical_stability
        assert results.total_arrests >= 0
        assert results.total_conversions >= 0

    def test_deterministic_results(self):
        """Test same seed produces identical results."""
        params = SimulationParameters.default()
        params = SimulationParameters(
            **{**params.__dict__, 'T_max': 10.0, 'random_seed': 42}
        )

        network = topology.create_scale_free_network(15, seed=42)

        engine1 = SimulationEngine(params)
        engine2 = SimulationEngine(params)

        results1 = engine1.run(network, verbose=False)
        results2 = engine2.run(network, verbose=False)

        assert results1.total_arrests == results2.total_arrests
        assert results1.total_conversions == results2.total_conversions
        assert np.allclose(
            results1.time_series.effectiveness,
            results2.time_series.effectiveness
        )

    def test_network_validation_errors(self):
        """Test validation catches invalid networks."""
        params = SimulationParameters.default()
        engine = SimulationEngine(params)

        # Empty network
        empty_network = NetworkState(t_current=0.0)
        errors = engine.validate_initial_network(empty_network)

        assert len(errors) > 0
        assert any('active actor' in e.lower() for e in errors)

    def test_performance_metrics(self):
        """Test performance metrics are tracked."""
        params = SimulationParameters.default()
        params = SimulationParameters(
            **{**params.__dict__, 'T_max': 5.0}
        )

        network = topology.create_scale_free_network(10, seed=42)
        engine = SimulationEngine(params)

        results = engine.run(network, verbose=False)
        metrics = engine.get_performance_metrics()

        assert 'wall_time' in metrics
        assert 'steps_per_second' in metrics
        assert 'events_generated' in metrics
        assert metrics['wall_time'] > 0


class TestAnalysisUtilities:
    """Test analysis utilities (US5)."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        params = SimulationParameters.default()
        params = SimulationParameters(
            **{**params.__dict__, 'T_max': 20.0}
        )

        network = topology.create_hierarchical_network([15, 8, 4], seed=42)
        engine = SimulationEngine(params)

        return engine.run(network, verbose=False)

    def test_conversion_statistics(self, sample_results):
        """Test conversion statistics computation."""
        stats = analysis.compute_conversion_statistics(sample_results)

        assert 'total_conversions' in stats
        assert 'mean_time_to_conversion' in stats
        assert 'conversion_rate' in stats
        assert stats['total_conversions'] >= 0

    def test_arrest_statistics(self, sample_results):
        """Test arrest statistics computation."""
        stats = analysis.compute_arrest_statistics(sample_results)

        assert 'total_arrests' in stats
        assert 'arrests_per_time_unit' in stats
        assert 'hierarchy_distribution' in stats
        assert stats['total_arrests'] >= 0

    def test_network_evolution_metrics(self, sample_results):
        """Test network evolution metrics computation."""
        metrics = analysis.compute_network_evolution_metrics(sample_results)

        assert 'initial_size' in metrics
        assert 'final_size' in metrics
        assert 'collapsed' in metrics
        assert metrics['initial_size'] > 0

    def test_fragility_cycle_correlation(self, sample_results):
        """Test fragility cycle correlation analysis."""
        corr = analysis.compute_fragility_cycle_correlation(sample_results)

        assert 'arrests_vs_risk' in corr
        assert 'risk_vs_trust' in corr
        assert 'conversions_vs_effectiveness' in corr

    def test_summary_report_generation(self, sample_results):
        """Test summary report generation."""
        report = analysis.generate_summary_report(sample_results)

        assert isinstance(report, str)
        assert 'SIMULATION ANALYSIS REPORT' in report
        assert 'NETWORK EVOLUTION' in report
        assert 'ARREST STATISTICS' in report
        assert 'CONVERSION STATISTICS' in report

    def test_compare_strategies(self):
        """Test strategy comparison functionality."""
        # Run two simulations with different params
        params_agg = SimulationParameters.aggressive_enforcement()
        params_agg = SimulationParameters(
            **{**params_agg.__dict__, 'T_max': 10.0}
        )

        params_con = SimulationParameters.conservative_enforcement()
        params_con = SimulationParameters(
            **{**params_con.__dict__, 'T_max': 10.0}
        )

        network = topology.create_scale_free_network(15, seed=42)

        engine_agg = SimulationEngine(params_agg)
        engine_con = SimulationEngine(params_con)

        results_agg = engine_agg.run(network, verbose=False)
        results_con = engine_con.run(network, verbose=False)

        comparison = analysis.compare_strategies(
            [results_agg, results_con],
            ['Aggressive', 'Conservative']
        )

        assert 'total_arrests' in comparison
        assert 'Aggressive' in comparison['total_arrests']
        assert 'Conservative' in comparison['total_arrests']


class TestExportFunctionality:
    """Test data export functionality."""

    def test_time_series_export(self, tmp_path):
        """Test time series CSV export."""
        params = SimulationParameters.default()
        params = SimulationParameters(
            **{**params.__dict__, 'T_max': 5.0}
        )

        network = topology.create_scale_free_network(10, seed=42)
        engine = SimulationEngine(params)
        results = engine.run(network, verbose=False)

        # Export to temporary file
        filepath = tmp_path / "test_time_series.csv"
        results.export_time_series(str(filepath))

        assert filepath.exists()
        assert filepath.stat().st_size > 0

    def test_events_export(self, tmp_path):
        """Test events JSON export."""
        params = SimulationParameters.default()
        params = SimulationParameters(
            **{**params.__dict__, 'T_max': 5.0}
        )

        network = topology.create_scale_free_network(10, seed=42)
        engine = SimulationEngine(params)
        results = engine.run(network, verbose=False)

        # Export to temporary file
        filepath = tmp_path / "test_events.json"
        results.export_events(str(filepath))

        assert filepath.exists()
        assert filepath.stat().st_size > 0


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
