"""
Complete Scenario Runner
========================

Tek komutla tüm analiz ve görselleştirmeleri üretir:
- Network görselleştirmeleri (static + interactive + ultra-advanced)
- Expert-level analizler (Sankey, motif, community, resilience)
- Stokastik süreç analizleri (path, Cox, SDE, CTMC)
- Monte Carlo simülasyonları
- LaTeX rapor (tablolar ve istatistikler)

Kullanım:
    python run_scenario.py --scenario aggressive
    python run_scenario.py --scenario custom --config my_config.py
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime
import json
import numpy as np

sys.path.insert(0, str(Path(__file__).parent))

from organized_crime_network.simulation import (
    SimulationEngine,
    SimulationParameters,
    topology
)
from organized_crime_network.simulation.visualization_network_advanced import (
    create_advanced_network_visualizations
)
from organized_crime_network.simulation.visualization_interactive import (
    create_interactive_visualizations
)
from organized_crime_network.simulation.visualization_ultra_advanced import (
    create_ultra_advanced_visualizations
)
from organized_crime_network.simulation.visualization_expert import (
    create_expert_visualizations
)
from organized_crime_network.simulation.visualization_stochastic import (
    create_stochastic_visualizations
)


# ============================================================================
# KALIBRASYON PARAMETRELERI - BURADAN DEĞİŞTİR
# ============================================================================

SCENARIOS = {
    "default": {
        "name": "Default Moderate Enforcement",
        "description": "Balanced enforcement with moderate parameters",
        "params": {
            # Arrest Process (Cox Process)
            "lambda_0": 0.1,        # Baseline arrest rate
            "kappa": 0.5,           # Hierarchical protection
            "gamma": 0.3,           # Operational risk multiplier
            "P_0": 1.0,             # Initial effectiveness
            "rho": 0.2,             # Effectiveness decay
            "eta_P": 0.1,           # Effectiveness jump per conversion

            # Trust Dynamics (Ornstein-Uhlenbeck SDE)
            "alpha": 0.5,           # Mean reversion rate
            "beta": 1.0,            # Risk sensitivity
            "sigma": 0.15,          # Volatility
            "Delta": 10.0,          # Memory time
            "delta_R": 0.1,         # Risk decay rate (1/Delta)

            # Informant Conversion (CTMC)
            "mu_LH": 0.3,           # High-trust conversion rate
            "mu_min": 0.01,         # Minimum conversion rate
            "mu_rng": 0.5,          # Range
            "theta": 2.0,           # Trust sensitivity
            "Y_bar_0": 0.0,         # Default logit-trust

            # Network Growth (TFPA)
            "w_min": 0.3,           # Minimum trust threshold
            "gamma_pa": 1.5,        # Preferential attachment strength

            # Simulation
            "T_max": 100.0,         # Simulation time
            "dt": 0.01,             # Time step
            "random_seed": 42,
        },
        "network": {
            "hierarchy": [60, 30, 10],  # L1 (operatives), L2 (mid), L3 (leaders)
            "seed": 42,
        },
        "monte_carlo": {
            "n_runs": 50,           # Monte Carlo simulations
            "n_cores": 1,           # Parallel cores (TODO)
        }
    },

    "aggressive": {
        "name": "Aggressive Law Enforcement",
        "description": "High arrest rate, strong effectiveness, rapid collapse",
        "params": {
            "lambda_0": 0.3,        # HIGH arrest rate (3x)
            "kappa": 0.3,           # LOW hierarchical protection
            "gamma": 0.5,           # HIGH operational risk
            "P_0": 1.2,             # HIGH effectiveness
            "rho": 0.1,             # SLOW decay (sustained)
            "eta_P": 0.15,          # LARGE jumps
            "alpha": 0.7,           # FAST mean reversion
            "beta": 1.5,            # HIGH risk sensitivity
            "sigma": 0.2,           # HIGH volatility
            "Delta": 5.0,           # SHORT memory
            "delta_R": 0.2,         # Fast risk decay (1/Delta)
            "mu_LH": 0.5,           # HIGH conversion
            "mu_min": 0.05,         # HIGHER minimum
            "mu_rng": 0.6,          # WIDER range
            "theta": 1.5,           # LOWER sensitivity (easier conversion)
            "Y_bar_0": 0.0,         # Default logit-trust
            "w_min": 0.3,           # Trust threshold
            "gamma_pa": 1.5,        # Preferential attachment
            "T_max": 50.0,          # Shorter (fast collapse)
            "dt": 0.01,
            "random_seed": 42,
        },
        "network": {
            "hierarchy": [60, 30, 10],
            "seed": 42,
        },
        "monte_carlo": {
            "n_runs": 50,
            "n_cores": 1,
        }
    },

    "conservative": {
        "name": "Conservative Limited Enforcement",
        "description": "Low arrest rate, limited resources, resilient network",
        "params": {
            "lambda_0": 0.05,       # LOW arrest rate
            "kappa": 0.7,           # HIGH protection
            "gamma": 0.2,           # LOW operational risk
            "P_0": 0.8,             # LOW effectiveness
            "rho": 0.3,             # FAST decay
            "eta_P": 0.08,          # SMALL jumps
            "alpha": 0.3,           # SLOW reversion
            "beta": 0.7,            # LOW risk sensitivity
            "sigma": 0.1,           # LOW volatility
            "Delta": 20.0,          # LONG memory
            "delta_R": 0.05,        # Slow risk decay (1/Delta)
            "mu_LH": 0.2,           # LOW conversion
            "mu_min": 0.005,        # VERY LOW minimum
            "mu_rng": 0.35,         # NARROW range
            "theta": 2.5,           # HIGH sensitivity (hard conversion)
            "Y_bar_0": 0.0,         # Default logit-trust
            "w_min": 0.4,           # Higher trust threshold
            "gamma_pa": 1.5,        # Preferential attachment
            "T_max": 200.0,         # Longer horizon
            "dt": 0.01,
            "random_seed": 42,
        },
        "network": {
            "hierarchy": [60, 30, 10],
            "seed": 42,
        },
        "monte_carlo": {
            "n_runs": 50,
            "n_cores": 1,
        }
    },

    "realistic": {
        "name": "Realistic Baseline",
        "description": "Literature-based realistic parameters",
        "params": {
            "lambda_0": 0.05,
            "kappa": 0.8,
            "gamma": 0.3,
            "P_0": 0.8,
            "rho": 0.2,
            "eta_P": 0.1,
            "alpha": 0.3,
            "beta": 1.0,
            "sigma": 0.08,
            "Delta": 20.0,
            "delta_R": 0.05,
            "mu_LH": 0.15,
            "mu_min": 0.005,
            "mu_rng": 0.35,
            "theta": 2.5,
            "Y_bar_0": 0.0,
            "w_min": 0.3,
            "gamma_pa": 1.5,
            "T_max": 500.0,
            "dt": 0.01,
            "random_seed": 42,
        },
        "network": {
            "hierarchy": [60, 30, 10],
            "seed": 42,
        },
        "monte_carlo": {
            "n_runs": 100,
            "n_cores": 1,
        }
    },
}


# ============================================================================
# MAIN SCENARIO RUNNER
# ============================================================================

class ScenarioRunner:
    """
    Complete scenario execution with all visualizations and reports.
    """

    def __init__(self, scenario_name: str, output_dir: str = None):
        self.scenario_name = scenario_name
        self.scenario = SCENARIOS[scenario_name]

        # Create output directory
        if output_dir is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_dir = f"results_{scenario_name}_{timestamp}"

        self.output_dir = Path(output_dir)
        self.create_directory_structure()

        # Store results
        self.single_run_result = None
        self.mc_results_list = []

    def create_directory_structure(self):
        """Create organized output directory structure."""
        print(f"\n{'='*80}")
        print(f"Creating output directory: {self.output_dir}")
        print(f"{'='*80}\n")

        dirs = [
            self.output_dir,
            self.output_dir / "01_network_static",
            self.output_dir / "02_network_interactive",
            self.output_dir / "03_network_ultra",
            self.output_dir / "04_expert_analysis",
            self.output_dir / "05_stochastic_analysis",
            self.output_dir / "06_monte_carlo",
            self.output_dir / "07_reports",
            self.output_dir / "08_data",
        ]

        for d in dirs:
            d.mkdir(parents=True, exist_ok=True)

        # Save scenario configuration
        with open(self.output_dir / "scenario_config.json", 'w') as f:
            json.dump(self.scenario, f, indent=2)

        print(f"[OK] Directory structure created\n")

    def run_single_simulation(self):
        """Run single simulation for visualization."""
        print(f"\n{'='*80}")
        print("STEP 1: Running Single Simulation")
        print(f"{'='*80}\n")

        # Create parameters
        params = SimulationParameters(**self.scenario['params'])

        # Create network
        network_config = self.scenario['network']
        network = topology.create_hierarchical_network(
            network_config['hierarchy'],
            seed=network_config['seed']
        )

        print(f"  Network: {len(network.V)} actors, {len(network.E)} edges")
        print(f"  Hierarchy: {network_config['hierarchy']}")
        print(f"  T_max: {params.T_max}")
        print()

        # Run simulation
        engine = SimulationEngine(params)
        self.single_run_result = engine.run(network, verbose=True)

        print(f"\n  [OK] Simulation complete")
        print(f"       Total arrests: {len([e for e in self.single_run_result.events if e.event_type.value == 'arrest'])}")
        print(f"       Total conversions: {len([e for e in self.single_run_result.events if e.event_type.value == 'conversion'])}")
        print()

        # Save results
        self.single_run_result.export_events(
            str(self.output_dir / "08_data" / "single_simulation_events.json")
        )
        self.single_run_result.export_time_series(
            str(self.output_dir / "08_data" / "single_simulation_timeseries.csv")
        )

    def run_monte_carlo(self):
        """Run Monte Carlo simulations."""
        print(f"\n{'='*80}")
        print("STEP 2: Running Monte Carlo Simulations")
        print(f"{'='*80}\n")

        n_runs = self.scenario['monte_carlo']['n_runs']
        params_dict = self.scenario['params'].copy()
        network_config = self.scenario['network']

        print(f"  Running {n_runs} Monte Carlo simulations...")
        print()

        for i in range(n_runs):
            # Change random seed for each run
            params_dict['random_seed'] = 42 + i
            params = SimulationParameters(**params_dict)

            # Create network
            network = topology.create_hierarchical_network(
                network_config['hierarchy'],
                seed=network_config['seed']
            )

            # Run simulation
            engine = SimulationEngine(params)
            result = engine.run(network, verbose=False)
            self.mc_results_list.append(result)

            if (i + 1) % 10 == 0:
                print(f"    Completed {i+1}/{n_runs} simulations...")

        print(f"\n  [OK] Monte Carlo complete: {len(self.mc_results_list)} runs\n")

    def generate_all_visualizations(self):
        """Generate all visualization types."""
        print(f"\n{'='*80}")
        print("STEP 3: Generating All Visualizations")
        print(f"{'='*80}\n")

        # 1. Advanced Network Static
        print("  [1/5] Advanced network visualizations (static)...")
        create_advanced_network_visualizations(
            self.single_run_result,
            output_dir=str(self.output_dir / "01_network_static"),
            show=False
        )

        # 2. Interactive Visualizations
        print("\n  [2/5] Interactive visualizations (HTML)...")
        create_interactive_visualizations(
            self.single_run_result,
            output_dir=str(self.output_dir / "02_network_interactive"),
            auto_open=False
        )

        # 3. Ultra-Advanced Visualizations
        print("\n  [3/5] Ultra-advanced visualizations...")
        create_ultra_advanced_visualizations(
            self.single_run_result,
            output_dir=str(self.output_dir / "03_network_ultra"),
            show=False
        )

        # 4. Expert-Level Analysis
        print("\n  [4/5] Expert-level analysis...")
        create_expert_visualizations(
            self.single_run_result,
            output_dir=str(self.output_dir / "04_expert_analysis"),
            show=False
        )

        # 5. Stochastic Analysis
        print("\n  [5/5] Stochastic process analysis...")
        create_stochastic_visualizations(
            self.mc_results_list[:20],  # Use subset for speed
            self.single_run_result,
            output_dir=str(self.output_dir / "05_stochastic_analysis"),
            show=False
        )

        print(f"\n  [OK] All visualizations generated\n")

    def generate_latex_report(self):
        """Generate comprehensive LaTeX report."""
        print(f"\n{'='*80}")
        print("STEP 4: Generating LaTeX Report")
        print(f"{'='*80}\n")

        from organized_crime_network.reporting.latex_generator import LatexReportGenerator

        generator = LatexReportGenerator(
            scenario_name=self.scenario['name'],
            scenario_description=self.scenario['description'],
            scenario_params=self.scenario['params'],
            single_result=self.single_run_result,
            mc_results=self.mc_results_list,
            output_dir=str(self.output_dir)
        )

        report_path = generator.generate_complete_report()

        print(f"\n  [OK] LaTeX report generated: {report_path}\n")

        return report_path

    def run(self):
        """Execute complete scenario."""
        start_time = datetime.now()

        print(f"\n{'#'*80}")
        print(f"# SCENARIO: {self.scenario['name']}")
        print(f"# {self.scenario['description']}")
        print(f"{'#'*80}")

        try:
            # Step 1: Single simulation
            self.run_single_simulation()

            # Step 2: Monte Carlo
            self.run_monte_carlo()

            # Step 3: Generate visualizations
            self.generate_all_visualizations()

            # Step 4: Generate LaTeX report
            latex_path = self.generate_latex_report()

            # Final summary
            elapsed = (datetime.now() - start_time).total_seconds()

            print(f"\n{'='*80}")
            print("SCENARIO EXECUTION COMPLETE")
            print(f"{'='*80}\n")
            print(f"  Elapsed time: {elapsed:.1f} seconds")
            print(f"  Output directory: {self.output_dir}")
            print(f"  LaTeX report: {latex_path}")
            print()
            print("  Generated files:")
            print(f"    - {len(list((self.output_dir / '01_network_static').glob('*.png')))} static network visualizations")
            print(f"    - {len(list((self.output_dir / '02_network_interactive').glob('*.html')))} interactive HTML files")
            print(f"    - {len(list((self.output_dir / '03_network_ultra').glob('*.png')))} ultra-advanced visualizations")
            print(f"    - {len(list((self.output_dir / '04_expert_analysis').glob('*.png')))} expert-level analyses")
            print(f"    - {len(list((self.output_dir / '05_stochastic_analysis').glob('*.png')))} stochastic analyses")
            print(f"    - 1 comprehensive LaTeX report")
            print()
            print(f"  To compile LaTeX report:")
            print(f"    cd {self.output_dir / '07_reports'}")
            print(f"    pdflatex scenario_report.tex")
            print(f"    pdflatex scenario_report.tex  (run twice for references)")
            print()

        except Exception as e:
            print(f"\n[ERROR] Scenario execution failed: {e}")
            import traceback
            traceback.print_exc()
            return False

        return True


# ============================================================================
# COMMAND LINE INTERFACE
# ============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Run complete organized crime network simulation scenario",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available scenarios:
  default      - Moderate enforcement (balanced)
  aggressive   - High arrest rate, rapid collapse
  conservative - Low enforcement, resilient network
  realistic    - Literature-based parameters

Examples:
  python run_scenario.py --scenario aggressive
  python run_scenario.py --scenario realistic --output my_results
        """
    )

    parser.add_argument(
        '--scenario',
        type=str,
        default='default',
        choices=list(SCENARIOS.keys()),
        help='Scenario to run (default: default)'
    )

    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory (default: results_SCENARIO_TIMESTAMP)'
    )

    parser.add_argument(
        '--list',
        action='store_true',
        help='List available scenarios and exit'
    )

    args = parser.parse_args()

    # List scenarios
    if args.list:
        print("\nAvailable Scenarios:")
        print("=" * 80)
        for name, config in SCENARIOS.items():
            print(f"\n  {name}")
            print(f"    {config['description']}")
            print(f"    Network: {config['network']['hierarchy']} actors")
            print(f"    Monte Carlo: {config['monte_carlo']['n_runs']} runs")
        print()
        return

    # Run scenario
    runner = ScenarioRunner(args.scenario, args.output)
    success = runner.run()

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
