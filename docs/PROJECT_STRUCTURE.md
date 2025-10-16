# Project Structure

This document provides a complete overview of the repository structure.

## 📁 Root Directory

```
organized-crime-network/
├── organized_crime_network/     # Main package
├── docs/                         # Documentation
├── examples/                     # Example scripts
├── tests/                        # Test suite
├── .specify/                     # Project specification
├── run_scenario.py              # Main scenario runner
├── requirements.txt             # Dependencies
├── README.md                    # Main documentation
├── LICENSE                      # MIT License
├── CONTRIBUTING.md              # Contribution guidelines
└── .gitignore                   # Git ignore rules
```

## 📦 Main Package Structure

```
organized_crime_network/
├── simulation/
│   ├── __init__.py
│   ├── engine.py                       # SimulationEngine - main orchestrator
│   ├── parameters.py                   # SimulationParameters with validation
│   ├── events.py                       # SimulationEvent dataclasses
│   ├── results.py                      # SimulationResults containers
│   ├── state.py                        # NetworkState management
│   ├── topology.py                     # Network topology generators
│   ├── arrest_process.py               # Cox process implementation
│   ├── trust_dynamics.py               # SDE/PDMP for trust evolution
│   ├── conversion_process.py           # CTMC for informant conversion
│   │
│   # Visualization modules
│   ├── visualization_network_advanced.py   # 5 advanced network visualizations
│   ├── visualization_interactive.py        # 3 interactive HTML visualizations
│   ├── visualization_ultra_advanced.py     # 3 ultra-advanced visualizations
│   ├── visualization_expert.py             # 4 expert-level analyses
│   └── visualization_stochastic.py         # 4 stochastic process analyses
│
└── reporting/
    ├── __init__.py
    └── latex_generator.py              # LaTeX report generation
```

## 📚 Documentation Directory

```
docs/
├── HIZLI_BASLANGIC.md              # Quick start guide (Turkish)
├── SCENARIO_GUIDE.md               # Scenario and parameter guide (Turkish)
├── README_SCENARIO_SYSTEM.md       # Complete system documentation (Turkish)
├── VISUALIZATION_GUIDE.md          # Visualization documentation
├── STOCHASTIC_PROCESSES_BEST_PRACTICES.md  # Implementation details
├── MATHEMATICAL_VERIFICATION_REPORT.md     # Verification report
├── TEST_SONUCLARI.md               # Test results (Turkish)
└── PROJECT_STRUCTURE.md            # This file
```

## 🔬 Examples Directory

```
examples/
├── main.py                         # Basic simulation example
├── demo_comprehensive.py           # Comprehensive demo with all features
├── test_multi_round.py            # Multi-round simulation example
└── test_strategy_comparison.py    # Strategy comparison example
```

## 🧪 Tests Directory

```
tests/
├── __init__.py
├── test_simulation.py             # Integration tests
├── test_parameters.py             # Parameter validation tests
├── test_processes.py              # Stochastic process tests
└── test_visualizations.py         # Visualization tests
```

## ⚙️ Core Components

### 1. SimulationEngine (`simulation/engine.py`)

Main orchestrator that:
- Initializes simulation state
- Runs time evolution loop
- Processes stochastic events
- Generates results

**Key methods:**
- `run(network, verbose=False)` - Execute simulation
- `step(dt)` - Single time step
- `process_event(event)` - Handle stochastic event

### 2. SimulationParameters (`simulation/parameters.py`)

Validated parameter container with:
- Constitutional domain validation
- Numerical stability checks
- Pre-configured strategy presets

**Key methods:**
- `default()` - Default parameters
- `aggressive_enforcement()` - High arrest strategy
- `conservative_enforcement()` - Low intervention strategy
- `validate_parameters()` - Domain validation

### 3. NetworkState (`simulation/state.py`)

Network state management:
- Actor states (Active, Arrested, Informant)
- Trust matrix Y_ij (logit-space)
- Risk values R_i
- Hierarchy levels

**State invariants:**
- Edges only between Active actors
- Trust defined only for existing edges
- Valid state transitions

### 4. Stochastic Processes

**Cox Process** (`simulation/arrest_process.py`):
- Non-homogeneous Poisson process
- State-dependent intensity λ_i(t)
- Hierarchical protection

**Trust SDE** (`simulation/trust_dynamics.py`):
- Ornstein-Uhlenbeck process
- Mean reversion with risk shifts
- Euler-Maruyama integration

**CTMC Conversion** (`simulation/conversion_process.py`):
- Two-stage Markov chain
- Trust-dependent rates
- Exponential waiting times

### 5. Visualization Modules

**Network Visualizations:**
- Circular hierarchy layout
- Force-directed multi-view
- 3D interactive network
- Temporal evolution grid
- Adjacency heatmap

**Expert Analysis:**
- Sankey state transitions
- Network motif analysis
- Community detection
- Resilience testing

**Stochastic Analysis:**
- Monte Carlo paths with confidence bands
- Cox process intensity
- Trust SDE phase space
- CTMC transition rates

### 6. LaTeX Report Generator (`reporting/latex_generator.py`)

Automated LaTeX report creation:
- Parameter tables
- Summary statistics
- Hierarchy analysis
- Monte Carlo aggregates
- Embedded figures

## 🚀 Entry Points

### Main Scenario Runner (`run_scenario.py`)

One-command execution of complete scenarios:

```python
python run_scenario.py --scenario aggressive
```

**Features:**
- 4 pre-configured scenarios
- Single simulation + Monte Carlo
- All visualizations (19+)
- LaTeX report generation
- Organized output structure

### Basic Simulation (`examples/main.py`)

Simple example showing core usage:

```python
python main.py
```

**Demonstrates:**
- Parameter creation
- Network initialization
- Simulation execution
- Results analysis

### Comprehensive Demo (`examples/demo_comprehensive.py`)

Full-featured demonstration:

```python
python demo_comprehensive.py
```

**Includes:**
- Multiple parameter presets
- Different network topologies
- Progress monitoring
- Strategy comparisons

## 📊 Output Structure

When running scenarios, outputs are organized as:

```
results_SCENARIO_TIMESTAMP/
├── scenario_config.json           # Configuration used
├── 01_network_static/             # 5 PNG visualizations
│   ├── network_circular_hierarchy.png
│   ├── network_force_directed_analysis.png
│   ├── network_adjacency_heatmap.png
│   ├── network_3d_visualization.png
│   └── network_temporal_evolution_grid.png
│
├── 02_network_interactive/        # 3 HTML visualizations
│   ├── interactive_3d_network.html
│   ├── animated_network_evolution.html
│   └── interactive_dashboard.html
│
├── 03_network_ultra/              # 3 PNG ultra-advanced
│   ├── radial_dendrogram.png
│   ├── network_metrics_dashboard.png
│   └── chord_diagram.png
│
├── 04_expert_analysis/            # 4 PNG expert analyses
│   ├── sankey_state_transitions.png
│   ├── network_motif_analysis.png
│   ├── community_evolution.png
│   └── network_resilience_analysis.png
│
├── 05_stochastic_analysis/        # 4 PNG stochastic analyses
│   ├── stochastic_path_analysis.png
│   ├── cox_process_intensity.png
│   ├── trust_sde_dynamics.png
│   └── ctmc_transition_analysis.png
│
├── 06_monte_carlo/                # Reserved for future use
│
├── 07_reports/                    # LaTeX reports
│   └── scenario_report.tex
│
└── 08_data/                       # Raw data exports
    ├── single_simulation_events.json
    └── single_simulation_timeseries.csv
```

## 🔧 Configuration Files

### `.gitignore`

Excludes:
- Python bytecode (`__pycache__/`, `*.pyc`)
- Virtual environments (`venv/`, `env/`)
- Test outputs (`results_*/`, `test_*/`)
- Temporary files (`*.tmp`, `*.log`)
- LaTeX build artifacts

### `requirements.txt`

Core dependencies:
- `numpy>=1.20` - Numerical computing
- `scipy>=1.7` - Scientific computing
- `matplotlib>=3.5` - Static plotting
- `seaborn>=0.11` - Statistical visualization
- `networkx>=2.6` - Network analysis
- `plotly>=5.0` - Interactive plots
- `pandas>=1.3` - Data manipulation

### `.specify/`

Project specification system:
- `memory/constitution.md` - Project principles
- `specs/*/` - Feature specifications
- Auto-generated documentation

## 📈 Development Workflow

1. **Make changes** in appropriate module
2. **Run tests**: `pytest`
3. **Format code**: `black .`
4. **Check linting**: `flake8`
5. **Run example**: `python run_scenario.py --scenario default`
6. **Verify outputs**: Check `results_*/`
7. **Update docs** if needed
8. **Commit and push**

## 🔍 Finding Code

### Need to modify...

- **Arrest rates**: `simulation/arrest_process.py`
- **Trust dynamics**: `simulation/trust_dynamics.py`
- **Conversion logic**: `simulation/conversion_process.py`
- **Network topology**: `simulation/topology.py`
- **Parameters**: `simulation/parameters.py`
- **Visualizations**: `simulation/visualization_*.py`
- **LaTeX reports**: `reporting/latex_generator.py`
- **Scenario configs**: `run_scenario.py` (SCENARIOS dict)

### Need documentation on...

- **Quick start**: `docs/HIZLI_BASLANGIC.md`
- **Parameters**: `docs/SCENARIO_GUIDE.md`
- **System overview**: `docs/README_SCENARIO_SYSTEM.md`
- **Visualizations**: `docs/VISUALIZATION_GUIDE.md`
- **Mathematics**: `gelis.tex`
- **Best practices**: `docs/STOCHASTIC_PROCESSES_BEST_PRACTICES.md`

## 🎯 Key Design Principles

1. **Mathematical Fidelity**: Implementation matches `gelis.tex` exactly
2. **Constitutional Validation**: All parameters validated against domains
3. **State Invariants**: Network state consistency maintained
4. **Numerical Stability**: Proper handling of edge cases
5. **Reproducibility**: Random seed control for deterministic results
6. **Modularity**: Clear separation of concerns
7. **Documentation**: Comprehensive docstrings and guides

## 📞 Support

For questions about structure:
- Check this document first
- Review module docstrings
- See `CONTRIBUTING.md` for guidelines
- Open an issue for clarification

---

**Last Updated**: 2025-10-16
**Project Version**: 1.0.0
