# Organized Crime Network Simulation 🚨

A comprehensive stochastic simulation framework for modeling law enforcement strategies against hierarchical criminal networks, featuring **19+ advanced visualizations**, Monte Carlo analysis, and automated LaTeX reporting.

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## ✨ Highlights

- 🎯 **One-Command Execution**: Generate complete analysis with a single command
- 📊 **19+ Visualizations**: Static, interactive HTML, and expert-level analyses
- 🔬 **Rigorous Mathematics**: Cox processes, SDEs, and CTMCs
- 📈 **Monte Carlo Support**: Parallel simulations with confidence intervals
- 📄 **Automated Reports**: Professional LaTeX documents with tables and figures
- ⚙️ **Easy Calibration**: Simple parameter tuning in one file

## 🚀 Quick Start

### Installation

```bash
git clone https://github.com/yourusername/organized-crime-network.git
cd organized-crime-network
pip install -r requirements.txt
```

### Run Your First Scenario

```bash
# Run aggressive enforcement scenario (~1 minute)
python run_scenario.py --scenario aggressive
```

This single command generates:
- ✅ 19+ visualizations (PNG + HTML)
- ✅ Monte Carlo analysis (50 runs)
- ✅ LaTeX report with tables
- ✅ JSON and CSV data exports

### View Results

```
results_aggressive_TIMESTAMP/
├── 01_network_static/         # 5 PNG visualizations
├── 02_network_interactive/    # 3 interactive HTML files
├── 03_network_ultra/          # 3 advanced PNG visualizations
├── 04_expert_analysis/        # 4 expert-level analyses
├── 05_stochastic_analysis/    # 4 stochastic process analyses
├── 07_reports/                # LaTeX report
│   └── scenario_report.tex
└── 08_data/                   # Raw data (JSON + CSV)
```

**Open HTML files in your browser for interactive 3D visualizations!**

## 📊 Example Visualizations

<table>
<tr>
<td width="50%">

### Network Analysis
- Circular hierarchy layout
- Force-directed multi-view
- 3D interactive network
- Temporal evolution (9 snapshots)
- Adjacency heatmap with clustering

</td>
<td width="50%">

### Stochastic Analysis
- Path analysis with confidence bands
- Cox process intensity
- Trust SDE dynamics
- CTMC transition analysis

</td>
</tr>
</table>

### Expert-Level Analysis
- **Sankey diagrams**: State transition flows
- **Motif analysis**: Triangles, feed-forward loops, reciprocity
- **Community detection**: Louvain algorithm over time
- **Resilience analysis**: Targeted attack vs. random failure

## 🎯 Available Scenarios

```bash
python run_scenario.py --list
```

| Scenario | Description | Parameters | Time |
|----------|-------------|------------|------|
| `aggressive` | High arrest rate, rapid collapse | λ₀=0.3, κ=0.3 | ~1 min |
| `default` | Moderate enforcement | λ₀=0.1, κ=0.5 | ~2 min |
| `conservative` | Low arrest rate, resilient network | λ₀=0.05, κ=0.7 | ~3 min |
| `realistic` | Literature-based parameters | λ₀=0.05, κ=0.8 | ~15 min |

## ⚙️ Customize Parameters

Edit `run_scenario.py` to create custom scenarios:

```python
SCENARIOS = {
    "my_custom": {
        "name": "My Custom Scenario",
        "params": {
            # Arrest Process (Cox)
            "lambda_0": 0.15,     # Baseline arrest rate
            "kappa": 0.6,         # Hierarchical protection

            # Trust Dynamics (SDE)
            "alpha": 0.4,         # Mean reversion rate
            "sigma": 0.12,        # Volatility
            "Delta": 15.0,        # Memory window

            # Conversion (CTMC)
            "mu_LH": 0.25,        # Conversion rate
            "theta": 2.2,         # Trust sensitivity

            # Simulation
            "T_max": 150.0,       # Time horizon
        }
    }
}
```

### Parameter Effects

| Parameter | Effect | Low → High |
|-----------|--------|------------|
| `lambda_0` | Arrest rate | Slow collapse → Fast collapse |
| `kappa` | Protection | Vulnerable → Resilient |
| `alpha` | Trust adaptation | Stable → Fragile |
| `sigma` | Volatility | Predictable → Noisy |
| `mu_LH` | Conversion rate | Few informants → Many informants |

## 🏗️ Architecture

```
organized_crime_network/
├── simulation/
│   ├── engine.py                        # Main simulation engine
│   ├── parameters.py                    # Parameter validation
│   ├── visualization_network_advanced.py
│   ├── visualization_interactive.py     # Interactive HTML
│   ├── visualization_ultra_advanced.py
│   ├── visualization_expert.py          # Expert analysis
│   └── visualization_stochastic.py      # Stochastic analysis
└── reporting/
    └── latex_generator.py               # LaTeX report generation

run_scenario.py                          # One-command scenario runner
```

## 🔬 Mathematical Foundation

### Four Coupled Stochastic Processes

1. **Cox Process** (Arrest Dynamics)
   ```
   λᵢ(t) = λ₀ × exp(-κ × level) × P(t)
   ```

2. **Ornstein-Uhlenbeck SDE** (Trust Dynamics)
   ```
   dYᵢⱼ = α(μ - Yᵢⱼ)dt + βRᵢ(t)dt + σdWᵢⱼ
   ```

3. **CTMC** (Informant Conversion)
   ```
   μᵢ(t) = μ_min + μ_rng × exp(-θ × wᵢⱼ)
   ```

4. **Hierarchical Network**
   - 3 levels: Leaders, mid-level, operatives
   - Trust-based connections
   - Dynamic topology

## 📈 Example Results

### Aggressive Enforcement (λ₀=0.3, κ=0.3)
- **Collapse time**: t ≈ 5-10
- **Arrest rate**: ~100%
- **Conversion rate**: ~25-30%
- **Effectiveness**: P(t) rises to 4.0+

### Conservative Enforcement (λ₀=0.05, κ=0.7)
- **Collapse time**: t ≈ 150-200+
- **Arrest rate**: ~40-60%
- **Conversion rate**: ~10-15%
- **Effectiveness**: P(t) stays near 1.0

## 📄 LaTeX Reports

Each scenario generates a comprehensive LaTeX report:

```latex
\documentclass{article}
\begin{document}

\section{Scenario Results}
- Parameter tables
- Summary statistics
- Hierarchy analysis
- Monte Carlo aggregates

\section{Visualizations}
- All figures embedded
- Publication-ready format

\end{document}
```

**Compile with:**
```bash
cd results_*/07_reports/
pdflatex scenario_report.tex
pdflatex scenario_report.tex  # Run twice
```

## 🧪 Testing

```bash
# Run basic simulation
python main.py

# Run comprehensive demo
python demo_comprehensive.py

# Run all tests
pytest tests/
```

## 📚 Documentation

- **[Quick Start](HIZLI_BASLANGIC.md)**: Fast introduction (Turkish)
- **[Scenario Guide](SCENARIO_GUIDE.md)**: Parameter calibration (Turkish)
- **[System Documentation](README_SCENARIO_SYSTEM.md)**: Complete system (Turkish)
- **[Mathematical Model](gelis.tex)**: Full mathematical framework
- **[Stochastic Best Practices](STOCHASTIC_PROCESSES_BEST_PRACTICES.md)**: Implementation details

## 🛠️ Requirements

- Python 3.9+
- NumPy >= 1.20
- SciPy >= 1.7
- Matplotlib >= 3.5
- Seaborn >= 0.11
- NetworkX >= 2.6
- Plotly >= 5.0
- pandas >= 1.3

Install all:
```bash
pip install -r requirements.txt
```

## 🤝 Contributing

Contributions welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Add tests for new features
4. Submit a pull request

See [CONTRIBUTING.md](CONTRIBUTING.md) for details.

## 📄 License

This project is licensed under the MIT License - see [LICENSE](LICENSE) for details.

## 🌟 Key Features

### 1. Network Visualizations (11 types)
- **Static**: Circular, force-directed, heatmap, 3D, temporal
- **Interactive**: Rotatable 3D, animations, dashboards
- **Ultra**: Radial dendrogram, metrics, chord diagram

### 2. Expert Analysis (4 types)
- Sankey state transitions
- Network motif analysis
- Community evolution
- Resilience testing

### 3. Stochastic Analysis (4 types)
- Monte Carlo paths with confidence bands
- Cox process intensity
- Trust SDE phase space
- CTMC transition rates

### 4. Automated Workflow
- Single-command execution
- Organized output structure
- LaTeX report generation
- Data export (JSON + CSV)

## 📊 Performance

| Scenario | Network Size | MC Runs | Total Time |
|----------|--------------|---------|------------|
| aggressive | 100 actors | 50 | ~1 minute |
| default | 100 actors | 50 | ~2 minutes |
| conservative | 100 actors | 50 | ~3 minutes |
| realistic | 100 actors | 100 | ~15 minutes |

## 🎓 Research Context

This simulation implements state-of-the-art stochastic modeling for:
- Law enforcement strategy optimization
- Criminal network resilience analysis
- Informant recruitment dynamics
- Trust-based network evolution

### Mathematical Rigor
- ✅ Exact formula implementation
- ✅ Parameter domain validation
- ✅ Numerical stability guarantees
- ✅ State invariant preservation

## 📧 Contact

- **Issues**: [GitHub Issues](https://github.com/yourusername/organized-crime-network/issues)
- **Email**: your.email@domain.com

## 🙏 Acknowledgments

Developed as part of research on law enforcement strategies against organized crime networks.

---

**Built with Python** | **Powered by NumPy, SciPy, NetworkX** | **Visualizations by Matplotlib & Plotly**

⭐ **Star this repo if you find it useful!**
