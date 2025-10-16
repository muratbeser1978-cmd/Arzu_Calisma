"""
LaTeX Report Generator

Generates comprehensive LaTeX reports with:
- Scenario description and parameters
- Summary statistics tables
- Detailed results tables
- References to all generated visualizations
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Any, Optional
from datetime import datetime

from ..simulation.results import SimulationResults


class LatexReportGenerator:
    """
    Generate comprehensive LaTeX report for simulation scenario.
    """

    def __init__(
        self,
        scenario_name: str,
        scenario_description: str,
        scenario_params: Dict[str, Any],
        single_result: SimulationResults,
        mc_results: List[SimulationResults],
        output_dir: str
    ):
        self.scenario_name = scenario_name
        self.scenario_description = scenario_description
        self.scenario_params = scenario_params
        self.single_result = single_result
        self.mc_results = mc_results
        self.output_dir = Path(output_dir)

    def _escape_latex(self, text: str) -> str:
        """Escape special LaTeX characters."""
        replacements = {
            '&': r'\&',
            '%': r'\%',
            '$': r'\$',
            '#': r'\#',
            '_': r'\_',
            '{': r'\{',
            '}': r'\}',
            '~': r'\textasciitilde{}',
            '^': r'\^{}',
            '\\': r'\textbackslash{}',
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def generate_parameter_table(self) -> str:
        """Generate LaTeX table for scenario parameters."""
        latex = r"""
\begin{table}[H]
\centering
\caption{Scenario Parameters}
\label{tab:parameters}
\begin{tabular}{llll}
\toprule
\textbf{Parameter} & \textbf{Symbol} & \textbf{Value} & \textbf{Description} \\
\midrule
"""

        param_descriptions = {
            'lambda_0': (r'$\lambda_0$', 'Baseline arrest rate'),
            'kappa': (r'$\kappa$', 'Hierarchical protection factor'),
            'alpha': (r'$\alpha$', 'Trust mean reversion rate'),
            'sigma': (r'$\sigma$', 'Trust volatility'),
            'Delta': (r'$\Delta$', 'Trust memory time'),
            'P_0': (r'$P_0$', 'Initial enforcement effectiveness'),
            'eta_P': (r'$\eta_P$', 'Effectiveness jump per conversion'),
            'mu_LH': (r'$\mu_{LH}$', 'High-trust conversion rate'),
            'mu_min': (r'$\mu_{min}$', 'Minimum conversion rate'),
            'mu_rng': (r'$\mu_{rng}$', 'Conversion rate range'),
            'theta': (r'$\theta$', 'Trust sensitivity parameter'),
            'T_max': (r'$T_{max}$', 'Simulation time horizon'),
            'dt': (r'$dt$', 'Time discretization step'),
        }

        for param_name, (symbol, description) in param_descriptions.items():
            if param_name in self.scenario_params:
                value = self.scenario_params[param_name]
                if isinstance(value, float):
                    value_str = f"{value:.3f}"
                else:
                    value_str = str(value)

                latex += f"{param_name} & {symbol} & {value_str} & {description} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def generate_single_run_summary_table(self) -> str:
        """Generate summary table for single simulation run."""
        result = self.single_result

        # Calculate statistics
        n_arrests = len([e for e in result.events if e.event_type.value == 'arrest'])
        n_conversions = len([e for e in result.events if e.event_type.value == 'conversion'])
        final_active = result.time_series.network_size[-1]
        final_arrested = result.time_series.arrested_count[-1]
        final_informants = result.time_series.informant_count[-1]
        final_trust = result.time_series.mean_trust[-1]
        final_effectiveness = result.final_effectiveness

        latex = r"""
\begin{table}[H]
\centering
\caption{Single Simulation Run - Summary Statistics}
\label{tab:single_summary}
\begin{tabular}{lr}
\toprule
\textbf{Metric} & \textbf{Value} \\
\midrule
"""

        metrics = [
            ("Total Arrests", n_arrests),
            ("Total Conversions", n_conversions),
            ("Final Active Actors", final_active),
            ("Final Arrested Actors", final_arrested),
            ("Final Informants", final_informants),
            (r"Final Mean Trust $\bar{Y}(T)$", f"{final_trust:.3f}"),
            (r"Final Effectiveness $P(T)$", f"{final_effectiveness:.3f}"),
            (r"Simulation Time $T_{max}$", f"{result.time_series.times[-1]:.1f}"),
        ]

        for metric, value in metrics:
            latex += f"{metric} & {value} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def generate_monte_carlo_summary_table(self) -> str:
        """Generate summary table for Monte Carlo results."""
        if not self.mc_results:
            return ""

        # Aggregate statistics
        final_active = [r.time_series.network_size[-1] for r in self.mc_results]
        final_arrested = [r.time_series.arrested_count[-1] for r in self.mc_results]
        final_informants = [r.time_series.informant_count[-1] for r in self.mc_results]
        total_arrests = [len([e for e in r.events if e.event_type.value == 'arrest'])
                        for r in self.mc_results]
        total_conversions = [len([e for e in r.events if e.event_type.value == 'conversion'])
                            for r in self.mc_results]

        # Collapse analysis
        collapse_threshold = self.mc_results[0].time_series.network_size[0] * 0.1
        collapse_times = []
        for r in self.mc_results:
            collapse_idx = np.where(np.array(r.time_series.network_size) < collapse_threshold)[0]
            if len(collapse_idx) > 0:
                collapse_times.append(r.time_series.times[collapse_idx[0]])

        collapse_rate = len(collapse_times) / len(self.mc_results)

        latex = r"""
\begin{table}[H]
\centering
\caption{Monte Carlo Simulation Results (""" + str(len(self.mc_results)) + r""" runs)}
\label{tab:monte_carlo}
\begin{tabular}{lrrr}
\toprule
\textbf{Metric} & \textbf{Mean} & \textbf{Std Dev} & \textbf{Median} \\
\midrule
"""

        metrics = [
            ("Final Active Actors", final_active),
            ("Final Arrested Actors", final_arrested),
            ("Final Informants", final_informants),
            ("Total Arrests", total_arrests),
            ("Total Conversions", total_conversions),
        ]

        for name, data in metrics:
            mean_val = np.mean(data)
            std_val = np.std(data)
            median_val = np.median(data)
            latex += f"{name} & {mean_val:.2f} & {std_val:.2f} & {median_val:.0f} \\\\\n"

        latex += r"""\midrule
"""

        # Collapse statistics
        if collapse_times:
            latex += f"Mean Collapse Time & {np.mean(collapse_times):.2f} & {np.std(collapse_times):.2f} & {np.median(collapse_times):.2f} \\\\\n"

        latex += f"Network Collapse Rate & {collapse_rate:.2%} & -- & -- \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def generate_hierarchy_analysis_table(self) -> str:
        """Generate table analyzing arrests/conversions by hierarchy level."""
        result = self.single_result

        # Count by hierarchy
        hierarchy_arrests = {1: 0, 2: 0, 3: 0}
        hierarchy_conversions = {1: 0, 2: 0, 3: 0}

        for event in result.events:
            h = event.details.get('hierarchy_level', 1)
            if event.event_type.value == 'arrest':
                hierarchy_arrests[h] += 1
            elif event.event_type.value == 'conversion':
                hierarchy_conversions[h] += 1

        latex = r"""
\begin{table}[H]
\centering
\caption{Arrests and Conversions by Hierarchy Level}
\label{tab:hierarchy}
\begin{tabular}{lrrr}
\toprule
\textbf{Hierarchy Level} & \textbf{Arrests} & \textbf{Conversions} & \textbf{Conversion Rate} \\
\midrule
"""

        for level in [1, 2, 3]:
            arrests = hierarchy_arrests[level]
            conversions = hierarchy_conversions[level]
            rate = conversions / arrests if arrests > 0 else 0

            level_name = {1: "L1 (Leaders)", 2: "L2 (Mid-level)", 3: "L3 (Operatives)"}[level]

            latex += f"{level_name} & {arrests} & {conversions} & {rate:.1%} \\\\\n"

        latex += r"""\midrule
"""

        total_arrests = sum(hierarchy_arrests.values())
        total_conversions = sum(hierarchy_conversions.values())
        total_rate = total_conversions / total_arrests if total_arrests > 0 else 0

        latex += f"\\textbf{{Total}} & \\textbf{{{total_arrests}}} & \\textbf{{{total_conversions}}} & \\textbf{{{total_rate:.1%}}} \\\\\n"

        latex += r"""\bottomrule
\end{tabular}
\end{table}
"""
        return latex

    def generate_complete_report(self) -> str:
        """Generate complete LaTeX report."""
        report_dir = self.output_dir / "07_reports"
        report_dir.mkdir(parents=True, exist_ok=True)

        report_path = report_dir / "scenario_report.tex"

        # Generate LaTeX content
        latex_content = r"""\documentclass[12pt,a4paper]{article}

\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{amsmath,amssymb}
\usepackage{graphicx}
\usepackage{booktabs}
\usepackage{float}
\usepackage[margin=2.5cm]{geometry}
\usepackage{hyperref}
\usepackage{subcaption}

\title{Organized Crime Network Simulation Report \\
\Large """ + self._escape_latex(self.scenario_name) + r"""}

\author{Stochastic Network Simulation Framework}
\date{""" + datetime.now().strftime("%Y-%m-%d %H:%M") + r"""}

\begin{document}

\maketitle

\begin{abstract}
This report presents the comprehensive results of the organized crime network simulation under the """ + self._escape_latex(self.scenario_name) + r""" scenario.
The simulation employs a stochastic mathematical framework incorporating Cox processes for arrests,
Ornstein-Uhlenbeck SDEs for trust dynamics, and continuous-time Markov chains for informant conversions.
Results include """ + str(len(self.mc_results)) + r""" Monte Carlo simulation runs and extensive network analysis.
\end{abstract}

\section{Scenario Description}

\subsection{Overview}
""" + self._escape_latex(self.scenario_description) + r"""

\subsection{Model Components}

The simulation framework consists of four main stochastic components:

\begin{enumerate}
    \item \textbf{Cox Process (Arrests):} Non-homogeneous Poisson process with intensity
    \[
    \lambda_i(t) = \lambda_0 \cdot \exp(-\kappa \cdot \ell_i) \cdot P(t)
    \]
    where $\ell_i$ is the hierarchy level and $P(t)$ is enforcement effectiveness.

    \item \textbf{Ornstein-Uhlenbeck SDE (Trust Dynamics):}
    \[
    dY_{ij}(t) = -\alpha (Y_{ij}(t) - \bar{Y}) dt + \sigma dW_{ij}(t)
    \]
    with memory parameter $\Delta$ influencing risk perception.

    \item \textbf{CTMC (Informant Conversion):} State transition process with trust-dependent rate
    \[
    \mu_i(t) = \mu_{min} + \mu_{rng} \cdot \sigma(\theta \cdot \bar{Y}_i(t_{arrest}))
    \]

    \item \textbf{Network Topology:} Hierarchical structure with preferential attachment growth.
\end{enumerate}

\section{Simulation Parameters}

""" + self.generate_parameter_table() + r"""

\section{Single Simulation Results}

\subsection{Summary Statistics}

""" + self.generate_single_run_summary_table() + r"""

\subsection{Hierarchy Analysis}

""" + self.generate_hierarchy_analysis_table() + r"""

\section{Monte Carlo Analysis}

\subsection{Aggregate Statistics}

""" + self.generate_monte_carlo_summary_table() + r"""

\subsection{Interpretation}

The Monte Carlo results provide insight into the stochastic variability of network collapse dynamics.
Key findings include:

\begin{itemize}
    \item Network collapse rate: """ + f"{len([r for r in self.mc_results if r.time_series.network_size[-1] < 5]) / len(self.mc_results):.1%}" + r"""
    \item Mean arrests: """ + f"{np.mean([len([e for e in r.events if e.event_type.value == 'arrest']) for r in self.mc_results]):.1f}" + r"""
    \item Conversion efficiency: """ + f"{np.mean([len([e for e in r.events if e.event_type.value == 'conversion']) for r in self.mc_results]) / (np.mean([len([e for e in r.events if e.event_type.value == 'arrest']) for r in self.mc_results]) + 1e-10):.1%}" + r"""
\end{itemize}

\section{Visualizations}

\subsection{Network Structure}

\begin{figure}[H]
\centering
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{../01_network_static/network_circular_hierarchy.png}
    \caption{Circular hierarchy layout}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{../01_network_static/network_3d_visualization.png}
    \caption{3D network visualization}
\end{subfigure}
\caption{Network topology visualizations showing hierarchical structure.}
\label{fig:network_topology}
\end{figure}

\subsection{Stochastic Process Analysis}

\begin{figure}[H]
\centering
\includegraphics[width=0.9\textwidth]{../05_stochastic_analysis/stochastic_path_analysis.png}
\caption{Stochastic path analysis with confidence bands showing trajectory variability across Monte Carlo runs.}
\label{fig:stochastic_paths}
\end{figure}

\begin{figure}[H]
\centering
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{../05_stochastic_analysis/cox_process_intensity.png}
    \caption{Cox process intensity $\lambda(t)$}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{../05_stochastic_analysis/trust_sde_dynamics.png}
    \caption{Trust SDE dynamics}
\end{subfigure}
\caption{Stochastic process components: arrest rate evolution and trust dynamics.}
\label{fig:stochastic_components}
\end{figure}

\subsection{Expert Analysis}

\begin{figure}[H]
\centering
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{../04_expert_analysis/sankey_state_transitions.png}
    \caption{State transitions (Sankey)}
\end{subfigure}
\hfill
\begin{subfigure}{0.48\textwidth}
    \includegraphics[width=\textwidth]{../04_expert_analysis/network_resilience_analysis.png}
    \caption{Network resilience}
\end{subfigure}
\caption{Expert-level analysis: state transitions and network resilience under targeted attacks.}
\label{fig:expert_analysis}
\end{figure}

\section{Conclusions}

This comprehensive simulation analysis provides quantitative insights into organized crime network dynamics
under the """ + self._escape_latex(self.scenario_name) + r""" scenario. The stochastic framework captures
the inherent uncertainty in enforcement outcomes while revealing systematic patterns in network collapse.

Key findings:
\begin{itemize}
    \item Network exhibits """ + ("high" if np.mean([r.time_series.network_size[-1] for r in self.mc_results]) < 5 else "moderate") + r""" fragility under current enforcement strategy
    \item Hierarchical structure provides """ + ("limited" if self.scenario_params.get('kappa', 0.5) < 0.5 else "significant") + r""" protection against disruption
    \item Informant conversion is """ + ("highly effective" if np.mean([len([e for e in r.events if e.event_type.value == 'conversion']) for r in self.mc_results]) > 20 else "moderately effective") + r""" as a disruption mechanism
\end{itemize}

\section*{References}

\begin{itemize}
    \item Mathematical framework: \texttt{gelis.tex}
    \item Simulation code: Stochastic Network Simulation Framework v1.0
    \item Generated: """ + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + r"""
\end{itemize}

\end{document}
"""

        # Write to file
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(latex_content)

        print(f"  [OK] LaTeX report written to: {report_path}")
        print(f"       Compile with: pdflatex scenario_report.tex")

        return str(report_path)
