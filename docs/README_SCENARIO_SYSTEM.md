# Komple Senaryo Çalıştırma Sistemi

## 🎯 Özellikler

✅ **Tek Komutla Tam Analiz**:
- Network görselleştirmeleri (11+ çeşit)
- Expert-level analizler (Sankey, motif, community, resilience)
- Stokastik süreç analizleri (Cox, SDE, CTMC)
- Monte Carlo simülasyonları
- **Profesyonel LaTeX Raporu** (tablolar, grafikler, istatistikler)

✅ **Kolay Kalibrasyon**:
- Tek dosyada tüm parametreler
- 4 hazır senaryo (default, aggressive, conservative, realistic)
- Yeni senaryo eklemek çok kolay

✅ **Organize Çıktı Yapısı**:
- 8 ayrı klasör (network, stochastic, expert, vs.)
- JSON + CSV veri çıktıları
- İnteraktif HTML görselleştirmeler

---

## 🚀 Hızlı Başlangıç

### Adım 1: Mevcut Senaryoları Görüntüle
```bash
python run_scenario.py --list
```

### Adım 2: Bir Senaryo Çalıştır
```bash
# Default (moderate enforcement)
python run_scenario.py --scenario default

# Aggressive enforcement
python run_scenario.py --scenario aggressive

# Conservative enforcement
python run_scenario.py --scenario conservative

# Realistic parameters
python run_scenario.py --scenario realistic
```

### Adım 3: Sonuçları İncele
```
results_SCENARIO_TIMESTAMP/
├── 01_network_static/       → PNG görsellerini aç
├── 02_network_interactive/  → HTML dosyalarını tarayıcıda aç
├── 05_stochastic_analysis/  → Stokastik analizleri incele
├── 07_reports/              → LaTeX raporunu derle
└── 08_data/                 → JSON/CSV verilere bak
```

---

## 📊 Çıktılar

### 1. Network Görselleştirmeleri (15+ çeşit)

**01_network_static/** (5 dosya):
- `network_circular_hierarchy.png` - Circular layout (hierarchy-based)
- `network_force_directed_analysis.png` - 4-panel force-directed
- `network_adjacency_heatmap.png` - Trust adjacency matrix + clustering
- `network_3d_visualization.png` - 3D network (Z = hierarchy)
- `network_temporal_evolution_grid.png` - 9 snapshot temporal evolution

**02_network_interactive/** (3 HTML):
- `interactive_3d_network.html` - Rotatable 3D (Plotly)
- `animated_network_evolution.html` - Play/pause animation
- `interactive_dashboard.html` - Comprehensive 2x2 dashboard

**03_network_ultra/** (3 dosya):
- `radial_dendrogram.png` - Radial hierarchical clustering
- `network_metrics_dashboard.png` - 10-panel metrics
- `chord_diagram.png` - Inter-hierarchy connections

### 2. Expert-Level Analizler (4 dosya)

**04_expert_analysis/**:
- `sankey_state_transitions.png` - Sankey flow diagram
- `network_motif_analysis.png` - Triangles, feed-forward loops, reciprocity
- `community_evolution.png` - Community detection over time
- `network_resilience_analysis.png` - Attack tolerance, critical nodes

### 3. Stokastik Süreç Analizleri (4 dosya)

**05_stochastic_analysis/**:
- `stochastic_path_analysis.png` - Confidence bands (90%, 50%)
- `cox_process_intensity.png` - Arrest rate λ(t) evolution
- `trust_sde_dynamics.png` - OU process, autocorrelation, phase space
- `ctmc_transition_analysis.png` - Conversion rates, waiting times

### 4. LaTeX Raporu

**07_reports/scenario_report.tex**:
- Senaryo açıklaması
- Parametre tabloları
- Summary statistics tabloları
- Hierarchy analysis tabloları
- Monte Carlo aggregate statistics
- Tüm görsellerin embed edilmiş versiyonu
- Professional academic format

**Derleme**:
```bash
cd results_*/07_reports/
pdflatex scenario_report.tex
pdflatex scenario_report.tex  # İki kez (referanslar için)
```

### 5. Ham Veri

**08_data/**:
- `single_simulation_events.json` - Complete event history
- `single_simulation_timeseries.csv` - Time series data

---

## ⚙️ Kalibrasyon Parametreleri

### `run_scenario.py` içinde parametreleri değiştir:

```python
SCENARIOS = {
    "my_scenario": {  # ← YENİ SENARYO İSMİ
        "name": "My Custom Scenario",
        "description": "İstediğin açıklama",
        "params": {
            # === ARREST PROCESS (Cox Process) ===
            "lambda_0": 0.1,      # Baseline arrest rate
            "kappa": 0.5,         # Hierarchical protection (0-1)

            # === TRUST DYNAMICS (Ornstein-Uhlenbeck SDE) ===
            "alpha": 0.5,         # Mean reversion rate
            "sigma": 0.15,        # Volatility/noise
            "Delta": 10.0,        # Memory time

            # === LAW ENFORCEMENT ===
            "P_0": 1.0,           # Initial effectiveness
            "eta_P": 0.1,         # Effectiveness jump per conversion

            # === CONVERSION (CTMC) ===
            "mu_LH": 0.3,         # High-trust conversion rate
            "mu_min": 0.01,       # Minimum rate
            "mu_rng": 0.5,        # Range
            "theta": 2.0,         # Trust sensitivity

            # === SIMULATION ===
            "T_max": 100.0,       # Simulation time
            "dt": 0.01,           # Time step
            "random_seed": 42,
        },
        "network": {
            "hierarchy": [60, 30, 10],  # [L3 operatives, L2 mid, L1 leaders]
            "seed": 42,
        },
        "monte_carlo": {
            "n_runs": 50,         # Monte Carlo simulations
            "n_cores": 1,
        }
    },
}
```

### Parametre Kılavuzu:

| Parametre | Düşük Değer | Yüksek Değer | Etki |
|-----------|-------------|--------------|------|
| `lambda_0` | 0.05 | 0.3 | Arrest rate (düşük → yavaş collapse) |
| `kappa` | 0.2 | 0.9 | Hierarchical protection (yüksek → resilient) |
| `alpha` | 0.2 | 0.8 | Trust mean reversion (düşük → stable) |
| `sigma` | 0.05 | 0.25 | Trust volatility (düşük → predictable) |
| `Delta` | 5 | 30 | Memory time (yüksek → long memory) |
| `P_0` | 0.6 | 1.5 | Enforcement effectiveness |
| `mu_LH` | 0.1 | 0.6 | Conversion rate (yüksek → many informants) |
| `T_max` | 30 | 500 | Simulation time |

---

## 📈 Örnek Senaryolar

### Scenario 1: **Aggressive Crackdown**
```python
"lambda_0": 0.4,    # Çok yüksek arrest
"kappa": 0.2,       # Düşük protection
"T_max": 30.0,      # Kısa zaman (hızlı collapse)
```
**Beklenen**: t~5-10'da tam collapse, %100 arrest rate

### Scenario 2: **Resilient Network**
```python
"lambda_0": 0.03,   # Çok düşük arrest
"kappa": 0.9,       # Çok yüksek protection
"alpha": 0.2,       # Yavaş trust decay
"Delta": 30.0,      # Uzun memory
"T_max": 500.0,     # Uzun simülasyon
```
**Beklenen**: t~200+'da yavaş collapse, düşük conversion

### Scenario 3: **High Informant Success**
```python
"mu_LH": 0.6,       # Çok yüksek conversion
"eta_P": 0.2,       # Büyük effectiveness jumps
"theta": 1.5,       # Kolay conversion
```
**Beklenen**: Cascade effect, yüksek informant sayısı

---

## 🏗️ Dizin Yapısı

```
istih/
├── run_scenario.py                      ← ANA ÇALIŞTIRMA SCRIPTI
├── SCENARIO_GUIDE.md                    ← Detaylı kullanım kılavuzu
├── README_SCENARIO_SYSTEM.md            ← Bu dosya
│
├── organized_crime_network/
│   ├── simulation/
│   │   ├── visualization_network_advanced.py
│   │   ├── visualization_interactive.py
│   │   ├── visualization_ultra_advanced.py
│   │   ├── visualization_expert.py
│   │   └── visualization_stochastic.py
│   └── reporting/
│       ├── __init__.py
│       └── latex_generator.py           ← LaTeX rapor oluşturucu
│
└── results_SCENARIO_TIMESTAMP/          ← OLUŞAN ÇIKTILAR
    ├── scenario_config.json
    ├── 01_network_static/               (5 PNG)
    ├── 02_network_interactive/          (3 HTML)
    ├── 03_network_ultra/                (3 PNG)
    ├── 04_expert_analysis/              (4 PNG)
    ├── 05_stochastic_analysis/          (4 PNG)
    ├── 06_monte_carlo/                  (reserved for future use)
    ├── 07_reports/
    │   └── scenario_report.tex          ← LaTeX raporu
    └── 08_data/
        ├── single_simulation_events.json
        └── single_simulation_timeseries.csv
```

---

## ⏱️ Execution Time

| Scenario | Single Run | Monte Carlo (50 runs) | Viz Generation | Total |
|----------|------------|------------------------|----------------|-------|
| Default | ~2-3s | ~2-3 min | ~30s | **~3-4 min** |
| Aggressive (T=50) | ~1s | ~1 min | ~30s | **~2 min** |
| Realistic (T=500) | ~15s | ~15 min | ~30s | **~16 min** |

**Not**: Monte Carlo parallel processing TODO (şimdilik sequential)

---

## 📄 LaTeX Rapor İçeriği

1. **Abstract**: Senaryo özeti
2. **Scenario Description**: Parametre açıklaması
3. **Model Components**: Mathematical framework
4. **Parameter Table**: Tüm kalibrasyon değerleri
5. **Single Run Results**:
   - Summary statistics table
   - Hierarchy analysis table
6. **Monte Carlo Analysis**:
   - Aggregate statistics (mean ± std)
   - Collapse rate, time analysis
7. **Visualizations**:
   - Network topology figures
   - Stochastic process figures
   - Expert analysis figures
8. **Conclusions**: Key findings

**Örnek Tablo**:
```latex
\begin{table}[H]
\caption{Monte Carlo Results (50 runs)}
\begin{tabular}{lrrr}
\toprule
Metric & Mean & Std Dev & Median \\
\midrule
Final Active Actors & 0.12 & 0.45 & 0.00 \\
Final Arrested & 17.32 & 6.28 & 18.00 \\
Total Arrests & 49.88 & 0.52 & 50.00 \\
Network Collapse Rate & 100.0\% & -- & -- \\
\bottomrule
\end{tabular}
\end{table}
```

---

## 🐛 Troubleshooting

### Problem: "Cannot import latex_generator"
**Çözüm**: `organized_crime_network/reporting/__init__.py` var mı kontrol et

### Problem: Memory error
**Çözüm**: Monte Carlo `n_runs` sayısını azalt (50 → 20)

### Problem: Çok yavaş
**Çözüm**:
- `T_max` azalt (500 → 100)
- Network size küçült ([60,30,10] → [30,15,5])
- `n_runs` azalt

### Problem: LaTeX derleme hatası
**Çözüm**:
- `pdflatex` yüklü mü kontrol et: `pdflatex --version`
- Grafiklerin yolları doğru mu kontrol et
- İki kez derle (references için)

---

## 🎓 İleri Seviye Kullanım

### Parametre Sweep (birden fazla değer test)

```bash
# Manuel loop
for lambda in 0.05 0.1 0.15 0.2 0.25 0.3
do
    # run_scenario.py içinde lambda_0'ı değiştir
    # Veya programmatic olarak yeni senaryo oluştur
    python run_scenario.py --scenario custom_lambda_$lambda
done
```

### Custom Analysis Script

```python
# Kendi analiz scriptini yaz
from organized_crime_network.simulation.results import SimulationResults
import json

# Load results
with open('results_*/08_data/single_simulation.json') as f:
    data = json.load(f)

# Custom analysis
# ...
```

---

## 📚 Ek Kaynaklar

- **Detaylı Parametre Kılavuzu**: `SCENARIO_GUIDE.md`
- **Matematiksel Model**: `gelis.tex`
- **Test Scriptleri**:
  - `test_expert_viz.py`
  - `test_stochastic_viz.py`
  - `test_all_new_viz.py`

---

## ✨ Özet: Tek Komut, Tam Analiz!

```bash
# 1. Senaryo seç
python run_scenario.py --scenario aggressive

# 2. Bekle (~3-5 dakika)

# 3. Sonuçları incele:
#    - 19+ görselleştirme
#    - LaTeX raporu
#    - JSON/CSV veriler
#    - İnteraktif HTML dashboards

# 4. LaTeX derleme (isteğe bağlı):
cd results_*/07_reports/
pdflatex scenario_report.tex
pdflatex scenario_report.tex
```

**Hepsi bu kadar!** 🎉

---

## 📞 İletişim

Sorular veya öneriler için issue aç: `github.com/your_repo/issues`
