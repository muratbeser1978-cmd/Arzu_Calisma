# Senaryo Çalıştırma Kılavuzu

## Hızlı Başlangıç

### 1. Hazır Senaryo Çalıştırma

```bash
# Varsayılan moderate enforcement senaryosu
python run_scenario.py --scenario default

# Agresif enforcement senaryosu
python run_scenario.py --scenario aggressive

# Konservatif enforcement senaryosu
python run_scenario.py --scenario conservative

# Gerçekçi literatür-bazlı parametreler
python run_scenario.py --scenario realistic
```

### 2. Mevcut Senaryoları Listeleme

```bash
python run_scenario.py --list
```

### 3. Özel Çıktı Dizini Belirtme

```bash
python run_scenario.py --scenario aggressive --output my_aggressive_run
```

---

## Kalibrasyon Parametrelerini Değiştirme

### Adım 1: `run_scenario.py` dosyasını aç

Dosyanın başında `SCENARIOS` dictionary'sini bulacaksın:

```python
SCENARIOS = {
    "default": {
        "name": "Default Moderate Enforcement",
        "description": "Balanced enforcement with moderate parameters",
        "params": {
            # BURADAN PARAMETRELERİ DEĞİŞTİR
            "lambda_0": 0.1,        # Baseline arrest rate
            "kappa": 0.5,           # Hierarchical protection
            ...
        },
        ...
    },
}
```

### Adım 2: Yeni Senaryo Ekle

```python
SCENARIOS = {
    # ... mevcut senaryolar ...

    "my_custom": {  # ← YENİ SENARYO ADI
        "name": "My Custom Scenario",
        "description": "Custom parameters for specific analysis",
        "params": {
            # Arrest Process
            "lambda_0": 0.15,       # DEĞİŞTİR: Arrest rate
            "kappa": 0.6,           # DEĞİŞTİR: Hierarchical protection

            # Trust Dynamics
            "alpha": 0.4,           # DEĞİŞTİR: Mean reversion
            "sigma": 0.12,          # DEĞİŞTİR: Volatility
            "Delta": 15.0,          # DEĞİŞTİR: Memory time

            # Law Enforcement
            "P_0": 0.9,             # DEĞİŞTİR: Initial effectiveness
            "eta_P": 0.12,          # DEĞİŞTİR: Effectiveness jump

            # Conversion
            "mu_LH": 0.25,          # DEĞİŞTİR: High-trust conversion
            "mu_min": 0.008,        # DEĞİŞTİR: Minimum conversion
            "mu_rng": 0.4,          # DEĞİŞTİR: Range
            "theta": 2.2,           # DEĞİŞTİR: Trust sensitivity

            # Network Growth (genelde 0)
            "beta_growth": 0.0,

            # Simulation
            "T_max": 150.0,         # DEĞİŞTİR: Simulation time
            "dt": 0.01,             # Time step (genelde sabit)
            "random_seed": 42,
        },
        "network": {
            "hierarchy": [60, 30, 10],  # DEĞİŞTİR: [operatives, mid-level, leaders]
            "seed": 42,
        },
        "monte_carlo": {
            "n_runs": 50,           # DEĞİŞTİR: Monte Carlo run sayısı
            "n_cores": 1,           # Parallelization (TODO)
        }
    },
}
```

### Adım 3: Yeni Senaryoyu Çalıştır

```bash
python run_scenario.py --scenario my_custom
```

---

## Parametre Açıklamaları

### Cox Process (Arrest Dynamics)
- **`lambda_0`**: Baseline arrest rate (düşük: 0.05, yüksek: 0.3)
  - Küçük değer → yavaş collapse
  - Büyük değer → hızlı collapse

- **`kappa`**: Hierarchical protection factor (0-1 arası)
  - 0 → hiç koruma yok
  - 1 → tam koruma
  - Leaders için: intensity = λ₀ × exp(-κ × level)

### Trust SDE (Ornstein-Uhlenbeck)
- **`alpha`**: Mean reversion rate (0.1-1.0)
  - Küçük → yavaş adaptation (resilient)
  - Büyük → hızlı adaptation (fragile)

- **`sigma`**: Volatility/noise (0.05-0.25)
  - Küçük → stabil trust
  - Büyük → noisy trust

- **`Delta`**: Memory time (5-30)
  - Küçük → kısa memory
  - Büyük → uzun memory

### Law Enforcement
- **`P_0`**: Initial effectiveness (0.5-1.5)
  - < 1.0 → zayıf enforcement
  - > 1.0 → güçlü enforcement

- **`eta_P`**: Effectiveness jump per conversion (0.05-0.2)
  - Küçük → informant'lar az faydalı
  - Büyük → informant'lar çok faydalı

### CTMC (Conversion Process)
- **`mu_LH`**: High-trust conversion rate (0.1-0.6)
  - Yüksek trust'lı arrested actors için conversion rate

- **`mu_min`**: Minimum conversion rate (0.001-0.05)
  - Düşük trust'lı actors için minimum rate

- **`mu_rng`**: Range (0.2-0.8)
  - mu_LH ile mu_min arasındaki fark

- **`theta`**: Trust sensitivity (1.0-3.0)
  - Küçük → trust'a az duyarlı
  - Büyük → trust'a çok duyarlı

### Simulation
- **`T_max`**: Total simulation time (30-500)
  - Kısa: 30-100 (fast collapse)
  - Uzun: 200-500 (long-term dynamics)

- **`dt`**: Time step (genelde 0.01, DEĞİŞTİRME!)

---

## Çıktı Dizin Yapısı

Senaryo çalıştırıldığında şu yapı oluşturulur:

```
results_SCENARIO_TIMESTAMP/
├── scenario_config.json              # Senaryo parametreleri
├── 01_network_static/                # Statik network görselleri
│   ├── network_circular_hierarchy.png
│   ├── network_force_directed_analysis.png
│   ├── network_adjacency_heatmap.png
│   ├── network_3d_visualization.png
│   └── network_temporal_evolution_grid.png
├── 02_network_interactive/           # İnteraktif HTML görseller
│   ├── interactive_3d_network.html
│   ├── animated_network_evolution.html
│   └── interactive_dashboard.html
├── 03_network_ultra/                 # Ultra-advanced görseller
│   ├── radial_dendrogram.png
│   ├── network_metrics_dashboard.png
│   └── chord_diagram.png
├── 04_expert_analysis/               # Expert-level analizler
│   ├── sankey_state_transitions.png
│   ├── network_motif_analysis.png
│   ├── community_evolution.png
│   └── network_resilience_analysis.png
├── 05_stochastic_analysis/           # Stokastik süreç analizleri
│   ├── stochastic_path_analysis.png
│   ├── cox_process_intensity.png
│   ├── trust_sde_dynamics.png
│   └── ctmc_transition_analysis.png
├── 06_monte_carlo/                   # Monte Carlo sonuçları (reserved)
├── 07_reports/                       # LaTeX raporlar
│   └── scenario_report.tex           # Ana rapor
└── 08_data/                          # Ham veri
    ├── single_simulation_events.json
    └── single_simulation_timeseries.csv
```

---

## LaTeX Rapor Derleme

```bash
cd results_SCENARIO_TIMESTAMP/07_reports/
pdflatex scenario_report.tex
pdflatex scenario_report.tex  # İki kez çalıştır (referanslar için)
```

PDF oluşturulduktan sonra:
```
07_reports/
├── scenario_report.tex
├── scenario_report.pdf      # ← OLUŞAN RAPOR
├── scenario_report.aux
└── scenario_report.log
```

---

## Senaryo Örnekleri

### Örnek 1: Yüksek Arrest Rate, Düşük Protection
```python
"high_arrest_low_protection": {
    "params": {
        "lambda_0": 0.4,        # Çok yüksek arrest
        "kappa": 0.2,           # Çok düşük protection
        # ... diğer parametreler ...
    }
}
```
**Beklenen Sonuç**: Çok hızlı collapse (t~5-10), %100 arrest rate

### Örnek 2: Düşük Arrest, Yüksek Resilience
```python
"low_arrest_high_resilience": {
    "params": {
        "lambda_0": 0.03,       # Çok düşük arrest
        "kappa": 0.9,           # Çok yüksek protection
        "alpha": 0.2,           # Yavaş trust decay
        "Delta": 30.0,          # Uzun memory
        "T_max": 500.0,         # Uzun simülasyon
    }
}
```
**Beklenen Sonuç**: Yavaş collapse (t~200+), düşük conversion rate

### Örnek 3: Yüksek Conversion Efficiency
```python
"high_conversion": {
    "params": {
        "mu_LH": 0.6,           # Çok yüksek conversion
        "mu_min": 0.1,          # Yüksek minimum
        "eta_P": 0.2,           # Büyük effectiveness jumps
        "theta": 1.5,           # Düşük sensitivity (kolay conversion)
    }
}
```
**Beklenen Sonuç**: Yüksek informant sayısı, cascade effect

---

## Troubleshooting

### Problem: "Scenario not found"
**Çözüm**: Senaryo adını kontrol et, `--list` ile mevcut senaryoları gör

### Problem: Memory error (Monte Carlo'da)
**Çözüm**: `n_runs` değerini düşür (100 → 50 → 20)

### Problem: Simulation çok yavaş
**Çözüm**:
- `T_max` değerini düşür
- `n_runs` değerini düşür
- Network size'ı küçült ([60,30,10] → [40,20,5])

### Problem: LaTeX derleme hatası
**Çözüm**:
- pdflatex yüklü mü kontrol et
- Grafik dosyalarının olduğundan emin ol
- İki kez derle (referanslar için)

---

## İleri Seviye: Parametre Sweep

Birden fazla parametre değeri test etmek için:

```bash
# Bash loop
for lambda in 0.05 0.1 0.15 0.2 0.25 0.3
do
    # Her lambda için yeni senaryo oluştur ve çalıştır
    # (TODO: otomatik parametre sweep özelliği)
done
```

---

## Notlar

1. **Deterministik Sonuçlar**: Aynı `random_seed` ile her zaman aynı sonuçları alırsın
2. **Monte Carlo Varyans**: Farklı seed'ler (42, 43, 44, ...) farklı trajectories verir
3. **Computational Cost**:
   - Single run: ~1-5 saniye
   - Monte Carlo (50 runs): ~1-3 dakika
   - Tüm görselleştirmeler: ~30 saniye
   - **Toplam**: ~3-5 dakika

4. **Disk Space**: Her senaryo ~20-50 MB (görselleştirmeler dahil)
