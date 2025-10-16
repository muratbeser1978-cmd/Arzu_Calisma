# Hızlı Başlangıç - Organized Crime Network Scenario System

## ✅ Sistem Hazır!

`beta_growth` parametresi hatası düzeltildi. Sistem artık tamamen çalışıyor.

## 🚀 Kullanım

### 1. Mevcut Senaryoları Görüntüle
```bash
python run_scenario.py --list
```

### 2. Bir Senaryo Çalıştır
```bash
# Aggressive enforcement (hızlı - 50 saniye)
python run_scenario.py --scenario aggressive

# Default moderate enforcement
python run_scenario.py --scenario default

# Conservative enforcement
python run_scenario.py --scenario conservative

# Realistic parameters (yavaş - 500 zaman adımı)
python run_scenario.py --scenario realistic
```

### 3. Sonuçları İncele
```bash
# Senaryo çalıştırdıktan sonra oluşan dizin:
results_SCENARIO_TIMESTAMP/
├── 01_network_static/         → 5 PNG görsel
├── 02_network_interactive/    → 3 HTML dosyası (tarayıcıda aç)
├── 03_network_ultra/          → 3 PNG görsel
├── 04_expert_analysis/        → 4 PNG analiz
├── 05_stochastic_analysis/    → 4 PNG stokastik analiz
├── 07_reports/                → LaTeX raporu
└── 08_data/                   → JSON + CSV veriler
```

## 📊 Ne Üretilir?

### Görselleştirmeler (19+ dosya):
1. **Network Static** (5 PNG):
   - Circular hierarchy layout
   - Force-directed multi-view
   - Adjacency matrix heatmap
   - 3D network visualization
   - Temporal evolution grid

2. **Interactive** (3 HTML):
   - Interactive 3D network (rotatable)
   - Animated network evolution (play/pause)
   - Interactive dashboard (2x2)

3. **Ultra-Advanced** (3 PNG):
   - Radial dendrogram
   - Network metrics dashboard
   - Chord diagram

4. **Expert Analysis** (4 PNG):
   - Sankey state transitions
   - Network motif analysis
   - Community evolution
   - Network resilience analysis

5. **Stochastic Analysis** (4 PNG):
   - Stochastic path analysis (confidence bands)
   - Cox process intensity
   - Trust SDE dynamics
   - CTMC transition analysis

### Rapor ve Veriler:
- **LaTeX Report**: `scenario_report.tex` (derlemek için: `pdflatex scenario_report.tex`)
- **JSON Events**: Tüm simülasyon olayları
- **CSV Time Series**: Zaman serisi verileri

## ⚙️ Parametreleri Değiştir

`run_scenario.py` dosyasını aç ve `SCENARIOS` dictionary'sinde parametreleri değiştir:

```python
SCENARIOS = {
    "default": {
        "params": {
            # Cox Process (Arrest)
            "lambda_0": 0.1,        # Arrest rate (↑ daha hızlı collapse)
            "kappa": 0.5,           # Hierarchical protection (↑ daha resilient)

            # Trust SDE
            "alpha": 0.5,           # Mean reversion (↑ daha hızlı adaptation)
            "sigma": 0.15,          # Volatility (↑ daha noisy)
            "Delta": 10.0,          # Memory time (↑ daha uzun hafıza)

            # Conversion CTMC
            "mu_LH": 0.3,           # Conversion rate (↑ daha fazla informant)
            "theta": 2.0,           # Trust sensitivity (↑ daha zor conversion)

            # Simulation
            "T_max": 100.0,         # Simulation time
            "dt": 0.01,             # Time step (DEĞİŞTİRME!)
        }
    }
}
```

### Yeni Senaryo Ekle:
Mevcut senaryolardan birini kopyala, yeni isim ver, parametreleri değiştir.

## ⏱️ Süre Tahminleri

| Senaryo | T_max | MC Runs | Toplam Süre |
|---------|-------|---------|-------------|
| aggressive | 50 | 50 | ~1 dakika |
| default | 100 | 50 | ~2 dakika |
| conservative | 200 | 50 | ~3 dakika |
| realistic | 500 | 100 | ~15 dakika |

## 📝 Test Edildi

```bash
# Test edilen komut:
python run_scenario.py --scenario aggressive --output test_aggressive_run

# Sonuç:
✅ 53.4 saniyede tamamlandı
✅ 19 görselleştirme oluşturuldu
✅ LaTeX raporu oluşturuldu
✅ JSON ve CSV veriler kaydedildi
```

## 🔧 Düzeltilen Hatalar

1. ✅ `beta_growth` parametresi kaldırıldı (desteklenmiyor)
2. ✅ Eksik parametreler eklendi:
   - `gamma`: Operational risk multiplier
   - `rho`: Effectiveness decay
   - `beta`: Risk sensitivity
   - `delta_R`: Risk decay rate (1/Delta)
   - `Y_bar_0`: Default logit-trust
   - `w_min`: Minimum trust threshold
   - `gamma_pa`: Preferential attachment

3. ✅ Method isimleri düzeltildi:
   - `export_json` → `export_events`
   - `export_csv` → `export_time_series`

## 📚 Daha Fazla Bilgi

- **Detaylı Kılavuz**: `SCENARIO_GUIDE.md`
- **Sistem Dokümantasyonu**: `README_SCENARIO_SYSTEM.md`
- **Matematiksel Model**: `gelis.tex`

## 🎉 Başarılı Test Çıktısı

```
================================================================================
SCENARIO EXECUTION COMPLETE
================================================================================

  Elapsed time: 53.4 seconds
  Output directory: test_aggressive_run
  LaTeX report: test_aggressive_run\07_reports\scenario_report.tex

  Generated files:
    - 5 static network visualizations
    - 3 interactive HTML files
    - 3 ultra-advanced visualizations
    - 4 expert-level analyses
    - 4 stochastic analyses
    - 1 comprehensive LaTeX report

  To compile LaTeX report:
    cd test_aggressive_run\07_reports
    pdflatex scenario_report.tex
    pdflatex scenario_report.tex  (run twice for references)
```

**Her şey çalışıyor! 🚀**
