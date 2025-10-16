# Simülasyon Görselleştirme Rehberi

Bu dokümanda, organize suç ağı simülasyonlarının detaylı görselleştirme özelliklerini nasıl kullanacağınız açıklanmaktadır.

## Genel Bakış

Simülasyon motoru, üç tür kapsamlı görselleştirme sağlar:

1. **Ağ Topolojisi Görselleştirmesi** - Ağ yapısı ve aktör durumları
2. **Zaman Serisi Analizi** - Metrikler üzerinden zaman içi evrim
3. **Strateji Karşılaştırması** - Birden fazla simülasyon stratejisinin yan yana analizi

## Hızlı Başlangıç

### Basit Kullanım

```python
from organized_crime_network.simulation import (
    SimulationEngine,
    SimulationParameters,
    visualize_results
)

# Simülasyonu çalıştır
params = SimulationParameters.default()
engine = SimulationEngine(params)
results = engine.run(initial_network, verbose=True)

# Tüm görselleştirmeleri otomatik oluştur
visualize_results(results, output_dir="my_plots", show=True)
```

Bu tek satır kod, üç adet PNG dosyası oluşturur:
- `my_plots/simulation_network_topology.png`
- `my_plots/simulation_time_series.png`
- `my_plots/simulation_event_distribution.png`

### Gelişmiş Kullanım

```python
from organized_crime_network.simulation import visualization

# Görselleştirici oluştur
viz = visualization.SimulationVisualizer()

# Sadece ağ topolojisi
viz.plot_network_topology(
    results.final_network,
    title="Final Network State",
    save_path="network.png",
    show=True
)

# Sadece zaman serileri
viz.plot_time_series(
    results,
    save_path="timeseries.png",
    show=True
)

# Sadece olay dağılımı
viz.plot_event_distribution(
    results,
    save_path="events.png",
    show=True
)
```

## Görselleştirme Tipleri

### 1. Ağ Topolojisi (Network Topology)

**Ne Gösterir:**
- Aktörlerin hiyerarşik yapısı (Level 1: Operatörler, Level 2: Orta Kademe, Level 3: Liderler)
- Aktör durumları (Aktif, Tutuklu, İhbarcı)
- Güven bağlantıları (edge thickness = güven seviyesi)

**Renk Kodları:**
- 🔴 Kırmızı: Level 1 (Operatives)
- 🔵 Mavi: Level 2 (Mid-level)
- 🟢 Yeşil: Level 3 (Leaders)

**Şekil Kodları:**
- ⚫ Daire: Aktif aktörler
- ⬜ Kare: Tutuklu aktörler
- ⚪ Beyaz daire (kenarlıklı): İhbarcılar

### 2. Zaman Serisi Analizi (Time Series)

Altı adet subplot içerir:

1. **Network Size Evolution** - Aktif aktör sayısı zamanla
2. **Cumulative Arrests & Conversions** - Kümülatif tutuklama ve dönüşümler
3. **Law Enforcement Effectiveness** - P(t) yasa uygulama etkinliği
4. **Average Trust Between Actors** - Ortalama güven seviyesi w(t)
5. **Average Risk Perception** - Ortalama risk algısı R(t)
6. **Largest Connected Component** - En büyük bağlı bileşen boyutu

### 3. Olay Dağılımı (Event Distribution)

İki adet subplot:

1. **Event Timeline** - Olayların zaman çizelgesi (tutuklama vs dönüşüm)
2. **Event Frequency Distribution** - Olay frekans histogramı

### 4. Strateji Karşılaştırması (Strategy Comparison)

Birden fazla simülasyon stratejisini karşılaştırır:

```python
# İki strateji çalıştır
params_aggressive = SimulationParameters.aggressive_enforcement()
params_conservative = SimulationParameters.conservative_enforcement()

engine_agg = SimulationEngine(params_aggressive)
engine_con = SimulationEngine(params_conservative)

results_agg = engine_agg.run(network)
results_con = engine_con.run(network)

# Karşılaştır
viz = visualization.SimulationVisualizer()
viz.plot_strategy_comparison(
    [results_agg, results_con],
    ['Aggressive', 'Conservative'],
    save_path="comparison.png",
    show=True
)
```

Altı adet karşılaştırma grafiği içerir:
1. Ağ boyutu evrimi
2. Tutuklama oranları
3. Yasa uygulama etkinliği
4. Toplam tutuklama & dönüşümler (bar chart)
5. Güven evrimi
6. Etkinlik değişimi (final - initial)

## Örnek Çıktılar

### main.py Çalıştırması

```bash
python main.py
```

Oluşturulan dosyalar:
- `visualization_output/simulation_network_topology.png`
- `visualization_output/simulation_time_series.png`
- `visualization_output/simulation_event_distribution.png`

### demo_comprehensive.py Çalıştırması

```bash
python demo_comprehensive.py
```

Oluşturulan dosyalar:
- `demo_visualizations/aggressive/simulation_network_topology.png`
- `demo_visualizations/aggressive/simulation_time_series.png`
- `demo_visualizations/aggressive/simulation_event_distribution.png`
- `demo_visualizations/conservative/simulation_network_topology.png`
- `demo_visualizations/conservative/simulation_time_series.png`
- `demo_visualizations/conservative/simulation_event_distribution.png`
- `demo_visualizations/strategy_comparison.png`

## Özelleştirme Seçenekleri

### Figür Boyutu

```python
viz.plot_time_series(
    results,
    figsize=(20, 12),  # Daha büyük grafik
    show=True
)
```

### DPI ve Kalite

```python
viz.plot_network_topology(
    network,
    save_path="network_hq.png",
    show=False  # Gösterme, sadece kaydet
)
# Ardından matplotlib savefig parametreleriyle kaydet
```

### Grafiği Göstermeden Kaydetme

```python
# show=False olarak ayarlayın
visualize_results(results, output_dir="plots", show=False)
```

## API Referansı

### `visualize_results(results, output_dir=".", show=True)`

Tüm görselleştirmeleri otomatik oluşturur.

**Parametreler:**
- `results` (SimulationResults): Simülasyon sonuçları
- `output_dir` (str): Çıktı dizini (varsayılan: ".")
- `show` (bool): Grafikleri göster (varsayılan: True)

### `SimulationVisualizer.plot_network_topology(...)`

Ağ topolojisi grafiği oluşturur.

**Parametreler:**
- `network_state` (NetworkState): Görselleştirilecek ağ
- `title` (str): Grafik başlığı
- `save_path` (Optional[str]): Kayıt yolu
- `figsize` (Tuple[int, int]): Figür boyutu (varsayılan: (12, 10))
- `show` (bool): Grafiği göster (varsayılan: True)

**Dönüş:** `plt.Figure`

### `SimulationVisualizer.plot_time_series(...)`

Zaman serisi analizi grafiği oluşturur.

**Parametreler:**
- `results` (SimulationResults): Simülasyon sonuçları
- `save_path` (Optional[str]): Kayıt yolu
- `figsize` (Tuple[int, int]): Figür boyutu (varsayılan: (14, 10))
- `show` (bool): Grafiği göster (varsayılan: True)

**Dönüş:** `plt.Figure`

### `SimulationVisualizer.plot_strategy_comparison(...)`

Strateji karşılaştırma grafiği oluşturur.

**Parametreler:**
- `results_list` (List[SimulationResults]): Karşılaştırılacak simülasyon sonuçları listesi
- `strategy_names` (List[str]): Her strateji için isimler
- `save_path` (Optional[str]): Kayıt yolu
- `figsize` (Tuple[int, int]): Figür boyutu (varsayılan: (16, 10))
- `show` (bool): Grafiği göster (varsayılan: True)

**Dönüş:** `plt.Figure`

### `SimulationVisualizer.plot_event_distribution(...)`

Olay dağılımı grafiği oluşturur.

**Parametreler:**
- `results` (SimulationResults): Simülasyon sonuçları
- `save_path` (Optional[str]): Kayıt yolu
- `figsize` (Tuple[int, int]): Figür boyutu (varsayılan: (14, 6))
- `show` (bool): Grafiği göster (varsayılan: True)

**Dönüş:** `plt.Figure`

## İpuçları ve En İyi Uygulamalar

### 1. Büyük Simülasyonlar için

Büyük simülasyonlarda (>100 aktör), ağ görselleştirmesi karmaşık olabilir:

```python
# Daha büyük figür kullanın
viz.plot_network_topology(
    network,
    figsize=(20, 20),  # Daha büyük
    save_path="big_network.png"
)
```

### 2. Batch Analiz

Birden fazla simülasyonu toplu analiz etmek için:

```python
results_list = []
strategy_names = []

for strategy_name, params in strategies.items():
    engine = SimulationEngine(params)
    results = engine.run(network)
    results_list.append(results)
    strategy_names.append(strategy_name)

# Hepsini karşılaştır
viz.plot_strategy_comparison(
    results_list,
    strategy_names,
    save_path="all_strategies.png"
)
```

### 3. Yayın Kalitesi Görseller

Yayın için yüksek kaliteli görseller:

```python
import matplotlib
matplotlib.rcParams['figure.dpi'] = 300  # Yüksek DPI
matplotlib.rcParams['savefig.dpi'] = 300

visualize_results(results, output_dir="publication_plots")
```

## Sorun Giderme

### "tight_layout" Uyarısı

Bu uyarı normal ve görmezden gelinebilir. Grafiklerin düzeni üzerinde etkisi yoktur.

### Grafik Görünmüyor

`show=False` parametresini `show=True` olarak değiştirin veya dosyaları kontrol edin:

```bash
ls visualization_output/
```

### Bellek Sorunları

Çok fazla grafik oluşturuyorsanız, her birinden sonra kapatın:

```python
import matplotlib.pyplot as plt

fig = viz.plot_time_series(results, show=False)
plt.close(fig)  # Belleği serbest bırak
```

## Gelecek Özellikler

Planlanan görselleştirmeler:
- [ ] Interaktif 3D ağ görselleştirmesi (plotly)
- [ ] Animasyonlu ağ evrimi (matplotlib.animation)
- [ ] Isı haritası görselleştirmeleri
- [ ] Pareto frontier analizi grafikleri
- [ ] Dashboard oluşturma (Streamlit/Dash)

## Katkıda Bulunma

Görselleştirme önerileri için lütfen issue açın veya pull request gönderin.

## Lisans

Bu görselleştirme sistemi, ana simülasyon motoruyla aynı lisans altındadır.
