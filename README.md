# AeroVision CFD v1.1

Professional CFD (Computational Fluid Dynamics) simulation interface with turbulence modeling and 3D visualization.

## Features

- **Physics Engine**: Navier-Stokes solver with multiple turbulence models
- **Turbulence Models**: Smagorinsky, Spalart-Allmaras, k-epsilon, k-omega SST
- **3D Visualization**: Interactive PyVista-based 3D viewer with streamlines
- **Post-Processing**: Cut planes, scalar fields, Q-Criterion visualization
- **Batch Processing**: Run multiple simulations with different parameters
- **Export Options**: CSV, JSON, VTK, PNG formats
- **Preset System**: Save/load simulation parameters
- **Comparison Mode**: Compare multiple simulation results
- **GPU Support**: Multi-vendor GPU acceleration — NVIDIA (cupy), Intel/AMD (PyOpenCL + custom kernels)
- **Multi-Language**: Turkish and English interface

## Installation

### From Release (Recommended)

Each release provides two packages:

- **Installer Version**: `AeroVision_CFD_Setup.exe` — built with Inno Setup
  from the onedir package. Install and run.
- **Portable Version**: `AeroVision_CFD_Single.exe` — single-file portable
  build (onefile). No installation needed, runs from USB.

Both packages embed all dependencies (numpy, scipy, PyQt6, pyvista, cupy,
pyopencl included) — no separate Python installation required.

### From Source
```bash
pip install -r requirements.txt
python main.py
```

### System Requirements
- **OS**: Windows 10/11 (64-bit)
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: NVIDIA (CUDA/cupy) or Intel/AMD (OpenCL) — optional, for GPU acceleration
- **Dependencies**: All included in the package

## Usage

### Keyboard Shortcuts
| Shortcut | Action |
|----------|--------|
| Ctrl+O | Load STL file |
| Ctrl+R / F5 | Start simulation |
| Ctrl+S | Save screenshot |
| Ctrl+E | Export CSV |
| Ctrl+Shift+E | Export VTK |
| Ctrl+P | Save preset |
| Ctrl+Shift+P | Load preset |
| Ctrl+B | Batch simulation |
| Ctrl+Q | Quit |

### Simulation Parameters

- **Wind Speed**: 5-150 m/s
- **Resolution**: Low (50×18×18) to Maximum (180×60×60)
- **Turbulence Model**: Smagorinsky, Spalart-Allmaras, k-epsilon, k-omega SST
- **Wall Functions**: Enable/disable wall function treatment

## Package Structure

```
AeroVision_CFD/
├── AeroVision_CFD.exe        # Main executable
├── _internal/                # Internal dependencies
│   ├── solver.py            # Physics engine
│   ├── ui_components.py     # UI components
│   ├── postprocess_ui.py    # Post-processing
│   ├── crash_logger.py      # Error logging
│   ├── numpy/               # NumPy library
│   ├── scipy/               # SciPy library
│   ├── PyQt6/               # PyQt6 library
│   ├── pyvista/             # PyVista library
│   ├── cupy/                # CuPy library (GPU support)
│   └── matplotlib/          # Matplotlib library
└── README.md                # This file
```

## Source Code Structure

```
aerovision_cfd/
├── main.py              # Main application window
├── solver.py            # Physics engine (Navier-Stokes, turbulence)
├── turbulence_models.py # SA / k-epsilon / k-omega SST implementations
├── gpu_backend.py       # GPU detection + array backend (cupy/pyopencl/numpy)
├── cl_backend.py        # OpenCL context/queue management
├── cl_solver.py         # OpenCL kernels + GPU solver (NS, CG, SA)
├── constants.py         # Physical constants + LRU caches
├── ui_components.py     # UI panel factory
├── postprocess_ui.py    # Post-processing interface
├── crash_logger.py      # Error logging
├── sweep.py             # CLI parameter sweep (wind x yaw x fluid)
├── gpu_selftest.py      # GPU verification (CPU parity + speedup report)
├── gpu_profiler.py      # GPU kernel timing profiler
├── validation/          # Scientific validation suite (sphere Cd, NACA 0012)
│   ├── validate_cd.py           # Sphere drag coefficient benchmark
│   ├── validate_airfoil.py      # NACA 0012 lift benchmark
│   ├── naca0012_stl.py          # NACA 0012 STL generator
│   ├── sphere_stl.py            # Sphere STL generator
│   └── report.py                # Generates validation/results/report.md
├── test_solver.py       # Unit tests
├── build.spec           # PyInstaller (onedir) configuration
├── build_onefile.spec   # PyInstaller (onefile) configuration
└── requirements.txt     # Dependencies
```

### Parameter Sweep (CLI)

```bash
# Single case
python sweep.py --stl model.stl --wind 10 --yaw 0 --fluid air --res 60x30x30 --iters 40

# Sweep: wind 5-15 (step 5), yaw 0-90 (step 45), fluids air+water
python sweep.py --stl model.stl --wind 5:15:5 --yaw 0:90:45 --fluid air,water \
    --profile --out results/sweep

# Outputs: sweep_report.md, sweep_results.csv, sweep_profiles.png (optional)
```

STL files must be in a centered coordinate system (tunnel (4, 2, 2) m, x is the flow direction).
Fluid presets: `air` (1.225 kg/m³), `water` (998.2 kg/m³), `co2` (1.87 kg/m³).

### Scientific Validation

```bash
# Sphere drag coefficient (Cd = 0.47 reference, Re = 4e5)
python validation/validate_cd.py --model sa --res 60x30x30 --iters 60
python validation/validate_cd.py --model sa --res 60x30x30 --iters 60 --wind 2.5

# NACA 0012 lift (Cl = 0.5 @ 8° reference)
python validation/validate_airfoil.py --model sa --alpha 8 --res 60x30x30 --iters 60

# Validation report (validation/results/report.md + results.csv)
python validation/report.py

# Validation tests (pytest)
python -m pytest test_solver.py -m validation
```

Results: sphere Cd = 0.449 (medium grid, 4.4% error), k-omega SST = 0.452.
NACA 0012 lift is reported as qualitative validation due to voxel geometry
limits (1-2 cell thickness) — see the "Limitations" section of the report.

## Physics Engine

### Navier-Stokes Solver
- Chorin projection method
- RK2 (Heun) time integration
- Conservative viscous formulation
- Rhie-Chow interpolation for pressure-velocity coupling
- Adaptive time stepping based on CFL condition

### Turbulence Models

#### Smagorinsky SGS
- Wall damping (Van Driest)
- Filter width: Δ = (Δx·Δy·Δz)^(1/3)

#### Spalart-Allmaras
- One-equation model
- Wall distance-based destruction term
- Upwind advection scheme

#### k-epsilon (Launder & Spalding 1974)
- Two-equation model (k, ε)
- Production limiter
- Upwind advection

#### k-omega SST (Menter 1994)
- Two-equation model (k, ω)
- Shear Stress Transport formulation
- F1/F2 blending functions
- Cross-diffusion term

## GPU Acceleration

The OpenCL path (Intel/AMD GPUs) runs the following on-device kernels (fp64):

- NS steps (RK2 stage1/combine)
- Pressure Poisson (device-resident CG solver)
- Velocity projection
- Spalart-Allmaras turbulence model

The k-epsilon / k-omega SST / Smagorinsky models and wall functions run on
the CPU (fp64) in hybrid mode. The NVIDIA path uses cupy with the same
numerical scheme.

GPU/CPU parity is verified on every session by `gpu_selftest.py`
(|dCd| = 0.0000, bitwise-identical to the CPU reference).

## Performance

Real benchmark results from `gpu_selftest.py` (sphere R = 0.25 m,
40×20×20 grid, 30 iterations, Spalart-Allmaras, wind 10 m/s) — GPU/CPU
parity |dCd| = 0.0000 on every system:

| Hardware | Backend | Time (30 iters) | Speedup |
|---|---|---|---|
| i5-3470 + RX 570 OC | CPU (reference) | 5.18 s | 1.0x |
| i5-3470 + RX 570 OC | GPU (OpenCL, fp64) | 1.47 s | **3.52x** |
| i7-13620H + RTX 4060 Laptop | CPU (reference) | 1.64 s | 1.0x |
| i7-13620H + RTX 4060 Laptop | GPU (cupy, fp64) | 1.38 s | **1.19x** |

The AMD desktop system shows a large 3.52x speedup. The laptop's strong
CPU (i7-13620H) narrows the gap on the small selftest grid — expect larger
GPU gains at higher resolutions. See `gpu_profiler.py` for per-kernel
timings on any system.

## Building from Source

### Prerequisites
- Python 3.10+
- pip

### Build Onedir (Installable Directory)
```bash
pip install pyinstaller
python -m PyInstaller build.spec --clean
# Output: dist/AeroVision_CFD/ (all dependencies embedded: numpy, scipy,
#        PyQt6, pyvista, cupy, pyopencl included)
```

### Build Onefile (Portable Single Exe)
```bash
python -m PyInstaller build_onefile.spec --clean
# Output: dist/AeroVision_CFD_Single.exe
```

### Setup Installer (Inno Setup)
1. Build the onedir package: `python -m PyInstaller build.spec --clean` (or
   download it from the GitHub Actions artifacts — see below).
2. Compile the ready-made script `AeroVision_CFD_Setup.iss` with Inno Setup
   Compiler (Source: `dist/AeroVision_CFD\*`, OutputBaseFilename:
   `AeroVision_CFD_Setup`, 64-bit, English + Turkish languages). Setup
   Compiler > Build.
3. The output `dist/AeroVision_CFD_Setup.exe` is the release installer.
   Keep `MyAppVersion` in sync with pyproject.toml.

### Automated Windows Builds (GitHub Actions)

PyInstaller cannot cross-compile: a Windows `.exe` must be built on Windows.
Instead of building locally, use the provided workflow
`.github/workflows/build.yml`:

1. Go to **Actions** > **Build Windows Packages** > **Run workflow** on the
   `main` branch (or push a `v*` tag — releases build automatically).
2. Download the artifacts from the finished run:
   - `AeroVision_CFD_onedir.zip` — folder distribution (Inno Setup source)
   - `AeroVision_CFD_Single.exe` — portable single-file build
3. Unzip the onedir artifact, point `AeroVision_CFD_Setup.iss` at it, and
   build the installer with Inno Setup.

### Output
- `dist/AeroVision_CFD/` - Folder distribution (onedir, Inno Setup source)
- `dist/AeroVision_CFD_Single.exe` - Single file executable (portable)

## Testing

```bash
# Unit tests
python -m pytest test_solver.py -q -m "not validation"

# Validation tests (sphere Cd + airfoil Cl; ~3 min)
python -m pytest test_solver.py -q -m validation

# All tests
python -m pytest test_solver.py

# Lint (ruff) and format
ruff check .
ruff format --check .
```

CI runs lint + format + tests on every push via GitHub Actions
(see `.github/workflows/ci.yml`).

## Troubleshooting

### GPU Issues

Both the OpenCL backend (Intel/AMD) and the cupy backend (NVIDIA) are
supported. Without a GPU, CPU mode is selected automatically.

```bash
# OpenCL GPU verification (self test)
python gpu_selftest.py                # full test: device + kernel + simulation
python gpu_selftest.py --quick        # device detection + kernel build only
python gpu_selftest.py --device intel # select a specific vendor
# Output: gpu_selftest_result.json (Cd comparison + speedup ratio)
```

Required for OpenCL (Intel/AMD): `pip install pyopencl`.

To verify the OpenCL kernels on CPU in a GPU-less environment (pocl):
```bash
sudo apt install pocl-opencl-icd
AV_OPENCL_ALLOW_CPU=1 python gpu_selftest.py --device intel
```
`AV_OPENCL_ALLOW_CPU=1` only enables CPU OpenCL devices (pocl) for testing;
it does not change production behavior.

If you have an NVIDIA GPU but cupy doesn't work:
1. Install CUDA Toolkit from the NVIDIA website
2. Install the matching cupy wheel: `pip install cupy-cuda12x`

### Performance Tips
- Use higher resolution for more accurate results
- k-epsilon and k-omega SST are more accurate but slower
- Enable wall functions for better near-wall accuracy

## License

GPL-3.0 — see [LICENSE](LICENSE).

---

# Türkçe

## Kurulum

### Release'den (Önerilen)

Her release iki seçenek sunar:

- **Kurulumcu Sürümü**: `AeroVision_CFD_Setup.exe` — Inno Setup ile paketlenmiş
  kurulum sihirbazı (onedir paketinden üretilir). Kur ve kullan.
- **Taşınabilir Sürüm**: `AeroVision_CFD_Single.exe` — tek dosyalık taşınabilir
  sürüm (onefile). Kurulum gerektirmez, USB'den çalışır.

Her iki paket de tüm bağımlılıkları gömülü taşır (numpy, scipy, PyQt6, pyvista,
cupy, pyopencl dahil) — ayrıca Python kurulumu gerekmez.

### Kaynak Koddan
```bash
pip install -r requirements.txt
python main.py
```

### Sistem Gereksinimleri
- **İşletim Sistemi**: Windows 10/11 (64-bit)
- **RAM**: En az 4GB, önerilen 8GB
- **GPU**: NVIDIA (CUDA/cupy) veya Intel/AMD (OpenCL) — isteğe bağlı, GPU hızlandırma için
- **Bağımlılıklar**: Paket içinde hazır gelir

## Kullanım

### Klavye Kısayolları
| Kısayol | Eylem |
|----------|--------|
| Ctrl+O | STL dosyası yükle |
| Ctrl+R / F5 | Simülasyonu başlat |
| Ctrl+S | Ekran görüntüsü kaydet |
| Ctrl+E | CSV dışa aktar |
| Ctrl+Shift+E | VTK dışa aktar |
| Ctrl+P | Hazır ayar kaydet |
| Ctrl+Shift+P | Hazır ayar yükle |
| Ctrl+B | Toplu simülasyon |
| Ctrl+Q | Çıkış |

### Simülasyon Parametreleri

- **Rüzgar Hızı**: 5-150 m/s
- **Çözünürlük**: Düşük (50×18×18) ile Maksimum (180×60×60)
- **Türbülans Modeli**: Smagorinsky, Spalart-Allmaras, k-epsilon, k-omega SST
- **Duvar Fonksiyonları**: Aç/kapat

## Kaynak Kod Yapısı

```
aerovision_cfd/
├── main.py              # Ana uygulama penceresi
├── solver.py            # Fizik motoru (Navier-Stokes, türbülans)
├── turbulence_models.py # SA / k-epsilon / k-omega SST uygulamaları
├── gpu_backend.py       # GPU tespiti + dizi altyapısı (cupy/pyopencl/numpy)
├── cl_backend.py        # OpenCL context/queue yönetimi
├── cl_solver.py         # OpenCL kernel'leri + GPU çözücü (NS, CG, SA)
├── constants.py         # Fiziksel sabitler + LRU cache'ler
├── ui_components.py     # UI panel fabrikası
├── postprocess_ui.py    # Son işleme arayüzü
├── crash_logger.py      # Hata kaydı
├── sweep.py             # CLI parametre taraması (wind x yaw x fluid)
├── gpu_selftest.py      # GPU doğrulama (CPU parite + hızlanma raporu)
├── gpu_profiler.py      # GPU kernel süre profileri
├── validation/          # Bilimsel doğrulama paketi (küre Cd, NACA 0012)
│   ├── validate_cd.py           # Küre sürüklenme katsayısı kıyaslaması
│   ├── validate_airfoil.py      # NACA 0012 kaldırma kıyaslaması
│   ├── naca0012_stl.py          # NACA 0012 STL üreteci
│   ├── sphere_stl.py            # Küre STL üreteci
│   └── report.py                # validation/results/report.md üretir
├── test_solver.py       # Birim testleri
├── build.spec           # PyInstaller (onedir) yapılandırması
├── build_onefile.spec   # PyInstaller (onefile) yapılandırması
└── requirements.txt     # Bağımlılıklar
```

### Parametre Taraması (CLI)

```bash
# Tek vaka
python sweep.py --stl model.stl --wind 10 --yaw 0 --fluid air --res 60x30x30 --iters 40

# Tarama: wind 5-15 (adım 5), yaw 0-90 (adım 45), akışkanlar air+water
python sweep.py --stl model.stl --wind 5:15:5 --yaw 0:90:45 --fluid air,water \
    --profile --out results/sweep

# Çıktılar: sweep_report.md, sweep_results.csv, sweep_profiles.png (opsiyonel)
```

STL dosyaları merkezli koordinat sisteminde olmalı (tünel (4, 2, 2) m, x akış yönü).
Akışkan hazır ayarları: `air` (1.225 kg/m³), `water` (998.2 kg/m³), `co2` (1.87 kg/m³).

### Bilimsel Doğrulama

```bash
# Küre sürüklenme katsayısı (Cd = 0.47 referans, Re = 4e5)
python validation/validate_cd.py --model sa --res 60x30x30 --iters 60
python validation/validate_cd.py --model sa --res 60x30x30 --iters 60 --wind 2.5

# NACA 0012 kaldırma (Cl = 0.5 @ 8° referans)
python validation/validate_airfoil.py --model sa --alpha 8 --res 60x30x30 --iters 60

# Doğrulama raporu (validation/results/report.md + results.csv)
python validation/report.py

# Doğrulama testleri (pytest)
python -m pytest test_solver.py -m validation
```

Sonuçlar: küre Cd = 0.449 (orta ağ, %4.4 hata), k-omega SST = 0.452.
NACA 0012 kaldırma, voxel geometri sınırı (1-2 hücre kalınlık) nedeniyle
niteliksel doğrulama olarak raporlanmıştır (raporun "Sınırlamalar" bölümü).

## Fizik Motoru

### Navier-Stokes Çözücü
- Chorin projeksiyon yöntemi
- RK2 (Heun) zaman entegrasyonu
- Korunumlu viskoz formülasyon
- Basınç-hız eşleşmesi için Rhie-Chow interpolasyonu
- CFL koşuluna dayalı uyarlanır zaman adımı

### Türbülans Modelleri

#### Smagorinsky SGS
- Duvar sönümlemesi (Van Driest)
- Filtre genişliği: Δ = (Δx·Δy·Δz)^(1/3)

#### Spalart-Allmaras
- Tek denklem modeli
- Duvar mesafesine dayalı yok etme terimi
- Upwind taşınım şeması

#### k-epsilon (Launder & Spalding 1974)
- İki denklem modeli (k, ε)
- Üretim sınırlayıcısı
- Upwind taşınım

#### k-omega SST (Menter 1994)
- İki denklem modeli (k, ω)
- Shear Stress Transport formülasyonu
- F1/F2 karışım fonksiyonları
- Çapraz difüzyon terimi

## GPU Hızlandırma

OpenCL yolu (Intel/AMD GPU'lar) şu işlemleri cihaz kernel'lerinde (fp64) çalıştırır:

- NS adımları (RK2 stage1/combine)
- Basınç Poisson (cihaz-yerleşik CG çözücü)
- Hız projeksiyonu
- Spalart-Allmaras türbülans modeli

k-epsilon / k-omega SST / Smagorinsky modelleri ve duvar fonksiyonları CPU
(fp64) tarafında hybrid olarak işlenir. NVIDIA yolu aynı sayısal şemayla
cupy kullanır.

GPU/CPU paritesi her oturumda `gpu_selftest.py` ile doğrulanır
(|dCd| = 0.0000, CPU referansıyla birebir).

## Performans

`gpu_selftest.py`'den gerçek benchmark sonuçları (küre R = 0.25 m,
40×20×20 ağ, 30 iterasyon, Spalart-Allmaras, 10 m/s rüzgar) — her sistemde
GPU/CPU paritesi |dCd| = 0.0000:

| Donanım | Backend | Süre (30 iter) | Hızlanma |
|---|---|---|---|
| i5-3470 + RX 570 OC | CPU (referans) | 5.18 s | 1.0x |
| i5-3470 + RX 570 OC | GPU (OpenCL, fp64) | 1.47 s | **3.52x** |
| i7-13620H + RTX 4060 Laptop | CPU (referans) | 1.64 s | 1.0x |
| i7-13620H + RTX 4060 Laptop | GPU (cupy, fp64) | 1.38 s | **1.19x** |

AMD masaüstü sisteminde büyük 3.52x hızlanma görülür. Laptop'un güçlü
CPU'su (i7-13620H) küçük selftest ağında farkı daraltır — daha yüksek
çözünürlüklerde GPU kazancının büyümesi beklenir. Her sistemde kernel bazlı
süreler için `gpu_profiler.py`'a bakın.

## Derleme

### Ön Koşullar
- Python 3.10+
- pip

### Onedir Derleme (Kurulabilir Dizin)
```bash
pip install pyinstaller
python -m PyInstaller build.spec --clean
# Çıktı: dist/AeroVision_CFD/ (tüm bağımlılıklar gömülü: numpy, scipy,
#        PyQt6, pyvista, cupy, pyopencl dahil)
```

### Onefile Derleme (Taşınabilir Tek Exe)
```bash
python -m PyInstaller build_onefile.spec --clean
# Çıktı: dist/AeroVision_CFD_Single.exe
```

### Kurulumcu (Inno Setup)
1. Onedir paketini derle: `python -m PyInstaller build.spec --clean` (veya
   aşağıdaki GitHub Actions artifact'lerinden indir).
2. Hazır betik `AeroVision_CFD_Setup.iss`'i Inno Setup Compiler ile derle
   (Source: `dist/AeroVision_CFD\*`, OutputBaseFilename: `AeroVision_CFD_Setup`,
   64-bit, İngilizce + Türkçe dilleri). Setup Compiler > Build.
3. Çıktı `dist/AeroVision_CFD_Setup.exe` — release kurulumcusudur.
   `MyAppVersion`'ı pyproject.toml ile aynı tutun.

### Otomatik Windows Derlemeleri (GitHub Actions)

PyInstaller çapraz derleme yapamaz: Windows `.exe` yalnızca Windows üzerinde
derlenebilir. Yerel derleme yerine sağlanan `.github/workflows/build.yml`
workflow'unu kullanın:

1. **Actions** > **Build Windows Packages** > **Run workflow** (main dalında)
   — veya `v*` tag'i gönderin (release'ler otomatik derlenir).
2. Biten run'dan artifact'leri indirin:
   - `AeroVision_CFD_onedir.zip` — klasör dağıtımı (Inno Setup kaynağı)
   - `AeroVision_CFD_Single.exe` — taşınabilir tek dosya
3. Onedir artifact'ini açın, `AeroVision_CFD_Setup.iss`'i ona yönlendirin ve
   Inno Setup ile kurulumcuyu derleyin.

### Çıktılar
- `dist/AeroVision_CFD/` - Klasör dağıtımı (onedir, Inno Setup kaynağı)
- `dist/AeroVision_CFD_Single.exe` - Tek dosyalık çalıştırılabilir (taşınabilir)

## Test

```bash
# Birim testleri
python -m pytest test_solver.py -q -m "not validation"

# Doğrulama testleri (küre Cd + kanat Cl; ~3 dk)
python -m pytest test_solver.py -q -m validation

# Tüm testler
python -m pytest test_solver.py

# Lint (ruff) ve format
ruff check .
ruff format --check .
```

CI, her push'ta lint + format + test'i GitHub Actions üzerinden çalıştırır
(bkz. `.github/workflows/ci.yml`).

## Sorun Giderme

### GPU Sorunları

OpenCL backend (Intel/AMD) ve cupy backend (NVIDIA) desteklenir. GPU olmadan
CPU modu otomatik devreye girer.

```bash
# OpenCL GPU doğrulaması (self test)
python gpu_selftest.py                # tam test: cihaz + kernel + simülasyon
python gpu_selftest.py --quick        # sadece cihaz tespiti + kernel derleme
python gpu_selftest.py --device intel # belirli vendor seç
# Çıktı: gpu_selftest_result.json (Cd karşılaştırması + hızlanma oranı)
```

OpenCL (Intel/AMD) için gerekli: `pip install pyopencl`.

GPU'suz ortamda OpenCL kernel'lerini CPU üzerinde doğrulamak için (pocl):
```bash
sudo apt install pocl-opencl-icd
AV_OPENCL_ALLOW_CPU=1 python gpu_selftest.py --device intel
```
`AV_OPENCL_ALLOW_CPU=1` yalnızca test amacıyla CPU OpenCL cihazlarını (pocl)
dikkate alır; üretim davranışını değiştirmez.

NVIDIA GPU'nuz var ama cupy çalışmıyorsa:
1. NVIDIA sitesinden CUDA Toolkit kurun
2. Eşleşen cupy wheel'ini kurun: `pip install cupy-cuda12x`

### Performans İpuçları
- Daha doğru sonuçlar için daha yüksek çözünürlük kullanın
- k-epsilon ve k-omega SST daha doğru ama daha yavaştır
- Duvar fonksiyonları duvar yakını doğruluğunu artırır

## Lisans

GPL-3.0 — bkz. [LICENSE](LICENSE).
