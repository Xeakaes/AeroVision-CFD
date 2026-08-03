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

Her release iki seçenek sunar:

- **Installer Version**: `AeroVision_CFD_Setup.exe` — Inno Setup ile paketlenmiş
  kurulum sihirbazı (onedir paketinden üretilir). Kur ve kullan.
- **Portable Version**: `AeroVision_CFD_Single.exe` — tek dosyalık taşınabilir
  sürüm (onefile). Kurulum gerektirmez, USB'den çalışır.

Her iki paket de tüm bağımlılıkları gömülü taşır (numpy, scipy, PyQt6, pyvista,
cupy, pyopencl dahil) — ayrıca Python kurulumu gerekmez.

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
├── gpu_selftest.py      # GPU dogrulama (CPU parite + hizlanma raporu)
├── gpu_profiler.py      # GPU kernel sure profileri
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
# Tek vaka
python sweep.py --stl model.stl --wind 10 --yaw 0 --fluid air --res 60x30x30 --iters 40

# Tarama: wind 5-15 (adim 5), yaw 0-90 (adim 45), akiskanlar air+water
python sweep.py --stl model.stl --wind 5:15:5 --yaw 0:90:45 --fluid air,water \
    --profile --out results/sweep

# Ciktilari: sweep_report.md, sweep_results.csv, sweep_profiles.png (opsiyonel)
```

STL dosyalari merkezli koordinat sisteminde olmali (tünel (4, 2, 2) m, x akis yonu).
Akiskan ayarlari: `air` (1.225 kg/m³), `water` (998.2 kg/m³), `co2` (1.87 kg/m³).

### Scientific Validation

```bash
# Kure suruklenme katsayisi (Cd = 0.47 referans, Re = 4e5)
python validation/validate_cd.py --model sa --res 60x30x30 --iters 60
python validation/validate_cd.py --model sa --res 60x30x30 --iters 60 --wind 2.5

# NACA 0012 kaldirma (Cl = 0.5 @ 8° referans)
python validation/validate_airfoil.py --model sa --alpha 8 --res 60x30x30 --iters 60

# Dogrulama raporu (validation/results/report.md + results.csv)
python validation/report.py

# Dogrulama testleri (pytest)
python -m pytest test_solver.py -m validation
```

Sonuclar: kure Cd = 0.449 (orta ag, %4.4 hata), k-omega SST = 0.452.
NACA 0012 kaldirma, voxel geometri siniri (1-2 hucre kalinlik) nedeniyle
niteliksel dogrulama olarak raporlanmistir (raporun "Sınırlamalar" bolumu).

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

## Building from Source

### Prerequisites
- Python 3.10+
- pip

### Build Onedir (Installable Directory)
```bash
pip install pyinstaller
python -m PyInstaller build.spec --clean
# Cikti: dist/AeroVision_CFD/ (tum bagimlilklar gomulu: numpy, scipy, PyQt6,
#        pyvista, cupy, pyopencl dahil)
```

### Build Onefile (Portable Single Exe)
```bash
python -m PyInstaller build_onefile.spec --clean
# Cikti: dist/AeroVision_CFD_Single.exe
```

### Setup Installer (Inno Setup)
1. `python -m PyInstaller build.spec --clean` ile `dist/AeroVision_CFD/` onedir paketini uret.
2. Inno Setup Compiler ile hazir betigi derle: `AeroVision_CFD_Setup.iss` (Source:
   `dist/AeroVision_CFD\*`, OutputBaseFilename: `AeroVision_CFD_Setup`, 64-bit + Turkiye
   dili dahil). Setup Compiler > Build.
3. Cikti `dist/AeroVision_CFD_Setup.exe` — release olarak yayinlanir. Versiyon
   (`MyAppVersion`) pyproject.toml ile ayni tutulmali.

### Output
- `dist/AeroVision_CFD/` - Folder distribution (onedir, Inno Setup kaynagi)
- `dist/AeroVision_CFD_Single.exe` - Single file executable (portable)

## Testing

```bash
# Birim testleri
python -m pytest test_solver.py -q -m "not validation"

# Dogrulama testleri (kure Cd + kanat Cl; ~3 dk)
python -m pytest test_solver.py -q -m validation

# Tum testler
python -m pytest test_solver.py

# Lint (ruff) ve format
ruff check .
ruff format --check .
```

CI, GitHub Actions uzerinden her push'ta lint + format + test calistirir
(bkz. `.github/workflows/ci.yml`).

## Troubleshooting

### GPU Issues

OpenCL backend (Intel/AMD) ve cupy backend (NVIDIA) desteklenir. GPU olmadan
CPU modu otomatik devreye girer.

```bash
# OpenCL GPU dogrulamasi (Faz 3 self test)
python gpu_selftest.py                # tam test: cihaz + kernel + simülasyon
python gpu_selftest.py --quick        # sadece cihaz tespiti + kernel derleme
python gpu_selftest.py --device intel # belirli vendor sec
# Cikti: gpu_selftest_result.json (Cd karsilastirmasi + hizlanma orani)
```

OpenCL yolu (Intel/AMD GPU): NS adimlari (RK2 stage1/combine), basinc
Poisson (cihaz-yerlesik CG), hiz projeksiyonu ve Spalart-Allmaras turbulans
modeli GPU kernel'larinda (fp64) calisir; k-epsilon / k-omega SST / Smagorinsky
modelleri ve duvar fonksiyonlari CPU (fp64) tarafinda hybrid olarak islenir.
Gerekli kurulum: `pip install pyopencl`.

GPU/CPU paritesi her oturumda `gpu_selftest.py` ile dogrulanir
(|dCd| = 0.0000, CPU referansiyla birebir).

GPU'suz ortamda OpenCL kernel'larini CPU uzerinde dogrulamak icin (pocl):
```bash
sudo apt install pocl-opencl-icd
AV_OPENCL_ALLOW_CPU=1 python gpu_selftest.py --device intel
```
`AV_OPENCL_ALLOW_CPU=1` yalnizca test amaciyla CPU OpenCL cihazlarini (pocl)
dikkate alir; uretim davranisini degistirmez.
If you have an NVIDIA GPU but cupy doesn't work:
1. Install CUDA Toolkit from NVIDIA website
2. Install matching cupy version: `pip install cupy-cuda12x`

### Performance Tips
- Use higher resolution for more accurate results
- k-epsilon and k-omega SST are more accurate but slower
- Enable wall functions for better near-wall accuracy

## License

GPL-3.0 — bkz. [LICENSE](LICENSE).
