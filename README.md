# AeroVision CFD v1.0

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
- **GPU Support**: Automatic CUDA/cupy acceleration (optional)
- **Multi-Language**: Turkish and English interface

## Installation

### From Source
```bash
pip install -r requirements.txt
python main.py
```

### From Package (Recommended)
No installation required. Just run the executable:

```
dist/AeroVision_CFD/AeroVision_CFD.exe
```

### System Requirements
- **OS**: Windows 10/11 (64-bit)
- **RAM**: 4GB minimum, 8GB recommended
- **GPU**: NVIDIA GPU with CUDA (optional, for GPU acceleration)
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
├── ui_components.py     # UI panel factory
├── postprocess_ui.py    # Post-processing interface
├── crash_logger.py      # Error logging
├── test_solver.py       # Unit tests
├── build.spec           # PyInstaller configuration
└── requirements.txt     # Dependencies
```

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

### Build Package
```bash
pip install pyinstaller
python -m PyInstaller build.spec --clean
```

### Output
- `dist/AeroVision_CFD/` - Folder distribution
- `dist/AeroVision_CFD_Single.exe` - Single file executable (optional)

## Testing

```bash
python -m pytest test_solver.py -v
```

## Troubleshooting

### GPU Issues
If you have an NVIDIA GPU but cupy doesn't work:
1. Install CUDA Toolkit from NVIDIA website
2. Install matching cupy version: `pip install cupy-cuda12x`

### Performance Tips
- Use higher resolution for more accurate results
- k-epsilon and k-omega SST are more accurate but slower
- Enable wall functions for better near-wall accuracy

## License

Internal use only.
