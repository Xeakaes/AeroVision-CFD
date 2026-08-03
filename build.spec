# AeroVision CFD - PyInstaller build configuration
# Kullanim: python -m PyInstaller build.spec --clean
# Windows:  dist/AeroVision_CFD/AeroVision_CFD.exe
# Linux:    dist/AeroVision_CFD/aerovision-cfd

import sys

from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hidden_imports = (
    collect_submodules("PyQt6")
    + collect_submodules("pyvista")
    + collect_submodules("pyvistaqt")
    + collect_submodules("matplotlib")
    + collect_submodules("scipy.sparse")
    + collect_submodules("stl")
    + collect_submodules("pyopencl")
    + collect_submodules("cupy")
    + ["gpu_backend", "cl_backend", "cl_solver", "turbulence_models", "constants", "ui_components", "postprocess_ui", "crash_logger"]
)

datas = collect_data_files("pyvista") + collect_data_files("matplotlib")

a = Analysis(
    ["main.py"],
    pathex=["."],
    binaries=[],
    datas=datas,
    hiddenimports=hidden_imports,
    hookspath=[],
    runtime_hooks=[],
    excludes=["IPython", "jupyter", "pytest"],
    noarchive=False,
)

pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="AeroVision_CFD",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="AeroVision_CFD",
)
