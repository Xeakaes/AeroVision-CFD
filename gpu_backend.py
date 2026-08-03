"""
AeroVision CFD - GPU/CPU Backend Secimi

Donanim tespiti ve array kutuphanesi (numpy/cupy) durumunu tek noktada tutar.
Runtime'da detect_gpu() ile backend degistirilebilir; tum tuketiciler bu
modulun atributlerini okur (import-time binding sorunu yasanmaz).

Oncelik sirasi:
  1. NVIDIA  → cupy (tam GPU hesaplama, fp64)
  2. Intel   → PyOpenCL (fp32 array + CPU fp64 CG solver)
  3. AMD     → PyOpenCL (fp32 array + CPU fp64 CG solver)
  4. Hicbiri → numpy (CPU)
"""

import numpy as np
import zlib

GPU_BACKEND = "numpy"
GPU_AVAILABLE = False
GPU_VENDOR = None
GPU_DEVICE_NAME = None

# Global referanslar — detect_gpu() icinde doldurulur
cp = np  # varsayilan: numpy
cu_csr = None
cu_diags = None
cu_cg = None
_cl_backend = None  # PyOpenCL backend referansi
LAST_ERROR = None  # son backend secim hatasi (teshis icin)


def detect_gpu(selection="auto"):
    """GPU donanimini tespit et ve uygun kutuphaneyi sec.

    Args:
        selection: "auto", "cpu", "nvidia", "amd", "intel"
    """
    global GPU_BACKEND, GPU_AVAILABLE, GPU_VENDOR, GPU_DEVICE_NAME, GPU
    global cp, cu_csr, cu_diags, cu_cg, _cl_backend, LAST_ERROR

    # ── CPU secildiyse ────────────────────────────────────────────────────
    if selection == "cpu":
        GPU_BACKEND = "numpy"
        GPU_VENDOR = "CPU"
        GPU_AVAILABLE = False
        GPU_DEVICE_NAME = "CPU (manuel secim)"
        cp = np
        cu_csr = None
        cu_diags = None
        cu_cg = None
        GPU = GPU_AVAILABLE
        print("[GPU] CPU secildi.")
        return

    # ── NVIDIA GPU — cupy ─────────────────────────────────────────────────
    if selection in ("auto", "nvidia"):
        try:
            import cupy as _cp
            import os

            if "CUDA_PATH" not in os.environ:
                base = r"C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA"
                if os.path.isdir(base):
                    versions = sorted(os.listdir(base), reverse=True)
                    if versions:
                        os.environ["CUDA_PATH"] = os.path.join(base, versions[0])
            _cp.cuda.Device(0).use()
            _ = _cp.array([1.0]) * 2.0
            from cupyx.scipy.sparse import csr_matrix as _cu_csr
            from cupyx.scipy.sparse import diags as _cu_diags
            from cupyx.scipy.sparse.linalg import cg as _cu_cg

            cp = _cp
            cu_csr = _cu_csr
            cu_diags = _cu_diags
            cu_cg = _cu_cg
            GPU_BACKEND = "cupy"
            GPU_AVAILABLE = True
            GPU_VENDOR = "NVIDIA"
            GPU_DEVICE_NAME = "NVIDIA GPU (cupy)"
            GPU = GPU_AVAILABLE
            print("[GPU] NVIDIA GPU bulundu — cupy modu etkin.")
            return
        except Exception:
            if selection == "nvidia":
                print("[GPU] NVIDIA GPU bulunamadi (cupy yuklenemedi).")

    # ── Intel/AMD GPU — PyOpenCL ──────────────────────────────────────────
    if selection in ("auto", "intel", "amd"):
        try:
            import cl_backend as _cl

            _cl.init(device_preference=selection if selection != "auto" else "auto")
            _cl_backend = _cl
            # cp = np olmali — slice islemleri PyOpenCL ile uyumsuz
            cp = np
            GPU_BACKEND = "opencl"
            GPU_AVAILABLE = True
            GPU_VENDOR = _cl.get_device_name()
            GPU_DEVICE_NAME = f"OpenCL GPU — {_cl.get_device_name()}"
            fp64_info = "fp64" if _cl.get_has_fp64() else "fp32"
            GPU = GPU_AVAILABLE
            print(f"[GPU] OpenCL GPU bulundu — {_cl.get_device_name()} ({fp64_info}).")
            return
        except Exception as e:
            if selection in ("intel", "amd"):
                print(f"[GPU] {selection.upper()} GPU bulunamadi (opencl: {e})")
            LAST_ERROR = f"opencl init: {e}"

    # ── Hic GPU bulunamadi ────────────────────────────────────────────────
    if selection != "auto":
        GPU_VENDOR = selection.upper()
        GPU_DEVICE_NAME = f"{selection.upper()} (bulunamadi)"
    else:
        GPU_VENDOR = "None"
        GPU_DEVICE_NAME = "Yok"
    GPU_BACKEND = "numpy"
    GPU_AVAILABLE = False
    cp = np
    cu_csr = None
    cu_diags = None
    cu_cg = None
    GPU = GPU_AVAILABLE
    print("[GPU] GPU bulunamadi — CPU modu etkin.")


detect_gpu()

# Kisa erisim — runtime'da guncel kalmasi icin tum okumalar gpu_backend.
#   ustunden yapilir (ornegin gpu_backend.GPU); bu sabitler ise UI gibi
#   import-aninda bilgi goruntuleyen taraflar icindir.
GPU = GPU_AVAILABLE


def _to_numpy(arr):
    """Herhangi bir array'i guvenli sekilde numpy'a donustur."""
    if arr is None:
        return np.zeros(1)
    # cupy array
    if hasattr(arr, "get"):
        try:
            return arr.get()
        except Exception:
            return np.asarray(arr)
    # torch tensor (gelecek icin)
    if hasattr(arr, "cpu") and hasattr(arr, "numpy"):
        try:
            return arr.cpu().numpy()
        except Exception:
            return np.asarray(arr)
    return np.asarray(arr)


def _hash_mask(mask_g):
    """Mask array'inin icerik ozeti — cache anahtari olusturmak icin."""
    mask_np = _to_numpy(mask_g) if GPU_AVAILABLE else np.asarray(mask_g)
    return zlib.crc32(mask_np.tobytes())


def get_gpu_summary():
    """UI'da gosterilecek GPU bilgi ozeti."""
    return {
        "backend": GPU_BACKEND,
        "available": GPU_AVAILABLE,
        "vendor": GPU_VENDOR,
        "device": GPU_DEVICE_NAME,
    }
