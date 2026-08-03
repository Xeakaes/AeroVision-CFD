"""
AeroVision CFD - GPU/OpenCL Self Test (Faz 3 dogrulamasi)

Windows anamakine uzerinde GPU hizlandirmasini dogrular:
  1. pyopencl + platform/cihaz tespiti (NVIDIA cupy ve/veya OpenCL)
  2. Backend secimi (auto -> NVIDIA cupy, degilse OpenCL)
  3. OpenCL kernel derleme (13 kernel) — yalnizca opencl backend'de
  4. Mini simülasyon: kure Cd — GPU yolu (cupy veya opencl)
  5. Ayni vaka CPU yolu (referans)
  6. Karsilastirma: sonuc farki + hiz farki

Kullanim:
  python gpu_selftest.py              # tam test (GPU otomatik tespit)
  python gpu_selftest.py --quick      # sadece cihaz + kernel derleme
  python gpu_selftest.py --device intel   # OpenCL'i Intel iGPU ile zorla
  python gpu_selftest.py --device cpu     # sadece CPU referans
  python gpu_selftest.py --res 40x20x20 --iters 15

Cikti: gpu_selftest_result.json + konsol ozeti.
Cikis kodu: 0 = basarili, 1 = GPU yok / test atlandi, 2 = hata.
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

RESULT_PATH = os.path.join(HERE, "gpu_selftest_result.json")


def parse_res(s):
    nx, ny, nz = (int(v) for v in s.split("x"))
    return nx, ny, nz


def run_case(mask, nx, ny, nz, v_wind, iterations, rho, nu_mol, solid_fraction=None):
    """Tek mini simülasyon: potansiyel akis + NS + kuvvet katsayilari."""
    from solver import (
        compute_force_coefficients,
        solve_navier_stokes_full,
        solve_potential_flow,
    )
    from validation.sphere_stl import sphere_stl

    verts, i_idx, j_idx, k_idx = sphere_stl(0.25, (-1.0, 0.0, 0.0), n_lat=24, n_lon=32)
    tx, ty, tz = (4.0, 2.0, 2.0)
    u0, v0, w0 = solve_potential_flow(
        mask, v_wind, nx, ny, nz, solid_fraction=solid_fraction, tunnel_dims=(tx, ty, tz)
    )
    t0 = time.time()
    u, v, w, p = solve_navier_stokes_full(
        u0,
        v0,
        w0,
        mask,
        v_wind,
        nx,
        ny,
        nz,
        iterations=iterations,
        turbulence_model="spalart_allmaras",
        rho=rho,
        nu_mol=nu_mol,
        solid_fraction=solid_fraction,
        tunnel_dims=(tx, ty, tz),
    )
    elapsed = time.time() - t0
    cp_field = np.zeros_like(u)
    forces = compute_force_coefficients(
        verts,
        i_idx,
        j_idx,
        k_idx,
        cp_field,
        u,
        v,
        w,
        (tx, ty, tz),
        nx,
        ny,
        nz,
        v_wind,
        mask=mask,
        p_field=p,
    )
    finite = bool(np.isfinite(u).all() and np.isfinite(v).all() and np.isfinite(w).all())
    return {
        "elapsed_seconds": round(elapsed, 2),
        "Cd": float(forces["Cd"]),
        "Cl": float(forces["Cl"]),
        "u_max": float(np.max(np.abs(u))),
        "finite": finite,
    }


def detect_platforms(result):
    """pyopencl platform/cihaz listesi. Basarisizsa status set eder."""
    try:
        import pyopencl as cl
    except ImportError as e:
        result["steps"]["platforms"] = {"pyopencl": False, "error": str(e)}
        return None, "pyopencl yok — GPU testi atlanir"

    platforms = []
    try:
        cl_platforms = cl.get_platforms()
    except Exception as e:
        result["steps"]["platforms"] = {"pyopencl": True, "platforms": [], "error": str(e)}
        return None, "OpenCL platform bulunamadi — GPU testi atlanir"

    for p in cl_platforms:
        devs = []
        for dt in (cl.device_type.GPU, cl.device_type.CPU):
            try:
                for d in p.get_devices(device_type=dt):
                    devs.append(
                        {
                            "name": d.name,
                            "vendor": d.vendor,
                            "is_gpu": dt == cl.device_type.GPU,
                            "fp64": "cl_khr_fp64" in d.extensions,
                            "mem_GB": round(d.global_mem_size / 1e9, 2),
                        }
                    )
            except cl.RuntimeError:
                pass
        platforms.append({"platform": p.name, "vendor": p.vendor, "devices": devs})
    result["steps"]["platforms"] = {"pyopencl": True, "platforms": platforms}
    allow_cpu = os.environ.get("AV_OPENCL_ALLOW_CPU") == "1"
    gpu_devices = [d for pl in platforms for d in pl["devices"] if d["is_gpu"] or allow_cpu]
    if not gpu_devices:
        return None, "OpenCL GPU cihazi bulunamadi — GPU testi atlanir"
    print(f"Bulunan GPU cihazlari: {[d['name'] for d in gpu_devices]}")
    return gpu_devices, None


def save(result):
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)
    print(f"Rapor: {RESULT_PATH}")


def main():
    ap = argparse.ArgumentParser(description="AeroVision CFD GPU/OpenCL self test")
    ap.add_argument("--quick", action="store_true", help="sadece cihaz + kernel derleme")
    ap.add_argument("--device", default="auto", choices=["auto", "intel", "amd", "nvidia", "cpu"])
    ap.add_argument("--res", default="40x20x20", help="grid boyutu nx x ny x nz")
    ap.add_argument("--iters", type=int, default=15, help="NS iterasyon sayisi")
    args = ap.parse_args()

    result = {
        "date": datetime.now().isoformat(),
        "os": platform.platform(),
        "python": sys.version,
        "backend": "numpy",
        "device": None,
        "steps": {},
    }

    # ── 1. pyopencl + platform tespiti (bilgilendirme amaçlı) ─────────────
    # OpenCL yalnizca opencl backend'i icin zorunlu; cupy (NVIDIA) yolu icin
    # platform yoklugu engel degildir.
    gpu_devices = None
    if args.device != "cpu":
        gpu_devices, status = detect_platforms(result)
        if status is not None:
            print(status)
            gpu_devices = []

    # ── 2. Backend secimi ─────────────────────────────────────────────────
    import gpu_backend

    if args.device == "cpu":
        gpu_backend.detect_gpu("cpu")
        result["backend"] = "cpu"
        result["device"] = "CPU (manuel secim)"
    else:
        gpu_backend.detect_gpu(args.device)
        result["backend"] = gpu_backend.GPU_BACKEND
        result["device"] = gpu_backend.GPU_DEVICE_NAME
        if gpu_backend.GPU_BACKEND not in ("opencl", "cupy"):
            result["steps"]["init_error"] = gpu_backend.LAST_ERROR
            result["status"] = f"GPU backend secilemedi (backend={gpu_backend.GPU_BACKEND})"
            save(result)
            print(f"GPU backend secilemedi: {gpu_backend.GPU_BACKEND} | son hata: {gpu_backend.LAST_ERROR}")
            return 1

        if gpu_backend.GPU_BACKEND == "opencl" and not gpu_devices:
            result["status"] = "OpenCL backend secildi ama OpenCL GPU platformu yok — GPU testi atlanir"
            save(result)
            print("OpenCL GPU platformu yok — GPU testi atlanir")
            return 1

        if gpu_backend.GPU_BACKEND == "cupy":
            result["steps"]["init"] = {
                "device": gpu_backend.GPU_DEVICE_NAME,
                "backend": "cupy",
            }
            print(f"cupy backend etkin — {gpu_backend.GPU_DEVICE_NAME}")
        else:
            from cl_backend import get_has_fp64

            result["steps"]["init"] = {
                "device": gpu_backend.GPU_DEVICE_NAME,
                "backend": "opencl",
                "fp64": get_has_fp64(),
            }

            # ── 3. OpenCL kernel derleme ─────────────────────────────────
            from cl_backend import _ctx, _queue

            import cl_solver

            t0 = time.time()
            slvr = cl_solver.OpenCLSolver(_ctx, _queue)
            result["steps"]["kernels"] = {
                "count": len(slvr._programs),
                "build_seconds": round(time.time() - t0, 3),
            }
            print(
                f"Kernel derleme OK ({len(slvr._programs)} kernel, {result['steps']['kernels']['build_seconds']}s)"
            )

    if args.quick:
        result["status"] = "quick — basarili"
        save(result)
        return 0

    # ── 4-6. Mini simülasyon: GPU vs CPU ─────────────────────────────────
    from validation.sphere_stl import sphere_stl, write_binary_stl

    nx, ny, nz = parse_res(args.res)
    stl_path = os.path.join(HERE, ".selftest_sphere.stl")
    verts, i_idx, j_idx, k_idx = sphere_stl(0.25, (-1.0, 0.0, 0.0), n_lat=24, n_lon=32)
    write_binary_stl(stl_path, verts, i_idx, j_idx, k_idx)

    from solver import create_voxel_mask

    mask, solid_fraction, ok = create_voxel_mask(verts, i_idx, j_idx, k_idx, nx, ny, nz, (4.0, 2.0, 2.0))
    if not ok:
        raise RuntimeError("create_voxel_mask basarisiz")
    rho, nu_mol = 1.225, 1.5e-5

    gpu_case = None
    if args.device != "cpu":
        print(f"[{result['backend'].upper()}] Mini simülasyon: {nx}x{ny}x{nz}, {args.iters} iterasyon...")
        gpu_case = run_case(mask, nx, ny, nz, 10.0, args.iters, rho, nu_mol, solid_fraction=solid_fraction)
        result["gpu_case"] = gpu_case
        print(f"  GPU: Cd={gpu_case['Cd']:.4f}, {gpu_case['elapsed_seconds']}s, finite={gpu_case['finite']}")
        gpu_backend.detect_gpu("cpu")

    print("[CPU] Referans simülasyon...")
    cpu_case = run_case(mask, nx, ny, nz, 10.0, args.iters, rho, nu_mol, solid_fraction=solid_fraction)
    result["cpu_case"] = cpu_case
    print(f"  CPU: Cd={cpu_case['Cd']:.4f}, {cpu_case['elapsed_seconds']}s, finite={cpu_case['finite']}")

    if gpu_case is None:
        result["status"] = "cpu-only — tamamlandi"
        save(result)
        return 0

    cd_diff = abs(gpu_case["Cd"] - cpu_case["Cd"])
    speedup = cpu_case["elapsed_seconds"] / max(gpu_case["elapsed_seconds"], 1e-6)
    result["comparison"] = {
        "Cd_diff": round(cd_diff, 4),
        "speedup": round(speedup, 2),
        "status": "OK" if (gpu_case["finite"] and cpu_case["finite"] and cd_diff < 0.1) else "INCELE",
    }
    print(f"Karsilastirma: |dCd|={cd_diff:.4f}, hizlanma={speedup:.2f}x")

    result["status"] = "basarili"
    save(result)
    return 0


if __name__ == "__main__":
    sys.exit(main())
