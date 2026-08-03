"""AeroVision CFD - GPU benchmark (Faz 6)

CPU / cupy / OpenCL yollarini buyuk gridde kiyaslar ve zaman dagilimini
profillestirir (NS_RHS, Poisson, transferler, turbulans).

Kullanim:
  python benchmark.py                 # tum backend'ler
  python benchmark.py --backends cpu  # secili
  python benchmark.py --res 150x60x60 --iters 30
  AV_OPENCL_ALLOW_CPU=1 python benchmark.py --backends opencl  # pocl (CPU OpenCL)

Cikti: benchmarks/results.json + konsol ozeti
"""

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime

HERE = os.path.dirname(os.path.abspath(__file__))
RESULT_DIR = os.path.join(HERE, "benchmarks")
RESULT_PATH = os.path.join(RESULT_DIR, "results.json")


def parse_res(s):
    parts = s.lower().split("x")
    if len(parts) != 3:
        raise ValueError(f"grid '{s}' gecersiz — nx x ny x nz beklenir")
    return int(parts[0]), int(parts[1]), int(parts[2])


def load_stl_case(nx, ny, nz):
    """Gercek STL kullanir; yoksa kure uretir."""
    from validation.sphere_stl import sphere_stl, write_binary_stl

    stl = None
    for name in os.listdir(HERE):
        if name.lower().endswith(".stl") and not name.startswith(".selftest"):
            stl = os.path.join(HERE, name)
            break
    if stl is not None and os.path.getsize(stl) > 0:
        from solver import create_voxel_mask
        from sweep import read_stl

        verts, i_idx, j_idx, k_idx = read_stl(stl)
        mask, solid_fraction, ok = create_voxel_mask(verts, i_idx, j_idx, k_idx, nx, ny, nz, (4.0, 2.0, 2.0))
        if not ok:
            print(f"[bench] STL voxelize basarisiz ({stl}) — kureye donuluyor")
            verts, i_idx, j_idx, k_idx = sphere_stl(0.25, (-1.0, 0.0, 0.0), n_lat=24, n_lon=32)
            mask, solid_fraction, ok = create_voxel_mask(
                verts, i_idx, j_idx, k_idx, nx, ny, nz, (4.0, 2.0, 2.0)
            )
        print(f"[bench] STL: {os.path.basename(stl)} ({len(verts)} verts)")
    else:
        stl_path = os.path.join(HERE, ".bench_sphere.stl")
        verts, i_idx, j_idx, k_idx = sphere_stl(0.25, (-1.0, 0.0, 0.0), n_lat=24, n_lon=32)
        write_binary_stl(stl_path, verts, i_idx, j_idx, k_idx)
        mask, solid_fraction, ok = create_voxel_mask(verts, i_idx, j_idx, k_idx, nx, ny, nz, (4.0, 2.0, 2.0))
        print("[bench] STL yok — kure kullanildi")
    return verts, i_idx, j_idx, k_idx, mask, solid_fraction, stl


def run_ns(
    verts,
    i_idx,
    j_idx,
    k_idx,
    mask,
    solid_fraction,
    nx,
    ny,
    nz,
    v_wind,
    iterations,
    rho,
    nu_mol,
    detail=False,
):
    """NS kosisu; zaman dagilimiyla birlikte sonuc dondurur."""
    from solver import compute_force_coefficients, solve_navier_stokes_full, solve_potential_flow

    tx, ty, tz = (4.0, 2.0, 2.0)
    t0 = time.time()
    u0, v0, w0 = solve_potential_flow(
        mask, v_wind, nx, ny, nz, solid_fraction=solid_fraction, tunnel_dims=(tx, ty, tz)
    )
    t_potential = time.time() - t0

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
    t_ns = time.time() - t0

    t0 = time.time()
    cp_field = __import__("numpy").zeros_like(u)
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
    t_forces = time.time() - t0

    return {
        "Cd": forces["Cd"],
        "finite": bool(__import__("numpy").isfinite(u).all()),
        "t_potential": round(t_potential, 3),
        "t_ns": round(t_ns, 3),
        "t_forces": round(t_forces, 3),
        "t_total": round(t_potential + t_ns + t_forces, 3),
    }


def bench_backend(
    name, verts, i_idx, j_idx, k_idx, mask, solid_fraction, nx, ny, nz, v_wind, iterations, rho, nu_mol
):
    import gpu_backend

    selection = {"cpu": "cpu", "cupy": "nvidia", "opencl": "intel"}[name]
    gpu_backend.detect_gpu(selection)
    print(
        f"\n[bench] === {name.upper()} (backend={gpu_backend.GPU_BACKEND}, device={gpu_backend.GPU_DEVICE_NAME}) ==="
    )
    res = run_ns(
        verts, i_idx, j_idx, k_idx, mask, solid_fraction, nx, ny, nz, v_wind, iterations, rho, nu_mol
    )
    res["backend"] = name
    res["device"] = gpu_backend.GPU_DEVICE_NAME
    ncells = nx * ny * nz
    res["s_per_step_1e6cells"] = round(res["t_ns"] / iterations / (ncells / 1e6), 4)
    print(
        f"[bench] {name}: Cd={res['Cd']:.4f} finite={res['finite']} | NS={res['t_ns']}s "
        f"({res['s_per_step_1e6cells']} s/adim/milyon-hucre) | toplam={res['t_total']}s"
    )
    return res


def main():
    ap = argparse.ArgumentParser(description="AeroVision CFD GPU benchmark")
    ap.add_argument("--backends", default="cpu,cupy,opencl", help="virgulle ayrilmis: cpu,cupy,opencl")
    ap.add_argument("--res", default="150x60x60", help="grid nx x ny x nz")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--wind", type=float, default=10.0)
    args = ap.parse_args()

    nx, ny, nz = parse_res(args.res)
    verts, i_idx, j_idx, k_idx, mask, solid_fraction, stl = load_stl_case(nx, ny, nz)
    rho, nu_mol = 1.225, 1.5e-5

    results = {
        "date": datetime.now().isoformat(),
        "os": platform.platform(),
        "python": sys.version,
        "grid": f"{nx}x{ny}x{nz}",
        "iters": args.iters,
        "stl": os.path.basename(stl) if stl else "sphere",
        "backends": {},
    }

    for name in [b.strip() for b in args.backends.split(",") if b.strip()]:
        try:
            results["backends"][name] = bench_backend(
                name,
                verts,
                i_idx,
                j_idx,
                k_idx,
                mask,
                solid_fraction,
                nx,
                ny,
                nz,
                args.wind,
                args.iters,
                rho,
                nu_mol,
            )
        except Exception as e:
            import traceback

            traceback.print_exc()
            results["backends"][name] = {"error": str(e)}

    if "cpu" in results["backends"] and "cupy" in results["backends"]:
        c, g = results["backends"]["cpu"], results["backends"]["cupy"]
        if "t_ns" in c and "t_ns" in g and c["t_ns"] > 0:
            results["speedup_cupy_vs_cpu"] = round(c["t_ns"] / max(g["t_ns"], 1e-6), 2)
    if "cpu" in results["backends"] and "opencl" in results["backends"]:
        c, g = results["backends"]["cpu"], results["backends"]["opencl"]
        if "t_ns" in c and "t_ns" in g and c["t_ns"] > 0:
            results["speedup_opencl_vs_cpu"] = round(c["t_ns"] / max(g["t_ns"], 1e-6), 2)

    os.makedirs(RESULT_DIR, exist_ok=True)
    with open(RESULT_PATH, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nRapor: {RESULT_PATH}")


if __name__ == "__main__":
    sys.exit(main())
