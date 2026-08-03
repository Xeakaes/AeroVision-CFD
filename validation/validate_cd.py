"""
Kure Cd dogrulama benchmark'i.

Gerçek pipeline (STL -> voxel mask -> potansiyel akis -> NS -> kuvvet entegrali)
ile kure suruklenme katsayisini hesaplar ve deneysel referansla (Cd ~ 0.47,
subcritical Re, 10^4 < Re < 2e5) karsilastirir.

Kullanim:
    python validation/validate_cd.py                 # tek vaka (medium)
    python validation/validate_cd.py --model k_epsilon
    python validation/validate_cd.py --mesh-study    # 3 cozunurluk + GCI
    python validation/validate_cd.py --quick         # dusuk cozunurluk smoke
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from solver import (
    create_voxel_mask,
    solve_potential_flow,
    solve_navier_stokes_full,
    compute_pressure_field,
    compute_force_coefficients,
)
from validation.sphere_stl import sphere_stl

REF_CD = 0.47  # deneysel referans (subcritical kure)

# (nx, ny, nz) cozunurluk kademeleri — tünel (4, 2, 2) m icin
RESOLUTIONS = {
    "quick": (60, 30, 30),
    "coarse": (70, 35, 35),
    "medium": (100, 50, 50),
    "fine": (140, 70, 70),
}
TUNNEL = (4.0, 2.0, 2.0)
# Merkezli koordinat sistemi: x in [-tx/2, tx/2], y in [-ty/2, ty/2], z in [-tz/2, tz/2]
CENTER = (-1.0, 0.0, 0.0)
RADIUS = 0.25
V_WIND = 10.0
MODEL_DEFAULT = "spalart_allmaras"
ITERS = {"quick": 10, "coarse": 30, "medium": 45, "fine": 60}


def run_case(res_key="medium", model=MODEL_DEFAULT, wind=10.0, verbose=True):
    """Tek cozunurlukte kure benchmark'i. Sonuc dict'i dondurur."""
    nx, ny, nz = RESOLUTIONS[res_key]
    iterations = ITERS[res_key]
    t0 = time.time()

    verts, i_idx, j_idx, k_idx = sphere_stl(RADIUS, CENTER, n_lat=48, n_lon=64)
    mask, solid_fraction, ok = create_voxel_mask(verts, i_idx, j_idx, k_idx, nx, ny, nz, TUNNEL)
    if not ok:
        raise RuntimeError("create_voxel_mask basarisiz")

    u, v, w = solve_potential_flow(mask, wind, nx, ny, nz, solid_fraction=solid_fraction, tunnel_dims=TUNNEL)
    u2, v2, w2, p = solve_navier_stokes_full(
        u,
        v,
        w,
        mask,
        wind,
        nx,
        ny,
        nz,
        iterations=iterations,
        turbulence_model=model,
        solid_fraction=solid_fraction,
        tunnel_dims=TUNNEL,
    )
    cp_field = compute_pressure_field(u2, v2, w2, wind)

    res = compute_force_coefficients(
        verts, i_idx, j_idx, k_idx, cp_field, u2, v2, w2, TUNNEL, nx, ny, nz, wind, mask=mask, p_field=p
    )

    A_front = np.pi * RADIUS**2  # kure icin frontal alan (A_ref bounding-box'tir)
    q_inf = 0.5 * 1.225 * wind**2
    cd_phys = res["drag_N"] / (q_inf * A_front) if np.isfinite(res["drag_N"]) else float("nan")
    Re = 1.225 * wind * (2 * RADIUS) / 1.5e-5

    result = {
        "res": res_key,
        "model": model,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "iters": iterations,
        "Re": float(Re),
        "Cd_solver": res["Cd"],
        "Cl_solver": res["Cl"],
        "Cd_physical": float(cd_phys),
        "drag_N": res["drag_N"],
        "A_ref": res["A_ref"],
        "A_frontal": float(A_front),
        "valid": bool(res["valid"]),
        "t_seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(
            f"[{res_key}] {model} | nx={nx} | Re={Re:.0f} | "
            f"Cd_physical={cd_phys:.3f} (ref {REF_CD}) | drag={res['drag_N']:.4f} N | {result['t_seconds']}s"
        )
    return result


def mesh_study(model=MODEL_DEFAULT, wind=10.0):
    """Uc cozunurlukte calistirip mesh bagimsizligini raporlar."""
    keys = ["coarse", "medium", "fine"]
    results = [run_case(k, model, wind) for k in keys]
    cds = [r["Cd_physical"] for r in results]
    print("\nMesh bagimsizlik (GCI yaklasik):")
    for k, c in zip(keys, cds, strict=False):
        print(f"  {k:8s}: Cd = {c:.4f}")
    if len(cds) == 3 and all(np.isfinite(c) for c in cds):
        f = (cds[2] - cds[1]) / (cds[1] - cds[0]) if abs(cds[1] - cds[0]) > 1e-9 else 0
        if 0 < f < 1:
            gci = 1.25 * abs(cds[1] - cds[0]) / (2**2 - 1)  # 2. derece varsayim
            print(f"  Cozum orani f={f:.2f} | yaklasik GCI={gci * 100:.2f}%")
        else:
            print(f"  Cozum orani f={f:.2f} (monotonik degil — mesh bagimsizligina ulasilmadi)")
    return results


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        choices=["spalart_allmaras", "k_epsilon", "k_omega_sst", "smagorinsky"],
    )
    ap.add_argument("--res", default="medium", choices=list(RESOLUTIONS))
    ap.add_argument("--mesh-study", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--wind", type=float, default=V_WIND, help="serbest akim hizi (m/s); Re'yi belirler")
    ap.add_argument("--out", default="validation/results/cd_sphere.json")
    args = ap.parse_args()

    if args.quick:
        args.res, args.mesh_study = "quick", False

    wind = args.wind
    results = mesh_study(args.model, wind) if args.mesh_study else [run_case(args.res, args.model, wind)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"ref_Cd": REF_CD, "results": results}, f, indent=2)
    print(f"\nKayit: {args.out}")


if __name__ == "__main__":
    main()
