"""
NACA 0012 kanat Cl/Cd dogrulama benchmark'i.

Gerçek pipeline (STL -> voxel mask -> potansiyel akis -> NS -> kuvvet entegrali)
ile NACA 0012 kaldirma/suruklenme katsayilarini hesaplar ve literatur
referanslariyla (XFOIL/deney, Re ~ 5e5) karsilastirir.

Kullanim:
    python validation/validate_airfoil.py                 # alpha=8, medium
    python validation/validate_airfoil.py --alpha 4 --res coarse
    python validation/validate_airfoil.py --sweep          # 0..12 derece tara
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
from validation.naca0012_stl import naca0012_wing

RHO = 1.225
NU = 1.5e-5

# (nx, ny, nz) cozunurlukler — tünel (4, 2, 2) m icin
RESOLUTIONS = {
    "quick": (60, 30, 30),
    "coarse": (80, 40, 40),
    "medium": (100, 50, 50),
}
TUNNEL = (4.0, 2.0, 2.0)
CENTER = (-0.8, 0.0, 0.0)  # kanat merkezi (merkezli koordinat sistemi)
CHORD = 0.6  # kiriş [m] — voxel cozunurlugu icin buyuk tutulur
SPAN = 2.0  # kanat açıklığı [m] (tam tünel genişliği)
V_WIND = 10.0  # Re ~ 4.9e5 (chord=0.6)
MODEL_DEFAULT = "spalart_allmaras"
ITERS = {"quick": 20, "coarse": 40, "medium": 60}

# Deneysel/XFOIL referanslari (Re ~ 2.5e5 - 5e5, linerer bolge)
REF_CL = {0: 0.00, 2: 0.24, 4: 0.50, 6: 0.74, 8: 0.90, 10: 1.05, 12: 1.18}
REF_CD = {0: 0.006, 4: 0.007, 8: 0.010, 12: 0.025}
CL_TOL = 0.25  # %25 tolerans (voxel geometri + duvar etkisi)

A_REF = SPAN * CHORD


def ref_interp(table, alpha_deg):
    keys = sorted(table)
    if alpha_deg <= keys[0]:
        return table[keys[0]]
    if alpha_deg >= keys[-1]:
        return table[keys[-1]]
    for a1, a2 in zip(keys, keys[1:], strict=False):
        if a1 <= alpha_deg <= a2:
            t = (alpha_deg - a1) / (a2 - a1)
            return table[a1] + t * (table[a2] - table[a1])
    return table[keys[-1]]


def run_case(alpha_deg=8.0, res_key="medium", model=MODEL_DEFAULT, verbose=True):
    nx, ny, nz = RESOLUTIONS[res_key]
    iterations = ITERS[res_key]
    t0 = time.time()

    verts, i_idx, j_idx, k_idx = naca0012_wing(
        c=CHORD, alpha_deg=alpha_deg, span=SPAN, center=CENTER, n_pts=80
    )
    mask, solid_fraction, ok = create_voxel_mask(verts, i_idx, j_idx, k_idx, nx, ny, nz, TUNNEL)
    if not ok:
        raise RuntimeError("create_voxel_mask basarisiz")

    u, v, w = solve_potential_flow(
        mask, V_WIND, nx, ny, nz, solid_fraction=solid_fraction, tunnel_dims=TUNNEL
    )
    u2, v2, w2, p = solve_navier_stokes_full(
        u,
        v,
        w,
        mask,
        V_WIND,
        nx,
        ny,
        nz,
        iterations=iterations,
        turbulence_model=model,
        solid_fraction=solid_fraction,
        tunnel_dims=TUNNEL,
    )
    cp_field = compute_pressure_field(u2, v2, w2, V_WIND)

    res = compute_force_coefficients(
        verts, i_idx, j_idx, k_idx, cp_field, u2, v2, w2, TUNNEL, nx, ny, nz, V_WIND, mask=mask, p_field=p
    )

    q_inf = 0.5 * RHO * V_WIND**2
    Cl = res["lift_N"] / (q_inf * A_REF) if np.isfinite(res["lift_N"]) else float("nan")
    Cd = res["drag_N"] / (q_inf * A_REF) if np.isfinite(res["drag_N"]) else float("nan")
    Re = RHO * V_WIND * CHORD / NU

    ref_cl = ref_interp(REF_CL, alpha_deg)
    ref_cd = ref_interp(REF_CD, alpha_deg)
    cl_ok = abs(Cl - ref_cl) <= CL_TOL * abs(ref_cl) + 0.05
    rel = (Cl - ref_cl) / ref_cl if ref_cl != 0 else float("nan")

    result = {
        "alpha": alpha_deg,
        "model": model,
        "res": res_key,
        "nx": nx,
        "ny": ny,
        "nz": nz,
        "iters": iterations,
        "Re": float(Re),
        "Cl": float(Cl),
        "Cd": float(Cd),
        "ref_Cl": ref_cl,
        "ref_Cd": ref_cd,
        "Cl_rel_err": float(rel),
        "Cl_ok": bool(cl_ok),
        "drag_N": res["drag_N"],
        "lift_N": res["lift_N"],
        "valid": bool(res["valid"]),
        "t_seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(
            f"[{res_key}] a={alpha_deg} deg | nx={nx} | Re={Re:.0f} | "
            f"Cl={Cl:.3f} (ref {ref_cl}) err=%{rel * 100:.1f} {'OK' if cl_ok else 'FAIL'} | "
            f"Cd={Cd:.4f} (ref {ref_cd}) | {result['t_seconds']}s"
        )
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument(
        "--model",
        default=MODEL_DEFAULT,
        choices=["spalart_allmaras", "k_epsilon", "k_omega_sst", "smagorinsky"],
    )
    ap.add_argument("--res", default="medium", choices=list(RESOLUTIONS))
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--out", default="validation/results/airfoil_naca0012.json")
    args = ap.parse_args()

    if args.sweep:
        results = [run_case(a, args.res, args.model) for a in [0, 4, 8, 12]]
    else:
        results = [run_case(args.alpha, args.res, args.model)]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as f:
        json.dump({"ref_Cl_table": REF_CL, "results": results}, f, indent=2)
    print(f"\nKayit: {args.out}")


if __name__ == "__main__":
    main()
