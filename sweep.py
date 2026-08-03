"""
Parametre taramasi (sweep) CLI — wind x yaw x akiskan kombinasyonlari.

Her vaka: STL -> voxel mask -> potansiyel akis -> NS -> kuvvet entegrali.
Sonuclar JSON + CSV + Markdown rapor olarak kaydedilir; u(y) hiz profili
grafigi istege bagli cizilir (--profile).

Kullanim:
    python sweep.py --stl model.stl --wind 5:30:5 --yaw 0:90:30 --fluid air,water
    python sweep.py --stl model.stl --wind 10 --yaw 0:90:45 --fluid co2 --profile
"""

import argparse
import json
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from solver import (
    TUNNEL_DIMS,
    create_voxel_mask,
    solve_potential_flow,
    solve_navier_stokes_full,
    compute_pressure_field,
    compute_force_coefficients,
    rotate_vertices,
    apply_offset,
)

FLUIDS = {
    "air": (1.225, 1.5e-5),
    "water": (998.2, 1.004e-6),
    "co2": (1.87, 1.47e-5),
}


def parse_range(spec, default):
    """'baslangic:bitis:adim' veya tek deger -> liste."""
    parts = str(spec).split(":")
    if len(parts) == 1:
        return [float(parts[0])]
    start, stop = float(parts[0]), float(parts[1])
    step = float(parts[2]) if len(parts) > 2 else (stop - start if stop != start else 1.0)
    if stop < start or step <= 0:
        raise ValueError(f"Gecersiz aralik: {spec}")
    vals = [start]
    while vals[-1] + step <= stop + 1e-9:
        vals.append(round(vals[-1] + step, 6))
    return vals


def read_stl(path):
    """Binary (veya ASCII) STL okur -> (verts, i_idx, j_idx, k_idx)."""
    with open(path, "rb") as f:
        data = f.read()
    if data[:5].lower() == b"solid" and b"facet" in data[:500]:
        return _read_ascii_stl(data)
    n = int.from_bytes(data[80:84], "little")
    buf = np.frombuffer(data[84 : 84 + n * 50], dtype=np.uint8).reshape(n, 50)
    tris = buf[:, :48].copy().view("<f4").reshape(n, 12)  # attr (2B) atilir
    v = tris[:, 3:].reshape(n, 3, 3)
    verts = v.reshape(-1, 3)
    base = np.arange(n) * 3
    return verts, base, base + 1, base + 2


def _read_ascii_stl(data):
    import re

    coords = re.findall(rb"vertex\s+([-\d.eE+]+)\s+([-\d.eE+]+)\s+([-\d.eE+]+)", data)
    v = np.array([[float(x) for x in c] for c in coords], dtype=np.float32)
    verts = v.reshape(-1, 3)
    base = np.arange(len(verts) // 3) * 3
    return verts, base, base + 1, base + 2


def run_case(
    verts,
    i_idx,
    j_idx,
    k_idx,
    wind,
    yaw,
    fluid_key,
    res=(60, 30, 30),
    iterations=30,
    tunnel=TUNNEL_DIMS,
    verbose=True,
):
    """Tek vaka: donus + mask + potential + NS + kuvvetler."""
    rho, nu_mol = FLUIDS[fluid_key]
    nx, ny, nz = res
    t0 = time.time()

    v = rotate_vertices(verts, yaw, 0.0, 0.0)
    mask, solid_fraction, ok = create_voxel_mask(v, i_idx, j_idx, k_idx, nx, ny, nz, tunnel)
    if not ok:
        raise RuntimeError("create_voxel_mask basarisiz")

    u, v_arr, w = solve_potential_flow(
        mask, wind, nx, ny, nz, solid_fraction=solid_fraction, tunnel_dims=tunnel
    )
    u, v_arr, w, p = solve_navier_stokes_full(
        u,
        v_arr,
        w,
        mask,
        wind,
        nx,
        ny,
        nz,
        solid_fraction=solid_fraction,
        iterations=iterations,
        turbulence_model="spalart_allmaras",
        rho=rho,
        nu_mol=nu_mol,
        tunnel_dims=tunnel,
    )

    cp = compute_pressure_field(u, v_arr, w, wind)
    res_f = compute_force_coefficients(
        v,
        i_idx,
        j_idx,
        k_idx,
        cp,
        u,
        v_arr,
        w,
        tunnel,
        nx,
        ny,
        nz,
        wind,
        mask=mask,
        p_field=p,
        rho=rho,
        nu_mol=nu_mol,
    )

    # u(y) profili: orta duzlem (y ekseni boyunca), x = kurek ortasi civari
    cx = int(nx / 2)
    u_profile = [float(u[cx, jy, nz // 2]) for jy in range(ny)]
    y_coords = [float(-tunnel[1] / 2 + jy * tunnel[1] / (ny - 1)) for jy in range(ny)]

    Re = rho * wind * (tunnel[0] * 0.1) / nu_mol  # karakteristik uzunluk ~ model olcegi
    result = {
        "name": f"{fluid_key}_yaw{yaw:g}_wind{wind:g}",
        "fluid": fluid_key,
        "yaw": yaw,
        "wind": wind,
        "Cd": res_f["Cd"],
        "Cl": res_f["Cl"],
        "drag_N": res_f["drag_N"],
        "lift_N": res_f["lift_N"],
        "A_ref": res_f["A_ref"],
        "Re": Re,
        "valid": res_f["valid"],
        "u_profile": u_profile,
        "y_coords": y_coords,
        "t_seconds": round(time.time() - t0, 1),
    }
    if verbose:
        print(
            f"  [{result['name']}] Cd={result['Cd']:.4f} Cl={result['Cl']:.4f} "
            f"Fd={result['drag_N']:.2f}N Fl={result['lift_N']:.2f}N | {result['t_seconds']}s"
        )
    return result


def write_report(results, out_dir, tunnel):
    """Markdown rapor + CSV yazar."""
    md = [
        "# AeroVision CFD — Parametre Taramasi Raporu",
        f"\n_Ozetimec: {time.strftime('%Y-%m-%d %H:%M')}_\n",
        f"Toplam vaka: **{len(results)}** | Tunel: {tunnel}\n",
        "\n| Vaka | Akiskan | Yaw | Wind (m/s) | Cd | Cl | Fd (N) | Fl (N) | Sure (s) |",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    csv_rows = [["vaka", "akiskan", "yaw", "wind", "Cd", "Cl", "Fd_N", "Fl_N", "sure_s"]]
    for r in results:
        md.append(
            f"| {r['name']} | {r['fluid']} | {r['yaw']:g} | {r['wind']:g} | "
            f"{r['Cd']:.4f} | {r['Cl']:.4f} | {r['drag_N']:.2f} | {r['lift_N']:.2f} | {r['t_seconds']} |"
        )
        csv_rows.append(
            [
                r["name"],
                r["fluid"],
                f"{r['yaw']:g}",
                f"{r['wind']:g}",
                f"{r['Cd']:.6f}",
                f"{r['Cl']:.6f}",
                f"{r['drag_N']:.4f}",
                f"{r['lift_N']:.4f}",
                f"{r['t_seconds']}",
            ]
        )

    # En iyi (min Cd) vaka
    valid = [r for r in results if r["valid"] and np.isfinite(r["Cd"])]
    if valid:
        best = min(valid, key=lambda r: r["Cd"])
        md.append(
            f"\n## En Dusuk Suruklenme\n\n**{best['name']}**: "
            f"Cd = {best['Cd']:.4f}, Fd = {best['drag_N']:.2f} N\n"
        )

    md_path = os.path.join(out_dir, "sweep_report.md")
    csv_path = os.path.join(out_dir, "sweep_results.csv")
    with open(md_path, "w") as f:
        f.write("\n".join(md) + "\n")
    with open(csv_path, "w", newline="") as f:
        import csv

        csv.writer(f).writerows(csv_rows)
    return md_path, csv_path


def plot_profiles(results, out_dir):
    """u(y) hiz profillerini tek PNG'de cizer (matplotlib Agg)."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    fig.patch.set_facecolor("#1e1e1e")
    ax.set_facecolor("#1e1e1e")
    for r in results:
        if r["valid"] and np.isfinite(r["Cd"]):
            ax.plot(r["u_profile"], r["y_coords"], label=f"{r['name']} (Cd={r['Cd']:.3f})")
    ax.set_title("u(y) Hiz Profili (orta duzlem)", color="white")
    ax.set_xlabel("u (m/s)", color="white")
    ax.set_ylabel("y (m)", color="white")
    ax.tick_params(colors="white")
    for spine in ax.spines.values():
        spine.set_color("#444")
    ax.legend(fontsize=7, facecolor="#1e1e1e", edgecolor="#444", labelcolor="white")
    fig.tight_layout()
    out = os.path.join(out_dir, "sweep_profiles.png")
    fig.savefig(out, facecolor="#1e1e1e")
    plt.close(fig)
    return out


def main():
    ap = argparse.ArgumentParser(description="AeroVision CFD parametre taramasi")
    ap.add_argument("--stl", required=True, help="STL dosyasi")
    ap.add_argument("--wind", default="10", help="ornek: 5:30:5")
    ap.add_argument("--yaw", default="0", help="ornek: 0:90:30")
    ap.add_argument("--fluid", default="air", help="virgulle ayrilmis: air,water,co2")
    ap.add_argument("--res", default="60x30x30", help="nx x ny x nz")
    ap.add_argument("--iters", type=int, default=30)
    ap.add_argument("--profile", action="store_true", help="u(y) grafigi ciz")
    ap.add_argument("--out", default="results")
    ap.add_argument("--ox", type=float, default=0.0)
    ap.add_argument("--oy", type=float, default=0.0)
    ap.add_argument("--oz", type=float, default=0.0)
    args = ap.parse_args()

    try:
        res = tuple(int(x) for x in args.res.split("x"))
        assert len(res) == 3
    except Exception:
        raise SystemExit("--res 'nx x ny x nz' biciminde olmali (ornek: 60x30x30)") from None

    winds = parse_range(args.wind, 10.0)
    yaws = parse_range(args.yaw, 0.0)
    fluids = [f.strip().lower() for f in args.fluid.split(",")]
    for f in fluids:
        if f not in FLUIDS:
            raise SystemExit(f"Bilinmeyen akiskan: {f} (air, water, co2)")

    print(f"STL: {args.stl} | wind={winds} | yaw={yaws} | fluid={fluids} | res={res}")
    verts, i_idx, j_idx, k_idx = read_stl(args.stl)
    verts = apply_offset(verts, args.ox, args.oy, args.oz)
    print(f"Geometri: {len(verts)} kok nokta, {len(i_idx)} ucgen\n")

    results = []
    for fluid in fluids:
        for yaw in yaws:
            for wind in winds:
                results.append(
                    run_case(verts, i_idx, j_idx, k_idx, wind, yaw, fluid, res, args.iters, verbose=True)
                )

    os.makedirs(args.out, exist_ok=True)
    with open(os.path.join(args.out, "sweep_results.json"), "w") as f:
        json.dump({"tunnel": list(TUNNEL_DIMS), "results": results}, f, indent=2)

    md_path, csv_path = write_report(results, args.out, TUNNEL_DIMS)
    print(f"\nRapor: {md_path}\nCSV:   {csv_path}")
    if args.profile:
        png = plot_profiles(results, args.out)
        print(f"Profil: {png}")


if __name__ == "__main__":
    main()
