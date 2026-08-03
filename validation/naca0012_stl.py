"""
Parametrik NACA 0012 kanat STL uretici — bilimsel dogrulama (Cl/Cd benchmark).

Iki kullanim modu:
  1. Dogrudan import:  `verts, i_idx, j_idx, k_idx = naca0012_wing(...)`
  2. STL dosyasi:      `python naca0012_stl.py --out naca0012.stl --alpha 8`
"""

import argparse
import os

import numpy as np


def naca0012_points(n_pts=60):
    """NACA 0012 profil koordinatlari (kiriş boyutunda, LE x=0, TE x=1).

    Ust yuzey: y > 0, alt yuzey: y < 0. Kapali TE (y(1) = 0).
    """
    beta = np.linspace(0.0, np.pi, n_pts)
    x_c = 0.5 * (1.0 - np.cos(beta))  # cos-spacing, LE'de yogun
    y_c = 0.6 * (0.2969 * np.sqrt(x_c) - 0.1260 * x_c - 0.3516 * x_c**2 + 0.2843 * x_c**3 - 0.1036 * x_c**4)
    pts_up = np.column_stack([x_c, y_c])
    pts_lo = np.column_stack([x_c[::-1], -y_c[::-1]])
    return np.vstack([pts_up, pts_lo])  # LE->TE (ust), TE->LE (alt)


def naca0012_wing(c=0.3, alpha_deg=8.0, span=2.0, center=(-0.5, 0.0, 0.0), n_pts=60):
    """NACA 0012 kanadi y-yonunde extrude eder (kaldirma +z ekseninde).

    Akis +x, kanat acikligi (span) y, kaldirma z.
    solver.compute_force_coefficients lift_N = F_total[2] (z) ile uyumlu.

    Args:
        c:        Kiris uzunlugu [m]
        alpha_deg: Hucum acisi [derece] — LE etrafinda x-z duzleminde doner
        span:     Kanat acikligi (y) [m]
        center:   Kanat merkezi (cx, cy, cz) [m]
        n_pts:    Profil nokta sayisi (ust+alt)

    Returns:
        verts: (n_tri*3, 3) float32 kok noktalari
        i_idx, j_idx, k_idx: her ucgenin kok nokta indeksleri (int32)
    """
    cx, cy, cz = center
    alpha = np.radians(alpha_deg)
    prof = naca0012_points(n_pts)
    prof[:, 0] *= c  # once olcekle
    prof[:, 1] *= c  # kalinlik z-yonune aktarilacak
    prof[:, 0] -= 0.25 * c  # sonra c/4'e kaydir (LE -> -c/4)
    x = prof[:, 0]
    z = prof[:, 1]
    rot_x = x * np.cos(alpha) - z * np.sin(alpha)
    rot_z = -x * np.sin(alpha) + z * np.cos(alpha)

    n = len(rot_x)
    y_lo, y_hi = -span / 2.0, span / 2.0

    # create_voxel_mask vertex mesafesi kullandigi icin y-yonunde yeterli
    # halka sarmali: halka araligi ~0.04 m (< voxel yarim kosegeni)
    n_rings = max(3, int(span / 0.04) + 1)
    y_rings = np.linspace(y_lo, y_hi, n_rings)

    rings = []
    for y in y_rings:
        rings.append(np.column_stack([rot_x + cx, np.full(n, y + cy), rot_z + cz]))

    verts = []
    tris = []
    idx = 0
    for r in range(n_rings - 1):  # halkalar arasi yan yuzey (profil
        r_lo, r_hi = rings[r], rings[r + 1]  # halkasi kapalidir: LE/TE diki
        for i in range(n):  # si kapak gerektirmez)
            j = (i + 1) % n
            a, b = r_lo[i], r_lo[j]
            c2, d = r_hi[j], r_hi[i]
            i0, i1, i2, i3 = idx, idx + 1, idx + 2, idx + 3
            verts.extend([a, b, c2])
            tris.append((i0, i1, i2))
            verts.extend([a, c2, d])
            tris.append((i0, i2, i3))
            idx += 4

    # Degenere (sifir alan) ucgenleri kaldir — kuvvet integralini kirletmesin
    verts = np.asarray(verts, dtype=np.float32)
    tris = np.asarray(tris, dtype=np.int32)
    ok_tris = []
    for t in range(len(tris)):
        a = verts[tris[t, 0]]
        b = verts[tris[t, 1]]
        c = verts[tris[t, 2]]
        if np.linalg.norm(np.cross(b - a, c - a)) > 1e-12:
            ok_tris.append(t)
    tris = tris[ok_tris]
    return verts, tris[:, 0], tris[:, 1], tris[:, 2]


def write_binary_stl(path, verts, i_idx, j_idx, k_idx, name=b"aero_naca0012"):
    """Binary STL dosyasi yazar (numpy-stl kullanmadan, bagimsiz)."""
    n = len(i_idx)
    header = bytearray(name.ljust(80, b"\0")[:80])
    header += np.uint32(n).tobytes()
    with open(path, "wb") as f:
        f.write(header)
        for t in range(n):
            a = verts[i_idx[t]]
            b = verts[j_idx[t]]
            c = verts[k_idx[t]]
            normal = np.cross(b - a, c - a)
            nl = np.linalg.norm(normal)
            normal = normal / nl if nl > 0 else np.zeros(3)
            tri = np.concatenate([normal, a, b, c]).astype(np.float32)
            f.write(tri.tobytes())
            f.write(np.uint16(0).tobytes())


def main():
    ap = argparse.ArgumentParser(description="NACA 0012 kanat STL uretici")
    ap.add_argument("--out", required=True)
    ap.add_argument("--c", type=float, default=0.3)
    ap.add_argument("--alpha", type=float, default=8.0)
    ap.add_argument("--span", type=float, default=2.0)
    ap.add_argument("--cx", type=float, default=-0.5)
    ap.add_argument("--cy", type=float, default=0.0)
    ap.add_argument("--cz", type=float, default=0.0)
    ap.add_argument("--n_pts", type=int, default=60)
    args = ap.parse_args()

    verts, i, j, k = naca0012_wing(args.c, args.alpha, args.span, (args.cx, args.cy, args.cz), args.n_pts)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_binary_stl(args.out, verts, i, j, k)
    print(f"{args.out}: {len(i)} ucgen, {len(verts)} kok noktasi")


if __name__ == "__main__":
    main()
