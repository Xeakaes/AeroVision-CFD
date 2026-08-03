"""
Parametrik kure STL uretici — bilimsel dogrulama (Cd benchmark) icin.

Iki kullanim modu:
  1. Dogrudan import:  `verts, i_idx, j_idx, k_idx = sphere_stl(...)`
  2. STL dosyasi:      `python sphere_stl.py --out sphere.stl --radius 0.25`
"""

import argparse
import os

import numpy as np


def sphere_stl(radius=0.25, center=(2.0, 1.0, 1.0), n_lat=48, n_lon=64):
    """Enlem/boylam grid'i uzerinde ucgenlesmis kure dondurur.

    Args:
        radius: Kure yaricapi [m]
        center: Merkez (cx, cy, cz) [m]
        n_lat:  Enlem dilimi sayisi (kutuplar dahil; kutuplarda cokgenler uclar)
        n_lon:  Boylam dilimi sayisi

    Returns:
        verts: (n_tri*3, 3) float32 kokte noktalari
        i_idx, j_idx, k_idx: her ucgenin kok nokta indeksleri (int32)
    """
    cx, cy, cz = center
    n_lat = max(6, int(n_lat))
    n_lon = max(8, int(n_lon))

    theta = np.linspace(0.0, 2.0 * np.pi, n_lon, endpoint=False)
    phi = np.linspace(0.0, np.pi, n_lat + 1)

    pts = []
    for p in phi:
        row = np.column_stack(
            [
                cx + radius * np.sin(p) * np.cos(theta),
                cy + radius * np.sin(p) * np.sin(theta),
                np.full(n_lon, cz + radius * np.cos(p)),
            ]
        )
        pts.append(row)

    verts = []
    tris = []
    idx = 0
    for i in range(n_lat):
        for j in range(n_lon):
            jp = (j + 1) % n_lon
            a = pts[i][j]
            b = pts[i + 1][j]
            c = pts[i + 1][jp]
            d = pts[i][jp]
            i0, i1, i2, i3 = idx, idx + 1, idx + 2, idx + 3
            verts.extend([a, b, c])
            tris.append((i0, i1, i2))
            verts.extend([a, c, d])
            tris.append((i0, i2, i3))
            idx += 4

    verts = np.asarray(verts, dtype=np.float32)
    tris = np.asarray(tris, dtype=np.int32)
    return verts, tris[:, 0], tris[:, 1], tris[:, 2]


def write_binary_stl(path, verts, i_idx, j_idx, k_idx, name=b"aero_sphere"):
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


def analytic_sphere_mask(nx, ny, nz, radius=0.25, center=(-1.0, 0.0, 0.0), tunnel_dims=(4.0, 2.0, 2.0)):
    """Analitik kure maskesi — STL tabanli pipeline ile karsilastirma icin.
    Merkezli koordinat sistemi kullanir (x in [-tx/2, tx/2], ...)."""
    xx, yy, zz = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    xs = np.linspace(-tunnel_dims[0] / 2, tunnel_dims[0] / 2, nx)
    ys = np.linspace(-tunnel_dims[1] / 2, tunnel_dims[1] / 2, ny)
    zs = np.linspace(-tunnel_dims[2] / 2, tunnel_dims[2] / 2, nz)
    dist = np.sqrt((xs[xx] - center[0]) ** 2 + (ys[yy] - center[1]) ** 2 + (zs[zz] - center[2]) ** 2)
    return dist <= radius


def main():
    ap = argparse.ArgumentParser(description="Parametrik kure STL uretici")
    ap.add_argument("--out", required=True)
    ap.add_argument("--radius", type=float, default=0.25)
    ap.add_argument("--cx", type=float, default=2.0)
    ap.add_argument("--cy", type=float, default=1.0)
    ap.add_argument("--cz", type=float, default=1.0)
    ap.add_argument("--n_lat", type=int, default=48)
    ap.add_argument("--n_lon", type=int, default=64)
    args = ap.parse_args()

    verts, i, j, k = sphere_stl(args.radius, (args.cx, args.cy, args.cz), args.n_lat, args.n_lon)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)) or ".", exist_ok=True)
    write_binary_stl(args.out, verts, i, j, k)
    print(f"{args.out}: {len(i)} ucgen, {len(verts)} kok noktasi")


if __name__ == "__main__":
    main()
