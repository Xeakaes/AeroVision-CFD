"""
AeroVision CFD v1.1 - Physics Engine Core

Features:
  - Mask:     Laser scan voxelization with cKDTree + sub-sample boundary detection
  - Poisson:  Vectorized sparse Laplacian with Jacobi preconditioner
  - Axes:     index=ij synchronization for interpolation consistency
  - Colors:   NaN cleanup for Cp computation stability
  - Forces:   Vertical force integration with trilinear interpolation
  - Turbulence: Smagorinsky SGS, Spalart-Allmaras, k-epsilon, k-omega SST models
  - GPU:      NVIDIA (cupy), Intel/AMD (PyOpenCL fp32 + CPU fp64) multi-vendor destek
"""

import os

import numpy as np
import scipy.spatial as spatial
from scipy.ndimage import binary_fill_holes, distance_transform_edt
from scipy.sparse import diags as sp_diags
from scipy.sparse import csr_matrix as sp_csr
from scipy.sparse.linalg import cg as scipy_cg

# ── Modüler çekirdekler ─────────────────────────────────────────────────────
# gpu_backend: GPU tespiti + array kutuphanesi durumu (runtime'da degisebilir;
#   bu yuzden fonksiyon govdeleri gpu_backend.GPU / gpu_backend.cp okur).
import gpu_backend
from gpu_backend import (  # noqa: F401  (main.py gerekli olan re-export'lar)
    GPU_BACKEND,
    GPU_AVAILABLE,
    GPU_VENDOR,
    GPU_DEVICE_NAME,
    cp,
    cu_csr,
    cu_diags,
    cu_cg,
    _to_numpy,
    _hash_mask,
)
from constants import (
    TUNNEL_DIMS,
    RHO,
    POISSON_TOL,
    POISSON_MAXIT,
    _CACHE_MAX_ENTRIES,  # noqa: F401  (test_solver.py re-export)
    _WALL_DISTANCE_CACHE,  # noqa: F401  (re-export)
    _POISSON_CACHE,
    _lru_insert,
)
from turbulence_models import (
    _smagorinsky_nu,
    solve_spalart_allmaras,
    solve_k_epsilon,
    solve_k_omega_sst,
    _build_wall_distance_cache,
    _apply_wall_functions,
)

# Eski isimle geriye uyumluluk (main.py runtime'da backend degistirebilir)
_detect_gpu = gpu_backend.detect_gpu

# ── Sabitler ────────────────────────────────────────────────────────────────
# Fizik sabitleri, LRU onbellekler ve _lru_insert artık constants modulunde.
# Geriye uyumluluk icin yukarida `from constants import ...` ile re-export
# edilmektedir.


# ════════════════════════════════════════════════════════════════════════════
# KATMAN 1 — MATEMATİKSEL KESİNLİKLİ LAZER TARAMA MASKESİ
# ════════════════════════════════════════════════════════════════════════════


def create_voxel_mask(
    verts: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    k_idx: np.ndarray,
    nx: int,
    ny: int,
    nz: int,
    tunnel_dims: tuple,
) -> tuple[np.ndarray, np.ndarray, bool]:
    """Sınır hücreleri için kısmi doluluk hesaplayarak daha doğru bir gömülü sınır maskesi sağlar.

    Sınır hücrelerinde 2x2x2 alt hücre örneklemesi kullanılır.

    Args:
        verts: STL vertex koordinatlari (N, 3)
        i_idx, j_idx, k_idx: Ucgen indeksleri
        nx, ny, nz: Grid boyutlari
        tunnel_dims: Tunel boyutlari (tx, ty, tz)

    Returns:
        mask: Boolean grid (nx, ny, nz)
        solid_fraction: Kismi doluluk orani (nx, ny, nz)
        success: Islem basarili mi
    """
    try:
        tx, ty, tz = tunnel_dims
        xs = np.linspace(-tx / 2, tx / 2, nx)
        ys = np.linspace(-ty / 2, ty / 2, ny)
        zs = np.linspace(-tz / 2, tz / 2, nz)

        tree = spatial.cKDTree(verts)
        dx, dy, dz = tx / (nx - 1), ty / (ny - 1), tz / (nz - 1)
        voxel_diagonal_half = np.sqrt(dx**2 + dy**2 + dz**2) / 2.0
        fine_diagonal_half = voxel_diagonal_half * 0.5

        offs = np.array([-0.25, 0.25], dtype=np.float64)
        ox, oy, oz = np.meshgrid(offs, offs, offs, indexing="ij")
        sub_offsets = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=-1)

        i_idx_all, j_idx_all, k_idx_all = np.meshgrid(
            np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij"
        )
        base_centers = np.column_stack([xs[i_idx_all.ravel()], ys[j_idx_all.ravel()], zs[k_idx_all.ravel()]])

        sub_points = base_centers[:, None, :] + sub_offsets[None, :, :] * np.array([dx, dy, dz])
        sub_points = sub_points.reshape(-1, 3)

        sub_dist, _ = tree.query(sub_points, k=1)
        sub_dist = sub_dist.reshape(nx * ny * nz, 8)
        fraction_flat = (sub_dist <= fine_diagonal_half).sum(axis=1) / 8.0

        mask_flat = fraction_flat >= 0.5
        mask = mask_flat.reshape((nx, ny, nz))
        solid_fraction = fraction_flat.reshape((nx, ny, nz))

        mask = binary_fill_holes(mask)
        solid_fraction[mask] = 1.0

        return mask, solid_fraction, True

    except Exception as e:
        print(f"[Maske Hatası] {e}")
        return np.zeros((nx, ny, nz), dtype=bool), np.zeros((nx, ny, nz), dtype=np.float64), False


# ════════════════════════════════════════════════════════════════════════════
# KATMAN 2 — POTANSİYEL AKIŞ
# ════════════════════════════════════════════════════════════════════════════


def solve_potential_flow(
    mask: np.ndarray,
    v_wind: float,
    nx: int,
    ny: int,
    nz: int,
    solid_fraction: np.ndarray | None = None,
    max_iter: int = 300,
    tol: float = 1e-3,
    tunnel_dims: tuple[float, float, float] | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Potansiyel akis cozumu - Laplace denklemi Jacobi iterasyonu ile.

    Args:
        mask: Boolean grid (nx, ny, nz) - katı hucreler
        v_wind: Ruzgar hizi (m/s)
        nx, ny, nz: Grid boyutlari
        solid_fraction: Kismi doluluk orani (nx, ny, nz)
        max_iter: Maksimum iterasyon sayisi
        tol: Yakinsama toleransi
        tunnel_dims: Tünel boyutlari (Lx, Ly, Lz); None ise TUNNEL_DIMS

    Returns:
        u, v, w: Hiz alanlari (nx, ny, nz)
    """
    tx, ty, tz = TUNNEL_DIMS if tunnel_dims is None else tuple(tunnel_dims)
    dx, dy, dz = tx / (nx - 1), ty / (ny - 1), tz / (nz - 1)

    xp = gpu_backend.cp if gpu_backend.GPU else np
    phi = xp.zeros((nx, ny, nz), dtype=xp.float64)
    x_phys = xp.linspace(-tx / 2, tx / 2, nx)
    for i in range(nx):
        phi[i, :, :] = float(x_phys[i]) * v_wind

    mask_g = xp.asarray(mask)
    if solid_fraction is not None:
        solid_fraction = xp.asarray(solid_fraction)
        mask_g = xp.logical_or(mask_g, solid_fraction >= 1.0)

    for _ in range(max_iter):
        phi_old = phi.copy()
        interior = (
            phi[2:, 1:-1, 1:-1]
            + phi[:-2, 1:-1, 1:-1]
            + phi[1:-1, 2:, 1:-1]
            + phi[1:-1, :-2, 1:-1]
            + phi[1:-1, 1:-1, 2:]
            + phi[1:-1, 1:-1, :-2]
        )
        # Katı hucreler de Jacobi relax (Neumann BC dphi/dn=0): phi tum
        # hucrelerde Laplace ile duzlestirilir; duvar yuzeyinde gradyan
        # boylece sifirlanir (girdi akiskanla sureklilik korunur).
        phi[1:-1, 1:-1, 1:-1] = interior / 6.0
        phi[0, :, :] = v_wind * (-tx / 2)
        phi[-1, :, :] = phi[-2, :, :]
        phi[:, 0, :] = phi[:, 1, :]
        phi[:, -1, :] = phi[:, -2, :]
        phi[:, :, 0] = phi[:, :, 1]
        phi[:, :, -1] = phi[:, :, -2]

        if float(xp.max(xp.abs(phi - phi_old))) < tol:
            break

    u = xp.zeros_like(phi)
    v = xp.zeros_like(phi)
    w = xp.zeros_like(phi)
    u[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2 * dx)
    v[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2 * dy)
    w[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2 * dz)
    u[0, :, :] = v_wind
    u[mask_g] = 0.0
    v[mask_g] = 0.0
    w[mask_g] = 0.0

    if gpu_backend.GPU:
        return _to_numpy(u), _to_numpy(v), _to_numpy(w)
    # Local cp eski backend kalintisi olabilir — cikti her zaman numpy olsun
    if hasattr(u, "get"):
        return _to_numpy(u), _to_numpy(v), _to_numpy(w)
    return u, v, w


def _assemble_poisson_matrix(mask_g, dx, dy, dz, nx, ny, nz):
    """Kararlı basınç Poisson sistemi için sparse Laplace matrisini hazırlar."""
    N = nx * ny * nz
    ax = float(1.0 / dx**2)
    ay = float(1.0 / dy**2)
    az = float(1.0 / dz**2)
    # POZITIF Laplacian (+L): diag=+2(ax+ay+az), komsular -ax/-ay/-az.
    # Negatif karsilik (-L) NSD'dir -> CG kirilir. +L SPD'dir ve OpenCL
    # kernel'inin A_p matrisiyle birebir aynidir (pAp > 0 guvence).
    # cozum ayni: +L p = -b  <=>  -L p = b  (rhs isareti _poisson_cg'de doner)
    diag_val = float(2.0 * (ax + ay + az))

    ox = ny * nz
    oy = nz
    oz = 1

    # COO tabanli vektorize kurulum: 3D-duyarli stencil, sarma (wrap-around)
    # komşuluklar hic olusmaz (OpenCL kernel'inin guard'lariyla birebir).
    idx = np.arange(N, dtype=np.int64)
    i3 = idx // ox
    j3 = (idx // oy) % ny
    k3 = idx % nz
    coefs = np.array([diag_val, -ax, -ax, -ay, -ay, -az, -az], dtype=np.float64)
    rows = np.broadcast_to(idx[:, None], (N, 7))
    offs = np.array([0, ox, -ox, oy, -oy, oz, -oz], dtype=np.int64)
    cols = rows + offs[None, :]
    valid = np.ones((N, 7), dtype=bool)
    valid[:, 1] = i3 < nx - 1
    valid[:, 2] = i3 > 0
    valid[:, 3] = j3 < ny - 1
    valid[:, 4] = j3 > 0
    valid[:, 5] = k3 < nz - 1
    valid[:, 6] = k3 > 0

    mask_flat = mask_g.ravel() if not gpu_backend.GPU else _to_numpy(mask_g).ravel()
    solid_idx = np.where(mask_flat)[0]
    valid[solid_idx, :] = False
    # row 0 saf dirichlet: komşu katkısı yok, diag=1
    if 0 not in solid_idx:
        valid[0, :] = False
        solid_idx = np.concatenate([solid_idx, np.array([0], dtype=np.int64)])
        solid_idx.sort()

    # Kolon tarafında da katı/row0 hücrelerine bağlantı olmasın:
    # p_solid = p[0] = 0 oldugundan katkıları sifirdir ve bu, matrisi
    # tam simetrik + pozitif tanimli yapar (OpenCL kernel ile birebir).
    col_mask = np.ones((N, 7), dtype=bool)
    col_mask[solid_idx, :] = False
    col_mask[0, :] = False
    valid = valid & col_mask[np.clip(cols, 0, N - 1), np.arange(7)[None, :]]

    # Katı yüzeylerde akı 0 (impenetrable) -> katı komşulukta Laplacian TEK
    # TARAFLI (Neumann): o yöndeki diag katkısı kaldırılır. Bu, RHS'teki
    # 0-akı maskesi ve project_correct'teki 0-dp maskesi ile birebir tutarlı
    # (aksi halde düzeltme sonrası div ~ O(1) kalır).
    guards = np.stack(
        [i3 < nx - 1, i3 > 0, j3 < ny - 1, j3 > 0, k3 < nz - 1, k3 > 0],
        axis=1,
    )
    nb_offsets = [ox, -ox, oy, -oy, oz, -oz]
    nb_coefs = [ax, ax, ay, ay, az, az]
    diag = np.full(N, diag_val, dtype=np.float64)
    for k in range(6):
        nb = np.clip(idx + nb_offsets[k], 0, N - 1)
        nb_solid = guards[:, k] & mask_flat[nb]
        diag -= nb_coefs[k] * nb_solid
    diag[solid_idx] = 1.0

    # katı satırlar (ve row 0): diag=1, rhs=0 -> p=0 (OpenCL kernel birebir)
    vals_mat = np.broadcast_to(coefs[None, :], (N, 7)).copy()
    vals_mat[:, 0] = diag
    rows_all = np.concatenate([rows[valid], solid_idx])
    cols_all = np.concatenate([cols[valid], solid_idx])
    vals_all = np.concatenate([vals_mat[valid], np.ones(solid_idx.size, dtype=np.float64)])
    A = sp_csr(
        (vals_all, (rows_all, cols_all)),
        shape=(N, N),
        dtype=np.float64,
    )

    return A, solid_idx


def _build_jacobi_preconditioner(A, solid_idx):
    """Poisson CG için Jacobi önşartlayıcıyı hazırlar."""
    diag = A.diagonal()
    inv_diag = np.zeros_like(diag, dtype=np.float64)
    nonzero = diag != 0.0
    inv_diag[nonzero] = 1.0 / diag[nonzero]
    inv_diag[solid_idx] = 1.0

    if gpu_backend.GPU and gpu_backend.GPU_BACKEND == "cupy":
        return gpu_backend.cu_diags(inv_diag, 0, shape=A.shape, format="csr")
    return sp_diags(inv_diag, 0, shape=A.shape, format="csr")


def _poisson_cg(rhs, mask_g, dx, dy, dz, nx, ny, nz, xp, x0=None):
    """Vektörize sparse Laplacian — CG çözücü (Basınç sızdırmazlık yamalı).

    x0 verilirse sicak baslangic kullanilir (adimlar arasi p yakin
    oldugundan iterasyon sayisi ~8x duser; cozum ayni, SPD sistem).
    """
    mask_hash = _hash_mask(mask_g)
    cache_key = (
        int(nx),
        int(ny),
        int(nz),
        round(float(dx), 8),
        round(float(dy), 8),
        round(float(dz), 8),
        int(mask_hash),
        gpu_backend.GPU_BACKEND,
    )

    if cache_key not in _POISSON_CACHE:
        A, solid_idx = _assemble_poisson_matrix(mask_g, dx, dy, dz, nx, ny, nz)
        M = _build_jacobi_preconditioner(A, solid_idx)
        _lru_insert(_POISSON_CACHE, cache_key, (A, M, solid_idx))
    else:
        A, M, solid_idx = _POISSON_CACHE[cache_key]
        _POISSON_CACHE.move_to_end(cache_key)

    rhs_np = rhs.ravel() if not gpu_backend.GPU else _to_numpy(rhs).ravel()
    rhs_np[solid_idx] = 0.0
    # Matris +L oldugundan cozumu korumak icin rhs isareti cevrilir: +L p = -b.
    rhs_np = -rhs_np
    # row 0 saf dirichlet (p[0]=0) — OpenCL kernel ile birebir.
    rhs_np[0] = 0.0

    x0_np = None
    if x0 is not None:
        x0_np = _to_numpy(x0).ravel().copy()
        x0_np[solid_idx] = 0.0
        x0_np[0] = 0.0

    if gpu_backend.GPU and gpu_backend.GPU_BACKEND == "cupy":
        A_gpu = gpu_backend.cu_csr(A)
        p_flat, info = gpu_backend.cu_cg(
            A_gpu,
            gpu_backend.cp.asarray(rhs_np),
            x0=gpu_backend.cp.asarray(x0_np) if x0_np is not None else None,
            rtol=POISSON_TOL,
            maxiter=POISSON_MAXIT,
            M=M,
        )
        p = _to_numpy(p_flat).reshape((nx, ny, nz))
    else:
        # CPU veya OpenCL (fp64 hassasiyet icin CPU'da coz)
        p_flat, info = scipy_cg(A, rhs_np, x0=x0_np, rtol=POISSON_TOL, maxiter=POISSON_MAXIT, M=M)
        p = p_flat.reshape((nx, ny, nz))

    if info != 0:
        print(
            f"[Poisson] WARNING: Convergence not achieved (info={info}), tolerance exceeded in {POISSON_MAXIT} iterations"
        )

    return p


def _solve_ns_opencl(
    u_np,
    v_np,
    w_np,
    mask,
    v_wind,
    nx,
    ny,
    nz,
    dx,
    dy,
    dz,
    nu_mol,
    solid_fraction,
    iterations,
    turbulence_model,
    use_wall_functions,
    turb_iterations=None,
    tunnel_dims=None,
    progress_callback=None,
):
    """OpenCL GPU ile Navier-Stokes cozumu — tam fizik.

    RK2 (Heun) + konservatif viskozite kernel'larda; turbulans viskozitesi,
    duvar fonksiyonlari ve Rhie-Chow basinc duzeltmesi CPU (fp64) tarafinda
    hesaplanir ve GPU'ya aktarilir (hybrid yaklasim).
    """
    import cl_solver
    import pyopencl.array as cl_array
    from cl_backend import _ctx, _queue

    slvr = cl_solver.OpenCLSolver(_ctx, _queue)
    inv_dx, inv_dy, inv_dz = 1.0 / dx, 1.0 / dy, 1.0 / dz
    inv_dx2, inv_dy2, inv_dz2 = 1.0 / (dx * dx), 1.0 / (dy * dy), 1.0 / (dz * dz)
    n_total = nx * ny * nz

    # Veriyi GPU'ya gonder (fp32)
    u_cl = slvr.to_device(u_np.astype(np.float32).ravel())
    v_cl = slvr.to_device(v_np.astype(np.float32).ravel())
    w_cl = slvr.to_device(w_np.astype(np.float32).ravel())
    mask_eff = np.asarray(mask)
    if solid_fraction is not None:
        mask_eff = mask_eff | (np.asarray(solid_fraction) >= 1.0)
    mask_cl = slvr.to_device(mask_eff.astype(np.float32).ravel())

    # Exact projection: RHS'te volume çarpanı yok (duzeltme operatörüyle birebir);
    # kısmi doluluk (solid_fraction) CPU yoluyla ayni: vol_weight = 1 - frac.
    if solid_fraction is not None:
        vol_np = (1.0 - np.asarray(solid_fraction)).clip(0.0, 1.0).astype(np.float32).ravel()
    else:
        vol_np = np.ones(n_total, dtype=np.float32)
    vol_cl = slvr.to_device(vol_np)

    # Turbulans alani — CPU'da hesaplanir, GPU'ya nu_eff olarak yuklenir
    # mask_g = mask | (solid_fraction >= 1.0) — CPU yolu (solver.py:799-801)
    # ile birebir. Kismi doluluk hucresi (0 < frac < 1) fluid sayilir.
    mask_g = np.asarray(mask_eff)
    d_wall = _build_wall_distance_cache(mask_g, dx, dy, dz, np)
    d_wall_cl = cl_array.to_device(_queue, d_wall.astype(np.float64).ravel())
    nu_eff_np = np.full(n_total, float(nu_mol), dtype=np.float32)
    nu_eff_cl = slvr.to_device(nu_eff_np)

    p_cl = cl_array.zeros(_queue, (n_total,), dtype=np.float32)

    # Dongu ici tamponlar — adim basina alloc YOK (kernel'ler out buffer'ina
    # yazar; sifir kullanimi kernel icinde).
    rhs1_u = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    rhs1_v = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    rhs1_w = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    rhs2_u = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    rhs2_v = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    rhs2_w = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    u_star_cl = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    v_star_cl = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    w_star_cl = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    u_new_cl = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    v_new_cl = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    w_new_cl = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    u_pre_cl = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    v_pre_cl = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    w_pre_cl = cl_array.empty(_queue, (n_total,), dtype=np.float32)
    proj_tmp = (
        cl_array.empty(_queue, (n_total,), dtype=np.float32),
        cl_array.empty(_queue, (n_total,), dtype=np.float32),
        cl_array.empty(_queue, (n_total,), dtype=np.float32),
    )

    # Katı hücrelerde hızı sıfırlamak için 1-mask (cihazda sabit)
    inv_mask_cl = slvr.to_device((1.0 - mask_eff.astype(np.float32).ravel()))

    # GPU Poisson hazirliklari: rhs tamponu + Jacobi onkosullayici (inv_diag)
    # Kernel pozitif tanimli sistemi (A_p) cozer; inv_diag buna uyar.
    rhs_cl = cl_array.zeros(_queue, (n_total,), dtype=np.float32)
    solid_flat = mask_eff.astype(np.float32).ravel()
    solid3 = solid_flat.reshape(nx, ny, nz)

    # Neumann diag: kati komsulukta o yonun katsayisi duser (CPU matrisi birebir)
    diag_np = np.full((nx, ny, nz), 2.0 * (inv_dx2 + inv_dy2 + inv_dz2), dtype=np.float32)
    nb = np.zeros_like(solid3)
    nb[:-1, :, :] = solid3[1:, :, :]
    diag_np -= inv_dx2 * nb
    nb = np.zeros_like(solid3)
    nb[1:, :, :] = solid3[:-1, :, :]
    diag_np -= inv_dx2 * nb
    nb = np.zeros_like(solid3)
    nb[:, :-1, :] = solid3[:, 1:, :]
    diag_np -= inv_dy2 * nb
    nb = np.zeros_like(solid3)
    nb[:, 1:, :] = solid3[:, :-1, :]
    diag_np -= inv_dy2 * nb
    nb = np.zeros_like(solid3)
    nb[:, :, :-1] = solid3[:, :, 1:]
    diag_np -= inv_dz2 * nb
    nb = np.zeros_like(solid3)
    nb[:, :, 1:] = solid3[:, :, :-1]
    diag_np -= inv_dz2 * nb

    inv_diag_np = np.zeros(n_total, dtype=np.float32)
    diag_flat = diag_np.ravel()
    nonzero = diag_flat != 0.0
    inv_diag_np[nonzero] = 1.0 / diag_flat[nonzero]
    inv_diag_np[solid_flat > 0.5] = 1.0
    inv_diag_np[0] = 1.0
    inv_diag_cl = slvr.to_device(inv_diag_np)

    # ── CFL parametreleri (CPU yolu ile ayni formül; dt her adimda yenilenir)
    print(f"[OpenCL-NS] {iterations} iterasyon, nx={nx}, ny={ny}, nz={nz}")

    turb_lower = str(turbulence_model).lower()
    TURB_IV = 3
    prev_residual = None
    u_np_prev = v_np_prev = w_np_prev = None

    # ── Ana dongu ──────────────────────────────────────────────────────────
    nu_eff_max = max(float(cl_array.max(nu_eff_cl).get()), float(nu_mol))
    for step in range(iterations):
        # ── Adaptif dt — CPU ile ayni formül, her adimda guncel hizdan ────
        # GPU tarafinda skaler max (yalnizca 3 skaler sync edilir)
        # MUTLAK deger (CPU: max|u|) — negatif hizlar dt'yi bozmasin.
        # abs_max3: uc |max|'i tek kernel'de — 6 host sync yerine 1.
        u_max = v_max = w_max = 1e-3
        u_max, v_max, w_max = slvr.abs_max3(u_cl, v_cl, w_cl, u_max, v_max, w_max, n_total)
        vel_max = max(u_max, v_max, w_max)
        if vel_max > 120:
            cfl_safety = 0.05
        elif vel_max > 60:
            cfl_safety = 0.08
        else:
            cfl_safety = 0.12
        dt_conv = cfl_safety * min(dx / u_max, dy / v_max, dz / w_max)
        dt_visc = cfl_safety / (2.0 * nu_eff_max * (inv_dx2 + inv_dy2 + inv_dz2) + 1e-12)
        dt = np.float32(min(dt_conv, dt_visc, 2e-3))

        # ── Turbulans viskozitesi ──────────────────────────────────────────
        # Spalart-Allmaras GPU'da (fp64, numpy yoluyla birebir); diger
        # modeller CPU'da hesaplanir, GPU'ya aktarilir (3 adimda 1).
        # nu_eff yalnizca burada degistigi icin nu_eff_max de burada yenilenir
        # (CPU her adimda ayni degerle hesaplar — parite korunur).
        if step % TURB_IV == 0:
            if turb_lower in ("spalart_allmaras", "spalart-allmaras"):
                sa_iters = 60 if turb_iterations is None else int(turb_iterations)
                nu_eff_cl = slvr.sa_solve(
                    u_cl,
                    v_cl,
                    w_cl,
                    mask_cl,
                    d_wall_cl,
                    dx,
                    dy,
                    dz,
                    float(nu_mol),
                    sa_iters,
                    nx,
                    ny,
                    nz,
                )
            else:
                u_np_t = slvr.to_numpy(u_cl).reshape(nx, ny, nz)
                v_np_t = slvr.to_numpy(v_cl).reshape(nx, ny, nz)
                w_np_t = slvr.to_numpy(w_cl).reshape(nx, ny, nz)
                if turb_lower in ("k_epsilon", "k-epsilon", "ke"):
                    ke_iters = 60 if turb_iterations is None else int(turb_iterations)
                    nu_t = solve_k_epsilon(
                        u_np_t, v_np_t, w_np_t, mask_g, dx, dy, dz, nu_mol=nu_mol, iterations=ke_iters, xp=np
                    )
                elif turb_lower in ("k_omega_sst", "k-omega-sst", "komega_sst", "sst"):
                    sst_iters = 60 if turb_iterations is None else int(turb_iterations)
                    nu_t = solve_k_omega_sst(
                        u_np_t, v_np_t, w_np_t, mask_g, dx, dy, dz, nu_mol=nu_mol, iterations=sst_iters, xp=np
                    )
                else:
                    nu_t = _smagorinsky_nu(u_np_t, v_np_t, w_np_t, dx, dy, dz, np, d_wall=d_wall)
                nu_eff_np = (float(nu_mol) + nu_t).astype(np.float32).ravel()
                nu_eff_cl = slvr.to_device(nu_eff_np)
            nu_eff_max = max(float(cl_array.max(nu_eff_cl).get()), float(nu_mol))

        # ── RK2 Stage 1: rhs1 ─────────────────────────────────────────────
        slvr.ns_rhs(
            u_cl,
            v_cl,
            w_cl,
            nu_eff_cl,
            vol_cl,
            inv_dx,
            inv_dy,
            inv_dz,
            inv_dx2,
            inv_dy2,
            inv_dz2,
            nx,
            ny,
            nz,
            out=(rhs1_u, rhs1_v, rhs1_w),
        )

        dt_f = np.float32(dt)
        half_dt_f = np.float32(0.5) * dt_f

        # u_star = (u_old + dt*rhs1) * inv_mask — tek kernel (alloc yok)
        slvr.rk2_stage1(u_star_cl, u_cl, rhs1_u, inv_mask_cl, dt_f, n_total)
        slvr.rk2_stage1(v_star_cl, v_cl, rhs1_v, inv_mask_cl, dt_f, n_total)
        slvr.rk2_stage1(w_star_cl, w_cl, rhs1_w, inv_mask_cl, dt_f, n_total)

        slvr.apply_tunnel_wall_bc(u_star_cl, v_star_cl, w_star_cl, nx, ny, nz)
        slvr.apply_inlet_bc(u_star_cl, v_star_cl, w_star_cl, v_wind, ny, nz)

        # ── RK2 Stage 2: rhs2 ─────────────────────────────────────────────
        slvr.ns_rhs(
            u_star_cl,
            v_star_cl,
            w_star_cl,
            nu_eff_cl,
            vol_cl,
            inv_dx,
            inv_dy,
            inv_dz,
            inv_dx2,
            inv_dy2,
            inv_dz2,
            nx,
            ny,
            nz,
            out=(rhs2_u, rhs2_v, rhs2_w),
        )

        # u_new = (u_old + half_dt*(rhs1+rhs2)) * vol * inv_mask — tek kernel
        slvr.rk2_combine(u_new_cl, u_cl, rhs1_u, rhs2_u, vol_cl, inv_mask_cl, half_dt_f, n_total)
        slvr.rk2_combine(v_new_cl, v_cl, rhs1_v, rhs2_v, vol_cl, inv_mask_cl, half_dt_f, n_total)
        slvr.rk2_combine(w_new_cl, w_cl, rhs1_w, rhs2_w, vol_cl, inv_mask_cl, half_dt_f, n_total)

        slvr.apply_tunnel_wall_bc(u_new_cl, v_new_cl, w_new_cl, nx, ny, nz)
        slvr.apply_inlet_bc(u_new_cl, v_new_cl, w_new_cl, v_wind, ny, nz)
        u_new_cl, v_new_cl, w_new_cl, p_cl = slvr.apply_outlet_bc(
            u_new_cl, v_new_cl, w_new_cl, p_cl, nx, ny, nz
        )

        # ── Basinc Poisson — GPU CG (stencil matvec), her adim ────────────
        # rk2_combine katı hücreleri zaten inv_mask ile sıfırlar; outlet BC
        # katı hücrelere dokunmaz — ekstra inv_mask gerekmez (CPU yolu ile
        # ayni deger: x*0=0). div_max icin duzeltme ONCESI hizlar saklanir.
        slvr.array_copy(u_pre_cl, u_new_cl, n_total)
        slvr.array_copy(v_pre_cl, v_new_cl, n_total)
        slvr.array_copy(w_pre_cl, w_new_cl, n_total)
        slvr.divergence_rhs(
            u_new_cl,
            v_new_cl,
            w_new_cl,
            rhs_cl,
            mask_cl,
            inv_dx,
            inv_dy,
            inv_dz,
            float(1.0 / dt),
            nx,
            ny,
            nz,
        )
        if os.environ.get("CFD_OLD_CG"):
            p_cl_new, info_cg, iters_cg = slvr.poisson_cg(
                rhs_cl,
                p_cl,
                mask_cl,
                inv_diag_cl,
                inv_mask_cl,
                inv_dx2,
                inv_dy2,
                inv_dz2,
                nx,
                ny,
                nz,
                tol=POISSON_TOL,
            )
        else:
            p_cl_new, info_cg, iters_cg = slvr.poisson_cg_device(
                rhs_cl,
                p_cl,
                mask_cl,
                inv_diag_cl,
                inv_mask_cl,
                inv_dx2,
                inv_dy2,
                inv_dz2,
                nx,
                ny,
                nz,
                tol=POISSON_TOL,
            )
        if info_cg != 0:
            # fp32 konverjans siniri — host fp64 CG'ye dus (dogruluk garantisi)
            u_np_t = slvr.to_numpy(u_new_cl).reshape(nx, ny, nz)
            v_np_t = slvr.to_numpy(v_new_cl).reshape(nx, ny, nz)
            w_np_t = slvr.to_numpy(w_new_cl).reshape(nx, ny, nz)
            mask_np_t = slvr.to_numpy(mask_cl).reshape(nx, ny, nz).astype(bool)

            u_fe = 0.5 * (u_np_t[2:, 1:-1, 1:-1] + u_np_t[1:-1, 1:-1, 1:-1])
            u_fw = 0.5 * (u_np_t[1:-1, 1:-1, 1:-1] + u_np_t[:-2, 1:-1, 1:-1])
            v_fn = 0.5 * (v_np_t[1:-1, 2:, 1:-1] + v_np_t[1:-1, 1:-1, 1:-1])
            v_fs = 0.5 * (v_np_t[1:-1, 1:-1, 1:-1] + v_np_t[1:-1, :-2, 1:-1])
            w_ft = 0.5 * (w_np_t[1:-1, 1:-1, 2:] + w_np_t[1:-1, 1:-1, 1:-1])
            w_fb = 0.5 * (w_np_t[1:-1, 1:-1, 1:-1] + w_np_t[1:-1, 1:-1, :-2])

            div_star = (u_fe - u_fw) * inv_dx + (v_fn - v_fs) * inv_dy + (w_ft - w_fb) * inv_dz

            rhs_np = np.zeros(n_total, dtype=np.float64)
            rhs_np.reshape(nx, ny, nz)[1:-1, 1:-1, 1:-1] = div_star / dt
            rhs_np[mask_np_t.ravel()] = 0.0

            p_np = _poisson_cg(rhs_np.reshape(nx, ny, nz), mask_np_t, dx, dy, dz, nx, ny, nz, np)
            p_np[mask_np_t] = 0.0
            p_cl_new = slvr.to_device(p_np.astype(np.float32).ravel())

        # Exact projection — yüzey hızı düzeltmesi (CPU yoluyla birebir)
        slvr.project_correct(
            u_new_cl,
            v_new_cl,
            w_new_cl,
            p_cl_new,
            mask_cl,
            dt_f,
            inv_dx,
            inv_dy,
            inv_dz,
            nx,
            ny,
            nz,
            tmp=proj_tmp,
        )

        # ── Duzeltme sonrasi: katı sifirla + sinir sartlarini yeniden uygula ──
        # project_correct sınır hücrelere dokunmaz — in-place mask (alloc yok)
        slvr.mask_inplace(u_new_cl, inv_mask_cl, n_total)
        slvr.mask_inplace(v_new_cl, inv_mask_cl, n_total)
        slvr.mask_inplace(w_new_cl, inv_mask_cl, n_total)
        slvr.apply_tunnel_wall_bc(u_new_cl, v_new_cl, w_new_cl, nx, ny, nz)
        slvr.apply_inlet_bc(u_new_cl, v_new_cl, w_new_cl, v_wind, ny, nz)
        u_new_cl, v_new_cl, w_new_cl, p_cl = slvr.apply_outlet_bc(
            u_new_cl, v_new_cl, w_new_cl, p_cl_new, nx, ny, nz
        )

        # CPU yolu ile ayni koruma: |vel| > 5*v_wind hucreleri sifirla
        slvr.clamp_velocity(u_new_cl, v_new_cl, w_new_cl, 5.0 * v_wind, nx, ny, nz)

        p_cl = p_cl_new

        # ── Duvar fonksiyonlari (CPU'da, 3 adimda 1) ──────────────────────
        if use_wall_functions and step % TURB_IV == 0:
            u_np_t = slvr.to_numpy(u_new_cl).reshape(nx, ny, nz)
            v_np_t = slvr.to_numpy(v_new_cl).reshape(nx, ny, nz)
            w_np_t = slvr.to_numpy(w_new_cl).reshape(nx, ny, nz)
            u_np_t, v_np_t, w_np_t = _apply_wall_functions(
                u_np_t, v_np_t, w_np_t, mask_g, d_wall, dx, dy, dz, nu_mol, np
            )
            u_new_cl = slvr.to_device(u_np_t.astype(np.float32).ravel())
            v_new_cl = slvr.to_device(v_np_t.astype(np.float32).ravel())
            w_new_cl = slvr.to_device(w_np_t.astype(np.float32).ravel())

        u_cl = u_new_cl
        v_cl = v_new_cl
        w_cl = w_new_cl

        # ── Yakinsama izleme + canli callback (10 adimda 1) ───────────────
        if step % 10 == 0:
            # CPU div_corr (solver.py:1098) ile birebir: düzeltilmiş yüzey
            # hızlarından (u_fe_c = u_fe - dt*dp_e). p de GPU'dan alinir.
            u_np_now = slvr.to_numpy(u_pre_cl).reshape(nx, ny, nz)
            v_np_now = slvr.to_numpy(v_pre_cl).reshape(nx, ny, nz)
            w_np_now = slvr.to_numpy(w_pre_cl).reshape(nx, ny, nz)
            p_np_now = slvr.to_numpy(p_cl_new).reshape(nx, ny, nz)
            # Katiya bitisik yuzeylerde NORMAL aki 0 (CPU m_e/m_w/... maskesi)
            m_e_d = mask_g[2:, 1:-1, 1:-1] | mask_g[1:-1, 1:-1, 1:-1]
            m_w_d = mask_g[1:-1, 1:-1, 1:-1] | mask_g[:-2, 1:-1, 1:-1]
            m_n_d = mask_g[1:-1, 2:, 1:-1] | mask_g[1:-1, 1:-1, 1:-1]
            m_s_d = mask_g[1:-1, 1:-1, 1:-1] | mask_g[1:-1, :-2, 1:-1]
            m_t_d = mask_g[1:-1, 1:-1, 2:] | mask_g[1:-1, 1:-1, 1:-1]
            m_b_d = mask_g[1:-1, 1:-1, 1:-1] | mask_g[1:-1, 1:-1, :-2]
            u_fe_d = np.where(m_e_d, 0.0, 0.5 * (u_np_now[2:, 1:-1, 1:-1] + u_np_now[1:-1, 1:-1, 1:-1]))
            u_fw_d = np.where(m_w_d, 0.0, 0.5 * (u_np_now[1:-1, 1:-1, 1:-1] + u_np_now[:-2, 1:-1, 1:-1]))
            v_fn_d = np.where(m_n_d, 0.0, 0.5 * (v_np_now[1:-1, 2:, 1:-1] + v_np_now[1:-1, 1:-1, 1:-1]))
            v_fs_d = np.where(m_s_d, 0.0, 0.5 * (v_np_now[1:-1, 1:-1, 1:-1] + v_np_now[1:-1, :-2, 1:-1]))
            w_ft_d = np.where(m_t_d, 0.0, 0.5 * (w_np_now[1:-1, 1:-1, 2:] + w_np_now[1:-1, 1:-1, 1:-1]))
            w_fb_d = np.where(m_b_d, 0.0, 0.5 * (w_np_now[1:-1, 1:-1, 1:-1] + w_np_now[1:-1, 1:-1, :-2]))
            dp_e_d = np.where(m_e_d, 0.0, (p_np_now[2:, 1:-1, 1:-1] - p_np_now[1:-1, 1:-1, 1:-1]) * inv_dx)
            dp_w_d = np.where(m_w_d, 0.0, (p_np_now[1:-1, 1:-1, 1:-1] - p_np_now[:-2, 1:-1, 1:-1]) * inv_dx)
            dp_n_d = np.where(m_n_d, 0.0, (p_np_now[1:-1, 2:, 1:-1] - p_np_now[1:-1, 1:-1, 1:-1]) * inv_dy)
            dp_s_d = np.where(m_s_d, 0.0, (p_np_now[1:-1, 1:-1, 1:-1] - p_np_now[1:-1, :-2, 1:-1]) * inv_dy)
            dp_t_d = np.where(m_t_d, 0.0, (p_np_now[1:-1, 1:-1, 2:] - p_np_now[1:-1, 1:-1, 1:-1]) * inv_dz)
            dp_b_d = np.where(m_b_d, 0.0, (p_np_now[1:-1, 1:-1, 1:-1] - p_np_now[1:-1, 1:-1, :-2]) * inv_dz)
            u_fe_c = u_fe_d - dt * dp_e_d
            u_fw_c = u_fw_d - dt * dp_w_d
            v_fn_c = v_fn_d - dt * dp_n_d
            v_fs_c = v_fs_d - dt * dp_s_d
            w_ft_c = w_ft_d - dt * dp_t_d
            w_fb_c = w_fb_d - dt * dp_b_d
            div_corr_d = (u_fe_c - u_fw_c) * inv_dx + (v_fn_c - v_fs_c) * inv_dy + (w_ft_c - w_fb_c) * inv_dz
            div_max = float(np.max(np.abs(div_corr_d)))
            if prev_residual is None:
                vel_residual = float(
                    np.max(np.abs(u_np_now)) + np.max(np.abs(v_np_now)) + np.max(np.abs(w_np_now))
                )
            else:
                vel_residual = float(
                    np.max(np.abs(u_np_now - u_np_prev))
                    + np.max(np.abs(v_np_now - v_np_prev))
                    + np.max(np.abs(w_np_now - w_np_prev))
                )
            print(
                f"[OpenCL-NS] step={step:4d}, dt={dt:.2e}, div_max={div_max:.2e}, residual={vel_residual:.4e}, "
                f"|u|_max={np.max(np.abs(u_np_now)):.4f}, "
                f"|v|_max={np.max(np.abs(v_np_now)):.4f}, |w|_max={np.max(np.abs(w_np_now)):.4f}"
            )
            u_np_prev, v_np_prev, w_np_prev = u_np_now, v_np_now, w_np_now
            if progress_callback is not None:
                progress_callback(int(step), float(vel_residual), float(dt))
            if step > 50 and vel_residual < 1e-4:
                print(f"[OpenCL-NS] Converged at step {step}: residual={vel_residual:.4e}")
                break

    # Sonucu dondur
    u_out = slvr.to_numpy(u_cl).reshape((nx, ny, nz))
    v_out = slvr.to_numpy(v_cl).reshape((nx, ny, nz))
    w_out = slvr.to_numpy(w_cl).reshape((nx, ny, nz))
    p_out = slvr.to_numpy(p_cl).reshape((nx, ny, nz))

    print(f"[OpenCL-NS] Tamamlandi — |u|_max={np.max(np.abs(u_out)):.4f}")
    return u_out, v_out, w_out, p_out


def solve_navier_stokes_full(
    u_np,
    v_np,
    w_np,
    mask,
    v_wind,
    nx,
    ny,
    nz,
    solid_fraction=None,
    iterations=300,
    turbulence_model="spalart_allmaras",
    use_wall_functions=False,
    turb_iterations=None,
    rho=None,
    nu_mol=None,
    tunnel_dims=None,
    progress_callback=None,
):
    """İnavier-Stokes çözücüsü — konservatif difüzyon, Rhie-Chow, RK2 zaman integrasyonu.

    v6.0 iyileştirmeleri:
    - Konservatif viskozite termi: div(nu_eff * grad(u))
    - Doğru viskoz CFL: nu_eff_max kullanarak
    - Her adımda Poisson çözümü (poisson_interval kaldırıldı)
    - Tünel duvar şartları (y/z eksenleri için slip boundary)
    - Van Leer limiter (minmod yerine)
    - RK2 (Heun) zaman integrasyonu
    - Poisson yakınsama kontrolü
    """
    tx, ty, tz = TUNNEL_DIMS if tunnel_dims is None else tuple(float(v) for v in tunnel_dims)
    dx, dy, dz = tx / (nx - 1), ty / (ny - 1), tz / (nz - 1)
    rho = RHO if rho is None else float(rho)
    nu_mol = 1.5e-5 if nu_mol is None else float(nu_mol)

    # ── OpenCL GPU yolu ────────────────────────────────────────────────────
    # Tam fizik: turbulans + duvar fonksiyonlari + Rhie-Chow hybrid olarak
    # CPU (fp64) tarafinda hesaplanir, RK2 + viskozite GPU kernel'larinda.
    if gpu_backend.GPU_BACKEND == "opencl":
        return _solve_ns_opencl(
            u_np,
            v_np,
            w_np,
            mask,
            v_wind,
            nx,
            ny,
            nz,
            dx,
            dy,
            dz,
            nu_mol,
            solid_fraction,
            iterations,
            turbulence_model,
            use_wall_functions,
            turb_iterations=turb_iterations,
            tunnel_dims=tunnel_dims,
            progress_callback=progress_callback,
        )

    xp = gpu_backend.cp if gpu_backend.GPU else np
    u = xp.asarray(u_np, dtype=xp.float64)
    v = xp.asarray(v_np, dtype=xp.float64)
    w = xp.asarray(w_np, dtype=xp.float64)
    mask_g = xp.asarray(mask)
    if solid_fraction is not None:
        solid_fraction = xp.asarray(solid_fraction)
        mask_g = xp.logical_or(mask_g, solid_fraction >= 1.0)
        volume_weight = xp.clip(1.0 - solid_fraction, 0.0, 1.0)
    else:
        volume_weight = xp.ones_like(u, dtype=xp.float64)
    d_wall = _build_wall_distance_cache(mask_g, dx, dy, dz, xp)

    # ── Van Leer limiter ──────────────────────────────────────────────────
    def _van_leer_limiter(a, b):
        return xp.where(a * b > 0, 2.0 * a * b / (a + b + 1e-12), 0.0)

    # ── Bounded QUICK scheme (3. derece, bounded) ────────────────────────
    def _bounded_quick(f, vel, dx_local):
        """Bounded QUICK scheme - 3. derece doğruluk, boundedness korunarak.

        f_ppm = QUICK face value with boundedness constraint.
        """
        sl = (slice(1, -1), slice(1, -1), slice(1, -1))
        # Upwind ve downwind gradyanlar
        grad_up = (f[sl] - f[:-2, 1:-1, 1:-1]) / dx_local
        grad_dn = (f[2:, 1:-1, 1:-1] - f[sl]) / dx_local

        # QUICK face value (3. derece)
        f_quick = xp.where(
            vel[sl] > 0,
            f[sl] + 0.5 * _van_leer_limiter(grad_up, grad_dn) * dx_local,
            f[sl] - 0.5 * _van_leer_limiter(grad_up, grad_dn) * dx_local,
        )

        # Boundedness: face value'i hücre ortalamaları arasında tut
        f_min = xp.minimum(f[sl], f[2:, 1:-1, 1:-1])
        f_max = xp.maximum(f[sl], f[2:, 1:-1, 1:-1])
        f_quick = xp.clip(f_quick, f_min, f_max)

        return f_quick

    # ── Konservatif viskozite termi ───────────────────────────────────────
    def _viscous_conservative(f, nu_eff_field):
        """d/dx(nu_eff * du/dx) + d/dy(nu_eff * dv/dy) + d/dz(nu_eff * dw/dz) — konservatif form."""
        # x ekseni
        nu_e = 0.5 * (nu_eff_field[2:, 1:-1, 1:-1] + nu_eff_field[1:-1, 1:-1, 1:-1])
        nu_w = 0.5 * (nu_eff_field[:-2, 1:-1, 1:-1] + nu_eff_field[1:-1, 1:-1, 1:-1])
        visc_x = (
            nu_e * (f[2:, 1:-1, 1:-1] - f[1:-1, 1:-1, 1:-1]) / dx
            - nu_w * (f[1:-1, 1:-1, 1:-1] - f[:-2, 1:-1, 1:-1]) / dx
        ) / dx
        # y ekseni
        nu_n = 0.5 * (nu_eff_field[1:-1, 2:, 1:-1] + nu_eff_field[1:-1, 1:-1, 1:-1])
        nu_s = 0.5 * (nu_eff_field[1:-1, :-2, 1:-1] + nu_eff_field[1:-1, 1:-1, 1:-1])
        visc_y = (
            nu_n * (f[1:-1, 2:, 1:-1] - f[1:-1, 1:-1, 1:-1]) / dy
            - nu_s * (f[1:-1, 1:-1, 1:-1] - f[1:-1, :-2, 1:-1]) / dy
        ) / dy
        # z ekseni
        nu_t_top = 0.5 * (nu_eff_field[1:-1, 1:-1, 2:] + nu_eff_field[1:-1, 1:-1, 1:-1])
        nu_b = 0.5 * (nu_eff_field[1:-1, 1:-1, :-2] + nu_eff_field[1:-1, 1:-1, 1:-1])
        visc_z = (
            nu_t_top * (f[1:-1, 1:-1, 2:] - f[1:-1, 1:-1, 1:-1]) / dz
            - nu_b * (f[1:-1, 1:-1, 1:-1] - f[1:-1, 1:-1, :-2]) / dz
        ) / dz
        return visc_x + visc_y + visc_z

    # ── Adveksiyon (Van Leer limiter ile) ─────────────────────────────────
    # Gradyan formu: u·∇f. Van Leer-limited gradyan, upwind eğimi ile
    # sınırlandırılmış gradyanın blend'i olmalı (smooth alanda tam eğimi
    # verir; 0.5*slope tek başına 1.5x/0.5x sapma üretirdi).
    def adv(f, u_vel, v_vel, w_vel):
        sl = (slice(1, -1), slice(1, -1), slice(1, -1))
        fb = (f[sl] - f[:-2, 1:-1, 1:-1]) / dx
        ff = (f[2:, 1:-1, 1:-1] - f[sl]) / dx
        slope_x = _van_leer_limiter(fb, ff)
        fx = xp.where(u_vel[sl] > 0, fb + 0.5 * (slope_x - fb), ff - 0.5 * (slope_x - ff))

        fb = (f[sl] - f[1:-1, :-2, 1:-1]) / dy
        ff = (f[1:-1, 2:, 1:-1] - f[sl]) / dy
        slope_y = _van_leer_limiter(fb, ff)
        fy = xp.where(v_vel[sl] > 0, fb + 0.5 * (slope_y - fb), ff - 0.5 * (slope_y - ff))

        fb = (f[sl] - f[1:-1, 1:-1, :-2]) / dz
        ff = (f[1:-1, 1:-1, 2:] - f[sl]) / dz
        slope_z = _van_leer_limiter(fb, ff)
        fz = xp.where(w_vel[sl] > 0, fb + 0.5 * (slope_z - fb), ff - 0.5 * (slope_z - ff))

        return u_vel[sl] * fx + v_vel[sl] * fy + w_vel[sl] * fz

    # ── Tünel duvar şartları ──────────────────────────────────────────────
    def _apply_tunnel_wall_bc(vel_u, vel_v, vel_w):
        """Y/z tünel duvarları için slip (serbest kayma) sınır şartı."""
        # Y eksenleri (sol/sag duvarlar): du/dy = 0, v = 0
        vel_u[:, 0, :] = vel_u[:, 1, :]
        vel_v[:, 0, :] = 0.0
        vel_w[:, 0, :] = vel_w[:, 1, :]
        vel_u[:, -1, :] = vel_u[:, -2, :]
        vel_v[:, -1, :] = 0.0
        vel_w[:, -1, :] = vel_w[:, -2, :]
        # Z eksenleri (üst/alt duvarlar): du/dz = 0, w = 0
        vel_u[:, :, 0] = vel_u[:, :, 1]
        vel_v[:, :, 0] = vel_v[:, :, 1]
        vel_w[:, :, 0] = 0.0
        vel_u[:, :, -1] = vel_u[:, :, -2]
        vel_v[:, :, -1] = vel_v[:, :, -2]
        vel_w[:, :, -1] = 0.0
        return vel_u, vel_v, vel_w

    def _apply_outlet_bc(vel_u, vel_v, vel_w, p_field, dx):
        """Basınç tabanlı çıkış sınır şartı: ∂u/∂n = 0, p = p_ref.

        Çıkışta sıfır gradyan (Neumann) koşulu + basınç referansı.
        """
        # Çıkış yüzeyi (x = nx-1): zero-gradient
        vel_u[-1, :, :] = vel_u[-2, :, :]
        vel_v[-1, :, :] = vel_v[-2, :, :]
        vel_w[-1, :, :] = vel_w[-2, :, :]
        # Basınç referansı: çıkış ortasında p = 0
        mid_y = vel_u.shape[1] // 2
        mid_z = vel_u.shape[2] // 2
        p_ref = p_field[-1, mid_y, mid_z]
        p_field[-1, :, :] = p_ref
        return vel_u, vel_v, vel_w, p_field

    # ── RK2 zaman integrasyonu fonksiyonu ─────────────────────────────────
    def _ns_rhs(u_in, v_in, w_in, nu_eff_field):
        """Navier-Stokes sağ tarafı hesaplar: -adv + visc."""
        u_rhs = xp.zeros_like(u_in)
        v_rhs = xp.zeros_like(v_in)
        w_rhs = xp.zeros_like(w_in)
        sl = (slice(1, -1), slice(1, -1), slice(1, -1))
        # Adveksiyon HIZLARI da taşınan alanla aynı andan gelmeli (Heun/RK2
        # doğruluğu için); k1 = RHS(u^n), k2 = RHS(u^mid) — her ikisinde de
        # transport hızı o anki alanın kendisidir.
        u_rhs[sl] = (
            -adv(u_in, u_in, v_in, w_in) + _viscous_conservative(u_in, nu_eff_field)
        ) * volume_weight[sl]
        v_rhs[sl] = (
            -adv(v_in, u_in, v_in, w_in) + _viscous_conservative(v_in, nu_eff_field)
        ) * volume_weight[sl]
        w_rhs[sl] = (
            -adv(w_in, u_in, v_in, w_in) + _viscous_conservative(w_in, nu_eff_field)
        ) * volume_weight[sl]
        return u_rhs, v_rhs, w_rhs

    # ── Rhie-Chow yüzey hızları ──────────────────────────────────────────
    def _rhie_chow_face_velocities(u_cell, v_cell, w_cell, p_field, dt_local):
        """Collocated grid'de checkerboard basıncı önlemek için Rhie-Chow enterpolasyonu.

        Yüzey hızları = lineer enterpolasyon + basınç düzeltmesi (önceki basınç ile).
        """
        # X eksen yüzeyleri (i+1/2)
        u_face_e = 0.5 * (u_cell[2:, 1:-1, 1:-1] + u_cell[1:-1, 1:-1, 1:-1]) - (
            dt_local / (2.0 * rho * dx)
        ) * (p_field[2:, 1:-1, 1:-1] - p_field[1:-1, 1:-1, 1:-1])
        u_face_w = 0.5 * (u_cell[1:-1, 1:-1, 1:-1] + u_cell[:-2, 1:-1, 1:-1]) - (
            dt_local / (2.0 * rho * dx)
        ) * (p_field[1:-1, 1:-1, 1:-1] - p_field[:-2, 1:-1, 1:-1])
        # Y eksen yüzeyleri (j+1/2)
        v_face_n = 0.5 * (v_cell[1:-1, 2:, 1:-1] + v_cell[1:-1, 1:-1, 1:-1]) - (
            dt_local / (2.0 * rho * dy)
        ) * (p_field[1:-1, 2:, 1:-1] - p_field[1:-1, 1:-1, 1:-1])
        v_face_s = 0.5 * (v_cell[1:-1, 1:-1, 1:-1] + v_cell[1:-1, :-2, 1:-1]) - (
            dt_local / (2.0 * rho * dy)
        ) * (p_field[1:-1, 1:-1, 1:-1] - p_field[1:-1, :-2, 1:-1])
        # Z eksen yüzeyleri (k+1/2)
        w_face_t = 0.5 * (w_cell[1:-1, 1:-1, 2:] + w_cell[1:-1, 1:-1, 1:-1]) - (
            dt_local / (2.0 * rho * dz)
        ) * (p_field[1:-1, 1:-1, 2:] - p_field[1:-1, 1:-1, 1:-1])
        w_face_b = 0.5 * (w_cell[1:-1, 1:-1, 1:-1] + w_cell[1:-1, 1:-1, :-2]) - (
            dt_local / (2.0 * rho * dz)
        ) * (p_field[1:-1, 1:-1, 1:-1] - p_field[1:-1, 1:-1, :-2])
        return u_face_e, u_face_w, v_face_n, v_face_s, w_face_t, w_face_b

    TURBULENCE_UPDATE_INTERVAL = 3
    prev_residual = None
    p_prev = None

    for step in range(iterations):
        u_max = max(float(xp.max(xp.abs(u))), 1e-3)
        v_max = max(float(xp.max(xp.abs(v))), 1e-3)
        w_max = max(float(xp.max(xp.abs(w))), 1e-3)

        # ── Adaptif CFL safety factor ──────────────────────────────────────
        # Hız arttıkça daha muhafazakar ol, hız azaldıkça serbest bırak
        vel_max = max(u_max, v_max, w_max)
        if vel_max > 120:
            cfl_safety = 0.05
        elif vel_max > 60:
            cfl_safety = 0.08
        else:
            cfl_safety = 0.12

        # ── Turbulans viskozitesi ─────────────────────────────────────────
        if step % TURBULENCE_UPDATE_INTERVAL == 0:
            turb_lower = str(turbulence_model).lower()
            if turb_lower in ("spalart_allmaras", "spalart-allmaras"):
                sa_iters = 60 if turb_iterations is None else int(turb_iterations)
                nu_t = solve_spalart_allmaras(
                    u, v, w, mask_g, dx, dy, dz, nu_mol=nu_mol, iterations=sa_iters, xp=xp
                )
            elif turb_lower in ("k_epsilon", "k-epsilon", "ke"):
                ke_iters = 60 if turb_iterations is None else int(turb_iterations)
                nu_t = solve_k_epsilon(u, v, w, mask_g, dx, dy, dz, nu_mol=nu_mol, iterations=ke_iters, xp=xp)
            elif turb_lower in ("k_omega_sst", "k-omega-sst", "komega_sst", "sst"):
                sst_iters = 60 if turb_iterations is None else int(turb_iterations)
                nu_t = solve_k_omega_sst(
                    u, v, w, mask_g, dx, dy, dz, nu_mol=nu_mol, iterations=sst_iters, xp=xp
                )
            else:
                nu_t = _smagorinsky_nu(u, v, w, dx, dy, dz, xp, d_wall=d_wall)
        nu_eff = nu_mol + nu_t

        # ── Doğru viskoz CFL ─────────────────────────────────────────────
        # 3D difüzyon CFL: dt <= C / (2*nu*(1/dx^2 + 1/dy^2 + 1/dz^2))
        nu_eff_max = float(xp.max(xp.abs(nu_eff)))
        dt_conv = cfl_safety * min(dx / u_max, dy / v_max, dz / w_max)
        dt_visc = cfl_safety / (2.0 * nu_eff_max * (1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2) + 1e-12)
        dt = min(dt_conv, dt_visc, 2e-3)

        u_old, v_old, w_old = u.copy(), v.copy(), w.copy()

        # ── RK2 (Heun) zaman integrasyonu ─────────────────────────────────
        # k1
        k1_u, k1_v, k1_w = _ns_rhs(u_old, v_old, w_old, nu_eff)
        # Ara adım
        u_mid = u_old + dt * k1_u
        v_mid = v_old + dt * k1_v
        w_mid = w_old + dt * k1_w
        u_mid[mask_g] = 0.0
        v_mid[mask_g] = 0.0
        w_mid[mask_g] = 0.0
        u_mid[0, :, :] = v_wind
        u_mid, v_mid, w_mid = _apply_tunnel_wall_bc(u_mid, v_mid, w_mid)
        # k2
        k2_u, k2_v, k2_w = _ns_rhs(u_mid, v_mid, w_mid, nu_eff)

        u = u_old + 0.5 * dt * (k1_u + k2_u)
        v = v_old + 0.5 * dt * (k1_v + k2_v)
        w = w_old + 0.5 * dt * (k1_w + k2_w)

        if solid_fraction is not None:
            u = u * volume_weight
            v = v * volume_weight
            w = w * volume_weight

        u[mask_g] = 0.0
        v[mask_g] = 0.0
        w[mask_g] = 0.0

        # ── Giriş/Çıkış/Tünel sınır şartları ──────────────────────────────
        u[0, :, :] = v_wind
        v[0, :, :] = 0.0
        w[0, :, :] = 0.0
        u[-1, :, :] = u[-2, :, :]
        v[-1, :, :] = v[-2, :, :]
        w[-1, :, :] = w[-2, :, :]
        u, v, w = _apply_tunnel_wall_bc(u, v, w)

        # ── Katı hücreleri RHS'den ÖNCE sıfırla ───────────────────────────
        # RHS yüzey hızları katı komşu hücrelerle tutarlı olsun (yüzey akısı 0);
        # aksi halde düzeltme sonrası sıfırlama yüzey diverjansını bozuyor.
        u[mask_g] = 0.0
        v[mask_g] = 0.0
        w[mask_g] = 0.0

        # ── Basınç Poisson — Rhie-Chow divergansı ────────────────────────

        # Yüzey hızları: lineer enterpolasyon (divergans için). Katıya bitişik
        # yüzeylerde NORMAL akı 0 olmalı (impenetrable body) — aksi halde
        # cisim gözenekli davranır (0.5*u_fluid sızıntısı). Bu yüzeyler
        # düzeltmede de 0 sabit tutulur.
        m_e = mask_g[2:, 1:-1, 1:-1] | mask_g[1:-1, 1:-1, 1:-1]
        m_w = mask_g[1:-1, 1:-1, 1:-1] | mask_g[:-2, 1:-1, 1:-1]
        m_n = mask_g[1:-1, 2:, 1:-1] | mask_g[1:-1, 1:-1, 1:-1]
        m_s = mask_g[1:-1, 1:-1, 1:-1] | mask_g[1:-1, :-2, 1:-1]
        m_t = mask_g[1:-1, 1:-1, 2:] | mask_g[1:-1, 1:-1, 1:-1]
        m_b = mask_g[1:-1, 1:-1, 1:-1] | mask_g[1:-1, 1:-1, :-2]

        u_fe = xp.where(m_e, 0.0, 0.5 * (u[2:, 1:-1, 1:-1] + u[1:-1, 1:-1, 1:-1]))
        u_fw = xp.where(m_w, 0.0, 0.5 * (u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1]))
        v_fn = xp.where(m_n, 0.0, 0.5 * (v[1:-1, 2:, 1:-1] + v[1:-1, 1:-1, 1:-1]))
        v_fs = xp.where(m_s, 0.0, 0.5 * (v[1:-1, 1:-1, 1:-1] + v[1:-1, :-2, 1:-1]))
        w_ft = xp.where(m_t, 0.0, 0.5 * (w[1:-1, 1:-1, 2:] + w[1:-1, 1:-1, 1:-1]))
        w_fb = xp.where(m_b, 0.0, 0.5 * (w[1:-1, 1:-1, 1:-1] + w[1:-1, 1:-1, :-2]))

        # Divergans — yüzey hızlarından
        div_star = (u_fe - u_fw) / dx + (v_fn - v_fs) / dy + (w_ft - w_fb) / dz

        rhs = xp.zeros((nx, ny, nz), dtype=xp.float64)
        # Exact projection: RHS'te volume çarpanı YOK — düzeltme operatörüyle
        # birebir tutarlı (L p = div/dt => duzeltme sonrasi div = 0).
        rhs[1:-1, 1:-1, 1:-1] = div_star / dt

        p_np = _poisson_cg(rhs, mask_g, dx, dy, dz, nx, ny, nz, xp, x0=p_prev if p_prev is not None else None)
        p = xp.asarray(p_np)
        p[mask_g] = 0.0
        p_prev = p.copy()

        # ── Basınç düzeltmesi — exact projection (yüzey hızı düzeltmesi) ──
        # Yüzey bazlı basınç gradyanı (checkerboard önleme). Katı yüzeylerde
        # düzeltme yok: akı 0 sabit kalır (maske).
        dp_dx_face_e = xp.where(m_e, 0.0, (p[2:, 1:-1, 1:-1] - p[1:-1, 1:-1, 1:-1]) / dx)
        dp_dx_face_w = xp.where(m_w, 0.0, (p[1:-1, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / dx)
        dp_dy_face_n = xp.where(m_n, 0.0, (p[1:-1, 2:, 1:-1] - p[1:-1, 1:-1, 1:-1]) / dy)
        dp_dy_face_s = xp.where(m_s, 0.0, (p[1:-1, 1:-1, 1:-1] - p[1:-1, :-2, 1:-1]) / dy)
        dp_dz_face_t = xp.where(m_t, 0.0, (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, 1:-1]) / dz)
        dp_dz_face_b = xp.where(m_b, 0.0, (p[1:-1, 1:-1, 1:-1] - p[1:-1, 1:-1, :-2]) / dz)

        # Tutarlı çift: Poisson (7-nokta) + yüzey basınç gradyanı ile YÜZEY
        # hızlarını düzelt; hücre merkezleri düzeltilmiş yüzeylerin ortalaması.
        # (Eski: hücre merkezine pointwise gradyan — operatör uyumsuzdu,
        #  düzeltme sonrası diverjans ~500'de kalıyordu.)
        u_fe_c = u_fe - dt * dp_dx_face_e
        u_fw_c = u_fw - dt * dp_dx_face_w
        v_fn_c = v_fn - dt * dp_dy_face_n
        v_fs_c = v_fs - dt * dp_dy_face_s
        w_ft_c = w_ft - dt * dp_dz_face_t
        w_fb_c = w_fb - dt * dp_dz_face_b

        u[1:-1, 1:-1, 1:-1] = 0.5 * (u_fw_c + u_fe_c)
        v[1:-1, 1:-1, 1:-1] = 0.5 * (v_fs_c + v_fn_c)
        w[1:-1, 1:-1, 1:-1] = 0.5 * (w_fb_c + w_ft_c)

        # Düzeltilmiş yüzeylerden diverjans — Poisson toleransına iner (~1e-3).
        # Katı yüzeyleri 0 akı taşıdığından katı hücrelerde doğal olarak 0'dır.
        div_corr = (u_fe_c - u_fw_c) / dx + (v_fn_c - v_fs_c) / dy + (w_ft_c - w_fb_c) / dz

        if use_wall_functions:
            u, v, w = _apply_wall_functions(u, v, w, mask_g, d_wall, dx, dy, dz, nu_mol, xp)

        # ── Son sınır şartları ────────────────────────────────────────────
        u[mask_g] = 0.0
        v[mask_g] = 0.0
        w[mask_g] = 0.0
        u[0, :, :] = v_wind
        v[0, :, :] = 0.0
        w[0, :, :] = 0.0
        u[-1, :, :] = u[-2, :, :]
        v[-1, :, :] = v[-2, :, :]
        w[-1, :, :] = w[-2, :, :]
        u, v, w = _apply_tunnel_wall_bc(u, v, w)

        # ── NaN/Inf kontrolü (gelişmiş) ──────────────────────────────────
        for arr in [u, v, w]:
            bad = xp.isnan(arr) | xp.isinf(arr) | (xp.abs(arr) > v_wind * 5.0)
            if bool(bad.any()):
                n_bad = int(bad.sum())
                if n_bad > 0:
                    print(f"[NS] WARNING: {n_bad} bad velocity values detected and clamped (step={step})")
                arr[bad] = 0.0
        # Basınç: sadece NaN/Inf temizle (büyük değerler fiziksel olabilir)
        bad_p = xp.isnan(p) | xp.isinf(p)
        if bool(bad_p.any()):
            n_bad_p = int(bad_p.sum())
            print(f"[NS] WARNING: {n_bad_p} NaN/Inf pressure values clamped (step={step})")
            p[bad_p] = 0.0

        # ── Yakınsama izleme ──────────────────────────────────────────────
        if step % 10 == 0:
            div_max = float(xp.max(xp.abs(div_corr)))

            # Velocity residual: değişim oranını hesapla
            u_change = float(xp.max(xp.abs(u - u_old)))
            v_change = float(xp.max(xp.abs(v - v_old)))
            w_change = float(xp.max(xp.abs(w - w_old)))
            vel_residual = u_change + v_change + w_change

            if prev_residual is not None and prev_residual > 0:
                conv_rate = vel_residual / prev_residual
                print(
                    f"[NS] step={step:4d}, dt={dt:.2e}, div_max={div_max:.2e}, residual={vel_residual:.4e}, rate={conv_rate:.4f}"
                )
            else:
                print(f"[NS] step={step:4d}, dt={dt:.2e}, div_max={div_max:.2e}, residual={vel_residual:.4e}")
            prev_residual = vel_residual

            # Canli izleme: UI/worker callback'i (step, residual, dt)
            if progress_callback is not None:
                progress_callback(int(step), float(vel_residual), float(dt))

            # ── Erken durdurma: yakınsama kontrolü ────────────────────────
            # Residual 1e-4'ün altına düştüyse ve divergence da küçükse dur
            if step > 50 and vel_residual < 1e-4 and div_max < 1e-3:
                print(f"[NS] Converged at step {step}: residual={vel_residual:.4e}, div_max={div_max:.4e}")
                break

    if gpu_backend.GPU:
        u_out, v_out, w_out = _to_numpy(u), _to_numpy(v), _to_numpy(w)
        p_out = _to_numpy(p)
    else:
        u_out, v_out, w_out = np.asarray(u), np.asarray(v), np.asarray(w)
        p_out = np.asarray(p)

    return u_out, v_out, w_out, p_out


# ════════════════════════════════════════════════════════════════════════════
# KATMAN 4 — BASINÇ ALANI VE İTÜ STANDARDI Cd/Cl ENTEGRASYONU
# ════════════════════════════════════════════════════════════════════════════


def compute_mesh_refinement(
    mask: np.ndarray, dx: float, dy: float, dz: float, tunnel_dims: tuple, refinement_level: int = 2
) -> np.ndarray:
    """Duvarlar yakinlastirma icin adaptif mesh refinement haritasi uretir.

    Her hucre icin refinement seviyesini hesaplar:
    - 0: Duvar yok, coarse grid
    - 1: Duvar yakininda, intermediate
    - 2: Cok yakin duvar, fine grid

    Args:
        mask: Boolean grid - kati hucreler
        dx, dy, dz: Grid araliklari
        tunnel_dims: Tunel boyutlari (tx, ty, tz)
        refinement_level: Maksimum refinement seviyesi

    Returns:
        refinement_map: Her hucre icin refinement seviyesi (nx, ny, nz)
    """
    mask_np = mask.get() if hasattr(mask, "get") else np.asarray(mask, dtype=bool)

    # Duvar mesafesi hesapla
    dist = distance_transform_edt(~mask_np, sampling=(dx, dy, dz))

    # Refinement eşik değerleri
    # 1. seviye: duvar mesafesi < 3 hücre boyutu
    # 2. seviye: duvar mesafesi < 1.5 hücre boyutu
    threshold_1 = 3.0 * max(dx, dy, dz)
    threshold_2 = 1.5 * max(dx, dy, dz)

    refinement_map = np.zeros_like(dist, dtype=np.int32)
    refinement_map[dist < threshold_1] = 1
    refinement_map[dist < threshold_2] = 2

    return refinement_map


def compute_adaptive_timesteps(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    nu_eff: np.ndarray,
    v_wind: float,
    cfl_target: float = 0.5,
) -> float:
    """Adaptif zaman adimi hesaplama - yerel CFL tabanli.

    Her hucre icin yerel hiz ve viskoziteye gore optimal dt hesaplar.

    Args:
        u, v, w: Hiz alanlari (nx, ny, nz)
        dx, dy, dz: Grid araliklari
        nu_eff: Efektif viskozite alani (nx, ny, nz)
        v_wind: Ruzgar hizi (m/s)
        cfl_target: Hedef CFL sayisi

    Returns:
        dt_global: Kuresel maksimum zaman adimi
    """
    xp = gpu_backend.cp if gpu_backend.GPU else np

    # Girdileri doğru tipe dönüştür
    u = xp.asarray(u, dtype=xp.float64)
    v = xp.asarray(v, dtype=xp.float64)
    w = xp.asarray(w, dtype=xp.float64)
    nu_eff = xp.asarray(nu_eff, dtype=xp.float64)

    u_local = xp.abs(u[1:-1, 1:-1, 1:-1]) + 1e-10
    v_local = xp.abs(v[1:-1, 1:-1, 1:-1]) + 1e-10
    w_local = xp.abs(w[1:-1, 1:-1, 1:-1]) + 1e-10
    nu_local = nu_eff[1:-1, 1:-1, 1:-1] + 1e-10

    # Konvektif CFL
    dt_conv = cfl_target / (u_local / dx + v_local / dy + w_local / dz + 1e-10)

    # Difüzyon CFL
    dt_visc = cfl_target / (2.0 * nu_local * (1.0 / dx**2 + 1.0 / dy**2 + 1.0 / dz**2) + 1e-10)

    # İkisinden küçüğünü al
    dt_local = xp.minimum(dt_conv, dt_visc)

    # Küresel maksimum ile sınırla
    dt_global = float(xp.min(dt_local))

    return dt_global


def compute_pressure_field(u: np.ndarray, v: np.ndarray, w: np.ndarray, v_wind: float) -> np.ndarray:
    """Bernoulli'den Cp = 1 - (|u|² / v_wind²) — GPU/CPU uyumluluğu icin veri korumasi icerir.

    Args:
        u, v, w: Hiz alanlari (nx, ny, nz)
        v_wind: Ruzgar hizi (m/s)

    Returns:
        cp_field: Basinc katsayisi alani (nx, ny, nz)
    """
    use_gpu = gpu_backend.GPU and any(hasattr(arr, "get") for arr in (u, v, w) if arr is not None)
    xp = gpu_backend.cp if use_gpu else np

    u = xp.asarray(u)
    v = xp.asarray(v)
    w = xp.asarray(w)

    vel_sq = u**2 + v**2 + w**2
    vel_sq = xp.clip(vel_sq, 0.0, (v_wind * 2.0) ** 2)
    cp_field = 1.0 - vel_sq / (v_wind**2 + 1e-6)

    cp_field = xp.where(xp.isnan(cp_field) | xp.isinf(cp_field), 0.0, cp_field)
    cp_field = xp.clip(cp_field, -10.0, 5.0)

    return cp_field


def compute_force_coefficients(
    verts: np.ndarray,
    i_idx: np.ndarray,
    j_idx: np.ndarray,
    k_idx: np.ndarray,
    cp_field: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    tunnel_dims: tuple,
    nx: int,
    ny: int,
    nz: int,
    v_wind: float,
    mask: np.ndarray | None = None,
    use_wall_functions: bool = False,
    stl_normals: np.ndarray | None = None,
    p_field: np.ndarray | None = None,
    rho: float | None = None,
    nu_mol: float | None = None,
) -> dict:
    """ITU Aerodinamik Kuvvet Entegrasyon Standardi.

    Duv fonksiyonu etkinse, yuze yakin hucrelerdeki duvar mesafesini
    kullanarak viskoz duvar kesme gerilmesini daha dogru tahmin eder.

    Args:
        verts: STL vertex koordinatlari (N, 3)
        i_idx, j_idx, k_idx: Ucgen indeksleri
        cp_field: Basinc katsayisi alani (nx, ny, nz)
        u, v, w: Hiz alanlari (nx, ny, nz)
        tunnel_dims: Tunel boyutlari (tx, ty, tz)
        nx, ny, nz: Grid boyutlari
        v_wind: Ruzgar hizi (m/s)
        mask: Boolean grid - kati hucreler
        use_wall_functions: Duv fonksiyonu kullanilsin mi
        stl_normals: STL normal vektorleri
        p_field: Basinç alani (varsa)

    Returns:
        dict: Kuvvet katsayilari ve detaylar
    """
    use_gpu = gpu_backend.GPU and any(
        hasattr(arr, "get") for arr in (u, v, w, cp_field, mask) if arr is not None
    )
    xp = gpu_backend.cp if use_gpu else np

    u = xp.asarray(u)
    v = xp.asarray(v)
    w = xp.asarray(w)
    cp_field = xp.asarray(cp_field)
    if mask is not None:
        mask = xp.asarray(mask)

    tx, ty, tz = tunnel_dims
    dx_g = tx / (nx - 1)
    dy_g = ty / (ny - 1)
    dz_g = tz / (nz - 1)
    rho = RHO if rho is None else float(rho)
    nu_mol = 1.5e-5 if nu_mol is None else float(nu_mol)
    q_inf = 0.5 * rho * v_wind**2

    v0 = verts[i_idx]
    v1 = verts[j_idx]
    v2 = verts[k_idx]
    centroids = (v0 + v1 + v2) / 3.0

    e1 = v1 - v0
    e2 = v2 - v0
    normals_area = np.cross(e1, e2)
    areas = np.linalg.norm(normals_area, axis=1) / 2.0

    # STL normal'lerini kullan (varsa) — cross product yerine
    if stl_normals is not None and len(stl_normals) == len(areas):
        normals_unit = stl_normals.copy()
    else:
        normals_unit = normals_area / (2.0 * areas[:, np.newaxis] + 1e-12)
        model_center = verts.mean(axis=0)
        outward = centroids - model_center
        flip = np.sum(normals_unit * outward, axis=1) < 0
        normals_unit[flip] *= -1

    # ── Yüzey koordinatları ─────────────────────────────────────────────
    fx = ((centroids[:, 0] + tx / 2) / tx * (nx - 1)).astype(np.float64)
    fy = ((centroids[:, 1] + ty / 2) / ty * (ny - 1)).astype(np.float64)
    fz = ((centroids[:, 2] + tz / 2) / tz * (nz - 1)).astype(np.float64)

    normals_unit_xp = xp.asarray(normals_unit)
    areas_xp = xp.asarray(areas)
    mask_np = (
        mask.get()
        if mask is not None and hasattr(mask, "get")
        else (np.asarray(mask, dtype=bool) if mask is not None else None)
    )

    # Basınç kuvveti: p_field mevcutsa doğrudan kullan.
    # Not: NS cozucusunun Poisson ciktisi KINEMATIK basinçtir (p ~ m²/s²);
    # gerçek basınç P = rho * p olduğundan kuvvet rho ile ölçeklenir.
    if p_field is not None:
        p_cpu = p_field.get() if hasattr(p_field, "get") else np.asarray(p_field)
        xi = np.clip(fx.astype(np.int32), 0, nx - 1)
        yi = np.clip(fy.astype(np.int32), 0, ny - 1)
        zi = np.clip(fz.astype(np.int32), 0, nz - 1)
        # Katı hücrelerde p = 0 (dirichlet). Yüzey noktalarının çoğu katıya
        # düşebildiğinden, katıdakiler normal yönünde ilk FLUID hücreye
        # kaydırılır (aksi halde F_pressure ~ 0 olur).
        if mask is not None:
            sx = np.sign(normals_unit[:, 0]).astype(np.int32)
            sy = np.sign(normals_unit[:, 1]).astype(np.int32)
            sz = np.sign(normals_unit[:, 2]).astype(np.int32)
            for _ in range(8):
                hit = mask_np[xi, yi, zi]
                if not hit.any():
                    break
                xi[hit] = np.clip(xi[hit] + sx[hit], 0, nx - 1)
                yi[hit] = np.clip(yi[hit] + sy[hit], 0, ny - 1)
                zi[hit] = np.clip(zi[hit] + sz[hit], 0, nz - 1)
        p_face_np = p_cpu[xi, yi, zi]
        F_pressure = -rho * xp.sum(
            xp.asarray(p_face_np)[:, xp.newaxis] * normals_unit_xp * areas_xp[:, xp.newaxis], axis=0
        )
    else:
        F_pressure = xp.zeros(3)

    # Viskoz kuvvet için hız: yüzeye 1.5 hücre dışarıdan örnekleme.
    # Katıya düşen örnekleme noktaları normal yönünde ilk FLUID hücreye
    # kaydırılır (katıda hız 0 olduğundan trilinear bozulmasın).
    fx_nb = np.clip(fx + normals_unit[:, 0] * (nx - 1) / tx * dx_g, 0, nx - 1)
    fy_nb = np.clip(fy + normals_unit[:, 1] * (ny - 1) / ty * dy_g, 0, ny - 1)
    fz_nb = np.clip(fz + normals_unit[:, 2] * (nz - 1) / tz * dz_g, 0, nz - 1)
    x0n = np.clip(fx_nb.astype(np.int32), 0, nx - 2)
    y0n = np.clip(fy_nb.astype(np.int32), 0, ny - 2)
    z0n = np.clip(fz_nb.astype(np.int32), 0, nz - 2)
    if mask is not None:
        sx = np.sign(normals_unit[:, 0]).astype(np.int32)
        sy = np.sign(normals_unit[:, 1]).astype(np.int32)
        sz = np.sign(normals_unit[:, 2]).astype(np.int32)
        for _ in range(8):
            hit = mask_np[x0n, y0n, z0n]
            if not hit.any():
                break
            x0n[hit] = np.clip(x0n[hit] + sx[hit], 0, nx - 2)
            y0n[hit] = np.clip(y0n[hit] + sy[hit], 0, ny - 2)
            z0n[hit] = np.clip(z0n[hit] + sz[hit], 0, nz - 2)
    xdn = fx_nb - x0n
    ydn = fy_nb - y0n
    zdn = fz_nb - z0n

    def _trilinear(field, x0a, y0a, z0a, xda, yda, zda):
        f = field.get() if hasattr(field, "get") else np.asarray(field)
        c000 = f[x0a, y0a, z0a]
        c100 = f[x0a + 1, y0a, z0a]
        c010 = f[x0a, y0a + 1, z0a]
        c110 = f[x0a + 1, y0a + 1, z0a]
        c001 = f[x0a, y0a, z0a + 1]
        c101 = f[x0a + 1, y0a, z0a + 1]
        c011 = f[x0a, y0a + 1, z0a + 1]
        c111 = f[x0a + 1, y0a + 1, z0a + 1]
        c00 = c000 * (1 - xda) + c100 * xda
        c01 = c001 * (1 - xda) + c101 * xda
        c10 = c010 * (1 - xda) + c110 * xda
        c11 = c011 * (1 - xda) + c111 * xda
        c0 = c00 * (1 - yda) + c10 * yda
        c1 = c01 * (1 - yda) + c11 * yda
        return c0 * (1 - zda) + c1 * zda

    u_nb_np = _trilinear(u, x0n, y0n, z0n, xdn, ydn, zdn)
    v_nb_np = _trilinear(v, x0n, y0n, z0n, xdn, ydn, zdn)
    w_nb_np = _trilinear(w, x0n, y0n, z0n, xdn, ydn, zdn)
    vel_nb = xp.column_stack([xp.asarray(u_nb_np), xp.asarray(v_nb_np), xp.asarray(w_nb_np)])

    if mask is not None and use_wall_functions:
        d_wall = _build_wall_distance_cache(mask_np, dx_g, dy_g, dz_g, xp)
        # Yüzeye en yakın hücrelerin duvar mesafesini al
        x0w = np.clip(fx.astype(np.int32), 0, nx - 1)
        y0w = np.clip(fy.astype(np.int32), 0, ny - 1)
        z0w = np.clip(fz.astype(np.int32), 0, nz - 1)
        d_wall_point = d_wall[x0w, y0w, z0w]
        d_wall_ref = xp.maximum(d_wall_point, 0.1 * xp.minimum(xp.minimum(dx_g, dy_g), dz_g))
    else:
        d_wall_ref = xp.full(fx.shape, xp.minimum(xp.minimum(dx_g, dy_g), dz_g), dtype=xp.float64)

    normal_comp = xp.sum(vel_nb * normals_unit_xp, axis=1, keepdims=True) * normals_unit_xp
    tangential_vel = vel_nb - normal_comp
    tangential_speed = xp.linalg.norm(tangential_vel, axis=1)
    t_hat = xp.zeros_like(tangential_vel)
    nonzero = tangential_speed > 0
    t_hat[nonzero] = tangential_vel[nonzero] / tangential_speed[nonzero][:, xp.newaxis]

    u_tau = xp.sqrt(xp.maximum(nu_mol * tangential_speed / xp.maximum(d_wall_ref, 1e-6), 0.0))
    tau_wall = rho * u_tau**2
    F_viscous = xp.sum(tau_wall[:, xp.newaxis] * t_hat * areas_xp[:, xp.newaxis], axis=0)

    F_total = F_pressure + F_viscous

    y_range = verts[:, 1].max() - verts[:, 1].min()
    z_range = verts[:, 2].max() - verts[:, 2].min()
    A_ref = max(y_range * z_range, 0.01)

    Cd = F_total[0] / (q_inf * A_ref + 1e-6)
    Cl = F_total[2] / (q_inf * A_ref + 1e-6)

    if use_gpu:
        Cd = float(_to_numpy(Cd))
        Cl = float(_to_numpy(Cl))
        drag_N = float(_to_numpy(F_total[0]))
        lift_N = float(_to_numpy(F_total[2]))
    else:
        drag_N = F_total[0]
        lift_N = F_total[2]

    invalid_result = False
    if not np.isfinite(Cd) or not np.isfinite(Cl) or not np.isfinite(drag_N) or not np.isfinite(lift_N):
        invalid_result = True

    if invalid_result:
        aerodynamic_status = "Hesaplama kararsız, sonuç güvenilir değil"
        efficiency_label = "Geçersiz sonuç"
        Cd = float("nan")
        Cl = float("nan")
        drag_N = float("nan")
        lift_N = float("nan")
    else:
        if lift_N > 2.0:
            aerodynamic_status = f"UPFORCE (Kaldırma): {abs(lift_N):.1f} N"
            efficiency_label = "Model Karakteri: Kanat profili / taşıma eğilimli"
        elif lift_N < -2.0:
            aerodynamic_status = f"DOWNFORCE (Yere Basma): {abs(lift_N):.1f} N"
            efficiency_label = "Model Karakteri: Yarış/SUV tipi / yere basma odaklı"
        else:
            aerodynamic_status = "NÖTR (Dikey kuvvet dengede)"
            efficiency_label = "Model Karakteri: Simetrik / akışkan geometri"

    return {
        "Cd": float(Cd) if np.isfinite(Cd) else float("nan"),
        "Cl": float(Cl) if np.isfinite(Cl) else float("nan"),
        "drag_N": float(drag_N) if np.isfinite(drag_N) else float("nan"),
        "lift_N": float(lift_N) if np.isfinite(lift_N) else float("nan"),
        "A_ref": float(A_ref),
        "status": aerodynamic_status,
        "label": efficiency_label,
        "valid": not invalid_result,
    }


def interpolate_cp_to_vertices(
    verts: np.ndarray, cp_field: np.ndarray, mask: np.ndarray, tunnel_dims: tuple, nx: int, ny: int, nz: int
) -> np.ndarray:
    """Bilineer/Trilineer enterpolasyon ile grid Cp → kose Cp (Senkronize Eksenler).

    Args:
        verts: Vertex koordinatlari (N, 3)
        cp_field: Basinc katsayisi alani (nx, ny, nz)
        mask: Boolean grid - kati hucreler
        tunnel_dims: Tunel boyutlari (tx, ty, tz)
        nx, ny, nz: Grid boyutlari

    Returns:
        vertex_cp: Her vertex icin Cp degeri (N,)
    """
    tx, ty, tz = tunnel_dims
    cp_cpu = cp_field.get() if hasattr(cp_field, "get") else np.asarray(cp_field)

    fx = (verts[:, 0] + tx / 2) / tx * (nx - 1)
    fy = (verts[:, 1] + ty / 2) / ty * (ny - 1)
    fz = (verts[:, 2] + tz / 2) / tz * (nz - 1)

    x0 = np.clip(fx.astype(np.int32), 0, nx - 2)
    y0 = np.clip(fy.astype(np.int32), 0, ny - 2)
    z0 = np.clip(fz.astype(np.int32), 0, nz - 2)
    x1, y1, z1 = x0 + 1, y0 + 1, z0 + 1

    xd, yd, zd = fx - x0, fy - y0, fz - z0

    c000, c100 = cp_cpu[x0, y0, z0], cp_cpu[x1, y0, z0]
    c010, c110 = cp_cpu[x0, y1, z0], cp_cpu[x1, y1, z0]
    c001, c101 = cp_cpu[x0, y0, z1], cp_cpu[x1, y0, z1]
    c011, c111 = cp_cpu[x0, y1, z1], cp_cpu[x1, y1, z1]

    c00 = c000 * (1 - xd) + c100 * xd
    c01 = c001 * (1 - xd) + c101 * xd
    c10 = c010 * (1 - xd) + c110 * xd
    c11 = c011 * (1 - xd) + c111 * xd
    c0 = c00 * (1 - yd) + c10 * yd
    c1 = c01 * (1 - yd) + c11 * yd

    return c0 * (1 - zd) + c1 * zd


def rotate_vertices(verts: np.ndarray, yaw_deg: float, pitch_deg: float, roll_deg: float) -> np.ndarray:
    """Vertex donusumu - Yaw, Pitch, Roll acilariyla.

    Args:
        verts: Vertex koordinatlari (N, 3)
        yaw_deg: Yuruyus acisi (derece)
        pitch_deg: Pitch acisi (derece)
        roll_deg: Roll acisi (derece)

    Returns:
        np.ndarray: Donmus vertex koordinatlari (N, 3)
    """
    yaw, pitch, roll = np.radians(yaw_deg), np.radians(pitch_deg), np.radians(roll_deg)
    R_z = np.array(
        [[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]], dtype=np.float32
    )
    R_y = np.array(
        [[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float32
    )
    R_x = np.array(
        [[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]], dtype=np.float32
    )
    return verts @ (R_z @ R_y @ R_x).T


def apply_offset(verts: np.ndarray, ox: float, oy: float, oz: float) -> np.ndarray:
    """Vertex ofseti uygula.

    Args:
        verts: Vertex koordinatlari (N, 3)
        ox, oy, oz: Ofset degerleri (metre)

    Returns:
        np.ndarray: Ofset uygulanmis vertex koordinatlari (N, 3)
    """
    return verts + np.array([ox, oy, oz], dtype=np.float32)


def compute_q_criterion(
    u: np.ndarray, v: np.ndarray, w: np.ndarray, dx: float, dy: float, dz: float, GPU: bool = True
) -> np.ndarray:
    """Q-Criterion hesaplama - Turbulans tespiti icin.

    Q = 0.5 * (|Omega|^2 - |S|^2)
    Omega: Rotasyon hizi tensoru
    S: Gerinim hizi tensoru

    Args:
        u, v, w: Hiz alanlari (nx, ny, nz)
        dx, dy, dz: Grid araliklari
        GPU: GPU kullanilsin mi

    Returns:
        np.ndarray: Q-Criterion alani (nx, ny, nz)
    """
    xp = gpu_backend.cp if GPU else np
    u, v, w = xp.asarray(_to_numpy(u)), xp.asarray(_to_numpy(v)), xp.asarray(_to_numpy(w))

    dudx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx)
    dudy = (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2 * dy)
    dudz = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * dz)

    dvdy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)
    dwdz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)

    extra_v = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2 * dx)
    dwdx = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dx)
    dvdz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2 * dz)
    dwdy = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2 * dy)

    omega_sq = (dudy - extra_v) ** 2 + (dudz - dwdx) ** 2 + (dvdz - dwdy) ** 2
    strain_sq = (
        dudx**2 + dvdy**2 + dwdz**2 + 0.5 * ((dudy + extra_v) ** 2 + (dudz + dwdx) ** 2 + (dvdz + dwdy) ** 2)
    )

    q = xp.zeros_like(u)
    q[1:-1, 1:-1, 1:-1] = 0.5 * (omega_sq - strain_sq)

    q_threshold = 1e-4
    q[xp.abs(q) < q_threshold] = 0.0

    return _to_numpy(q) if GPU else q
