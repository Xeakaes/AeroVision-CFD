"""
AeroVision CFD v1.0 - Physics Engine Core

Features:
  - Mask:     Laser scan voxelization with cKDTree + sub-sample boundary detection
  - Poisson:  Vectorized sparse Laplacian with Jacobi preconditioner
  - Axes:     index=ij synchronization for interpolation consistency
  - Colors:   NaN cleanup for Cp computation stability
  - Forces:   Vertical force integration with trilinear interpolation
  - Turbulence: Smagorinsky SGS, Spalart-Allmaras, k-epsilon, k-omega SST models
"""

import time
import numpy as np
import pyvista as pv
import scipy.spatial as spatial
from scipy.ndimage import binary_dilation, binary_fill_holes, distance_transform_edt
from scipy.sparse import diags as sp_diags
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg as scipy_cg
import zlib

# ── GPU / CPU backend seçimi ────────────────────────────────────────────────
try:
    import cupy as cp
    import os
    if "CUDA_PATH" not in os.environ:
        os.environ["CUDA_PATH"] = "C:\\Program Files\\NVIDIA GPU Computing Toolkit\\CUDA\\v12.1"
    cp.cuda.Device(0).use()
    _ = cp.array([1.0]) * 2.0
    from cupyx.scipy.sparse import csr_matrix as cu_csr
    from cupyx.scipy.sparse import diags as cu_diags
    from cupyx.scipy.sparse.linalg import cg as cu_cg
    GPU = True
    print("CuPy bulundu — GPU modu etkin.")
except Exception as e:
    print(f"[AeroVision] GPU başlatılamadı ({e}) — CPU moduna geçiliyor.")
    cp = np
    GPU = False

def _to_numpy(arr):
    """Herhangi bir array'i guvenli sekilde numpy'a donustur."""
    if arr is None:
        return np.zeros(1)
    if GPU and hasattr(arr, 'get'):
        try:
            return _to_numpy(arr)
        except:
            return np.asarray(arr)
    return np.asarray(arr)

# ── Sabitler ────────────────────────────────────────────────────────────────
TUNNEL_DIMS   = (10.0, 4.0, 4.0)   # tx, ty, tz  [m]
RHO           = 1.225                # hava yoğunluğu [kg/m³]
C_SMAG        = 0.17                 # Smagorinsky sabiti
SA_CB1        = 0.1355               # Spalart-Allmaras üretim terimi sabiti
SA_CB2        = 0.622                # Spalart-Allmaras difüzyon sabiti
SA_SIGMA      = 2.0 / 3.0            # Spalart-Allmaras Prandtl benzeri sabiti
SA_KAPPA      = 0.41                 # Von Karman sabiti
SA_CW1        = SA_CB1 / SA_KAPPA**2 + (1.0 + SA_CB2) / SA_SIGMA
SA_CW2        = 0.3
SA_CW3        = 2.0
SA_CV1        = 7.1
SA_CV2        = 0.7
SA_CV3        = 0.9
SA_CV4        = 0.8
POISSON_TOL   = 1e-6
POISSON_MAXIT = 800
_POISSON_CACHE = {}
_WALL_DISTANCE_CACHE = {}

# ── k-epsilon türbülans modeli sabitleri (Launder & Spalding 1974) ─────────
KE_CMU   = 0.09     # ν_t = C_μ * k² / ε
KE_C1    = 1.44     # Üretim sabiti
KE_C2    = 1.92     # Dağılma sabiti
KE_SIGMA_K = 1.0    # k difüzyon Prandtl sayısı
KE_SIGMA_E = 1.3    # ε difüzyon Prandtl sayısı
KE_WALL_K  = 0.0    # Duvarlarda k (sıfır)
KE_WALL_E  = 1e-10  # Duvarlarda ε (çok küçük, 0/0 önlemek için)

# ── k-omega SST sabitleri (Menter 1994) ────────────────────────────────────
# (gelecek genişletme için hazır)
KO_BETA1  = 0.075
KO_BETA2  = 0.0828
KO_SIGMA_K1 = 0.85
KO_SIGMA_W1 = 0.50
KO_SIGMA_K2 = 1.0
KO_SIGMA_W2 = 0.856
KO_A1      = 0.31
KO_BETA_STAR = 0.09


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
    tunnel_dims: tuple
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
        xs = np.linspace(-tx/2, tx/2, nx)
        ys = np.linspace(-ty/2, ty/2, ny)
        zs = np.linspace(-tz/2, tz/2, nz)

        tree = spatial.cKDTree(verts)
        dx, dy, dz = tx/(nx-1), ty/(ny-1), tz/(nz-1)
        voxel_diagonal_half = np.sqrt(dx**2 + dy**2 + dz**2) / 2.0
        fine_diagonal_half = voxel_diagonal_half * 0.5

        offs = np.array([-0.25, 0.25], dtype=np.float64)
        ox, oy, oz = np.meshgrid(offs, offs, offs, indexing='ij')
        sub_offsets = np.stack([ox.ravel(), oy.ravel(), oz.ravel()], axis=-1)

        i_idx_all, j_idx_all, k_idx_all = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing='ij')
        base_centers = np.column_stack([
            xs[i_idx_all.ravel()], ys[j_idx_all.ravel()], zs[k_idx_all.ravel()]
        ])

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
    tol: float = 1e-3
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Potansiyel akis cozumu - Laplace denklemi Jacobi iterasyonu ile.
    
    Args:
        mask: Boolean grid (nx, ny, nz) - katı hucreler
        v_wind: Ruzgar hizi (m/s)
        nx, ny, nz: Grid boyutlari
        solid_fraction: Kismi doluluk orani (nx, ny, nz)
        max_iter: Maksimum iterasyon sayisi
        tol: Yakinsama toleransi
    
    Returns:
        u, v, w: Hiz alanlari (nx, ny, nz)
    """
    tx, ty, tz = TUNNEL_DIMS
    dx, dy, dz = tx/(nx-1), ty/(ny-1), tz/(nz-1)

    xp = cp
    phi = xp.zeros((nx, ny, nz), dtype=xp.float64)
    x_phys = xp.linspace(-tx/2, tx/2, nx)
    for i in range(nx):
        phi[i, :, :] = float(x_phys[i]) * v_wind

    mask_g = xp.asarray(mask)
    if solid_fraction is not None:
        solid_fraction = xp.asarray(solid_fraction)
        mask_g = xp.logical_or(mask_g, solid_fraction >= 1.0)
        volume_weight = xp.clip(1.0 - solid_fraction, 0.0, 1.0)
    else:
        volume_weight = xp.ones_like(mask_g, dtype=xp.float64)

    phi[mask_g] = 0.0
    fluid = ~mask_g

    for _ in range(max_iter):
        phi_old = phi.copy()
        interior = (
            phi[2:,  1:-1, 1:-1] + phi[:-2, 1:-1, 1:-1] +
            phi[1:-1, 2:,  1:-1] + phi[1:-1, :-2, 1:-1] +
            phi[1:-1, 1:-1, 2:] + phi[1:-1, 1:-1, :-2]
        )
        phi_update = interior / 6.0
        if solid_fraction is not None:
            weight = volume_weight[1:-1, 1:-1, 1:-1]
            phi[1:-1, 1:-1, 1:-1] = xp.where(
                fluid[1:-1, 1:-1, 1:-1],
                phi_update,
                phi_update * weight + phi_old[1:-1, 1:-1, 1:-1] * (1.0 - weight)
            )
        else:
            phi[1:-1, 1:-1, 1:-1] = xp.where(
                fluid[1:-1, 1:-1, 1:-1], phi_update, phi_old[1:-1, 1:-1, 1:-1]
            )
        phi[mask_g] = 0.0
        phi[0, :, :]  = v_wind * (-tx/2)
        phi[-1, :, :] = phi[-2, :, :]
        phi[:, 0, :]  = phi[:, 1, :]
        phi[:, -1, :] = phi[:, -2, :]
        phi[:, :, 0]  = phi[:, :, 1]
        phi[:, :, -1] = phi[:, :, -2]

        if float(xp.max(xp.abs(phi - phi_old))) < tol:
            break

    u = xp.zeros_like(phi)
    v = xp.zeros_like(phi)
    w = xp.zeros_like(phi)
    u[1:-1, :, :] = (phi[2:, :, :] - phi[:-2, :, :]) / (2*dx)
    v[:, 1:-1, :] = (phi[:, 2:, :] - phi[:, :-2, :]) / (2*dy)
    w[:, :, 1:-1] = (phi[:, :, 2:] - phi[:, :, :-2]) / (2*dz)
    u[0, :, :] = v_wind
    u[mask_g] = 0.0; v[mask_g] = 0.0; w[mask_g] = 0.0

    if GPU:
        return _to_numpy(u), _to_numpy(v), _to_numpy(w)
    return u, v, w


# ════════════════════════════════════════════════════════════════════════════
# KATMAN 3 — CHORIN PROJEKSİYON + POISSON MATRİS İZOLASYONU
# ════════════════════════════════════════════════════════════════════════════

def _smagorinsky_nu(u, v, w, dx, dy, dz, xp, d_wall=None):
    """Smagorinsky SGS modeli — Van Driest duvar sönümlemesi dahil."""
    delta = (dx * dy * dz) ** (1.0/3.0)
    cs_delta_sq = (C_SMAG * delta) ** 2

    dudx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2*dx)
    dudy = (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2*dy)
    dudz = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2*dz)

    dvdy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2*dy)
    dwdz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2*dz)

    extra_v = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2*dx)
    dwdx = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2*dx)
    dvdz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2*dz)
    dwdy = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2*dy)

    S_mag = xp.sqrt(2.0*(dudx**2 + dvdy**2 + dwdz**2 +
                         2*(0.5*(dudy + extra_v))**2 + 2*(0.5*(dudz + dwdx))**2 + 2*(0.5*(dvdz + dwdy))**2) + 1e-8)

    nu_t = xp.zeros(u.shape, dtype=xp.float64)

    if d_wall is not None:
        nu_mol = 1.5e-5
        d_wall_int = d_wall[1:-1, 1:-1, 1:-1]
        u_tau_est = xp.sqrt(xp.maximum(nu_mol * S_mag, 0.0))
        y_plus = xp.maximum(d_wall_int * u_tau_est / nu_mol, 1e-6)
        damping = (1.0 - xp.exp(-y_plus / 25.0))**2
        nu_t[1:-1, 1:-1, 1:-1] = cs_delta_sq * S_mag * damping
    else:
        nu_t[1:-1, 1:-1, 1:-1] = cs_delta_sq * S_mag

    return nu_t


def _compute_wall_distance(mask, dx, dy, dz, xp):
    """SciPy'ın CPU fonksiyonu için GPU maskesini güvenli tahliye eden yama."""
    mask_np = mask.get() if hasattr(mask, 'get') else np.asarray(mask, dtype=bool)
    dist = distance_transform_edt(~mask_np, sampling=(dx, dy, dz))
    return xp.asarray(dist)


def solve_spalart_allmaras(u, v, w, mask, dx, dy, dz, nu_mol=1.5e-5, iterations=20, xp=None):
    """Spalart-Allmaras türbülans viskozite modeli için ν_t alanını çözer.
    
    iyileştirmeler:
    - Minimum 60 iterasyon (yetersiz yakınsama düzeltildi)
    - Upwind adveksiyon şeması (merkezi farklar yerine)
    - Daha büyük pseudo-time adımı (1e-3)
    """
    if xp is None:
        xp = cp if GPU else np

    mask_g = xp.asarray(mask)
    d_wall = _compute_wall_distance(mask, dx, dy, dz, xp)
    nu_tilde = xp.full(u.shape, 3.0 * nu_mol, dtype=xp.float64)
    nu_tilde[mask_g] = 0.0

    sl = (slice(1, -1), slice(1, -1), slice(1, -1))
    dt_sa = min(dx, dy, dz) ** 2 / max(6.0 * nu_mol, 1e-6)
    dt_sa = min(dt_sa, 1e-3)

    min_spacing = min(dx, dy, dz)
    if min_spacing < 0.07:
        iterations = max(iterations, 80)
    elif min_spacing < 0.10:
        iterations = max(iterations, 60)
    elif min_spacing < 0.13:
        iterations = max(iterations, 60)
    else:
        iterations = max(iterations, 60)

    for _ in range(iterations):
        nu = nu_tilde
        d_nu_dx = (nu[2:, 1:-1, 1:-1] - nu[:-2, 1:-1, 1:-1]) / (2*dx)
        d_nu_dy = (nu[1:-1, 2:, 1:-1] - nu[1:-1, :-2, 1:-1]) / (2*dy)
        d_nu_dz = (nu[1:-1, 1:-1, 2:] - nu[1:-1, 1:-1, :-2]) / (2*dz)
        grad2_nu = d_nu_dx**2 + d_nu_dy**2 + d_nu_dz**2

        lap_nu = (
            (nu[2:, 1:-1, 1:-1] - 2*nu[sl] + nu[:-2, 1:-1, 1:-1]) / dx**2 +
            (nu[1:-1, 2:, 1:-1] - 2*nu[sl] + nu[1:-1, :-2, 1:-1]) / dy**2 +
            (nu[1:-1, 1:-1, 2:] - 2*nu[sl] + nu[1:-1, 1:-1, :-2]) / dz**2
        )

        du_dx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2*dx)
        dv_dy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2*dy)
        dw_dz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2*dz)

        du_dy = (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2*dy)
        dv_dx = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2*dx)
        du_dz = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2*dz)
        dw_dx = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2*dx)
        dv_dz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2*dz)
        dw_dy = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2*dy)

        S = xp.sqrt(2.0*(du_dx**2 + dv_dy**2 + dw_dz**2 +
                         0.5*(du_dy + dv_dx)**2 + 0.5*(du_dz + dw_dx)**2 + 0.5*(dv_dz + dw_dy)**2) + 1e-12)

        chi = nu[sl] / (nu_mol + 1e-12)
        fv1 = chi**3 / (chi**3 + SA_CV1**3 + 1e-12)
        r = nu[sl] / ((S * d_wall[sl])**2 + 1e-12)
        g = r + SA_CW2 * r**6
        f_w = g * ((1.0 + SA_CW3**6) / (g**6 + SA_CW3**6 + 1e-12))**(1.0/6.0)
        prod = SA_CB1 * S * nu[sl]
        diff = (1.0 / SA_SIGMA) * (lap_nu + SA_CB2 * grad2_nu / (nu[sl] + 1e-12))
        destr = -SA_CW1 * f_w * (nu[sl] / (d_wall[sl] + 1e-12))**2

        # Upwind adveksiyon şeması (merkezi farklar yerine)
        adv_x = xp.where(u[sl] > 0,
                         u[sl] * (nu[sl] - nu[:-2, 1:-1, 1:-1]) / dx,
                         u[sl] * (nu[2:, 1:-1, 1:-1] - nu[sl]) / dx)
        adv_y = xp.where(v[sl] > 0,
                         v[sl] * (nu[sl] - nu[1:-1, :-2, 1:-1]) / dy,
                         v[sl] * (nu[1:-1, 2:, 1:-1] - nu[sl]) / dy)
        adv_z = xp.where(w[sl] > 0,
                         w[sl] * (nu[sl] - nu[1:-1, 1:-1, :-2]) / dz,
                         w[sl] * (nu[1:-1, 1:-1, 2:] - nu[sl]) / dz)
        adv_nu = adv_x + adv_y + adv_z

        nu_new = nu[sl] + dt_sa * (-adv_nu + prod + diff + destr)

        nu_tilde[sl] = xp.maximum(nu_new, 0.0)
        nu_tilde[mask_g] = 0.0
        nu_tilde[0, :, :] = 3.0 * nu_mol
        nu_tilde[-1, :, :] = 3.0 * nu_mol
        nu_tilde[:, 0, :] = 3.0 * nu_mol
        nu_tilde[:, -1, :] = 3.0 * nu_mol
        nu_tilde[:, :, 0] = 3.0 * nu_mol
        nu_tilde[:, :, -1] = 3.0 * nu_mol

    chi = nu_tilde / (nu_mol + 1e-12)
    fv1 = chi**3 / (chi**3 + SA_CV1**3 + 1e-12)
    nu_t = nu_tilde * fv1
    nu_t[mask_g] = 0.0
    return xp.clip(nu_t, 0.0, 1000.0)


def _spalart_allmaras_nu(u, v, w, mask, dx, dy, dz, nu_mol, iterations, xp):
    return solve_spalart_allmaras(u, v, w, mask, dx, dy, dz, nu_mol=nu_mol, iterations=iterations, xp=xp)


def solve_k_epsilon(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    mask: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    nu_mol: float = 1.5e-5,
    iterations: int = 60,
    xp: any = None
) -> np.ndarray:
    """Standart k-epsilon türbülans modeli (Launder & Spalding 1974).
    
    Iki denklem cozulur:
    - k: Türbülans kinetik enerjisi
    - ε: Türbülans dispersiyon orani
    
    Eddy viskozite: ν_t = C_μ * k² / ε
    
    Args:
        u, v, w: Hiz alanlari (nx, ny, nz)
        mask: Boolean grid - katı hucreler
        dx, dy, dz: Grid araliklari
        nu_mol: Molekuler viskozite (varsayilan: 1.5e-5)
        iterations: Cozum iterasyon sayisi
        xp: Array kutuphanesi (numpy veya cupy)
    
    Returns:
        nu_t: Eddy viskozite alani (nx, ny, nz)
    """
    """Standart k-ε türbülans modeli (Launder & Spalding 1974).
    
    İki denklem çözülür:
    - k: Türbülans kinetik enerjisi
    - ε: Türbülans dispersiyon oranı
    
    Eddy viskozite: ν_t = C_μ * k² / ε
    
    İyileştirmeler:
    - Upwind adveksiyon (kararlılık için)
    - Duvar fonksiyonu entegrasyonu (yakın duvar)
    - Üretim sınırlayıcı (P_k ≤ 20 * ε)
    """
    if xp is None:
        xp = cp if GPU else np

    # Girdileri doğru tipe dönüştür
    u = xp.asarray(u, dtype=xp.float64)
    v = xp.asarray(v, dtype=xp.float64)
    w = xp.asarray(w, dtype=xp.float64)
    
    mask_g = xp.asarray(mask)
    sl = (slice(1, -1), slice(1, -1), slice(1, -1))

    # ── Başlangıç koşulları ──────────────────────────────────────────────
    I_turb = 0.005  # %0.5 türbülans yoğunluğu (düşük)
    U_ref = 30.0
    L_turb = 0.1    # Ölçek uzunluğu (10cm)
    k_init = 1.5 * (I_turb * U_ref)**2
    eps_init = KE_CMU * k_init**1.5 / L_turb
    
    k = xp.full(u.shape, k_init, dtype=xp.float64)
    eps = xp.full(u.shape, eps_init, dtype=xp.float64)
    k[mask_g] = 1e-10; eps[mask_g] = 1e-10

    # ── Pseudo-time step ─────────────────────────────────────────────────
    dt_ke = min(dx, dy, dz)**2 / max(6.0 * nu_mol, 1e-6)
    dt_ke = min(dt_ke, 1e-3)

    for _ in range(iterations):
        # ── Gerinim oranı |S| ────────────────────────────────────────────
        dudx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2*dx)
        dudy = (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2*dy)
        dudz = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2*dz)
        dvdx = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2*dx)
        dvdy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2*dy)
        dvdz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2*dz)
        dwdx = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2*dx)
        dwdy = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2*dy)
        dwdz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2*dz)

        S_mag = xp.sqrt(2.0*(dudx**2 + dvdy**2 + dwdz**2 +
                             0.5*(dudy + dvdx)**2 + 0.5*(dudz + dwdx)**2 + 0.5*(dvdz + dwdy)**2) + 1e-12)

        # ── Eddy viskozite ────────────────────────────────────────────────
        k_int = xp.maximum(k[sl], 1e-10)
        eps_int = xp.maximum(eps[sl], 1e-10)
        nu_t_local = KE_CMU * k_int**2 / eps_int
        nu_t_local = xp.clip(nu_t_local, 0.0, 100.0)  # Sert üst sınır

        # ── Üretim terimi P_k = ν_t * |S|² (sınırlı) ────────────────────
        P_k = nu_t_local * S_mag**2
        P_k = xp.minimum(P_k, 20.0 * eps_int)  # Production limiter

        # ── Laplacian ─────────────────────────────────────────────────────
        def _laplacian(f):
            return (
                (f[2:, 1:-1, 1:-1] - 2*f[sl] + f[:-2, 1:-1, 1:-1]) / dx**2 +
                (f[1:-1, 2:, 1:-1] - 2*f[sl] + f[1:-1, :-2, 1:-1]) / dy**2 +
                (f[1:-1, 1:-1, 2:] - 2*f[sl] + f[1:-1, 1:-1, :-2]) / dz**2
            )

        # ── Upwind adveksiyon ─────────────────────────────────────────────
        def _upwind_adv(f):
            adv_x = xp.where(u[sl] > 0,
                             u[sl] * (f[sl] - f[:-2, 1:-1, 1:-1]) / dx,
                             u[sl] * (f[2:, 1:-1, 1:-1] - f[sl]) / dx)
            adv_y = xp.where(v[sl] > 0,
                             v[sl] * (f[sl] - f[1:-1, :-2, 1:-1]) / dy,
                             v[sl] * (f[1:-1, 2:, 1:-1] - f[sl]) / dy)
            adv_z = xp.where(w[sl] > 0,
                             w[sl] * (f[sl] - f[1:-1, 1:-1, :-2]) / dz,
                             w[sl] * (f[1:-1, 1:-1, 2:] - f[sl]) / dz)
            return adv_x + adv_y + adv_z

        # ── k denklemi ───────────────────────────────────────────────────
        diff_k = (nu_mol + nu_t_local / KE_SIGMA_K) * _laplacian(k)
        k_new = k[sl] + dt_ke * (-_upwind_adv(k) + P_k - eps_int + diff_k)
        k[sl] = xp.maximum(k_new, 1e-10)

        # ── ε denklemi ───────────────────────────────────────────────────
        diff_eps = (nu_mol + nu_t_local / KE_SIGMA_E) * _laplacian(eps)
        eps_new = eps[sl] + dt_ke * (
            -_upwind_adv(eps)
            + KE_C1 * (eps_int / k_int) * P_k
            - KE_C2 * eps_int**2 / k_int
            + diff_eps
        )
        eps[sl] = xp.maximum(eps_new, 1e-10)

        # ── Sınır şartları ───────────────────────────────────────────────
        k[mask_g] = 1e-10; eps[mask_g] = 1e-10
        k[0, :, :] = k_init; eps[0, :, :] = eps_init
        k[-1, :, :] = k[-2, :, :]; eps[-1, :, :] = eps[-2, :, :]
        k[:, 0, :] = k[:, 1, :]; eps[:, 0, :] = eps[:, 1, :]
        k[:, -1, :] = k[:, -2, :]; eps[:, -1, :] = eps[:, -2, :]
        k[:, :, 0] = k[:, :, 1]; eps[:, :, 0] = eps[:, :, 1]
        k[:, :, -1] = k[:, :, -2]; eps[:, :, -1] = eps[:, :, -2]

        k = xp.where(xp.isnan(k) | xp.isinf(k), 1e-10, k)
        eps = xp.where(xp.isnan(eps) | xp.isinf(eps), 1e-10, eps)

    # ── Son eddy viskozite ────────────────────────────────────────────────
    nu_t = KE_CMU * xp.maximum(k, 0.0)**2 / xp.maximum(eps, 1e-10)
    nu_t[mask_g] = 0.0
    return xp.clip(nu_t, 0.0, 100.0)


def solve_k_omega_sst(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    mask: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    nu_mol: float = 1.5e-5,
    iterations: int = 60,
    xp: any = None
) -> np.ndarray:
    """k-omega SST türbülans modeli (Menter 1994).
    
    Iki denklem cozulur:
    - k: Türbülans kinetik enerjisi
    - omega: Türbülans frekansi (spesifik dissipation)
    
    SST = Shear Stress Transport - ic bolge (k-omega) ile dis bolge (k-epsilon) arasinda gecis.
    Eddy viskozite: ν_t = a1 * k / max(a1*omega, S*F2)
    
    Iyilestirmeler:
    - Van Driest duvar sonumlemesi (yakin duvar)
    - F1/F2 blend fonksiyonlari (ic/dis bolge gecisi)
    - Production limiter (P_k <= 10 * beta* * k * omega)
    - Upwind adveksiyon (kararlilik icin)
    
    Args:
        u, v, w: Hiz alanlari (nx, ny, nz)
        mask: Boolean grid - kati hucreler
        dx, dy, dz: Grid araliklari
        nu_mol: Molekuler viskozite (varsayilan: 1.5e-5)
        iterations: Cozum iterasyon sayisi
        xp: Array kutuphanesi (numpy veya cupy)
    
    Returns:
        nu_t: Eddy viskozite alani (nx, ny, nz)
    """
    if xp is None:
        xp = cp if GPU else np

    # Girdileri doğru tipe dönüştür
    u = xp.asarray(u, dtype=xp.float64)
    v = xp.asarray(v, dtype=xp.float64)
    w = xp.asarray(w, dtype=xp.float64)
    
    mask_g = xp.asarray(mask)
    sl = (slice(1, -1), slice(1, -1), slice(1, -1))

    # ── Başlangıç koşulları ──────────────────────────────────────────────
    I_turb = 0.005
    U_ref = 30.0
    L_turb = 0.1
    k_init = 1.5 * (I_turb * U_ref)**2
    omega_init = k_init**0.5 / L_turb
    
    k = xp.full(u.shape, k_init, dtype=xp.float64)
    omega = xp.full(u.shape, omega_init, dtype=xp.float64)
    k[mask_g] = 1e-10; omega[mask_g] = 1e-10

    # ── Duvar mesafesi ───────────────────────────────────────────────────
    d_wall = _compute_wall_distance(mask, dx, dy, dz, xp)

    # ── Pseudo-time step ─────────────────────────────────────────────────
    dt_sst = min(dx, dy, dz)**2 / max(6.0 * nu_mol, 1e-6)
    dt_sst = min(dt_sst, 1e-3)

    for _ in range(iterations):
        # ── Gerinim oranı |S| ────────────────────────────────────────────
        dudx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2*dx)
        dudy = (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2*dy)
        dudz = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2*dz)
        dvdx = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2*dx)
        dvdy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2*dy)
        dvdz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2*dz)
        dwdx = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2*dx)
        dwdy = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2*dy)
        dwdz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2*dz)

        S_mag = xp.sqrt(2.0*(dudx**2 + dvdy**2 + dwdz**2 +
                             0.5*(dudy + dvdx)**2 + 0.5*(dudz + dwdx)**2 + 0.5*(dvdz + dwdy)**2) + 1e-12)

        # ── F2 blend fonksiyonu (duvar yakınlığı) ────────────────────────
        d_wall_int = d_wall[1:-1, 1:-1, 1:-1]
        omega_int = xp.maximum(omega[sl], 1e-10)
        CD_komega = xp.maximum(2.0 * KO_SIGMA_K2 / omega_int * 
                               (dudx**2 + dudy**2 + dudz**2 + dvdx**2 + dvdy**2 + dvdz**2 +
                                dwdx**2 + dwdy**2 + dwdz**2), 1e-20)
        
        arg2 = xp.maximum(2.0 * KO_BETA_STAR * omega_int / (KO_A1 * xp.maximum(1e-10, 
               xp.sqrt(xp.maximum(0.0, dudx**2 + dvdy**2 + dwdz**2 + 
                                   0.5*(dudy + dvdx)**2 + 0.5*(dudz + dwdx)**2 + 0.5*(dvdz + dwdy)**2)))), 
               500.0 * nu_mol / (d_wall_int**2 * omega_int + 1e-20))
        F2 = xp.tanh(arg2**2)

        # ── F1 blend fonksiyonu (iç/dış bölge geçişi) ───────────────────
        arg1 = xp.minimum(xp.maximum(d_wall_int * xp.sqrt(xp.maximum(omega_int, 1e-10)) / 
                         (KO_A1 * xp.maximum(1e-10, 
                          xp.sqrt(xp.maximum(0.0, dudx**2 + dvdy**2 + dwdz**2 +
                                              0.5*(dudy + dvdx)**2 + 0.5*(dudz + dwdx)**2 + 0.5*(dvdz + dwdy)**2)))), 
                         500.0 * nu_mol / (d_wall_int**2 * omega_int + 1e-20)),
                         4.0 * KO_SIGMA_K2 * omega_int / (CD_komega * d_wall_int**2 + 1e-20))
        F1 = xp.tanh(arg1**4)

        # ── Beta ve sigma blend ──────────────────────────────────────────
        beta = F1 * KO_BETA1 + (1.0 - F1) * KO_BETA2
        sigma_k = F1 * KO_SIGMA_K1 + (1.0 - F1) * KO_SIGMA_K2
        sigma_w = F1 * KO_SIGMA_W1 + (1.0 - F1) * KO_SIGMA_W2

        # ── Eddy viskozite ──────────────────────────────────────────────
        nu_t_local = KO_A1 * k[sl] / xp.maximum(KO_A1 * omega[sl], S_mag * F2 + 1e-20)
        nu_t_local = xp.clip(nu_t_local, 0.0, 100.0)

        # ── Üretim terimi (sınırlı) ─────────────────────────────────────
        P_k = nu_t_local * S_mag**2
        P_k = xp.minimum(P_k, 10.0 * KO_BETA_STAR * k[sl] * omega[sl])

        # ── Laplacian ─────────────────────────────────────────────────────
        def _laplacian(f):
            return (
                (f[2:, 1:-1, 1:-1] - 2*f[sl] + f[:-2, 1:-1, 1:-1]) / dx**2 +
                (f[1:-1, 2:, 1:-1] - 2*f[sl] + f[1:-1, :-2, 1:-1]) / dy**2 +
                (f[1:-1, 1:-1, 2:] - 2*f[sl] + f[1:-1, 1:-1, :-2]) / dz**2
            )

        # ── Upwind adveksiyon ─────────────────────────────────────────────
        def _upwind_adv(f):
            adv_x = xp.where(u[sl] > 0,
                             u[sl] * (f[sl] - f[:-2, 1:-1, 1:-1]) / dx,
                             u[sl] * (f[2:, 1:-1, 1:-1] - f[sl]) / dx)
            adv_y = xp.where(v[sl] > 0,
                             v[sl] * (f[sl] - f[1:-1, :-2, 1:-1]) / dy,
                             v[sl] * (f[1:-1, 2:, 1:-1] - f[sl]) / dy)
            adv_z = xp.where(w[sl] > 0,
                             w[sl] * (f[sl] - f[1:-1, 1:-1, :-2]) / dz,
                             w[sl] * (f[1:-1, 1:-1, 2:] - f[sl]) / dz)
            return adv_x + adv_y + adv_z

        # ── k denklemi ───────────────────────────────────────────────────
        diff_k = (nu_mol + nu_t_local / sigma_k) * _laplacian(k)
        k_new = k[sl] + dt_sst * (-_upwind_adv(k) + P_k - KO_BETA_STAR * omega[sl] * k[sl] + diff_k)
        k[sl] = xp.maximum(k_new, 1e-10)

        # ── omega denklemi ───────────────────────────────────────────────
        diff_omega = (nu_mol + nu_t_local / sigma_w) * _laplacian(omega)
        omega_new = omega[sl] + dt_sst * (
            -_upwind_adv(omega)
            + beta / nu_t_local * P_k
            - beta * omega[sl]**2
            + diff_omega
            + (1.0 - F1) * CD_komega  # Cross-diffusion
        )
        omega[sl] = xp.maximum(omega_new, 1e-10)

        # ── Sınır şartları ───────────────────────────────────────────────
        k[mask_g] = 1e-10; omega[mask_g] = 1e-10
        k[0, :, :] = k_init; omega[0, :, :] = omega_init
        k[-1, :, :] = k[-2, :, :]; omega[-1, :, :] = omega[-2, :, :]
        k[:, 0, :] = k[:, 1, :]; omega[:, 0, :] = omega[:, 1, :]
        k[:, -1, :] = k[:, -2, :]; omega[:, -1, :] = omega[:, -2, :]
        k[:, :, 0] = k[:, :, 1]; omega[:, :, 0] = omega[:, :, 1]
        k[:, :, -1] = k[:, :, -2]; omega[:, :, -1] = omega[:, :, -2]

        k = xp.where(xp.isnan(k) | xp.isinf(k), 1e-10, k)
        omega = xp.where(xp.isnan(omega) | xp.isinf(omega), 1e-10, omega)

    # ── Son eddy viskozite ────────────────────────────────────────────────
    omega_final = xp.maximum(omega, 1e-10)
    
    # Gerinim oranını tam grid üzerinde hesapla
    dudx_full = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2*dx)
    dvdy_full = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2*dy)
    dwdz_full = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2*dz)
    S_mag_full = xp.sqrt(2.0*(dudx_full**2 + dvdy_full**2 + dwdz_full**2) + 1e-12)
    
    # Interior indeksleri kullan
    omega_int = omega_final[sl]
    k_int = k[sl]
    d_wall_int = d_wall[sl]
    
    F2_final = xp.tanh(xp.maximum(2.0 * KO_BETA_STAR * omega_int / 
                        (KO_A1 * xp.maximum(1e-10, S_mag_full)), 500.0 * nu_mol / (d_wall_int**2 * omega_int + 1e-20))**2)
    nu_t_int = KO_A1 * xp.maximum(k_int, 0.0) / xp.maximum(KO_A1 * omega_int, S_mag_full * F2_final + 1e-20)
    
    nu_t = xp.zeros_like(u)
    nu_t[sl] = xp.clip(nu_t_int, 0.0, 100.0)
    nu_t[mask_g] = 0.0
    return nu_t


def _assemble_poisson_matrix(mask_g, dx, dy, dz, nx, ny, nz):
    """Kararlı basınç Poisson sistemi için sparse Laplace matrisini hazırlar."""
    N = nx * ny * nz
    ax = float(1.0 / dx**2)
    ay = float(1.0 / dy**2)
    az = float(1.0 / dz**2)
    diag_val = float(-2.0 * (ax + ay + az))

    ox = ny * nz
    oy = nz
    oz = 1

    diagonals = [
        np.full(N,           diag_val),
        np.full(N - ox,      ax),
        np.full(N - ox,      ax),
        np.full(N - oy,      ay),
        np.full(N - oy,      ay),
        np.full(N - oz,      az),
        np.full(N - oz,      az),
    ]
    A = sp_diags(diagonals, [0, ox, -ox, oy, -oy, oz, -oz], shape=(N, N), format='csr')

    mask_flat = mask_g.ravel() if not GPU else _to_numpy(mask_g).ravel()
    solid_idx = np.where(mask_flat)[0]

    A = A.tolil()
    if solid_idx.size:
        A[solid_idx, :] = 0.0
    if 0 not in solid_idx:
        A[0, :] = 0.0

    A = A.tocsr()
    diag = A.diagonal()
    diag[solid_idx] = 1.0
    if 0 not in solid_idx:
        diag[0] = 1.0
    A.setdiag(diag)

    return A, solid_idx


def _build_jacobi_preconditioner(A, solid_idx):
    """Poisson CG için Jacobi önşartlayıcıyı hazırlar."""
    diag = A.diagonal()
    inv_diag = np.zeros_like(diag, dtype=np.float64)
    nonzero = diag != 0.0
    inv_diag[nonzero] = 1.0 / diag[nonzero]
    inv_diag[solid_idx] = 1.0

    if GPU:
        return cu_diags(inv_diag, 0, shape=A.shape, format='csr')
    return sp_diags(inv_diag, 0, shape=A.shape, format='csr')


def _hash_mask(mask_g):
    mask_np = _to_numpy(mask_g) if GPU else np.asarray(mask_g)
    return zlib.adler32(mask_np.tobytes())


def _build_wall_distance_cache(mask, dx, dy, dz, xp):
    """Duvar mesafesi hesaplamasını cache'leyip tekrar kullanım için saklar."""
    cache_key = (
        mask.shape,
        round(float(dx), 8),
        round(float(dy), 8),
        round(float(dz), 8),
        _hash_mask(mask)
    )
    if cache_key not in _WALL_DISTANCE_CACHE:
        mask_np = mask.get() if hasattr(mask, 'get') else np.asarray(mask, dtype=bool)
        dist = distance_transform_edt(~mask_np, sampling=(dx, dy, dz))
        _WALL_DISTANCE_CACHE[cache_key] = dist
    dist = _WALL_DISTANCE_CACHE[cache_key]
    return xp.asarray(dist) if xp is not np else np.asarray(dist)


def _apply_wall_functions(u, v, w, mask, d_wall, dx, dy, dz, nu_mol, xp):
    """Duvar komşu hücrelerine log-law wall function boundary condition uygular."""
    mask_np = mask.get() if hasattr(mask, 'get') else np.asarray(mask, dtype=bool)
    structure_6 = np.zeros((3, 3, 3), dtype=bool)
    structure_6[1, 1, 0] = True
    structure_6[1, 1, 2] = True
    structure_6[1, 0, 1] = True
    structure_6[1, 2, 1] = True
    structure_6[0, 1, 1] = True
    structure_6[2, 1, 1] = True
    wall_adj_np = binary_dilation(mask_np, structure=structure_6) & ~mask_np
    wall_adj = xp.asarray(wall_adj_np)

    sl = (slice(1, -1), slice(1, -1), slice(1, -1))
    wall_cells = wall_adj[sl]
    if not bool(wall_cells.any()):
        return u, v, w

    u_slice = u[sl]; v_slice = v[sl]; w_slice = w[sl]
    d_wall_local = xp.maximum(d_wall[sl], 1e-6)

    vel_slice = xp.sqrt(u_slice**2 + v_slice**2 + w_slice**2) + 1e-12
    t_hat_u = u_slice / vel_slice
    t_hat_v = v_slice / vel_slice
    t_hat_w = w_slice / vel_slice

    u_tau = xp.sqrt(nu_mol * vel_slice / d_wall_local)
    y_plus = xp.maximum(u_tau * d_wall_local / nu_mol, 1e-6)

    log_layer = y_plus > 11.5
    u_plus = xp.where(log_layer,
                      (1.0 / SA_KAPPA) * xp.log(y_plus) + 5.0,
                      y_plus)
    u_wall = u_plus * nu_mol / d_wall_local

    blend = xp.clip(0.15 + 0.35 * xp.tanh((y_plus - 5.0) / 10.0), 0.15, 0.5)
    u_target = u_wall * t_hat_u
    v_target = u_wall * t_hat_v
    w_target = u_wall * t_hat_w

    u_new = xp.where(wall_cells, (1.0 - blend) * u_slice + blend * u_target, u_slice)
    v_new = xp.where(wall_cells, (1.0 - blend) * v_slice + blend * v_target, v_slice)
    w_new = xp.where(wall_cells, (1.0 - blend) * w_slice + blend * w_target, w_slice)

    u[sl] = u_new
    v[sl] = v_new
    w[sl] = w_new
    return u, v, w


def _poisson_cg(rhs, mask_g, dx, dy, dz, nx, ny, nz, xp):
    """Vektörize sparse Laplacian — CG çözücü (Basınç sızdırmazlık yamalı)."""
    mask_hash = _hash_mask(mask_g)
    cache_key = (
        int(nx), int(ny), int(nz),
        round(float(dx), 8), round(float(dy), 8), round(float(dz), 8),
        int(mask_hash)
    )

    if cache_key not in _POISSON_CACHE:
        A, solid_idx = _assemble_poisson_matrix(mask_g, dx, dy, dz, nx, ny, nz)
        M = _build_jacobi_preconditioner(A, solid_idx)
        _POISSON_CACHE[cache_key] = (A, M, solid_idx)
    else:
        A, M, solid_idx = _POISSON_CACHE[cache_key]

    rhs_np = rhs.ravel() if not GPU else _to_numpy(rhs).ravel()
    rhs_np[solid_idx] = 0.0

    if GPU:
        A_gpu = cu_csr(A)
        p_flat, info = cu_cg(A_gpu, cp.asarray(rhs_np), rtol=POISSON_TOL, maxiter=POISSON_MAXIT, M=M)
        p = _to_numpy(p_flat).reshape((nx, ny, nz))
    else:
        p_flat, info = scipy_cg(A, rhs_np, rtol=POISSON_TOL, maxiter=POISSON_MAXIT, M=M)
        p = p_flat.reshape((nx, ny, nz))

    if info != 0:
        print(f"[Poisson] WARNING: Convergence not achieved (info={info}), tolerance exceeded in {POISSON_MAXIT} iterations")

    return p


def solve_navier_stokes_full(u_np, v_np, w_np, mask, v_wind, nx, ny, nz, solid_fraction=None, iterations=300, turbulence_model="spalart_allmaras", use_wall_functions=False, turb_iterations=None):
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
    tx, ty, tz = TUNNEL_DIMS
    dx, dy, dz = tx/(nx-1), ty/(ny-1), tz/(nz-1)
    nu_mol = 1.5e-5

    xp = cp if GPU else np

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

    p = xp.zeros((nx, ny, nz), dtype=xp.float64)
    p_old = xp.zeros((nx, ny, nz), dtype=xp.float64)  # Rhie-Chow için önceki basınç
    d_wall = _build_wall_distance_cache(mask_g, dx, dy, dz, xp)

    # ── Van Leer limiter ──────────────────────────────────────────────────
    def _van_leer_limiter(a, b):
        return xp.where(a * b > 0, 2.0 * a * b / (a + b + 1e-12), 0.0)

    # ── Bounded QUICK scheme (3. derece, bounded) ────────────────────────
    def _bounded_quick(f, vel, dx_local):
        """Bounded QUICK scheme - 3. derece doğruluk, boundedness korunarak.
        
        f_ppm = QUICK face value with boundedness constraint.
        """
        sl = (slice(1,-1), slice(1,-1), slice(1,-1))
        # Upwind ve downwind gradyanlar
        grad_up = (f[sl] - f[:-2, 1:-1, 1:-1]) / dx_local
        grad_dn = (f[2:, 1:-1, 1:-1] - f[sl]) / dx_local
        
        # QUICK face value (3. derece)
        f_quick = xp.where(vel[sl] > 0,
                           f[sl] + 0.5 * _van_leer_limiter(grad_up, grad_dn) * dx_local,
                           f[sl] - 0.5 * _van_leer_limiter(grad_up, grad_dn) * dx_local)
        
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
        visc_x = (nu_e * (f[2:, 1:-1, 1:-1] - f[1:-1, 1:-1, 1:-1]) / dx -
                  nu_w * (f[1:-1, 1:-1, 1:-1] - f[:-2, 1:-1, 1:-1]) / dx) / dx
        # y ekseni
        nu_n = 0.5 * (nu_eff_field[1:-1, 2:, 1:-1] + nu_eff_field[1:-1, 1:-1, 1:-1])
        nu_s = 0.5 * (nu_eff_field[1:-1, :-2, 1:-1] + nu_eff_field[1:-1, 1:-1, 1:-1])
        visc_y = (nu_n * (f[1:-1, 2:, 1:-1] - f[1:-1, 1:-1, 1:-1]) / dy -
                  nu_s * (f[1:-1, 1:-1, 1:-1] - f[1:-1, :-2, 1:-1]) / dy) / dy
        # z ekseni
        nu_t_top = 0.5 * (nu_eff_field[1:-1, 1:-1, 2:] + nu_eff_field[1:-1, 1:-1, 1:-1])
        nu_b = 0.5 * (nu_eff_field[1:-1, 1:-1, :-2] + nu_eff_field[1:-1, 1:-1, 1:-1])
        visc_z = (nu_t_top * (f[1:-1, 1:-1, 2:] - f[1:-1, 1:-1, 1:-1]) / dz -
                  nu_b * (f[1:-1, 1:-1, 1:-1] - f[1:-1, 1:-1, :-2]) / dz) / dz
        return visc_x + visc_y + visc_z

    # ── Adveksiyon (Van Leer limiter ile) ─────────────────────────────────
    def adv(f):
        sl = (slice(1,-1), slice(1,-1), slice(1,-1))
        fb = (f[sl] - f[:-2, 1:-1, 1:-1]) / dx
        ff = (f[2:, 1:-1, 1:-1] - f[sl]) / dx
        slope_x = _van_leer_limiter(fb, ff)
        fx = xp.where(u_old[sl] > 0, fb + 0.5 * slope_x, ff - 0.5 * slope_x)

        fb = (f[sl] - f[1:-1, :-2, 1:-1]) / dy
        ff = (f[1:-1, 2:, 1:-1] - f[sl]) / dy
        slope_y = _van_leer_limiter(fb, ff)
        fy = xp.where(v_old[sl] > 0, fb + 0.5 * slope_y, ff - 0.5 * slope_y)

        fb = (f[sl] - f[1:-1, 1:-1, :-2]) / dz
        ff = (f[1:-1, 1:-1, 2:] - f[sl]) / dz
        slope_z = _van_leer_limiter(fb, ff)
        fz = xp.where(w_old[sl] > 0, fb + 0.5 * slope_z, ff - 0.5 * slope_z)

        return u_old[sl]*fx + v_old[sl]*fy + w_old[sl]*fz

    # ── Tünel duvar şartları ──────────────────────────────────────────────
    def _apply_tunnel_wall_bc(vel_u, vel_v, vel_w):
        """Y/z tünel duvarları için slip (serbest kayma) sınır şartı."""
        # Y eksenleri (sol/sag duvarlar): du/dy = 0, v = 0
        vel_u[:, 0, :] = vel_u[:, 1, :];  vel_v[:, 0, :] = 0.0;  vel_w[:, 0, :] = vel_w[:, 1, :]
        vel_u[:, -1, :] = vel_u[:, -2, :]; vel_v[:, -1, :] = 0.0; vel_w[:, -1, :] = vel_w[:, -2, :]
        # Z eksenleri (üst/alt duvarlar): du/dz = 0, w = 0
        vel_u[:, :, 0] = vel_u[:, :, 1];   vel_v[:, :, 0] = vel_v[:, :, 1];  vel_w[:, :, 0] = 0.0
        vel_u[:, :, -1] = vel_u[:, :, -2]; vel_v[:, :, -1] = vel_v[:, :, -2]; vel_w[:, :, -1] = 0.0
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
        sl = (slice(1,-1), slice(1,-1), slice(1,-1))
        u_rhs[sl] = (-adv(u_in) + _viscous_conservative(u_in, nu_eff_field)) * volume_weight[sl]
        v_rhs[sl] = (-adv(v_in) + _viscous_conservative(v_in, nu_eff_field)) * volume_weight[sl]
        w_rhs[sl] = (-adv(w_in) + _viscous_conservative(w_in, nu_eff_field)) * volume_weight[sl]
        return u_rhs, v_rhs, w_rhs

    # ── Rhie-Chow yüzey hızları ──────────────────────────────────────────
    def _rhie_chow_face_velocities(u_cell, v_cell, w_cell, p_field, dt_local):
        """Collocated grid'de checkerboard basıncı önlemek için Rhie-Chow enterpolasyonu.
        
        Yüzey hızları = lineer enterpolasyon + basınç düzeltmesi (önceki basınç ile).
        """
        rho = RHO
        # X eksen yüzeyleri (i+1/2)
        u_face_e = 0.5 * (u_cell[2:, 1:-1, 1:-1] + u_cell[1:-1, 1:-1, 1:-1]) \
                   - (dt_local / (2.0 * rho * dx)) * (p_field[2:, 1:-1, 1:-1] - p_field[1:-1, 1:-1, 1:-1])
        u_face_w = 0.5 * (u_cell[1:-1, 1:-1, 1:-1] + u_cell[:-2, 1:-1, 1:-1]) \
                   - (dt_local / (2.0 * rho * dx)) * (p_field[1:-1, 1:-1, 1:-1] - p_field[:-2, 1:-1, 1:-1])
        # Y eksen yüzeyleri (j+1/2)
        v_face_n = 0.5 * (v_cell[1:-1, 2:, 1:-1] + v_cell[1:-1, 1:-1, 1:-1]) \
                   - (dt_local / (2.0 * rho * dy)) * (p_field[1:-1, 2:, 1:-1] - p_field[1:-1, 1:-1, 1:-1])
        v_face_s = 0.5 * (v_cell[1:-1, 1:-1, 1:-1] + v_cell[1:-1, :-2, 1:-1]) \
                   - (dt_local / (2.0 * rho * dy)) * (p_field[1:-1, 1:-1, 1:-1] - p_field[1:-1, :-2, 1:-1])
        # Z eksen yüzeyleri (k+1/2)
        w_face_t = 0.5 * (w_cell[1:-1, 1:-1, 2:] + w_cell[1:-1, 1:-1, 1:-1]) \
                   - (dt_local / (2.0 * rho * dz)) * (p_field[1:-1, 1:-1, 2:] - p_field[1:-1, 1:-1, 1:-1])
        w_face_b = 0.5 * (w_cell[1:-1, 1:-1, 1:-1] + w_cell[1:-1, 1:-1, :-2]) \
                   - (dt_local / (2.0 * rho * dz)) * (p_field[1:-1, 1:-1, 1:-1] - p_field[1:-1, 1:-1, :-2])
        return u_face_e, u_face_w, v_face_n, v_face_s, w_face_t, w_face_b

    TURBULENCE_UPDATE_INTERVAL = 3
    prev_residual = None
    
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
                nu_t = solve_spalart_allmaras(u, v, w, mask_g, dx, dy, dz, nu_mol=nu_mol, iterations=sa_iters, xp=xp)
            elif turb_lower in ("k_epsilon", "k-epsilon", "ke"):
                ke_iters = 60 if turb_iterations is None else int(turb_iterations)
                nu_t = solve_k_epsilon(u, v, w, mask_g, dx, dy, dz, nu_mol=nu_mol, iterations=ke_iters, xp=xp)
            elif turb_lower in ("k_omega_sst", "k-omega-sst", "komega_sst", "sst"):
                sst_iters = 60 if turb_iterations is None else int(turb_iterations)
                nu_t = solve_k_omega_sst(u, v, w, mask_g, dx, dy, dz, nu_mol=nu_mol, iterations=sst_iters, xp=xp)
            else:
                nu_t = _smagorinsky_nu(u, v, w, dx, dy, dz, xp, d_wall=d_wall)
        nu_eff = nu_mol + nu_t

        # ── Doğru viskoz CFL ─────────────────────────────────────────────
        nu_eff_max = float(xp.max(xp.abs(nu_eff)))
        dt_conv = cfl_safety * min(dx/u_max, dy/v_max, dz/w_max)
        dt_visc = cfl_safety * min(dx**2, dy**2, dz**2) / (2.0 * nu_eff_max + 1e-12)
        dt = min(dt_conv, dt_visc, 5e-4)

        u_old, v_old, w_old = u.copy(), v.copy(), w.copy()

        # ── RK2 (Heun) zaman integrasyonu ─────────────────────────────────
        # k1
        k1_u, k1_v, k1_w = _ns_rhs(u_old, v_old, w_old, nu_eff)
        # Ara adım
        u_mid = u_old + dt * k1_u
        v_mid = v_old + dt * k1_v
        w_mid = w_old + dt * k1_w
        u_mid[mask_g] = 0.0; v_mid[mask_g] = 0.0; w_mid[mask_g] = 0.0
        u_mid[0, :, :] = v_wind
        u_mid, v_mid, w_mid = _apply_tunnel_wall_bc(u_mid, v_mid, w_mid)
        # k2
        k2_u, k2_v, k2_w = _ns_rhs(u_mid, v_mid, w_mid, nu_eff)

        u = u_old + 0.5 * dt * (k1_u + k2_u)
        v = v_old + 0.5 * dt * (k1_v + k2_v)
        w = w_old + 0.5 * dt * (k1_w + k2_w)

        if solid_fraction is not None:
            u = u * volume_weight; v = v * volume_weight; w = w * volume_weight

        u[mask_g] = 0.0; v[mask_g] = 0.0; w[mask_g] = 0.0

        # ── Giriş/Çıkış/Tünel sınır şartları ──────────────────────────────
        u[0, :, :] = v_wind; v[0, :, :] = 0.0; w[0, :, :] = 0.0
        u[-1, :, :] = u[-2, :, :]; v[-1, :, :] = v[-2, :, :]; w[-1, :, :] = w[-2, :, :]
        u, v, w = _apply_tunnel_wall_bc(u, v, w)

        # ── Basınç Poisson — Rhie-Chow divergansı ────────────────────────
        t0 = time.time()
        
        # Yüzey hızları: lineer enterpolasyon (divergans için)
        u_fe = 0.5 * (u[2:, 1:-1, 1:-1] + u[1:-1, 1:-1, 1:-1])
        u_fw = 0.5 * (u[1:-1, 1:-1, 1:-1] + u[:-2, 1:-1, 1:-1])
        v_fn = 0.5 * (v[1:-1, 2:, 1:-1] + v[1:-1, 1:-1, 1:-1])
        v_fs = 0.5 * (v[1:-1, 1:-1, 1:-1] + v[1:-1, :-2, 1:-1])
        w_ft = 0.5 * (w[1:-1, 1:-1, 2:] + w[1:-1, 1:-1, 1:-1])
        w_fb = 0.5 * (w[1:-1, 1:-1, 1:-1] + w[1:-1, 1:-1, :-2])
        
        # Divergans — yüzey hızlarından
        div_star = (u_fe - u_fw) / dx + (v_fn - v_fs) / dy + (w_ft - w_fb) / dz
        
        rhs = xp.zeros((nx, ny, nz), dtype=xp.float64)
        rhs[1:-1, 1:-1, 1:-1] = div_star / dt * volume_weight[1:-1, 1:-1, 1:-1]

        p_np = _poisson_cg(rhs, mask_g, dx, dy, dz, nx, ny, nz, xp)
        p = xp.asarray(p_np)
        p[mask_g] = 0.0

        # ── Basınç düzeltmesi — Rhie-Chow yüzey gradienti ────────────────
        # Yüzey bazlı basınç gradyanı (checkerboard önleme)
        dp_dx_face_e = (p[2:, 1:-1, 1:-1] - p[1:-1, 1:-1, 1:-1]) / dx
        dp_dx_face_w = (p[1:-1, 1:-1, 1:-1] - p[:-2, 1:-1, 1:-1]) / dx
        dp_dy_face_n = (p[1:-1, 2:, 1:-1] - p[1:-1, 1:-1, 1:-1]) / dy
        dp_dy_face_s = (p[1:-1, 1:-1, 1:-1] - p[1:-1, :-2, 1:-1]) / dy
        dp_dz_face_t = (p[1:-1, 1:-1, 2:] - p[1:-1, 1:-1, 1:-1]) / dz
        dp_dz_face_b = (p[1:-1, 1:-1, 1:-1] - p[1:-1, 1:-1, :-2]) / dz
        
        # Hücre merkezi basınç düzeltmesi: yüzey gradyanlarının ortalaması
        u[1:-1, 1:-1, 1:-1] -= dt * 0.5 * (dp_dx_face_e + dp_dx_face_w)
        v[1:-1, 1:-1, 1:-1] -= dt * 0.5 * (dp_dy_face_n + dp_dy_face_s)
        w[1:-1, 1:-1, 1:-1] -= dt * 0.5 * (dp_dz_face_t + dp_dz_face_b)

        if use_wall_functions:
            u, v, w = _apply_wall_functions(u, v, w, mask_g, d_wall, dx, dy, dz, nu_mol, xp)

        # ── Son sınır şartları ────────────────────────────────────────────
        u[mask_g] = 0.0; v[mask_g] = 0.0; w[mask_g] = 0.0
        u[0, :, :] = v_wind; v[0, :, :] = 0.0; w[0, :, :] = 0.0
        u[-1, :, :] = u[-2, :, :]; v[-1, :, :] = v[-2, :, :]; w[-1, :, :] = w[-2, :, :]
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

        # Bir sonraki adım için basıncı kaydet (Rhie-Chow)
        p_old = p.copy()

        # ── Yakınsama izleme ──────────────────────────────────────────────
        if step % 10 == 0:
            div_max = float(xp.max(xp.abs(div_star)))
            
            # Velocity residual: değişim oranını hesapla
            u_change = float(xp.max(xp.abs(u - u_old)))
            v_change = float(xp.max(xp.abs(v - v_old)))
            w_change = float(xp.max(xp.abs(w - w_old)))
            vel_residual = u_change + v_change + w_change
            
            if prev_residual is not None and prev_residual > 0:
                conv_rate = vel_residual / prev_residual
                print(f"[NS] step={step:4d}, dt={dt:.2e}, div_max={div_max:.2e}, residual={vel_residual:.4e}, rate={conv_rate:.4f}")
            else:
                print(f"[NS] step={step:4d}, dt={dt:.2e}, div_max={div_max:.2e}, residual={vel_residual:.4e}")
            prev_residual = vel_residual
            
            # ── Erken durdurma: yakınsama kontrolü ────────────────────────
            # Residual 1e-4'ün altına düştüyse ve divergence da küçükse dur
            if step > 50 and vel_residual < 1e-4 and div_max < 1e-3:
                print(f"[NS] Converged at step {step}: residual={vel_residual:.4e}, div_max={div_max:.4e}")
                break

    if GPU:
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
    mask: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    tunnel_dims: tuple,
    refinement_level: int = 2
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
    mask_np = mask.get() if hasattr(mask, 'get') else np.asarray(mask, dtype=bool)
    
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
    cfl_target: float = 0.5
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
    xp = cp if GPU else np
    
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
    dt_conv = cfl_target / (u_local/dx + v_local/dy + w_local/dz + 1e-10)
    
    # Difüzyon CFL
    dt_visc = cfl_target * min(dx**2, dy**2, dz**2) / (2.0 * nu_local + 1e-10)
    
    # İkisinden küçüğünü al
    dt_local = xp.minimum(dt_conv, dt_visc)
    
    # Küresel maksimum ile sınırla
    dt_global = float(xp.min(dt_local))
    
    return dt_global


def compute_pressure_field(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    v_wind: float
) -> np.ndarray:
    """Bernoulli'den Cp = 1 - (|u|² / v_wind²) — GPU/CPU uyumluluğu icin veri korumasi icerir.
    
    Args:
        u, v, w: Hiz alanlari (nx, ny, nz)
        v_wind: Ruzgar hizi (m/s)
    
    Returns:
        cp_field: Basinc katsayisi alani (nx, ny, nz)
    """
    use_gpu = GPU and any(hasattr(arr, 'get') for arr in (u, v, w) if arr is not None)
    xp = cp if use_gpu else np

    u = xp.asarray(u)
    v = xp.asarray(v)
    w = xp.asarray(w)

    vel_sq = u**2 + v**2 + w**2
    vel_sq = xp.clip(vel_sq, 0.0, (v_wind * 2.0)**2)
    cp_field = 1.0 - vel_sq / (v_wind ** 2 + 1e-6)

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
    p_field: np.ndarray | None = None
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
    use_gpu = GPU and any(
        hasattr(arr, 'get') for arr in (u, v, w, cp_field, mask) if arr is not None
    )
    xp = cp if use_gpu else np

    u = xp.asarray(u)
    v = xp.asarray(v)
    w = xp.asarray(w)
    cp_field = xp.asarray(cp_field)
    if mask is not None:
        mask = xp.asarray(mask)

    tx, ty, tz = tunnel_dims
    dx_g = tx / (nx-1)
    dy_g = ty / (ny-1)
    dz_g = tz / (nz-1)
    q_inf = 0.5 * RHO * v_wind**2

    v0 = verts[i_idx]; v1 = verts[j_idx]; v2 = verts[k_idx]
    centroids = (v0 + v1 + v2) / 3.0

    e1 = v1 - v0; e2 = v2 - v0
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
    fx = ((centroids[:,0] + tx/2) / tx * (nx-1)).astype(np.float64)
    fy = ((centroids[:,1] + ty/2) / ty * (ny-1)).astype(np.float64)
    fz = ((centroids[:,2] + tz/2) / tz * (nz-1)).astype(np.float64)

    normals_unit_xp = xp.asarray(normals_unit)
    areas_xp = xp.asarray(areas)
    
    # Basınç kuvveti: p_field mevcutsa doğrudan kullan
    if p_field is not None:
        p_cpu = p_field.get() if hasattr(p_field, 'get') else np.asarray(p_field)
        p_face_np = np.zeros(len(centroids))
        for i in range(len(centroids)):
            xi = int(np.clip(fx[i], 0, nx-1))
            yi = int(np.clip(fy[i], 0, ny-1))
            zi = int(np.clip(fz[i], 0, nz-1))
            p_face_np[i] = p_cpu[xi, yi, zi]
        F_pressure = -xp.sum(
            xp.asarray(p_face_np)[:, xp.newaxis] * normals_unit_xp * areas_xp[:, xp.newaxis], axis=0
        )
    else:
        F_pressure = xp.zeros(3)

    nu_mol = 1.5e-5
    # Viskoz kuvvet için hız: yüzeye 1.5 hücre dışarıdan örnekleme
    fx_nb = np.clip(fx + normals_unit[:,0] * (nx-1) / tx * dx_g, 0, nx-1)
    fy_nb = np.clip(fy + normals_unit[:,1] * (ny-1) / ty * dy_g, 0, ny-1)
    fz_nb = np.clip(fz + normals_unit[:,2] * (nz-1) / tz * dz_g, 0, nz-1)
    x0n = np.clip(fx_nb.astype(np.int32), 0, nx-2)
    y0n = np.clip(fy_nb.astype(np.int32), 0, ny-2)
    z0n = np.clip(fz_nb.astype(np.int32), 0, nz-2)
    xdn = fx_nb - x0n; ydn = fy_nb - y0n; zdn = fz_nb - z0n
    
    def _trilinear(field, x0a, y0a, z0a, xda, yda, zda):
        f = field.get() if hasattr(field, 'get') else np.asarray(field)
        c000 = f[x0a, y0a, z0a]; c100 = f[x0a+1, y0a, z0a]
        c010 = f[x0a, y0a+1, z0a]; c110 = f[x0a+1, y0a+1, z0a]
        c001 = f[x0a, y0a, z0a+1]; c101 = f[x0a+1, y0a, z0a+1]
        c011 = f[x0a, y0a+1, z0a+1]; c111 = f[x0a+1, y0a+1, z0a+1]
        c00 = c000*(1-xda) + c100*xda; c01 = c001*(1-xda) + c101*xda
        c10 = c010*(1-xda) + c110*xda; c11 = c011*(1-xda) + c111*xda
        c0 = c00*(1-yda) + c10*yda; c1 = c01*(1-yda) + c11*yda
        return c0*(1-zda) + c1*zda
    
    u_nb_np = _trilinear(u, x0n, y0n, z0n, xdn, ydn, zdn)
    v_nb_np = _trilinear(v, x0n, y0n, z0n, xdn, ydn, zdn)
    w_nb_np = _trilinear(w, x0n, y0n, z0n, xdn, ydn, zdn)
    vel_nb = xp.column_stack([xp.asarray(u_nb_np), xp.asarray(v_nb_np), xp.asarray(w_nb_np)])

    if mask is not None and use_wall_functions:
        mask_np = mask.get() if hasattr(mask, 'get') else np.asarray(mask, dtype=bool)
        d_wall = _build_wall_distance_cache(mask_np, dx_g, dy_g, dz_g, xp)
        # Yüzeye en yakın hücrelerin duvar mesafesini al
        x0w = np.clip(fx.astype(np.int32), 0, nx-1)
        y0w = np.clip(fy.astype(np.int32), 0, ny-1)
        z0w = np.clip(fz.astype(np.int32), 0, nz-1)
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
    tau_wall = RHO * u_tau**2
    F_viscous = xp.sum(tau_wall[:, xp.newaxis] * t_hat * areas_xp[:, xp.newaxis], axis=0)

    F_total = F_pressure + F_viscous

    y_range = verts[:,1].max() - verts[:,1].min()
    z_range = verts[:,2].max() - verts[:,2].min()
    A_ref   = max(y_range * z_range, 0.01)

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
        Cd = float('nan')
        Cl = float('nan')
        drag_N = float('nan')
        lift_N = float('nan')
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
        "Cd": float(Cd) if np.isfinite(Cd) else float('nan'),
        "Cl": float(Cl) if np.isfinite(Cl) else float('nan'),
        "drag_N": float(drag_N) if np.isfinite(drag_N) else float('nan'),
        "lift_N": float(lift_N) if np.isfinite(lift_N) else float('nan'),
        "A_ref": float(A_ref),
        "status": aerodynamic_status,
        "label": efficiency_label,
        "valid": not invalid_result,
    }


def interpolate_cp_to_vertices(
    verts: np.ndarray,
    cp_field: np.ndarray,
    mask: np.ndarray,
    tunnel_dims: tuple,
    nx: int,
    ny: int,
    nz: int
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
    cp_cpu = cp_field.get() if hasattr(cp_field, 'get') else np.asarray(cp_field)

    fx = (verts[:,0] + tx/2) / tx * (nx-1)
    fy = (verts[:,1] + ty/2) / ty * (ny-1)
    fz = (verts[:,2] + tz/2) / tz * (nz-1)

    x0 = np.clip(fx.astype(np.int32), 0, nx-2)
    y0 = np.clip(fy.astype(np.int32), 0, ny-2)
    z0 = np.clip(fz.astype(np.int32), 0, nz-2)
    x1, y1, z1 = x0+1, y0+1, z0+1

    xd, yd, zd = fx - x0, fy - y0, fz - z0

    c000, c100 = cp_cpu[x0, y0, z0], cp_cpu[x1, y0, z0]
    c010, c110 = cp_cpu[x0, y1, z0], cp_cpu[x1, y1, z0]
    c001, c101 = cp_cpu[x0, y0, z1], cp_cpu[x1, y0, z1]
    c011, c111 = cp_cpu[x0, y1, z1], cp_cpu[x1, y1, z1]

    c00 = c000*(1-xd) + c100*xd; c01 = c001*(1-xd) + c101*xd
    c10 = c010*(1-xd) + c110*xd; c11 = c011*(1-xd) + c111*xd
    c0  = c00*(1-yd)  + c10*yd;  c1  = c01*(1-yd)  + c11*yd

    return c0*(1-zd) + c1*zd


def rotate_vertices(
    verts: np.ndarray,
    yaw_deg: float,
    pitch_deg: float,
    roll_deg: float
) -> np.ndarray:
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
    R_z = np.array([[np.cos(yaw), -np.sin(yaw), 0], [np.sin(yaw), np.cos(yaw), 0], [0, 0, 1]], dtype=np.float32)
    R_y = np.array([[np.cos(pitch), 0, np.sin(pitch)], [0, 1, 0], [-np.sin(pitch), 0, np.cos(pitch)]], dtype=np.float32)
    R_x = np.array([[1, 0, 0], [0, np.cos(roll), -np.sin(roll)], [0, np.sin(roll), np.cos(roll)]], dtype=np.float32)
    return verts @ (R_z @ R_y @ R_x).T

def apply_offset(
    verts: np.ndarray,
    ox: float,
    oy: float,
    oz: float
) -> np.ndarray:
    """Vertex ofseti uygula.
    
    Args:
        verts: Vertex koordinatlari (N, 3)
        ox, oy, oz: Ofset degerleri (metre)
    
    Returns:
        np.ndarray: Ofset uygulanmis vertex koordinatlari (N, 3)
    """
    return verts + np.array([ox, oy, oz], dtype=np.float32)


def compute_q_criterion(
    u: np.ndarray,
    v: np.ndarray,
    w: np.ndarray,
    dx: float,
    dy: float,
    dz: float,
    GPU: bool = True
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
    xp = cp if GPU else np
    u, v, w = xp.asarray(u), xp.asarray(v), xp.asarray(w)
    
    dudx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2*dx)
    dudy = (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2*dy)
    dudz = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2*dz)
    
    dvdy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2*dy)
    dwdz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2*dz)

    extra_v = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2*dx)
    dwdx = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2*dx)
    dvdz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2*dz)
    dwdy = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2*dy)
    
    omega_sq = (dudy - extra_v)**2 + (dudz - dwdx)**2 + (dvdz - dwdy)**2
    strain_sq = dudx**2 + dvdy**2 + dwdz**2 + 0.5*((dudy + extra_v)**2 + (dudz + dwdx)**2 + (dvdz + dwdy)**2)
    
    q = xp.zeros_like(u)
    q[1:-1, 1:-1, 1:-1] = 0.5 * (omega_sq - strain_sq)

    q_threshold = 1e-4
    q[xp.abs(q) < q_threshold] = 0.0

    return _to_numpy(q) if GPU else q