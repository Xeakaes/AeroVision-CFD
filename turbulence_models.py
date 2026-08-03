"""
AeroVision CFD - Turbulans Modelleri

Smagorinsky SGS, Spalart-Allmaras, k-epsilon ve k-omega SST eddy viskozite
modelleri ile duvar fonksiyonu yardimcilari. Sadece constants ve gpu_backend
modullerine bagimlidir.
"""

import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

import gpu_backend
from constants import (
    C_SMAG,
    SA_CB1,
    SA_CB2,
    SA_SIGMA,
    SA_KAPPA,
    SA_CW1,
    SA_CW2,
    SA_CW3,
    SA_CV1,
    KE_CMU,
    KE_C1,
    KE_C2,
    KE_SIGMA_K,
    KE_SIGMA_E,
    KO_BETA1,
    KO_BETA2,
    KO_SIGMA_K1,
    KO_SIGMA_W1,
    KO_SIGMA_K2,
    KO_SIGMA_W2,
    KO_A1,
    KO_BETA_STAR,
    _lru_insert,
    _WALL_DISTANCE_CACHE,
)


def _smagorinsky_nu(u, v, w, dx, dy, dz, xp, d_wall=None):
    """Smagorinsky SGS modeli — Van Driest duvar sönümlemesi dahil."""
    delta = (dx * dy * dz) ** (1.0 / 3.0)
    cs_delta_sq = (C_SMAG * delta) ** 2

    dudx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx)
    dudy = (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2 * dy)
    dudz = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * dz)

    dvdy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)
    dwdz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)

    extra_v = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2 * dx)
    dwdx = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dx)
    dvdz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2 * dz)
    dwdy = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2 * dy)

    S_mag = xp.sqrt(
        2.0
        * (
            dudx**2
            + dvdy**2
            + dwdz**2
            + 2 * (0.5 * (dudy + extra_v)) ** 2
            + 2 * (0.5 * (dudz + dwdx)) ** 2
            + 2 * (0.5 * (dvdz + dwdy)) ** 2
        )
        + 1e-8
    )

    nu_t = xp.zeros(u.shape, dtype=xp.float64)

    if d_wall is not None:
        nu_mol = 1.5e-5
        d_wall_int = d_wall[1:-1, 1:-1, 1:-1]
        u_tau_est = xp.sqrt(xp.maximum(nu_mol * S_mag, 0.0))
        y_plus = xp.maximum(d_wall_int * u_tau_est / nu_mol, 1e-6)
        damping = (1.0 - xp.exp(-y_plus / 25.0)) ** 2
        nu_t[1:-1, 1:-1, 1:-1] = cs_delta_sq * S_mag * damping
    else:
        nu_t[1:-1, 1:-1, 1:-1] = cs_delta_sq * S_mag

    return nu_t


def _compute_wall_distance(mask, dx, dy, dz, xp):
    """SciPy'ın CPU fonksiyonu için GPU maskesini güvenli tahliye eden yama."""
    mask_np = mask.get() if hasattr(mask, "get") else np.asarray(mask, dtype=bool)
    dist = distance_transform_edt(~mask_np, sampling=(dx, dy, dz))
    return xp.asarray(dist)


def solve_spalart_allmaras(u, v, w, mask, dx, dy, dz, nu_mol=1.5e-5, iterations=20, xp=None, use_fused=True):
    """Spalart-Allmaras türbülans viskozite modeli için ν_t alanını çözer.

    iyileştirmeler:
    - Minimum 60 iterasyon (yetersiz yakınsama düzeltildi)
    - Upwind adveksiyon şeması (merkezi farklar yerine)
    - Daha büyük pseudo-time adımı (1e-3)
    - use_fused: cupy'de tek RawKernel ile iterasyon (kernel fırtınası önleme)
    """
    if xp is None:
        xp = gpu_backend.cp if gpu_backend.GPU else np

    if use_fused and gpu_backend.GPU and gpu_backend.GPU_BACKEND == "cupy":
        return _solve_sa_fused(u, v, w, mask, dx, dy, dz, nu_mol=nu_mol, iterations=iterations, xp=xp)

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
        iterations = max(iterations, 70)
    elif min_spacing < 0.13:
        iterations = max(iterations, 60)
    else:
        iterations = max(iterations, 50)

    for _ in range(iterations):
        nu = nu_tilde
        d_nu_dx = (nu[2:, 1:-1, 1:-1] - nu[:-2, 1:-1, 1:-1]) / (2 * dx)
        d_nu_dy = (nu[1:-1, 2:, 1:-1] - nu[1:-1, :-2, 1:-1]) / (2 * dy)
        d_nu_dz = (nu[1:-1, 1:-1, 2:] - nu[1:-1, 1:-1, :-2]) / (2 * dz)
        grad2_nu = d_nu_dx**2 + d_nu_dy**2 + d_nu_dz**2

        lap_nu = (
            (nu[2:, 1:-1, 1:-1] - 2 * nu[sl] + nu[:-2, 1:-1, 1:-1]) / dx**2
            + (nu[1:-1, 2:, 1:-1] - 2 * nu[sl] + nu[1:-1, :-2, 1:-1]) / dy**2
            + (nu[1:-1, 1:-1, 2:] - 2 * nu[sl] + nu[1:-1, 1:-1, :-2]) / dz**2
        )

        du_dx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx)
        dv_dy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)
        dw_dz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)

        du_dy = (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2 * dy)
        dv_dx = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2 * dx)
        du_dz = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * dz)
        dw_dx = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dx)
        dv_dz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2 * dz)
        dw_dy = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2 * dy)

        S = xp.sqrt(
            2.0
            * (
                du_dx**2
                + dv_dy**2
                + dw_dz**2
                + 0.5 * (du_dy + dv_dx) ** 2
                + 0.5 * (du_dz + dw_dx) ** 2
                + 0.5 * (dv_dz + dw_dy) ** 2
            )
            + 1e-12
        )

        chi = nu[sl] / (nu_mol + 1e-12)
        fv1 = chi**3 / (chi**3 + SA_CV1**3 + 1e-12)
        r = xp.minimum(nu[sl] / ((S * d_wall[sl]) ** 2 + 1e-12), 10.0)
        g = r + SA_CW2 * r**6
        g = xp.minimum(g, 1e5)
        f_w = g * ((1.0 + SA_CW3**6) / (g**6 + SA_CW3**6 + 1e-12)) ** (1.0 / 6.0)
        f_w = xp.minimum(f_w, 2.5)
        prod = SA_CB1 * S * nu[sl]
        # Difuzyon: (1/sigma)*[div((nu+nu_tilde)*grad(nu_tilde)) + Cb2*|grad|^2]
        #   = (1/sigma)*[(nu+nu_tilde)*lap + (1+Cb2)*|grad|^2]
        # (nu+nu_tilde) carpani eksikti — Cb2 terimi de yanlislikla tabana bolunuyordu.
        diff = (1.0 / SA_SIGMA) * ((nu[sl] + nu_mol) * lap_nu + (1.0 + SA_CB2) * grad2_nu)
        destr = -SA_CW1 * f_w * (nu[sl] / (d_wall[sl] + 1e-12)) ** 2

        # Upwind adveksiyon şeması (merkezi farklar yerine)
        adv_x = xp.where(
            u[sl] > 0, u[sl] * (nu[sl] - nu[:-2, 1:-1, 1:-1]) / dx, u[sl] * (nu[2:, 1:-1, 1:-1] - nu[sl]) / dx
        )
        adv_y = xp.where(
            v[sl] > 0, v[sl] * (nu[sl] - nu[1:-1, :-2, 1:-1]) / dy, v[sl] * (nu[1:-1, 2:, 1:-1] - nu[sl]) / dy
        )
        adv_z = xp.where(
            w[sl] > 0, w[sl] * (nu[sl] - nu[1:-1, 1:-1, :-2]) / dz, w[sl] * (nu[1:-1, 1:-1, 2:] - nu[sl]) / dz
        )
        adv_nu = adv_x + adv_y + adv_z

        nu_new = nu[sl] + dt_sa * (-adv_nu + prod + diff + destr)

        nu_tilde[sl] = xp.maximum(nu_new, 0.0)
        nu_tilde[sl] = xp.minimum(nu_tilde[sl], 1e3 * nu_mol)  # patlama korumasi
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
    xp: any = None,
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
        xp = gpu_backend.cp if gpu_backend.GPU else np

    # Girdileri doğru tipe dönüştür
    u = xp.asarray(u, dtype=xp.float64)
    v = xp.asarray(v, dtype=xp.float64)
    w = xp.asarray(w, dtype=xp.float64)

    mask_g = xp.asarray(mask)
    sl = (slice(1, -1), slice(1, -1), slice(1, -1))

    # ── Başlangıç koşulları ──────────────────────────────────────────────
    # Freestream eddy viskozitesi nu_t = Cmu^0.25 * sqrt(k) * L ~ 1e-4 m2/s
    # civarinda kalacak sekilde secilir; buyuk degerler (eski: U_ref=30,
    # L_turb=0.1) alani nu_t ~ 0.02 ile doldurup akisi asiri sonumler.
    I_turb = 0.002  # %0.2 türbülans yoğunluğu (düşük)
    U_ref = 10.0
    L_turb = 0.02  # Ölçek uzunluğu (2cm)
    k_init = 1.5 * (I_turb * U_ref) ** 2
    eps_init = KE_CMU * k_init**1.5 / L_turb

    k = xp.full(u.shape, k_init, dtype=xp.float64)
    eps = xp.full(u.shape, eps_init, dtype=xp.float64)
    k[mask_g] = 1e-10
    eps[mask_g] = 1e-10

    # ── Pseudo-time step ─────────────────────────────────────────────────
    dt_ke = min(dx, dy, dz) ** 2 / max(6.0 * nu_mol, 1e-6)
    dt_ke = min(dt_ke, 1e-3)

    for _ in range(iterations):
        # ── Gerinim oranı |S| ────────────────────────────────────────────
        dudx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx)
        dudy = (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2 * dy)
        dudz = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * dz)
        dvdx = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2 * dx)
        dvdy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)
        dvdz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2 * dz)
        dwdx = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dx)
        dwdy = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2 * dy)
        dwdz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)

        S_mag = xp.sqrt(
            2.0
            * (
                dudx**2
                + dvdy**2
                + dwdz**2
                + 0.5 * (dudy + dvdx) ** 2
                + 0.5 * (dudz + dwdx) ** 2
                + 0.5 * (dvdz + dwdy) ** 2
            )
            + 1e-12
        )

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
                (f[2:, 1:-1, 1:-1] - 2 * f[sl] + f[:-2, 1:-1, 1:-1]) / dx**2
                + (f[1:-1, 2:, 1:-1] - 2 * f[sl] + f[1:-1, :-2, 1:-1]) / dy**2
                + (f[1:-1, 1:-1, 2:] - 2 * f[sl] + f[1:-1, 1:-1, :-2]) / dz**2
            )

        # ── Upwind adveksiyon ─────────────────────────────────────────────
        def _upwind_adv(f):
            adv_x = xp.where(
                u[sl] > 0, u[sl] * (f[sl] - f[:-2, 1:-1, 1:-1]) / dx, u[sl] * (f[2:, 1:-1, 1:-1] - f[sl]) / dx
            )
            adv_y = xp.where(
                v[sl] > 0, v[sl] * (f[sl] - f[1:-1, :-2, 1:-1]) / dy, v[sl] * (f[1:-1, 2:, 1:-1] - f[sl]) / dy
            )
            adv_z = xp.where(
                w[sl] > 0, w[sl] * (f[sl] - f[1:-1, 1:-1, :-2]) / dz, w[sl] * (f[1:-1, 1:-1, 2:] - f[sl]) / dz
            )
            return adv_x + adv_y + adv_z

        # ── k denklemi ───────────────────────────────────────────────────
        diff_k = (nu_mol + nu_t_local / KE_SIGMA_K) * _laplacian(k)
        k_new = k[sl] + dt_ke * (-_upwind_adv(k) + P_k - eps_int + diff_k)
        k[sl] = xp.maximum(k_new, 1e-10)

        # ── ε denklemi ───────────────────────────────────────────────────
        diff_eps = (nu_mol + nu_t_local / KE_SIGMA_E) * _laplacian(eps)
        eps_new = eps[sl] + dt_ke * (
            -_upwind_adv(eps) + KE_C1 * (eps_int / k_int) * P_k - KE_C2 * eps_int**2 / k_int + diff_eps
        )
        eps[sl] = xp.maximum(eps_new, 1e-10)

        # ── Sınır şartları ───────────────────────────────────────────────
        k[mask_g] = 1e-10
        eps[mask_g] = 1e-10
        k[0, :, :] = k_init
        eps[0, :, :] = eps_init
        k[-1, :, :] = k[-2, :, :]
        eps[-1, :, :] = eps[-2, :, :]
        k[:, 0, :] = k[:, 1, :]
        eps[:, 0, :] = eps[:, 1, :]
        k[:, -1, :] = k[:, -2, :]
        eps[:, -1, :] = eps[:, -2, :]
        k[:, :, 0] = k[:, :, 1]
        eps[:, :, 0] = eps[:, :, 1]
        k[:, :, -1] = k[:, :, -2]
        eps[:, :, -1] = eps[:, :, -2]

        k = xp.where(xp.isnan(k) | xp.isinf(k), 1e-10, k)
        eps = xp.where(xp.isnan(eps) | xp.isinf(eps), 1e-10, eps)

    # ── Son eddy viskozite ────────────────────────────────────────────────
    nu_t = KE_CMU * xp.maximum(k, 0.0) ** 2 / xp.maximum(eps, 1e-10)
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
    xp: any = None,
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
        xp = gpu_backend.cp if gpu_backend.GPU else np

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
    k_init = 1.5 * (I_turb * U_ref) ** 2
    omega_init = k_init**0.5 / L_turb

    k = xp.full(u.shape, k_init, dtype=xp.float64)
    omega = xp.full(u.shape, omega_init, dtype=xp.float64)
    k[mask_g] = 1e-10
    omega[mask_g] = 1e-10

    # ── Duvar mesafesi ───────────────────────────────────────────────────
    d_wall = _compute_wall_distance(mask, dx, dy, dz, xp)

    # ── Pseudo-time step ─────────────────────────────────────────────────
    dt_sst = min(dx, dy, dz) ** 2 / max(6.0 * nu_mol, 1e-6)
    dt_sst = min(dt_sst, 1e-3)

    for _ in range(iterations):
        # ── Gerinim oranı |S| ────────────────────────────────────────────
        dudx = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx)
        dudy = (u[1:-1, 2:, 1:-1] - u[1:-1, :-2, 1:-1]) / (2 * dy)
        dudz = (u[1:-1, 1:-1, 2:] - u[1:-1, 1:-1, :-2]) / (2 * dz)
        dvdx = (v[2:, 1:-1, 1:-1] - v[:-2, 1:-1, 1:-1]) / (2 * dx)
        dvdy = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)
        dvdz = (v[1:-1, 1:-1, 2:] - v[1:-1, 1:-1, :-2]) / (2 * dz)
        dwdx = (w[2:, 1:-1, 1:-1] - w[:-2, 1:-1, 1:-1]) / (2 * dx)
        dwdy = (w[1:-1, 2:, 1:-1] - w[1:-1, :-2, 1:-1]) / (2 * dy)
        dwdz = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)

        S_mag = xp.sqrt(
            2.0
            * (
                dudx**2
                + dvdy**2
                + dwdz**2
                + 0.5 * (dudy + dvdx) ** 2
                + 0.5 * (dudz + dwdx) ** 2
                + 0.5 * (dvdz + dwdy) ** 2
            )
            + 1e-12
        )

        # ── F2 blend fonksiyonu (duvar yakınlığı) ────────────────────────
        d_wall_int = d_wall[1:-1, 1:-1, 1:-1]
        omega_int = xp.maximum(omega[sl], 1e-10)
        # Cross-diffusion: CD_kw = max(2*sigma_w2/omega * grad(k).grad(omega), 1e-10)
        # (hiz gradyanlariyla degil, k-omega gradyanlarinin nokta carpimiyla)
        dk_dx = (k[2:, 1:-1, 1:-1] - k[:-2, 1:-1, 1:-1]) / (2 * dx)
        dk_dy = (k[1:-1, 2:, 1:-1] - k[1:-1, :-2, 1:-1]) / (2 * dy)
        dk_dz = (k[1:-1, 1:-1, 2:] - k[1:-1, 1:-1, :-2]) / (2 * dz)
        do_dx = (omega[2:, 1:-1, 1:-1] - omega[:-2, 1:-1, 1:-1]) / (2 * dx)
        do_dy = (omega[1:-1, 2:, 1:-1] - omega[1:-1, :-2, 1:-1]) / (2 * dy)
        do_dz = (omega[1:-1, 1:-1, 2:] - omega[1:-1, 1:-1, :-2]) / (2 * dz)
        CD_komega = xp.maximum(
            2.0 * KO_SIGMA_K2 / omega_int * (dk_dx * do_dx + dk_dy * do_dy + dk_dz * do_dz),
            1e-10,
        )

        arg2 = xp.maximum(
            2.0
            * KO_BETA_STAR
            * omega_int
            / (
                KO_A1
                * xp.maximum(
                    1e-10,
                    xp.sqrt(
                        xp.maximum(
                            0.0,
                            dudx**2
                            + dvdy**2
                            + dwdz**2
                            + 0.5 * (dudy + dvdx) ** 2
                            + 0.5 * (dudz + dwdx) ** 2
                            + 0.5 * (dvdz + dwdy) ** 2,
                        )
                    ),
                )
            ),
            500.0 * nu_mol / (d_wall_int**2 * omega_int + 1e-20),
        )
        F2 = xp.tanh(arg2**2)

        # ── F1 blend fonksiyonu (iç/dış bölge geçişi) ───────────────────
        arg1 = xp.minimum(
            xp.maximum(
                d_wall_int
                * xp.sqrt(xp.maximum(omega_int, 1e-10))
                / (
                    KO_A1
                    * xp.maximum(
                        1e-10,
                        xp.sqrt(
                            xp.maximum(
                                0.0,
                                dudx**2
                                + dvdy**2
                                + dwdz**2
                                + 0.5 * (dudy + dvdx) ** 2
                                + 0.5 * (dudz + dwdx) ** 2
                                + 0.5 * (dvdz + dwdy) ** 2,
                            )
                        ),
                    )
                ),
                500.0 * nu_mol / (d_wall_int**2 * omega_int + 1e-20),
            ),
            4.0 * KO_SIGMA_K2 * omega_int / (CD_komega * d_wall_int**2 + 1e-20),
        )
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
                (f[2:, 1:-1, 1:-1] - 2 * f[sl] + f[:-2, 1:-1, 1:-1]) / dx**2
                + (f[1:-1, 2:, 1:-1] - 2 * f[sl] + f[1:-1, :-2, 1:-1]) / dy**2
                + (f[1:-1, 1:-1, 2:] - 2 * f[sl] + f[1:-1, 1:-1, :-2]) / dz**2
            )

        # ── Upwind adveksiyon ─────────────────────────────────────────────
        def _upwind_adv(f):
            adv_x = xp.where(
                u[sl] > 0, u[sl] * (f[sl] - f[:-2, 1:-1, 1:-1]) / dx, u[sl] * (f[2:, 1:-1, 1:-1] - f[sl]) / dx
            )
            adv_y = xp.where(
                v[sl] > 0, v[sl] * (f[sl] - f[1:-1, :-2, 1:-1]) / dy, v[sl] * (f[1:-1, 2:, 1:-1] - f[sl]) / dy
            )
            adv_z = xp.where(
                w[sl] > 0, w[sl] * (f[sl] - f[1:-1, 1:-1, :-2]) / dz, w[sl] * (f[1:-1, 1:-1, 2:] - f[sl]) / dz
            )
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
            - beta * omega[sl] ** 2
            + diff_omega
            + (1.0 - F1) * CD_komega  # Cross-diffusion
        )
        omega[sl] = xp.maximum(omega_new, 1e-10)

        # ── Sınır şartları ───────────────────────────────────────────────
        k[mask_g] = 1e-10
        omega[mask_g] = 1e-10
        k[0, :, :] = k_init
        omega[0, :, :] = omega_init
        k[-1, :, :] = k[-2, :, :]
        omega[-1, :, :] = omega[-2, :, :]
        k[:, 0, :] = k[:, 1, :]
        omega[:, 0, :] = omega[:, 1, :]
        k[:, -1, :] = k[:, -2, :]
        omega[:, -1, :] = omega[:, -2, :]
        k[:, :, 0] = k[:, :, 1]
        omega[:, :, 0] = omega[:, :, 1]
        k[:, :, -1] = k[:, :, -2]
        omega[:, :, -1] = omega[:, :, -2]

        k = xp.where(xp.isnan(k) | xp.isinf(k), 1e-10, k)
        omega = xp.where(xp.isnan(omega) | xp.isinf(omega), 1e-10, omega)

    # ── Son eddy viskozite ────────────────────────────────────────────────
    omega_final = xp.maximum(omega, 1e-10)

    # Gerinim oranını tam grid üzerinde hesapla
    dudx_full = (u[2:, 1:-1, 1:-1] - u[:-2, 1:-1, 1:-1]) / (2 * dx)
    dvdy_full = (v[1:-1, 2:, 1:-1] - v[1:-1, :-2, 1:-1]) / (2 * dy)
    dwdz_full = (w[1:-1, 1:-1, 2:] - w[1:-1, 1:-1, :-2]) / (2 * dz)
    S_mag_full = xp.sqrt(2.0 * (dudx_full**2 + dvdy_full**2 + dwdz_full**2) + 1e-12)

    # Interior indeksleri kullan
    omega_int = omega_final[sl]
    k_int = k[sl]
    d_wall_int = d_wall[sl]

    F2_final = xp.tanh(
        xp.maximum(
            2.0 * KO_BETA_STAR * omega_int / (KO_A1 * xp.maximum(1e-10, S_mag_full)),
            500.0 * nu_mol / (d_wall_int**2 * omega_int + 1e-20),
        )
        ** 2
    )
    nu_t_int = KO_A1 * xp.maximum(k_int, 0.0) / xp.maximum(KO_A1 * omega_int, S_mag_full * F2_final + 1e-20)

    nu_t = xp.zeros_like(u)
    nu_t[sl] = xp.clip(nu_t_int, 0.0, 100.0)
    nu_t[mask_g] = 0.0
    return nu_t


def _build_wall_distance_cache(mask, dx, dy, dz, xp):
    """Duvar mesafesi hesaplamasını cache'leyip tekrar kullanım için saklar."""
    cache_key = (
        mask.shape,
        round(float(dx), 8),
        round(float(dy), 8),
        round(float(dz), 8),
        gpu_backend._hash_mask(mask),
    )
    if cache_key not in _WALL_DISTANCE_CACHE:
        mask_np = mask.get() if hasattr(mask, "get") else np.asarray(mask, dtype=bool)
        dist = distance_transform_edt(~mask_np, sampling=(dx, dy, dz))
        _lru_insert(_WALL_DISTANCE_CACHE, cache_key, dist)
    dist = _WALL_DISTANCE_CACHE[cache_key]
    _WALL_DISTANCE_CACHE.move_to_end(cache_key)
    return xp.asarray(dist) if xp is not np else np.asarray(dist)


def _apply_wall_functions(u, v, w, mask, d_wall, dx, dy, dz, nu_mol, xp):
    """Duvar komşu hücrelerine log-law wall function boundary condition uygular."""
    mask_np = mask.get() if hasattr(mask, "get") else np.asarray(mask, dtype=bool)
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

    u_slice = u[sl]
    v_slice = v[sl]
    w_slice = w[sl]
    d_wall_local = xp.maximum(d_wall[sl], 1e-6)

    vel_slice = xp.sqrt(u_slice**2 + v_slice**2 + w_slice**2) + 1e-12
    t_hat_u = u_slice / vel_slice
    t_hat_v = v_slice / vel_slice
    t_hat_w = w_slice / vel_slice

    u_tau = xp.sqrt(nu_mol * vel_slice / d_wall_local)
    y_plus = xp.maximum(u_tau * d_wall_local / nu_mol, 1e-6)

    log_layer = y_plus > 11.5
    u_plus = xp.where(log_layer, (1.0 / SA_KAPPA) * xp.log(y_plus) + 5.0, y_plus)
    # u_plus = u/u_tau oldugundan duvar hedef hizi u_wall = u_plus * u_tau
    # (nu/d ile carpma boyutsal olarak yanlis faktor verirdi).
    u_wall = u_plus * u_tau

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


# ── Fused SA (cupy) ────────────────────────────────────────────────────────────
# Python elementwise yoluyla birebir ayni matematik; iterasyon basina ~35 kernel
# yerine tek RawKernel. fp64 ve fp32 varyantlari derlenir.
_SA_STEP_SOURCE = r"""
#define T_CAST(x) (({t})(x))

extern "C" __global__ void sa_step(
    {t}* __restrict__ out, const {t}* __restrict__ nu_in,
    const {t}* __restrict__ u, const {t}* __restrict__ v, const {t}* __restrict__ w,
    const {t}* __restrict__ d_wall, const {t}* __restrict__ solid,
    {t} dx, {t} dy, {t} dz, {t} dt_sa, {t} nu_mol,
    {t} cb1, {t} cb2, {t} sigma, {t} cw1, {t} cw2, {t} cw3_6, {t} nu_max, {t} nu_bnd,
    int nx, int ny, int nz)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= nx * ny * nz) return;
    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);
    int ox = ny * nz;
    int oy = nz;

    if (i == 0 || i == nx - 1 || j == 0 || j == ny - 1 || k == 0 || k == nz - 1) {
        out[idx] = nu_bnd;
        return;
    }
    if (solid[idx] > T_CAST(0.5)) { out[idx] = T_CAST(0.0); return; }

    int ce = idx + ox, cw = idx - ox;
    int cn = idx + oy, cs = idx - oy;
    int ct = idx + 1, cb = idx - 1;

    {t} nu = nu_in[idx];
    {t} nu_e = nu_in[ce], nu_w = nu_in[cw];
    {t} nu_n = nu_in[cn], nu_s = nu_in[cs];
    {t} nu_t = nu_in[ct], nu_b = nu_in[cb];

    {t} d_nu_dx = (nu_e - nu_w) / (T_CAST(2.0) * dx);
    {t} d_nu_dy = (nu_n - nu_s) / (T_CAST(2.0) * dy);
    {t} d_nu_dz = (nu_t - nu_b) / (T_CAST(2.0) * dz);
    {t} grad2 = d_nu_dx * d_nu_dx + d_nu_dy * d_nu_dy + d_nu_dz * d_nu_dz;
    {t} lap_nu = (nu_e - T_CAST(2.0) * nu + nu_w) / (dx * dx)
               + (nu_n - T_CAST(2.0) * nu + nu_s) / (dy * dy)
               + (nu_t - T_CAST(2.0) * nu + nu_b) / (dz * dz);

    {t} du_dx = (u[ce] - u[cw]) / (T_CAST(2.0) * dx);
    {t} dv_dy = (v[cn] - v[cs]) / (T_CAST(2.0) * dy);
    {t} dw_dz = (w[ct] - w[cb]) / (T_CAST(2.0) * dz);
    {t} du_dy = (u[cn] - u[cs]) / (T_CAST(2.0) * dy);
    {t} dv_dx = (v[ce] - v[cw]) / (T_CAST(2.0) * dx);
    {t} du_dz = (u[ct] - u[cb]) / (T_CAST(2.0) * dz);
    {t} dw_dx = (w[ce] - w[cw]) / (T_CAST(2.0) * dx);
    {t} dv_dz = (v[ct] - v[cb]) / (T_CAST(2.0) * dz);
    {t} dw_dy = (w[cn] - w[cs]) / (T_CAST(2.0) * dy);

    {t} strain = sqrt(T_CAST(2.0) * (du_dx * du_dx + dv_dy * dv_dy + dw_dz * dw_dz
        + T_CAST(0.5) * (du_dy + dv_dx) * (du_dy + dv_dx)
        + T_CAST(0.5) * (du_dz + dw_dx) * (du_dz + dw_dx)
        + T_CAST(0.5) * (dv_dz + dw_dy) * (dv_dz + dw_dy)) + T_CAST(1e-12));

    {t} r = fmin(nu / ((strain * d_wall[idx]) * (strain * d_wall[idx]) + T_CAST(1e-12)), T_CAST(10.0));
    {t} g = fmin(r + cw2 * pow(r, T_CAST(6.0)), T_CAST(1e5));
    {t} fw = fmin(g * pow((T_CAST(1.0) + cw3_6) / (pow(g, T_CAST(6.0)) + cw3_6 + T_CAST(1e-12)), T_CAST(1.0) / T_CAST(6.0)), T_CAST(2.5));
    {t} prod = cb1 * strain * nu;
    {t} diff = (T_CAST(1.0) / sigma) * ((nu + nu_mol) * lap_nu + (T_CAST(1.0) + cb2) * grad2);
    {t} dw2 = nu / (d_wall[idx] + T_CAST(1e-12));
    {t} destr = -cw1 * fw * dw2 * dw2;

    {t} ue = u[idx], ve = v[idx], we = w[idx];
    {t} adv_x = ue > T_CAST(0.0) ? ue * (nu - nu_w) / dx : ue * (nu_e - nu) / dx;
    {t} adv_y = ve > T_CAST(0.0) ? ve * (nu - nu_s) / dy : ve * (nu_n - nu) / dy;
    {t} adv_z = we > T_CAST(0.0) ? we * (nu - nu_b) / dz : we * (nu_t - nu) / dz;

    {t} nu_new = nu + dt_sa * (-(adv_x + adv_y + adv_z) + prod + diff + destr);
    out[idx] = fmin(fmax(nu_new, T_CAST(0.0)), nu_max);
}
"""

_SA_KERNELS = {}


def _solve_sa_fused(u, v, w, mask, dx, dy, dz, nu_mol=1.5e-5, iterations=20, xp=None):
    """Tek RawKernel ile SA iterasyonu (cupy/fp64+fp32)."""
    if xp is None:
        xp = gpu_backend.cp
    n = u.size
    nx, ny, nz = u.shape

    d_wall = _compute_wall_distance(mask, dx, dy, dz, xp)
    dt_sa = min(dx, dy, dz) ** 2 / max(6.0 * nu_mol, 1e-6)
    dt_sa = min(dt_sa, 1e-3)

    min_spacing = min(dx, dy, dz)
    if min_spacing < 0.07:
        iterations = max(iterations, 80)
    elif min_spacing < 0.10:
        iterations = max(iterations, 70)
    elif min_spacing < 0.13:
        iterations = max(iterations, 60)
    else:
        iterations = max(iterations, 50)

    nu_bnd = 3.0 * nu_mol
    nu_max = 1e3 * nu_mol
    cw3_6 = SA_CW3**6

    if u.dtype == xp.float64:
        t = "double"

        def sc(a):
            return xp.float64(a)

        nu_in = xp.full(u.shape, nu_bnd, dtype=xp.float64)
        u_k, v_k, w_k = u, v, w
        d_k = d_wall
        solid = xp.asarray(mask, dtype=xp.float64)
    else:
        t = "float"

        def sc(a):
            return xp.float32(a)

        nu_in = xp.full(u.shape, nu_bnd, dtype=xp.float32)
        u_k = u.astype(xp.float32)
        v_k = v.astype(xp.float32)
        w_k = w.astype(xp.float32)
        d_k = d_wall.astype(xp.float32)
        solid = xp.asarray(mask, dtype=xp.float32)
    nu_in[mask] = 0.0
    nu_out = xp.empty_like(nu_in)

    kernel = _SA_KERNELS.get(t)
    if kernel is None:
        kernel = xp.RawKernel(_SA_STEP_SOURCE.replace("{t}", t), "sa_step")
        _SA_KERNELS[t] = kernel

    args = (
        u_k,
        v_k,
        w_k,
        d_k,
        solid,
        sc(dx),
        sc(dy),
        sc(dz),
        sc(dt_sa),
        sc(nu_mol),
        sc(SA_CB1),
        sc(SA_CB2),
        sc(SA_SIGMA),
        sc(SA_CW1),
        sc(SA_CW2),
        sc(cw3_6),
        sc(nu_max),
        sc(nu_bnd),
        xp.int32(nx),
        xp.int32(ny),
        xp.int32(nz),
    )
    block = 256
    grid = (n + block - 1) // block
    for _ in range(iterations):
        kernel((grid,), (block,), (nu_out, nu_in) + args)
        nu_in, nu_out = nu_out, nu_in
    nu_tilde = nu_in

    chi = nu_tilde / sc(nu_mol + 1e-12)
    fv1 = chi**3 / (chi**3 + sc(SA_CV1**3 + 1e-12))
    nu_t = nu_tilde * fv1
    nu_t[mask] = 0.0
    nu_t = xp.clip(nu_t, 0.0, 1000.0)
    if t == "float":
        nu_t = nu_t.astype(xp.float64)
    return nu_t
