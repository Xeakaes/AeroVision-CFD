"""
AeroVision CFD - OpenCL Solver Kernels
Intel/AMD GPU'larda Navier-Stokes cozumu icin tum gerekli kernel'ler.
Slice islemlerinin hepsi custom OpenCL kernel olarak yeniden yazildi.
"""

import numpy as np
import pyopencl as cl
import pyopencl.array as cl_array

_ABSWGS = 128  # abs_max3 paralel reduce workgroup boyutu


# ═══════════════════════════════════════════════════════════════════════════════
# KERNEL SOURCE CODE
# ═══════════════════════════════════════════════════════════════════════════════

_STENCIL6_KERNEL = """
// 6-nokta Laplacian: out[i,j,k] = (sum of 6 neighbors) / 6
__kernel void stencil6(
    __global float *out, __global const float *phi,
    int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    int total = nx * ny * nz;
    if (idx >= total) return;

    int k = idx % nz;
    int j = (idx / nz) % ny;
    int i = idx / (ny * nz);

    if (i < 1 || i >= nx-1 || j < 1 || j >= ny-1 || k < 1 || k >= nz-1) {
        out[idx] = phi[idx];
        return;
    }

    float sum = phi[((i+1)*ny + j)*nz + k] + phi[((i-1)*ny + j)*nz + k]
              + phi[(i*ny + (j+1))*nz + k] + phi[(i*ny + (j-1))*nz + k]
              + phi[(i*ny + j)*nz + (k+1)] + phi[(i*ny + j)*nz + (k-1)];

    out[idx] = sum / 6.0f;
}
"""

_WHERE_KERNEL = """
// phi_new[idx] = fluid[idx] ? update[idx] : phi_old[idx]
__kernel void where_fluid(
    __global float *phi_new, __global const float *update,
    __global const float *phi_old, __global const float *fluid,
    int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= nx * ny * nz) return;

    if (fluid[idx] > 0.5f)
        phi_new[idx] = update[idx];
    else
        phi_new[idx] = phi_old[idx];
}
"""

_GRADIENT_KERNEL = """
__kernel void gradient_x(
    __global float *out, __global const float *phi,
    float inv_2dx, int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= nx * ny * nz) return;
    int k = idx % nz; int j = (idx / nz) % ny; int i = idx / (ny * nz);
    if (i >= 1 && i < nx - 1)
        out[idx] = (phi[((i+1)*ny+j)*nz+k] - phi[((i-1)*ny+j)*nz+k]) * inv_2dx;
    else
        out[idx] = 0.0f;
}

__kernel void gradient_y(
    __global float *out, __global const float *phi,
    float inv_2dy, int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= nx * ny * nz) return;
    int k = idx % nz; int j = (idx / nz) % ny; int i = idx / (ny * nz);
    if (j >= 1 && j < ny - 1)
        out[idx] = (phi[(i*ny+(j+1))*nz+k] - phi[(i*ny+(j-1))*nz+k]) * inv_2dy;
    else
        out[idx] = 0.0f;
}

__kernel void gradient_z(
    __global float *out, __global const float *phi,
    float inv_2dz, int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= nx * ny * nz) return;
    int k = idx % nz; int j = (idx / nz) % ny; int i = idx / (ny * nz);
    if (k >= 1 && k < nz - 1)
        out[idx] = (phi[(i*ny+j)*nz+(k+1)] - phi[(i*ny+j)*nz+(k-1)]) * inv_2dz;
    else
        out[idx] = 0.0f;
}
"""

_VISCOSITY_KERNEL = """
// Konservatif viskozite: d/dx(nu_eff * df/dx) + d/dy(...) + d/dz(...)
__kernel void viscous_x(
    __global float *out, __global const float *f,
    __global const float *nu, float inv_dx2, int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    int ix = nx-2, iy = ny-2, iz = nz-2;
    if (idx >= ix * iy * iz) return;
    int ck = idx % iz; int cj = (idx / iz) % iy; int ci = idx / (iy * iz);
    int i = ci+1, j = cj+1, k = ck+1;
    int ce = ((i+1)*ny+j)*nz+k, cc = (i*ny+j)*nz+k, cw = ((i-1)*ny+j)*nz+k;
    float nu_e = 0.5f * (nu[ce] + nu[cc]);
    float nu_w = 0.5f * (nu[cw] + nu[cc]);
    out[idx] = (nu_e*(f[ce]-f[cc]) - nu_w*(f[cc]-f[cw])) * inv_dx2;
}

__kernel void viscous_y(
    __global float *out, __global const float *f,
    __global const float *nu, float inv_dy2, int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    int ix = nx-2, iy = ny-2, iz = nz-2;
    if (idx >= ix * iy * iz) return;
    int ck = idx % iz; int cj = (idx / iz) % iy; int ci = idx / (iy * iz);
    int i = ci+1, j = cj+1, k = ck+1;
    int cn = (i*ny+(j+1))*nz+k, cc = (i*ny+j)*nz+k, cs = (i*ny+(j-1))*nz+k;
    float nu_n = 0.5f * (nu[cn] + nu[cc]);
    float nu_s = 0.5f * (nu[cs] + nu[cc]);
    out[idx] = (nu_n*(f[cn]-f[cc]) - nu_s*(f[cc]-f[cs])) * inv_dy2;
}

__kernel void viscous_z(
    __global float *out, __global const float *f,
    __global const float *nu, float inv_dz2, int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    int ix = nx-2, iy = ny-2, iz = nz-2;
    if (idx >= ix * iy * iz) return;
    int ck = idx % iz; int cj = (idx / iz) % iy; int ci = idx / (iy * iz);
    int i = ci+1, j = cj+1, k = ck+1;
    int ct = (i*ny+j)*nz+(k+1), cc = (i*ny+j)*nz+k, cb = (i*ny+j)*nz+(k-1);
    float nu_t = 0.5f * (nu[ct] + nu[cc]);
    float nu_b = 0.5f * (nu[cb] + nu[cc]);
    out[idx] = (nu_t*(f[ct]-f[cc]) - nu_b*(f[cc]-f[cb])) * inv_dz2;
}
"""

_NS_RHS_KERNEL = """
// NS sag taraf: rhs = (-adv(f) + visc(f)) * vol_weight
// n_total boyutunda output — sinir hucresde 0 yazar
// Adveksiyon: CPU'daki Van Leer gradyan formuyla birebir:
//   fb = (f[i]-f[i-1])*inv_dx, ff = (f[i+1]-f[i])*inv_dx
//   slope = VL(fb, ff) (harmonik limiter)
//   u>0: grad = fb + 0.5*(slope-fb)   u<=0: grad = ff - 0.5*(slope-ff)
__kernel void ns_rhs(
    __global float *rhs_u, __global float *rhs_v, __global float *rhs_w,
    __global const float *u_in, __global const float *v_in, __global const float *w_in,
    __global const float *nu_eff, __global const float *vol_weight,
    float inv_dx, float inv_dy, float inv_dz,
    float inv_dx2, float inv_dy2, float inv_dz2,
    int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    int total = nx * ny * nz;
    if (idx >= total) return;

    int k = idx % nz; int j = (idx / nz) % ny; int i = idx / (ny * nz);

    if (i < 1 || i >= nx-1 || j < 1 || j >= ny-1 || k < 1 || k >= nz-1) {
        rhs_u[idx] = 0.0f;
        rhs_v[idx] = 0.0f;
        rhs_w[idx] = 0.0f;
        return;
    }

    int cc = idx;
    int ce=((i+1)*ny+j)*nz+k, cw=((i-1)*ny+j)*nz+k;
    int cn=(i*ny+(j+1))*nz+k, cs=(i*ny+(j-1))*nz+k;
    int ct=(i*ny+j)*nz+(k+1), cb=(i*ny+j)*nz+(k-1);

    float ue=u_in[cc], ve=v_in[cc], we=w_in[cc];

    float f_u_cw = u_in[cw], f_u_cc = u_in[cc], f_u_ce = u_in[ce];
    float f_u_cs = u_in[cs], f_u_cn = u_in[cn], f_u_cb = u_in[cb], f_u_ct = u_in[ct];
    float f_v_cw = v_in[cw], f_v_cc = v_in[cc], f_v_ce = v_in[ce];
    float f_v_cs = v_in[cs], f_v_cn = v_in[cn], f_v_cb = v_in[cb], f_v_ct = v_in[ct];
    float f_w_cw = w_in[cw], f_w_cc = w_in[cc], f_w_ce = w_in[ce];
    float f_w_cs = w_in[cs], f_w_cn = w_in[cn], f_w_cb = w_in[cb], f_w_ct = w_in[ct];

    float fb_u_x = (f_u_cc - f_u_cw) * inv_dx, ff_u_x = (f_u_ce - f_u_cc) * inv_dx;
    float fb_v_x = (f_v_cc - f_v_cw) * inv_dx, ff_v_x = (f_v_ce - f_v_cc) * inv_dx;
    float fb_w_x = (f_w_cc - f_w_cw) * inv_dx, ff_w_x = (f_w_ce - f_w_cc) * inv_dx;
    float fb_u_y = (f_u_cc - f_u_cs) * inv_dy, ff_u_y = (f_u_cn - f_u_cc) * inv_dy;
    float fb_v_y = (f_v_cc - f_v_cs) * inv_dy, ff_v_y = (f_v_cn - f_v_cc) * inv_dy;
    float fb_w_y = (f_w_cc - f_w_cs) * inv_dy, ff_w_y = (f_w_cn - f_w_cc) * inv_dy;
    float fb_u_z = (f_u_cc - f_u_cb) * inv_dz, ff_u_z = (f_u_ct - f_u_cc) * inv_dz;
    float fb_v_z = (f_v_cc - f_v_cb) * inv_dz, ff_v_z = (f_v_ct - f_v_cc) * inv_dz;
    float fb_w_z = (f_w_cc - f_w_cb) * inv_dz, ff_w_z = (f_w_ct - f_w_cc) * inv_dz;

    float sl_u_x = (fb_u_x * ff_u_x > 0.0f) ? 2.0f * fb_u_x * ff_u_x / (fb_u_x + ff_u_x + 1e-12f) : 0.0f;
    float sl_v_x = (fb_v_x * ff_v_x > 0.0f) ? 2.0f * fb_v_x * ff_v_x / (fb_v_x + ff_v_x + 1e-12f) : 0.0f;
    float sl_w_x = (fb_w_x * ff_w_x > 0.0f) ? 2.0f * fb_w_x * ff_w_x / (fb_w_x + ff_w_x + 1e-12f) : 0.0f;
    float sl_u_y = (fb_u_y * ff_u_y > 0.0f) ? 2.0f * fb_u_y * ff_u_y / (fb_u_y + ff_u_y + 1e-12f) : 0.0f;
    float sl_v_y = (fb_v_y * ff_v_y > 0.0f) ? 2.0f * fb_v_y * ff_v_y / (fb_v_y + ff_v_y + 1e-12f) : 0.0f;
    float sl_w_y = (fb_w_y * ff_w_y > 0.0f) ? 2.0f * fb_w_y * ff_w_y / (fb_w_y + ff_w_y + 1e-12f) : 0.0f;
    float sl_u_z = (fb_u_z * ff_u_z > 0.0f) ? 2.0f * fb_u_z * ff_u_z / (fb_u_z + ff_u_z + 1e-12f) : 0.0f;
    float sl_v_z = (fb_v_z * ff_v_z > 0.0f) ? 2.0f * fb_v_z * ff_v_z / (fb_v_z + ff_v_z + 1e-12f) : 0.0f;
    float sl_w_z = (fb_w_z * ff_w_z > 0.0f) ? 2.0f * fb_w_z * ff_w_z / (fb_w_z + ff_w_z + 1e-12f) : 0.0f;

    float g_u_x = (ue > 0.0f) ? fb_u_x + 0.5f * (sl_u_x - fb_u_x) : ff_u_x - 0.5f * (sl_u_x - ff_u_x);
    float g_v_x = (ue > 0.0f) ? fb_v_x + 0.5f * (sl_v_x - fb_v_x) : ff_v_x - 0.5f * (sl_v_x - ff_v_x);
    float g_w_x = (ue > 0.0f) ? fb_w_x + 0.5f * (sl_w_x - fb_w_x) : ff_w_x - 0.5f * (sl_w_x - ff_w_x);
    float g_u_y = (ve > 0.0f) ? fb_u_y + 0.5f * (sl_u_y - fb_u_y) : ff_u_y - 0.5f * (sl_u_y - ff_u_y);
    float g_v_y = (ve > 0.0f) ? fb_v_y + 0.5f * (sl_v_y - fb_v_y) : ff_v_y - 0.5f * (sl_v_y - ff_v_y);
    float g_w_y = (ve > 0.0f) ? fb_w_y + 0.5f * (sl_w_y - fb_w_y) : ff_w_y - 0.5f * (sl_w_y - ff_w_y);
    float g_u_z = (we > 0.0f) ? fb_u_z + 0.5f * (sl_u_z - fb_u_z) : ff_u_z - 0.5f * (sl_u_z - ff_u_z);
    float g_v_z = (we > 0.0f) ? fb_v_z + 0.5f * (sl_v_z - fb_v_z) : ff_v_z - 0.5f * (sl_v_z - ff_v_z);
    float g_w_z = (we > 0.0f) ? fb_w_z + 0.5f * (sl_w_z - fb_w_z) : ff_w_z - 0.5f * (sl_w_z - ff_w_z);

    float adv_u = ue * g_u_x + ve * g_u_y + we * g_u_z;
    float adv_v = ue * g_v_x + ve * g_v_y + we * g_v_z;
    float adv_w = ue * g_w_x + ve * g_w_y + we * g_w_z;

    float nu_e=0.5f*(nu_eff[ce]+nu_eff[cc]), nu_w=0.5f*(nu_eff[cw]+nu_eff[cc]);
    float nu_n=0.5f*(nu_eff[cn]+nu_eff[cc]), nu_s=0.5f*(nu_eff[cs]+nu_eff[cc]);
    float nu_t=0.5f*(nu_eff[ct]+nu_eff[cc]), nu_b=0.5f*(nu_eff[cb]+nu_eff[cc]);

    float visc_u = (nu_e*(u_in[ce]-u_in[cc]) - nu_w*(u_in[cc]-u_in[cw]))*inv_dx2
                 + (nu_n*(u_in[cn]-u_in[cc]) - nu_s*(u_in[cc]-u_in[cs]))*inv_dy2
                 + (nu_t*(u_in[ct]-u_in[cc]) - nu_b*(u_in[cc]-u_in[cb]))*inv_dz2;
    float visc_v = (nu_e*(v_in[ce]-v_in[cc]) - nu_w*(v_in[cc]-v_in[cw]))*inv_dx2
                 + (nu_n*(v_in[cn]-v_in[cc]) - nu_s*(v_in[cc]-v_in[cs]))*inv_dy2
                 + (nu_t*(v_in[ct]-v_in[cc]) - nu_b*(v_in[cc]-v_in[cb]))*inv_dz2;
    float visc_w = (nu_e*(w_in[ce]-w_in[cc]) - nu_w*(w_in[cc]-w_in[cw]))*inv_dx2
                 + (nu_n*(w_in[cn]-w_in[cc]) - nu_s*(w_in[cc]-w_in[cs]))*inv_dy2
                 + (nu_t*(w_in[ct]-w_in[cc]) - nu_b*(w_in[cc]-w_in[cb]))*inv_dz2;

    float vw = vol_weight[cc];
    rhs_u[idx] = (-adv_u + visc_u) * vw;
    rhs_v[idx] = (-adv_v + visc_v) * vw;
    rhs_w[idx] = (-adv_w + visc_w) * vw;
}
"""

_RK2_UPDATE_KERNEL = """
// RK2 (Heun) zaman guncelleme: u_new = u_old + 0.5*dt*(rhs1 + rhs2)
__kernel void rk2_update(
    __global float *u_out,
    __global const float *u_old, __global const float *rhs1, __global const float *rhs2,
    float half_dt, int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= nx * ny * nz) return;
    u_out[idx] = u_old[idx] + half_dt * (rhs1[idx] + rhs2[idx]);
}

// Fused RK2 stage 1: u_out = (u_old + dt*rhs) * inv_mask
// (eski: u_star = u_cl + dt_f*rhs1; u_star = u_star*inv_mask — ayni islem
//  sirasi, bitwise ayni sonuc; 2 enqueue yerine 1)
__kernel void rk2_stage1(
    __global float *u_out, __global const float *u_old,
    __global const float *rhs, __global const float *inv_mask,
    float dt, int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    u_out[idx] = (u_old[idx] + dt * rhs[idx]) * inv_mask[idx];
}

// Fused RK2 final: u_new = (u_old + half_dt*(rhs1+rhs2)) * vol * inv_mask
// (eski: (u_cl + half_dt*(rhs1+rhs2))*vol_cl sonra *inv_mask — ayni sirasi,
//  bitwise ayni sonuc; 7 enqueue yerine 1)
__kernel void rk2_combine(
    __global float *u_out, __global const float *u_old,
    __global const float *rhs1, __global const float *rhs2,
    __global const float *vol, __global const float *inv_mask,
    float half_dt, int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    u_out[idx] = (u_old[idx] + half_dt * (rhs1[idx] + rhs2[idx])) * vol[idx] * inv_mask[idx];
}
"""

_TUNNEL_WALL_BC_KERNEL = """
// Tünel duvar şartları (slip boundary)
__kernel void tunnel_wall_bc(
    __global float *u, __global float *v, __global float *w,
    int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= nx * ny * nz) return;
    int k = idx % nz; int j = (idx / nz) % ny; int i = idx / (ny * nz);

    // Y duvarlari (j=0 ve j=ny-1)
    if (j == 0 || j == ny-1) {
        v[idx] = 0.0f;
    }
    // Z duvarlari (k=0 ve k=nz-1)
    if (k == 0 || k == nz-1) {
        w[idx] = 0.0f;
    }
}
"""

_WALL_BC_COPY_KERNEL = """
// Slip boundary: duvar hizini komsu hucreden kopyala
// Y duvarlari (j=0, j=ny-1): u VE w komsudan kopyalanir (v=0 tunnel kernel'de).
// Z duvarlari (k=0, k=nz-1): u VE v komsudan kopyalanir (w=0 tunnel kernel'de;
// w kopyasi YAPILMAZ — tunnel kernel'inin sifirlamasini ezerdi).
// Sirasiyla calisirlar (tunnel -> Y -> Z) — Z, Y'nin yazdigi degerleri okur
// (kopse hucrelerinde mevcut 8-ayri-kernel davranisiyla birebir ayni).
__kernel void wall_bc_copy_y(
    __global float *u, __global float *w, int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= nx * ny * nz) return;
    int k = idx % nz; int j = (idx / nz) % ny; int i = idx / (ny * nz);
    if (j == 0) {
        u[idx] = u[(i*ny + 1)*nz + k];
        w[idx] = w[(i*ny + 1)*nz + k];
    } else if (j == ny-1) {
        u[idx] = u[(i*ny + (ny-2))*nz + k];
        w[idx] = w[(i*ny + (ny-2))*nz + k];
    }
}

__kernel void wall_bc_copy_z(
    __global float *u, __global float *v, int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= nx * ny * nz) return;
    int k = idx % nz; int j = (idx / nz) % ny; int i = idx / (ny * nz);
    if (k == 0) {
        u[idx] = u[(i*ny + j)*nz + 1];
        v[idx] = v[(i*ny + j)*nz + 1];
    } else if (k == nz-1) {
        u[idx] = u[(i*ny + j)*nz + (nz-2)];
        v[idx] = v[(i*ny + j)*nz + (nz-2)];
    }
}
"""

_DIVERGENCE_KERNEL = """
// Basinc kaynigi: div = dU/dx + dV/dy + dW/dz (Rhie-Chow face velocity'den)
__kernel void compute_divergence(
    __global float *div,
    __global const float *u_e, __global const float *u_w,
    __global const float *v_n, __global const float *v_s,
    __global const float *w_t, __global const float *w_b,
    float inv_dx, float inv_dy, float inv_dz,
    int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    int ix = nx-2, iy = ny-2, iz = nz-2;
    if (idx >= ix * iy * iz) return;
    div[idx] = (u_e[idx] - u_w[idx]) * inv_dx
             + (v_n[idx] - v_s[idx]) * inv_dy
             + (w_t[idx] - w_b[idx]) * inv_dz;
}
"""

_COPY_KERNEL = """
// out[:] = in[:] (flat copy)
__kernel void array_copy(
    __global float *out, __global const float *in, int n)
{
    int idx = get_global_id(0);
    if (idx < n) out[idx] = in[idx];
}

// u[:] *= inv_mask[:] — in-place katı sifirlama (project sonrasi)
__kernel void mask_inplace(
    __global float *u, __global const float *inv_mask, int n)
{
    int idx = get_global_id(0);
    if (idx < n) u[idx] *= inv_mask[idx];
}
"""

_ABS_MAX_KERNEL = """
// Reduction: bul ve dondur |arr[i]|'nin maksimumu
// (Basit tek-pass — buyuk diziler icin optimizasyon gerekir)
__kernel void abs_max_reduce(
    __global const float *arr, __global float *result, int n)
{
    int idx = get_global_id(0);
    if (idx != 0) return;
    float mx = 0.0f;
    for (int i = 0; i < n; i++) {
        float v = fabs(arr[i]);
        if (v > mx) mx = v;
    }
    result[0] = mx;
}

// Uc dizinin |max|'ini TEK geciste alir (dt hesabi — host sync 6x->1x).
// Paralel: per-block |max|'lar partial buffer'a, son tek-pass reduce ile
// tek (3,) buffer'a. (eski tek-thread dongusu 16000 eleman icin yavasti)
#define ABSWGS 128
__kernel void abs_max3_reduce(
    __global const float *a1, __global const float *a2, __global const float *a3,
    __global float *partial, int n)
{
    int idx = get_global_id(0);
    float m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
    if (idx < n) {
        m1 = fabs(a1[idx]);
        m2 = fabs(a2[idx]);
        m3 = fabs(a3[idx]);
    }
    __local float l1[ABSWGS], l2[ABSWGS], l3[ABSWGS];
    int lid = get_local_id(0);
    l1[lid] = m1; l2[lid] = m2; l3[lid] = m3;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = ABSWGS / 2; s > 0; s >>= 1) {
        if (lid < s) {
            if (l2[lid + s] > l2[lid]) l2[lid] = l2[lid + s];
            if (l3[lid + s] > l3[lid]) l3[lid] = l3[lid + s];
            if (l1[lid + s] > l1[lid]) l1[lid] = l1[lid + s];
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        int g = get_group_id(0);
        partial[g * 3 + 0] = l1[0];
        partial[g * 3 + 1] = l2[0];
        partial[g * 3 + 2] = l3[0];
    }
}

__kernel void abs_max3_final(
    __global float *out, __global const float *partial, int nwg)
{
    if (get_global_id(0) != 0) return;
    float m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
    for (int i = 0; i < nwg; i++) {
        float v1 = partial[i * 3 + 0]; if (v1 > m1) m1 = v1;
        float v2 = partial[i * 3 + 1]; if (v2 > m2) m2 = v2;
        float v3 = partial[i * 3 + 2]; if (v3 > m3) m3 = v3;
    }
    out[0] = m1; out[1] = m2; out[2] = m3;
}
"""

_WALL_DISTANCE_KERNEL = """
// BFS tabanli duvar mesafesi (basit yaklasim)
__kernel void wall_distance_init(
    __global float *d_wall, __global const char *mask,
    int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= nx * ny * nz) return;
    d_wall[idx] = (mask[idx] != 0) ? 0.0f : 1e6f;
}
"""

_NS_DIVERGENCE_RHS_KERNEL = """
// Basinc Poisson sag tarafi: rhs = div_star * inv_dt * vol
// div_star, yuzey-enterpolasyonlu hizlardan (CPU yolu ile birebir):
//   u_fe = 0.5*(u[i+1] + u[i]) vb.
//   RHS = div/dt — volume carpani YOK (CPU ile birebir; kismi doluluk
//   hucrelerinde CPU'da da carpan yok). Kati hucrede rhs = 0 (katida
//   inv_mask ile ayrica sifirlanir, burada da guvenlik icin).
__kernel void ns_divergence_rhs(
    __global float *rhs,
    __global const float *u, __global const float *v, __global const float *w,
    __global const float *solid,
    float inv_dx, float inv_dy, float inv_dz, float inv_dt,
    int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= nx * ny * nz) return;
    rhs[idx] = 0.0f;
    int k = idx % nz; int j = (idx / nz) % ny; int i = idx / (ny * nz);
    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1 || k < 1 || k >= nz - 1) return;
    if (solid[idx] > 0.5f) return;

    int c = idx;
    int ce = ((i + 1) * ny + j) * nz + k, cw = ((i - 1) * ny + j) * nz + k;
    int cn = (i * ny + (j + 1)) * nz + k, cs = (i * ny + (j - 1)) * nz + k;
    int ct = (i * ny + j) * nz + (k + 1), cb = (i * ny + j) * nz + (k - 1);

    // Katiya bitisik yuzeylerde NORMAL aki 0 (impenetrable — CPU m_e/m_w/...).
    float u_fe = (solid[ce] > 0.5f || solid[c] > 0.5f) ? 0.0f : 0.5f * (u[ce] + u[c]);
    float u_fw = (solid[cw] > 0.5f || solid[c] > 0.5f) ? 0.0f : 0.5f * (u[c] + u[cw]);
    float v_fn = (solid[cn] > 0.5f || solid[c] > 0.5f) ? 0.0f : 0.5f * (v[cn] + v[c]);
    float v_fs = (solid[cs] > 0.5f || solid[c] > 0.5f) ? 0.0f : 0.5f * (v[c] + v[cs]);
    float w_ft = (solid[ct] > 0.5f || solid[c] > 0.5f) ? 0.0f : 0.5f * (w[ct] + w[c]);
    float w_fb = (solid[cb] > 0.5f || solid[c] > 0.5f) ? 0.0f : 0.5f * (w[c] + w[cb]);

    float div = (u_fe - u_fw) * inv_dx + (v_fn - v_fs) * inv_dy + (w_ft - w_fb) * inv_dz;
    rhs[idx] = div * inv_dt;
}
"""

_POISSON_STENCIL_KERNEL = """
// 7 nokta Laplacian matvec — pozitif tanimli karsilik (pAp > 0 guvence):
//   A_p = -A_cpu  (diag = +2(ax+ay+az), komsular -ax/-ay/-az)
//   ayni cozum: A_p x = -b  <=>  A_cpu x = b
//   satir 0 (idx==0): dirichlet, out = 0
//   katı satirlar: out = 0 (diag=1, rhs=0 -> p=0)
//   sinir satirlari: kayip komsu katkisi yok (tam diag korunur)
//   kati komsuluk: Neumann (tek tarafli) — o yonun diag katkisi duser;
//   RHS'teki 0-aki maskesi ve project_correct 0-dp maskesiyle tutarli.
__kernel void poisson_matvec(
    __global float *out, __global const float *p, __global const float *solid,
    float ax, float ay, float az, int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= nx * ny * nz) return;
    if (idx == 0 || solid[idx] > 0.5f) { out[idx] = 0.0f; return; }

    int k = idx % nz; int j = (idx / nz) % ny; int i = idx / (ny * nz);
    int ce = ((i + 1) * ny + j) * nz + k, cw = ((i - 1) * ny + j) * nz + k;
    int cn = (i * ny + (j + 1)) * nz + k, cs = (i * ny + (j - 1)) * nz + k;
    int ct = (i * ny + j) * nz + (k + 1), cb = (i * ny + j) * nz + (k - 1);

    float acc = 2.0f * (ax + ay + az) * p[idx];
    if (i < nx - 1) { if (solid[ce] > 0.5f) acc -= ax * p[idx]; else acc -= ax * p[ce]; }
    if (i > 0)      { if (solid[cw] > 0.5f) acc -= ax * p[idx]; else acc -= ax * p[cw]; }
    if (j < ny - 1) { if (solid[cn] > 0.5f) acc -= ay * p[idx]; else acc -= ay * p[cn]; }
    if (j > 0)      { if (solid[cs] > 0.5f) acc -= ay * p[idx]; else acc -= ay * p[cs]; }
    if (k < nz - 1) { if (solid[ct] > 0.5f) acc -= az * p[idx]; else acc -= az * p[ct]; }
    if (k > 0)      { if (solid[cb] > 0.5f) acc -= az * p[idx]; else acc -= az * p[cb]; }
    out[idx] = acc;
}

// ── CG fused update'ler (host sync ve kernel sayisini azaltir) ──────────
// cg_init: r = b - A*x ; z = r*inv_diag ; p = z  (tek gecis)
// cg_xrz:  x += alpha*p ; r -= alpha*Ap ; z = r*inv_diag (tek gecis)
// cg_beta: p = z + beta*p (tek gecis)
__kernel void cg_init(
    __global float *r, __global float *z, __global float *p,
    __global const float *b, __global const float *ap,
    __global const float *inv_diag, int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    float ri = b[idx] - ap[idx];
    r[idx] = ri;
    float zi = ri * inv_diag[idx];
    z[idx] = zi;
    p[idx] = zi;
}

__kernel void cg_xrz(
    __global float *x, __global float *r, __global float *z,
    __global const float *p, __global const float *ap,
    __global const float *inv_diag, float alpha, int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    x[idx] += alpha * p[idx];
    float ri = r[idx] - alpha * ap[idx];
    r[idx] = ri;
    z[idx] = ri * inv_diag[idx];
}

__kernel void cg_beta(
    __global float *p, __global const float *z, float beta, int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    p[idx] = z[idx] + beta * p[idx];
}

// ── Device-resident CG (iterasyon basina host sync YOK) ──────────────────
// 5 kernel/iter: cg_matvec_dot + cg_reduce_alpha + cg_xrz_dot +
// cg_reduce_beta + cg_pbeta. Skalerler (alpha/beta/rz/bnorm) ve status
// [2]={conv,fail} cihazda tutulur; host ~16 iterasyonda bir status'u okur
// (~5100 host sync yerine ~170). Siralama ayni tek kuyrukta garanti —
// race yok. Toplama sirasi farkli oldugundan alfa/beta ~1e-7 duzeyinde
// degisir; cozum ayni tolerans/kriterde (tol*b_norm), CPU ile parite
// korunur.
#define CGLOCAL 128

__kernel void cg_warm(
    __global float *x, __global float *p,
    __global const float *x0, __global const float *inv_mask, int n)
{
    int idx = get_global_id(0);
    if (idx < n) {
        float xi = x0[idx] * inv_mask[idx];
        x[idx] = xi;
        p[idx] = xi;
    }
}

__kernel void cg_final(
    __global float *x, __global const float *inv_mask, int n)
{
    int idx = get_global_id(0);
    if (idx < n) x[idx] *= inv_mask[idx];
}

__kernel void cg_bgen(
    __global float *b, __global const float *rhs, __global const float *inv_mask,
    __global float *partial_bb, int n)
{
    int idx = get_global_id(0);
    float v = 0.0f;
    if (idx < n) {
        float bi = -rhs[idx] * inv_mask[idx];
        b[idx] = bi;
        v = bi * bi;
    }
    __local float lsum[CGLOCAL];
    int lid = get_local_id(0);
    lsum[lid] = v;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = CGLOCAL / 2; s > 0; s >>= 1) {
        if (lid < s) lsum[lid] += lsum[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial_bb[get_group_id(0)] = lsum[0];
}

__kernel void cg_reduce_bnorm(
    __global float *bnorm, __global const float *partial_bb, int nwg)
{
    int idx = get_global_id(0);
    __local float lsum[CGLOCAL];
    int lid = get_local_id(0);
    lsum[lid] = (idx < nwg) ? partial_bb[idx] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = CGLOCAL / 2; s > 0; s >>= 1) {
        if (lid < s) lsum[lid] += lsum[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) bnorm[0] = sqrt(lsum[0]);
}

__kernel void cg_matvec_dot(
    __global float *out, __global const float *p, __global const float *solid,
    float ax, float ay, float az, int nx, int ny, int nz,
    __global float *partial_pAp, int n)
{
    int idx = get_global_id(0);
    float acc = 0.0f;
    if (idx < n) {
        if (idx != 0 && solid[idx] <= 0.5f) {
            int k = idx % nz; int j = (idx / nz) % ny; int i = idx / (ny * nz);
            int ce = ((i + 1) * ny + j) * nz + k, cw = ((i - 1) * ny + j) * nz + k;
            int cn = (i * ny + (j + 1)) * nz + k, cs = (i * ny + (j - 1)) * nz + k;
            int ct = (i * ny + j) * nz + (k + 1), cb = (i * ny + j) * nz + (k - 1);
            acc = 2.0f * (ax + ay + az) * p[idx];
            if (i < nx - 1) { if (solid[ce] > 0.5f) acc -= ax * p[idx]; else acc -= ax * p[ce]; }
            if (i > 0)      { if (solid[cw] > 0.5f) acc -= ax * p[idx]; else acc -= ax * p[cw]; }
            if (j < ny - 1) { if (solid[cn] > 0.5f) acc -= ay * p[idx]; else acc -= ay * p[cn]; }
            if (j > 0)      { if (solid[cs] > 0.5f) acc -= ay * p[idx]; else acc -= ay * p[cs]; }
            if (k < nz - 1) { if (solid[ct] > 0.5f) acc -= az * p[idx]; else acc -= az * p[ct]; }
            if (k > 0)      { if (solid[cb] > 0.5f) acc -= az * p[idx]; else acc -= az * p[cb]; }
        }
        out[idx] = acc;
        acc = p[idx] * acc;
    }
    __local float lsum[CGLOCAL];
    int lid = get_local_id(0);
    lsum[lid] = acc;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = CGLOCAL / 2; s > 0; s >>= 1) {
        if (lid < s) lsum[lid] += lsum[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) partial_pAp[get_group_id(0)] = lsum[0];
}

__kernel void cg_init_dot(
    __global float *r, __global float *z, __global float *p,
    __global const float *b, __global const float *ap,
    __global const float *inv_diag,
    __global float *partial_rz, __global float *partial_rr, int n)
{
    int idx = get_global_id(0);
    float vrz = 0.0f, vrr = 0.0f;
    if (idx < n) {
        float ri = b[idx] - ap[idx];
        r[idx] = ri;
        float zi = ri * inv_diag[idx];
        z[idx] = zi;
        p[idx] = zi;
        vrz = ri * zi;
        vrr = ri * ri;
    }
    __local float lsumA[CGLOCAL], lsumB[CGLOCAL];
    int lid = get_local_id(0);
    lsumA[lid] = vrz;
    lsumB[lid] = vrr;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = CGLOCAL / 2; s > 0; s >>= 1) {
        if (lid < s) { lsumA[lid] += lsumA[lid + s]; lsumB[lid] += lsumB[lid + s]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        partial_rz[get_group_id(0)] = lsumA[0];
        partial_rr[get_group_id(0)] = lsumB[0];
    }
}

__kernel void cg_reduce_alpha(
    __global float *alpha, __global const float *rz, __global int *status,
    __global const float *partial_pAp, int nwg)
{
    int idx = get_global_id(0);
    __local float lsum[CGLOCAL];
    int lid = get_local_id(0);
    lsum[lid] = (idx < nwg) ? partial_pAp[idx] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = CGLOCAL / 2; s > 0; s >>= 1) {
        if (lid < s) lsum[lid] += lsum[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        float pAp = lsum[0];
        status[0] = 0;
        if (pAp <= 0.0f) { status[1] = 1; alpha[0] = 0.0f; }
        else { status[1] = 0; alpha[0] = rz[0] / pAp; }
    }
}

__kernel void cg_xrz_dot(
    __global float *x, __global float *r, __global float *z,
    __global const float *p, __global const float *ap,
    __global const float *inv_diag, __global const float *alpha,
    __global float *partial_rz, __global float *partial_rr, int n)
{
    int idx = get_global_id(0);
    float vrz = 0.0f, vrr = 0.0f;
    if (idx < n) {
        float a = alpha[0];
        x[idx] += a * p[idx];
        float ri = r[idx] - a * ap[idx];
        r[idx] = ri;
        float zi = ri * inv_diag[idx];
        z[idx] = zi;
        vrz = ri * zi;
        vrr = ri * ri;
    }
    __local float lsumA[CGLOCAL], lsumB[CGLOCAL];
    int lid = get_local_id(0);
    lsumA[lid] = vrz;
    lsumB[lid] = vrr;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = CGLOCAL / 2; s > 0; s >>= 1) {
        if (lid < s) { lsumA[lid] += lsumA[lid + s]; lsumB[lid] += lsumB[lid + s]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        partial_rz[get_group_id(0)] = lsumA[0];
        partial_rr[get_group_id(0)] = lsumB[0];
    }
}

__kernel void cg_reduce_beta(
    __global float *beta, __global float *rz, __global int *status,
    __global const float *partial_rz, __global const float *partial_rr,
    float tol2b, int nwg)
{
    int idx = get_global_id(0);
    __local float lsumA[CGLOCAL], lsumB[CGLOCAL];
    int lid = get_local_id(0);
    lsumA[lid] = (idx < nwg) ? partial_rz[idx] : 0.0f;
    lsumB[lid] = (idx < nwg) ? partial_rr[idx] : 0.0f;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = CGLOCAL / 2; s > 0; s >>= 1) {
        if (lid < s) { lsumA[lid] += lsumA[lid + s]; lsumB[lid] += lsumB[lid + s]; }
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    if (lid == 0) {
        float rz_new = lsumA[0];
        if (rz_new < 0.0f) rz_new = 0.0f;
        beta[0] = rz_new / (rz[0] + 1e-30f);
        rz[0] = rz_new;
        status[0] = (lsumB[0] <= tol2b) ? 1 : 0;
        status[1] = 0;
    }
}

__kernel void cg_pbeta(
    __global float *p, __global const float *z, __global const float *beta, int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    p[idx] = z[idx] + beta[0] * p[idx];
}
"""

_PROJECT_CORRECT_KERNEL = """
// Exact projection — CPU yolu ile birebir:
//   yuzey hizlari yuzey basinç gradyaniyla duzeltilir,
//   hucre merkezleri duzeltilmis yuzeylerin ortalamasi olur.
//   u0/v0/w0 = duzeltme ONCESI hizlar (ayri tampon — race yok).
//   Kati yuzeylerde dp duzeltmesi 0 (impenetrable — CPU m_e/m_w/... maskesi).
//   Duzeltme sonrasi yuzey diverjansi ~Poisson toleransina iner.
__kernel void project_correct(
    __global float *u, __global float *v, __global float *w,
    __global const float *u0, __global const float *v0, __global const float *w0,
    __global const float *p, __global const float *solid,
    float dt, float inv_dx, float inv_dy, float inv_dz,
    int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= nx * ny * nz) return;
    int k = idx % nz; int j = (idx / nz) % ny; int i = idx / (ny * nz);
    if (i < 1 || i >= nx - 1 || j < 1 || j >= ny - 1 || k < 1 || k >= nz - 1) return;

    int c = idx;
    int ce = ((i + 1) * ny + j) * nz + k, cw = ((i - 1) * ny + j) * nz + k;
    int cn = (i * ny + (j + 1)) * nz + k, cs = (i * ny + (j - 1)) * nz + k;
    int ct = (i * ny + j) * nz + (k + 1), cb = (i * ny + j) * nz + (k - 1);

    // Katiya bitisik yuzeylerde NORMAL aki 0 (impenetrable — CPU m_e/m_w/...).
    float u_fe = (solid[ce] > 0.5f || solid[c] > 0.5f) ? 0.0f : 0.5f * (u0[ce] + u0[c]);
    float u_fw = (solid[cw] > 0.5f || solid[c] > 0.5f) ? 0.0f : 0.5f * (u0[c] + u0[cw]);
    float v_fn = (solid[cn] > 0.5f || solid[c] > 0.5f) ? 0.0f : 0.5f * (v0[cn] + v0[c]);
    float v_fs = (solid[cs] > 0.5f || solid[c] > 0.5f) ? 0.0f : 0.5f * (v0[c] + v0[cs]);
    float w_ft = (solid[ct] > 0.5f || solid[c] > 0.5f) ? 0.0f : 0.5f * (w0[ct] + w0[c]);
    float w_fb = (solid[cb] > 0.5f || solid[c] > 0.5f) ? 0.0f : 0.5f * (w0[c] + w0[cb]);

    // 0-dp maskesi (CPU m_e/m_w/... = solid[komşu] OR solid[c])
    float dp_e = (solid[ce] > 0.5f || solid[c] > 0.5f) ? 0.0f : (p[ce] - p[c]) * inv_dx;
    float dp_w = (solid[cw] > 0.5f || solid[c] > 0.5f) ? 0.0f : (p[c] - p[cw]) * inv_dx;
    float dp_n = (solid[cn] > 0.5f || solid[c] > 0.5f) ? 0.0f : (p[cn] - p[c]) * inv_dy;
    float dp_s = (solid[cs] > 0.5f || solid[c] > 0.5f) ? 0.0f : (p[c] - p[cs]) * inv_dy;
    float dp_t = (solid[ct] > 0.5f || solid[c] > 0.5f) ? 0.0f : (p[ct] - p[c]) * inv_dz;
    float dp_b = (solid[cb] > 0.5f || solid[c] > 0.5f) ? 0.0f : (p[c] - p[cb]) * inv_dz;

    u[c] = 0.5f * ((u_fw - dt * dp_w) + (u_fe - dt * dp_e));
    v[c] = 0.5f * ((v_fs - dt * dp_s) + (v_fn - dt * dp_n));
    w[c] = 0.5f * ((w_fb - dt * dp_b) + (w_ft - dt * dp_t));
}
"""

_APPLY_BC_X0_KERNEL = """
// Giris kosulu: u[0,:,:] = v_wind, v[0,:,:] = 0, w[0,:,:] = 0
__kernel void inlet_bc(
    __global float *u, __global float *v, __global float *w,
    float v_wind, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= ny * nz) return;
    u[idx] = v_wind;
    v[idx] = 0.0f;
    w[idx] = 0.0f;
}
"""

_OUTLET_BC_KERNEL = """
// Cikis: zero-gradient hizlar (CPU yoluyla birebir: u[-1]=u[-2] vb. —
// p'ye DOKUNULMAZ, CPU'da outlet p'yi degistirmez)
__kernel void outlet_bc(
    __global float *u, __global float *v, __global float *w,
    int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= ny * nz) return;
    int last = (nx - 1) * ny * nz + idx;
    int prev = (nx - 2) * ny * nz + idx;
    u[last] = u[prev];
    v[last] = v[prev];
    w[last] = w[prev];
}
"""

_VELOCITY_CLAMP_KERNEL = """
// Hiz clamp'i — CPU yolunun 5*v_wind siniriyla birebir:
// |vel| > limit olan hucreler sifirlanir (kararsizlik korumasi).
__kernel void velocity_clamp(
    __global float *u, __global float *v, __global float *w,
    float limit, int nx, int ny, int nz)
{
    int idx = get_global_id(0);
    if (idx >= nx * ny * nz) return;
    if (fabs(u[idx]) > limit) u[idx] = 0.0f;
    if (fabs(v[idx]) > limit) v[idx] = 0.0f;
    if (fabs(w[idx]) > limit) w[idx] = 0.0f;
}
"""

# Spalart-Allmaras turbulans — GPU'da fp64, turbulence_models.solve_spalart_allmaras
# (numpy yolu) ile birebir ayni matematik. u/v/w girisleri float32, tum aritmetik
# double; sabitler (cw1, cw2, cw3_6, cv1_3, sigma_inv) Python tarafindan hesaplanip
# parametre olarak gecilir. adim basina ping-pong (sa_step) + son adimda
# nu_eff = nu_mol + clip(nu_tilde*fv1, 0, 1000) (sa_finalize).
_SA_KERNEL = """
#pragma OPENCL EXTENSION cl_khr_fp64 : enable

// nu_tilde baslangici: solid hucrelerde 0, digerlerinde nu_bnd (=3*nu_mol)
__kernel void sa_init(
    __global double *out, __global const float *solid, double nu_bnd, int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    out[idx] = (solid[idx] > 0.5f) ? 0.0 : nu_bnd;
}

__kernel void sa_step(
    __global double *out, __global const double *nu_in,
    __global const float *u, __global const float *v, __global const float *w,
    __global const double *d_wall, __global const float *solid,
    double dx, double dy, double dz, double dt_sa, double nu_mol,
    double cb1, double cb2, double sigma_inv, double cw1, double cw2,
    double cw3_6, double cv1_3, double nu_max, double nu_bnd,
    int nx, int ny, int nz)
{
    int idx = get_global_id(0);
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
    if (solid[idx] > 0.5f) {
        out[idx] = 0.0;
        return;
    }

    int ce = idx + ox, cw = idx - ox;
    int cn = idx + oy, cs = idx - oy;
    int ct = idx + 1, cb = idx - 1;

    double nu = nu_in[idx];
    double nu_e = nu_in[ce], nu_w = nu_in[cw];
    double nu_n = nu_in[cn], nu_s = nu_in[cs];
    double nu_t = nu_in[ct], nu_b = nu_in[cb];

    double d_nu_dx = (nu_e - nu_w) / (2.0 * dx);
    double d_nu_dy = (nu_n - nu_s) / (2.0 * dy);
    double d_nu_dz = (nu_t - nu_b) / (2.0 * dz);
    double grad2_nu = d_nu_dx * d_nu_dx + d_nu_dy * d_nu_dy + d_nu_dz * d_nu_dz;
    double lap_nu = (nu_e - 2.0 * nu + nu_w) / (dx * dx)
                  + (nu_n - 2.0 * nu + nu_s) / (dy * dy)
                  + (nu_t - 2.0 * nu + nu_b) / (dz * dz);

    double ue = (double)u[idx], ve = (double)v[idx], we = (double)w[idx];
    double du_dx = ((double)u[ce] - (double)u[cw]) / (2.0 * dx);
    double dv_dy = ((double)v[cn] - (double)v[cs]) / (2.0 * dy);
    double dw_dz = ((double)w[ct] - (double)w[cb]) / (2.0 * dz);
    double du_dy = ((double)u[cn] - (double)u[cs]) / (2.0 * dy);
    double dv_dx = ((double)v[ce] - (double)v[cw]) / (2.0 * dx);
    double du_dz = ((double)u[ct] - (double)u[cb]) / (2.0 * dz);
    double dw_dx = ((double)w[ce] - (double)w[cw]) / (2.0 * dx);
    double dv_dz = ((double)v[ct] - (double)v[cb]) / (2.0 * dz);
    double dw_dy = ((double)w[cn] - (double)w[cs]) / (2.0 * dy);

    double S = sqrt(2.0 * (du_dx * du_dx + dv_dy * dv_dy + dw_dz * dw_dz
        + 0.5 * (du_dy + dv_dx) * (du_dy + dv_dx)
        + 0.5 * (du_dz + dw_dx) * (du_dz + dw_dx)
        + 0.5 * (dv_dz + dw_dy) * (dv_dz + dw_dy)) + 1e-12);

    double chi = nu / (nu_mol + 1e-12);
    double chi3 = pow(chi, 3.0);
    double fv1 = chi3 / (chi3 + cv1_3 + 1e-12);
    double r = fmin(nu / ((S * d_wall[idx]) * (S * d_wall[idx]) + 1e-12), 10.0);
    double g = fmin(r + cw2 * pow(r, 6.0), 1e5);
    double f_w = fmin(g * pow((1.0 + cw3_6) / (pow(g, 6.0) + cw3_6 + 1e-12), 1.0 / 6.0), 2.5);

    double prod = cb1 * S * nu;
    double diff = sigma_inv * ((nu + nu_mol) * lap_nu + (1.0 + cb2) * grad2_nu);
    double d_w = d_wall[idx] + 1e-12;
    double destr = -cw1 * f_w * (nu / d_w) * (nu / d_w);

    double adv = (ue > 0.0 ? ue * (nu - nu_w) / dx : ue * (nu_e - nu) / dx)
               + (ve > 0.0 ? ve * (nu - nu_s) / dy : ve * (nu_n - nu) / dy)
               + (we > 0.0 ? we * (nu - nu_b) / dz : we * (nu_t - nu) / dz);

    double nu_new = nu + dt_sa * (-adv + prod + diff + destr);
    out[idx] = fmin(fmax(nu_new, 0.0), nu_max);
}

// nu_eff = nu_mol + clip(nu_tilde*fv1, 0, 1000); solid hucrelerde nu_mol
__kernel void sa_finalize(
    __global float *nu_eff, __global const double *nu_tilde,
    __global const float *solid, double nu_mol, double cv1_3, int n)
{
    int idx = get_global_id(0);
    if (idx >= n) return;
    if (solid[idx] > 0.5f) {
        nu_eff[idx] = (float)nu_mol;
        return;
    }
    double chi = nu_tilde[idx] / (nu_mol + 1e-12);
    double chi3 = pow(chi, 3.0);
    double fv1 = chi3 / (chi3 + cv1_3 + 1e-12);
    double nu_t = fmin(fmax(nu_tilde[idx] * fv1, 0.0), 1000.0);
    nu_eff[idx] = (float)(nu_mol + nu_t);
}
"""


# ═══════════════════════════════════════════════════════════════════════════════
# SOLVER CLASS
# ═══════════════════════════════════════════════════════════════════════════════

# Kernel adi -> program adi (cache icin)
_KERNEL_PROGRAMS = {
    "stencil6": "stencil6",
    "where_fluid": "where",
    "gradient_x": "gradient",
    "gradient_y": "gradient",
    "gradient_z": "gradient",
    "viscous_x": "viscous",
    "viscous_y": "viscous",
    "viscous_z": "viscous",
    "ns_rhs": "ns_rhs",
    "rk2_update": "rk2",
    "rk2_stage1": "rk2",
    "rk2_combine": "rk2",
    "tunnel_wall_bc": "wall_bc",
    "wall_bc_copy_y": "wall_copy",
    "wall_bc_copy_z": "wall_copy",
    "compute_divergence": "divergence",
    "array_copy": "array_copy",
    "mask_inplace": "array_copy",
    "wall_distance_init": "wall_dist",
    "inlet_bc": "inlet_bc",
    "ns_divergence_rhs": "poisson_rhs",
    "poisson_matvec": "poisson_stencil",
    "outlet_bc": "outlet",
    "velocity_clamp": "vel_clamp",
    "project_correct": "project",
    "cg_init": "poisson_stencil",
    "cg_xrz": "poisson_stencil",
    "cg_beta": "poisson_stencil",
    "cg_bgen": "poisson_stencil",
    "cg_warm": "poisson_stencil",
    "cg_final": "poisson_stencil",
    "cg_reduce_bnorm": "poisson_stencil",
    "cg_matvec_dot": "poisson_stencil",
    "cg_init_dot": "poisson_stencil",
    "cg_reduce_alpha": "poisson_stencil",
    "cg_xrz_dot": "poisson_stencil",
    "cg_reduce_beta": "poisson_stencil",
    "cg_pbeta": "poisson_stencil",
    "abs_max3": "absmax",
    "abs_max3_reduce": "absmax",
    "abs_max3_final": "absmax",
    "sa_init": "sa",
    "sa_step": "sa",
    "sa_finalize": "sa",
}


class OpenCLSolver:
    """Intel/AMD GPU icin OpenCL Navier-Stokes cozucusu."""

    def __init__(self, ctx, queue):
        self.ctx = ctx
        self.queue = queue
        self._programs = {}
        self._kernels = {}
        self._cg_workspace = None
        self._max3_ws = None
        self._compile_all()

    def _compile(self, name, source):
        self._programs[name] = cl.Program(self.ctx, source).build()

    def _kernel(self, name):
        """Kernel nesnesini cache'ler — RepeatedKernelRetrieval maliyetinden kacinir."""
        k = self._kernels.get(name)
        if k is None:
            k = cl.Kernel(self._programs[_KERNEL_PROGRAMS[name]], name)
            self._kernels[name] = k
        return k

    def _compile_all(self):
        self._compile("stencil6", _STENCIL6_KERNEL)
        self._compile("where", _WHERE_KERNEL)
        self._compile("gradient", _GRADIENT_KERNEL)
        self._compile("viscous", _VISCOSITY_KERNEL)
        self._compile("ns_rhs", _NS_RHS_KERNEL)
        self._compile("rk2", _RK2_UPDATE_KERNEL)
        self._compile("wall_bc", _TUNNEL_WALL_BC_KERNEL)
        self._compile("wall_copy", _WALL_BC_COPY_KERNEL)
        self._compile("divergence", _DIVERGENCE_KERNEL)
        self._compile("array_copy", _COPY_KERNEL)
        self._compile("wall_dist", _WALL_DISTANCE_KERNEL)
        self._compile("inlet_bc", _APPLY_BC_X0_KERNEL)
        self._compile("poisson_rhs", _NS_DIVERGENCE_RHS_KERNEL)
        self._compile("poisson_stencil", _POISSON_STENCIL_KERNEL)
        self._compile("outlet", _OUTLET_BC_KERNEL)
        self._compile("vel_clamp", _VELOCITY_CLAMP_KERNEL)
        self._compile("project", _PROJECT_CORRECT_KERNEL)
        self._compile("absmax", _ABS_MAX_KERNEL)
        self._compile("sa", _SA_KERNEL)
        print(f"[OpenCL] {len(self._programs)} kernel derlendi.")

    def _inner_n(self, nx, ny, nz):
        return (nx - 2) * (ny - 2) * (nz - 2)

    def _flat_n(self, nx, ny, nz):
        return nx * ny * nz

    # ── Array kopyalama ────────────────────────────────────────────────────

    def array_copy(self, dst, src, n):
        self._kernel("array_copy")(self.queue, (n,), None, dst.data, src.data, np.int32(n))

    def mask_inplace(self, a_cl, inv_mask_cl, n):
        """a[:] *= inv_mask[:] — katı hucreri sifirla (alloc yok)."""
        self._kernel("mask_inplace")(self.queue, (n,), None, a_cl.data, inv_mask_cl.data, np.int32(n))

    def to_device(self, arr):
        if isinstance(arr, cl_array.Array):
            return arr
        arr = np.asarray(arr, dtype=np.float32)
        return cl_array.to_device(self.queue, arr)

    def to_numpy(self, arr):
        if isinstance(arr, cl_array.Array):
            return np.asarray(arr.get(), dtype=np.float64)
        return np.asarray(arr, dtype=np.float64)

    # ── Potansiyel akis: Jacobi ────────────────────────────────────────────

    def potential_flow_jacobi(self, phi_cl, fluid_cl, nx, ny, nz):
        """Tek Jacobi adimi: phi_new = stencil6(phi_old), sadece fluid hucrelerde."""
        n = self._flat_n(nx, ny, nz)
        update = cl_array.empty(self.queue, (n,), dtype=np.float32)
        phi_new = cl_array.empty(self.queue, (n,), dtype=np.float32)
        self._kernel("stencil6")(
            self.queue, (n,), None, update.data, phi_cl.data, np.int32(nx), np.int32(ny), np.int32(nz)
        )
        self._kernel("where_fluid")(
            self.queue,
            (n,),
            None,
            phi_new.data,
            update.data,
            phi_cl.data,
            fluid_cl.data,
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )
        return phi_new

    # ── Gradient ───────────────────────────────────────────────────────────

    def gradient_x(self, phi_cl, inv_2dx, nx, ny, nz):
        out = cl_array.zeros(self.queue, (self._flat_n(nx, ny, nz),), dtype=np.float32)
        self._kernel("gradient_x")(
            self.queue,
            (self._flat_n(nx, ny, nz),),
            None,
            out.data,
            phi_cl.data,
            np.float32(inv_2dx),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )
        return out

    def gradient_y(self, phi_cl, inv_2dy, nx, ny, nz):
        out = cl_array.zeros(self.queue, (self._flat_n(nx, ny, nz),), dtype=np.float32)
        self._kernel("gradient_y")(
            self.queue,
            (self._flat_n(nx, ny, nz),),
            None,
            out.data,
            phi_cl.data,
            np.float32(inv_2dy),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )
        return out

    def gradient_z(self, phi_cl, inv_2dz, nx, ny, nz):
        out = cl_array.zeros(self.queue, (self._flat_n(nx, ny, nz),), dtype=np.float32)
        self._kernel("gradient_z")(
            self.queue,
            (self._flat_n(nx, ny, nz),),
            None,
            out.data,
            phi_cl.data,
            np.float32(inv_2dz),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )
        return out

    # ── Viskozite ──────────────────────────────────────────────────────────

    def viscous_x(self, f_cl, nu_cl, inv_dx2, nx, ny, nz):
        n = self._inner_n(nx, ny, nz)
        out = cl_array.empty(self.queue, (n,), dtype=np.float32)
        self._kernel("viscous_x")(
            self.queue,
            (n,),
            None,
            out.data,
            f_cl.data,
            nu_cl.data,
            np.float32(inv_dx2),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )
        return out

    def viscous_y(self, f_cl, nu_cl, inv_dy2, nx, ny, nz):
        n = self._inner_n(nx, ny, nz)
        out = cl_array.empty(self.queue, (n,), dtype=np.float32)
        self._kernel("viscous_y")(
            self.queue,
            (n,),
            None,
            out.data,
            f_cl.data,
            nu_cl.data,
            np.float32(inv_dy2),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )
        return out

    def viscous_z(self, f_cl, nu_cl, inv_dz2, nx, ny, nz):
        n = self._inner_n(nx, ny, nz)
        out = cl_array.empty(self.queue, (n,), dtype=np.float32)
        self._kernel("viscous_z")(
            self.queue,
            (n,),
            None,
            out.data,
            f_cl.data,
            nu_cl.data,
            np.float32(inv_dz2),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )
        return out

    # ── NS_rhs: adv + visc tek kernel'da ──────────────────────────────────

    def ns_rhs(
        self,
        u_cl,
        v_cl,
        w_cl,
        nu_cl,
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
        out=None,
    ):
        """NS sag taraf. out verilirse alloc YAPILMAZ (dongu ici buffer'lar).
        Kernel tum hucreri yazar (sinirlarda 0) — empty guvenli."""
        n = self._flat_n(nx, ny, nz)
        if out is None:
            rhs_u = cl_array.empty(self.queue, (n,), dtype=np.float32)
            rhs_v = cl_array.empty(self.queue, (n,), dtype=np.float32)
            rhs_w = cl_array.empty(self.queue, (n,), dtype=np.float32)
        else:
            rhs_u, rhs_v, rhs_w = out
        self._kernel("ns_rhs")(
            self.queue,
            (n,),
            None,
            rhs_u.data,
            rhs_v.data,
            rhs_w.data,
            u_cl.data,
            v_cl.data,
            w_cl.data,
            nu_cl.data,
            vol_cl.data,
            np.float32(inv_dx),
            np.float32(inv_dy),
            np.float32(inv_dz),
            np.float32(inv_dx2),
            np.float32(inv_dy2),
            np.float32(inv_dz2),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )
        return rhs_u, rhs_v, rhs_w

    # ── RK2 guncelleme ────────────────────────────────────────────────────

    def rk2_update(self, u_out, u_old, rhs1, rhs2, half_dt, n):
        self._kernel("rk2_update")(
            self.queue,
            (n,),
            None,
            u_out.data,
            u_old.data,
            rhs1.data,
            rhs2.data,
            np.float32(half_dt),
            np.int32(1),
            np.int32(1),
            np.int32(n),
        )

    def rk2_stage1(self, u_out, u_old, rhs, inv_mask, dt, n):
        """u_out = (u_old + dt*rhs) * inv_mask — tek kernel."""
        self._kernel("rk2_stage1")(
            self.queue,
            (n,),
            None,
            u_out.data,
            u_old.data,
            rhs.data,
            inv_mask.data,
            np.float32(dt),
            np.int32(n),
        )

    def rk2_combine(self, u_out, u_old, rhs1, rhs2, vol, inv_mask, half_dt, n):
        """u_out = (u_old + half_dt*(rhs1+rhs2)) * vol * inv_mask — tek kernel."""
        self._kernel("rk2_combine")(
            self.queue,
            (n,),
            None,
            u_out.data,
            u_old.data,
            rhs1.data,
            rhs2.data,
            vol.data,
            inv_mask.data,
            np.float32(half_dt),
            np.int32(n),
        )

    # ── Tünel duvar sarti ──────────────────────────────────────────────────

    def apply_tunnel_wall_bc(self, u_cl, v_cl, w_cl, nx, ny, nz):
        n = self._flat_n(nx, ny, nz)
        self._kernel("tunnel_wall_bc")(
            self.queue, (n,), None, u_cl.data, v_cl.data, w_cl.data, np.int32(nx), np.int32(ny), np.int32(nz)
        )
        # Slip copy — Y duvarlari: u VE w (v=0 tunnel kernel'de) — 1 enqueue
        self._kernel("wall_bc_copy_y")(
            self.queue, (n,), None, u_cl.data, w_cl.data, np.int32(nx), np.int32(ny), np.int32(nz)
        )
        # Slip copy — Z duvarlari: u VE v (w=0 tunnel kernel'de; w kopyasi YOK)
        self._kernel("wall_bc_copy_z")(
            self.queue, (n,), None, u_cl.data, v_cl.data, np.int32(nx), np.int32(ny), np.int32(nz)
        )

    # ── Giris kosulu ───────────────────────────────────────────────────────

    def apply_inlet_bc(self, u_cl, v_cl, w_cl, v_wind, ny, nz):
        n = ny * nz
        self._kernel("inlet_bc")(
            self.queue,
            (n,),
            None,
            u_cl.data,
            v_cl.data,
            w_cl.data,
            np.float32(v_wind),
            np.int32(ny),
            np.int32(nz),
        )

    # ── Cikis kosulu (kernel — host transferi yok) ────────────────────────

    def apply_outlet_bc(self, u_cl, v_cl, w_cl, p_cl, nx, ny, nz):
        n = ny * nz
        self._kernel("outlet_bc")(
            self.queue,
            (n,),
            None,
            u_cl.data,
            v_cl.data,
            w_cl.data,
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )
        return u_cl, v_cl, w_cl, p_cl

    def clamp_velocity(self, u_cl, v_cl, w_cl, limit, nx, ny, nz):
        """|vel| > limit hucreleri sifirlar (CPU yolundaki 5*v_wind korumasi)."""
        n = self._flat_n(nx, ny, nz)
        self._kernel("velocity_clamp")(
            self.queue,
            (n,),
            None,
            u_cl.data,
            v_cl.data,
            w_cl.data,
            np.float32(limit),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )
        return u_cl, v_cl, w_cl

    def project_correct(
        self, u_cl, v_cl, w_cl, p_cl, solid_cl, dt, inv_dx, inv_dy, inv_dz, nx, ny, nz, tmp=None
    ):
        """Exact projection: yuzey hizlarini yuzey basinç gradyaniyla duzelt."""
        n = self._flat_n(nx, ny, nz)
        if tmp is None:
            tmp_u = cl_array.empty_like(u_cl)
            tmp_v = cl_array.empty_like(v_cl)
            tmp_w = cl_array.empty_like(w_cl)
        else:
            tmp_u, tmp_v, tmp_w = tmp
        self.array_copy(tmp_u, u_cl, n)
        self.array_copy(tmp_v, v_cl, n)
        self.array_copy(tmp_w, w_cl, n)
        self._kernel("project_correct")(
            self.queue,
            (n,),
            None,
            u_cl.data,
            v_cl.data,
            w_cl.data,
            tmp_u.data,
            tmp_v.data,
            tmp_w.data,
            p_cl.data,
            solid_cl.data,
            np.float32(dt),
            np.float32(inv_dx),
            np.float32(inv_dy),
            np.float32(inv_dz),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )
        return u_cl, v_cl, w_cl

    # ── Poisson sag tarafi (GPU tek geciste) ───────────────────────────────

    def abs_max3(self, a_cl, b_cl, c_cl, out1, out2, out3, n):
        """Uc dizinin |max|'ini paralel reduce ile alir (dt hesabi icin).
        out1/out2/out3: host tarafinda degistirilebilen float objeler.
        Tamponlar cache'lenir — her adimda alloc yok; tek (3,) get."""
        ws = self._max3_ws
        if ws is None:
            nwg = (n + _ABSWGS - 1) // _ABSWGS
            partial = cl_array.empty(self.queue, (nwg * 3,), dtype=np.float32)
            out = cl_array.empty(self.queue, (3,), dtype=np.float32)
            self._max3_ws = (nwg, partial, out)
            nwg, partial, out = ws = self._max3_ws
        else:
            nwg, partial, out = ws
        self._kernel("abs_max3_reduce")(
            self.queue, (n,), (_ABSWGS,), a_cl.data, b_cl.data, c_cl.data, partial.data, np.int32(n)
        )
        self._kernel("abs_max3_final")(self.queue, (1,), None, out.data, partial.data, np.int32(nwg))
        self.queue.finish()
        vals = out.get()
        return max(float(vals[0]), 1e-3), max(float(vals[1]), 1e-3), max(float(vals[2]), 1e-3)

    def divergence_rhs(self, u_cl, v_cl, w_cl, rhs_cl, solid_cl, inv_dx, inv_dy, inv_dz, inv_dt, nx, ny, nz):
        n = self._flat_n(nx, ny, nz)
        self._kernel("ns_divergence_rhs")(
            self.queue,
            (n,),
            None,
            rhs_cl.data,
            u_cl.data,
            v_cl.data,
            w_cl.data,
            solid_cl.data,
            np.float32(inv_dx),
            np.float32(inv_dy),
            np.float32(inv_dz),
            np.float32(inv_dt),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )

    # ── Poisson matvec (GPU stencil) ───────────────────────────────────────

    def poisson_matvec(self, out_cl, p_cl, solid_cl, ax, ay, az, nx, ny, nz):
        n = self._flat_n(nx, ny, nz)
        self._kernel("poisson_matvec")(
            self.queue,
            (n,),
            None,
            out_cl.data,
            p_cl.data,
            solid_cl.data,
            np.float32(ax),
            np.float32(ay),
            np.float32(az),
            np.int32(nx),
            np.int32(ny),
            np.int32(nz),
        )

    # ── Konjuge gradyan — Jacobi onkosullayicili, GPU stencil matvec ──────
    # CPU scipy_cg ile ayni matris/tolerans; sadece matvec GPU'da.
    # Basarisizlikta (fp32 siniri) info != 0 doner — cagiran host'a duser.

    def poisson_cg(
        self,
        rhs_cl,
        x_init_cl,
        solid_cl,
        inv_diag_cl,
        inv_mask_cl,
        ax,
        ay,
        az,
        nx,
        ny,
        nz,
        tol=1e-4,
        maxiter=800,
    ):
        n = self._flat_n(nx, ny, nz)
        q = self.queue
        # A_p = -A_cpu oldugundan b_p = -b (cozum ayni)
        b = -(rhs_cl * inv_mask_cl)
        x = x_init_cl * inv_mask_cl
        Ap = cl_array.empty(q, (n,), dtype=np.float32)
        r = cl_array.empty(q, (n,), dtype=np.float32)
        z = cl_array.empty(q, (n,), dtype=np.float32)
        p = cl_array.empty(q, (n,), dtype=np.float32)

        b_norm = float(np.sqrt(float(cl_array.dot(b, b).get())))
        if b_norm < 1e-30:
            x.fill(0.0)
            return x, 0, 0

        self.poisson_matvec(Ap, x, solid_cl, ax, ay, az, nx, ny, nz)
        self._kernel("cg_init")(
            q, (n,), None, r.data, z.data, p.data, b.data, Ap.data, inv_diag_cl.data, np.int32(n)
        )
        rz = float(cl_array.dot(r, z).get())
        if rz < 0.0:
            rz = 0.0

        iters = 0
        r_norm = float("inf")
        for _ in range(maxiter):
            iters += 1
            self.poisson_matvec(Ap, p, solid_cl, ax, ay, az, nx, ny, nz)
            pAp = float(cl_array.dot(p, Ap).get())
            if pAp <= 0.0:
                break
            alpha = rz / pAp
            self._kernel("cg_xrz")(
                q,
                (n,),
                None,
                x.data,
                r.data,
                z.data,
                p.data,
                Ap.data,
                inv_diag_cl.data,
                np.float32(alpha),
                np.int32(n),
            )
            rz_new = float(cl_array.dot(r, z).get())
            if rz_new < 0.0:
                rz_new = 0.0
            beta = rz_new / (rz + 1e-30)
            self._kernel("cg_beta")(q, (n,), None, p.data, z.data, np.float32(beta), np.int32(n))
            rz = rz_new
            # Konverjans kontrolü her 4 iterasyonda bir — host sync'i 3x azaltir;
            # geciken durma doğruluğu bozmaz (tol zaten saglanmis olur).
            if iters % 4 == 0:
                r_norm = float(np.sqrt(float(cl_array.dot(r, r).get())))
                if r_norm <= tol * b_norm:
                    break

        x *= inv_mask_cl
        return x, (0 if r_norm <= tol * b_norm else 1), iters

    # ── Cihaz-yerleşik CG — iterasyon başına host sync YOK ────────────────
    # Ayni matris/tolerans/kriter (r_norm <= tol*b_norm); yalniz skalerler ve
    # konverjans bayragi cihazda tutulur. Host ~16 iterasyonda bir status'u
    # okur; pAp<=0 hali (fail) CPU fallback'ini tetikler.

    def _cg_ws_build(self, q, key, n, nwg, wgs, ax, ay, az, nx, ny, nz):
        """Cihaz CG workspace'i — buffer'lar ve kernel arg'lari BIR KEZ kurulur;
        iterasyon dongusunde sadece enqueue cagrisi kalir (pyopencl arg parse
        ve allocator overhead'i devre disi)."""
        f32, i32 = np.float32, np.int32

        def mk(size, dt=np.float32):
            return cl_array.empty(q, (size,), dtype=dt)

        ws = {"key": key}
        ws["b"] = mk(n)
        ws["x"] = mk(n)
        ws["Ap"] = mk(n)
        ws["r"] = mk(n)
        ws["z"] = mk(n)
        ws["p"] = mk(n)
        ws["part_pAp"] = mk(nwg)
        ws["part_rz"] = mk(nwg)
        ws["part_rr"] = mk(nwg)
        ws["part_bb"] = mk(nwg)
        ws["alpha"] = mk(1)
        ws["beta"] = mk(1)
        ws["rz"] = mk(1)
        ws["bnorm"] = mk(1)
        ws["status"] = mk(2, np.int32)
        b, x, Ap, r, z, p = ws["b"], ws["x"], ws["Ap"], ws["r"], ws["z"], ws["p"]
        part_pAp, part_rz, part_rr, part_bb = ws["part_pAp"], ws["part_rz"], ws["part_rr"], ws["part_bb"]
        alpha, beta, rz, bnorm, status = ws["alpha"], ws["beta"], ws["rz"], ws["bnorm"], ws["status"]

        def kk(name, dtypes):
            kobj = self._kernel(name)
            kobj.set_scalar_arg_dtypes(dtypes)
            return kobj

        k_bgen = ws["k_bgen"] = kk("cg_bgen", [None, None, None, None, i32])
        ws["k_warm"] = kk("cg_warm", [None, None, None, None, i32])
        ws["k_final"] = kk("cg_final", [None, None, i32])
        k_bnorm = ws["k_bnorm"] = kk("cg_reduce_bnorm", [None, None, i32])
        k_mdot = ws["k_mdot"] = kk(
            "cg_matvec_dot", [None, None, None, f32, f32, f32, i32, i32, i32, None, i32]
        )
        k_init_dot = ws["k_init_dot"] = kk(
            "cg_init_dot", [None, None, None, None, None, None, None, None, i32]
        )
        k_ralpha = ws["k_ralpha"] = kk("cg_reduce_alpha", [None, None, None, None, i32])
        k_xrz_dot = ws["k_xrz_dot"] = kk(
            "cg_xrz_dot", [None, None, None, None, None, None, None, None, None, i32]
        )
        k_rbeta = ws["k_rbeta"] = kk("cg_reduce_beta", [None, None, None, None, None, f32, i32])
        k_pbeta = ws["k_pbeta"] = kk("cg_pbeta", [None, None, None, i32])

        # Degismeyen arg'lar — bir kez set edilir.
        k_bgen.set_arg(0, b.data)
        k_bgen.set_arg(3, part_bb.data)
        k_bgen.set_arg(4, i32(n))
        ws["k_warm"].set_arg(0, x.data)
        ws["k_warm"].set_arg(1, p.data)
        ws["k_final"].set_arg(0, x.data)
        k_bnorm.set_arg(0, bnorm.data)
        k_bnorm.set_arg(1, part_bb.data)
        k_bnorm.set_arg(2, i32(nwg))
        k_mdot.set_arg(0, Ap.data)
        k_mdot.set_arg(1, p.data)
        k_mdot.set_arg(3, f32(ax))
        k_mdot.set_arg(4, f32(ay))
        k_mdot.set_arg(5, f32(az))
        k_mdot.set_arg(6, i32(nx))
        k_mdot.set_arg(7, i32(ny))
        k_mdot.set_arg(8, i32(nz))
        k_mdot.set_arg(9, part_pAp.data)
        k_mdot.set_arg(10, i32(n))
        k_init_dot.set_arg(0, r.data)
        k_init_dot.set_arg(1, z.data)
        k_init_dot.set_arg(2, p.data)
        k_init_dot.set_arg(3, b.data)
        k_init_dot.set_arg(4, Ap.data)
        k_init_dot.set_arg(6, part_rz.data)
        k_init_dot.set_arg(7, part_rr.data)
        k_init_dot.set_arg(8, i32(n))
        k_ralpha.set_arg(0, alpha.data)
        k_ralpha.set_arg(1, rz.data)
        k_ralpha.set_arg(2, status.data)
        k_ralpha.set_arg(3, part_pAp.data)
        k_ralpha.set_arg(4, i32(nwg))
        k_xrz_dot.set_arg(0, x.data)
        k_xrz_dot.set_arg(1, r.data)
        k_xrz_dot.set_arg(2, z.data)
        k_xrz_dot.set_arg(3, p.data)
        k_xrz_dot.set_arg(4, Ap.data)
        k_xrz_dot.set_arg(6, alpha.data)
        k_xrz_dot.set_arg(7, part_rz.data)
        k_xrz_dot.set_arg(8, part_rr.data)
        k_xrz_dot.set_arg(9, i32(n))
        k_rbeta.set_arg(0, beta.data)
        k_rbeta.set_arg(1, rz.data)
        k_rbeta.set_arg(2, status.data)
        k_rbeta.set_arg(3, part_rz.data)
        k_rbeta.set_arg(4, part_rr.data)
        k_rbeta.set_arg(6, i32(nwg))
        k_pbeta.set_arg(0, p.data)
        k_pbeta.set_arg(1, z.data)
        k_pbeta.set_arg(2, beta.data)
        k_pbeta.set_arg(3, i32(n))
        return ws

    def poisson_cg_device(
        self,
        rhs_cl,
        x_init_cl,
        solid_cl,
        inv_diag_cl,
        inv_mask_cl,
        ax,
        ay,
        az,
        nx,
        ny,
        nz,
        tol=1e-4,
        maxiter=800,
    ):
        n = self._flat_n(nx, ny, nz)
        q = self.queue
        wgs = 128
        nwg = (n + wgs - 1) // wgs

        ws = self._cg_workspace
        key = (n, wgs, ax, ay, az, nx, ny, nz)
        if ws is None or ws["key"] != key:
            ws = self._cg_ws_build(q, key, n, nwg, wgs, ax, ay, az, nx, ny, nz)
            self._cg_workspace = ws
        x = ws["x"]
        bnorm = ws["bnorm"]
        status = ws["status"]
        k_bgen, k_bnorm, k_mdot = ws["k_bgen"], ws["k_bnorm"], ws["k_mdot"]
        k_warm, k_final = ws["k_warm"], ws["k_final"]
        k_init_dot, k_ralpha, k_xrz_dot, k_rbeta, k_pbeta = (
            ws["k_init_dot"],
            ws["k_ralpha"],
            ws["k_xrz_dot"],
            ws["k_rbeta"],
            ws["k_pbeta"],
        )
        f32 = np.float32

        # Cagri basina degisen arg'lar.
        k_bgen.set_arg(1, rhs_cl.data)
        k_bgen.set_arg(2, inv_mask_cl.data)
        k_warm.set_arg(2, x_init_cl.data)
        k_warm.set_arg(3, inv_mask_cl.data)
        k_final.set_arg(1, inv_mask_cl.data)
        k_mdot.set_arg(2, solid_cl.data)
        k_init_dot.set_arg(5, inv_diag_cl.data)
        k_xrz_dot.set_arg(5, inv_diag_cl.data)

        k_warm.set_arg(4, np.int32(n))
        k_final.set_arg(2, np.int32(n))
        cl.enqueue_nd_range_kernel(q, k_warm, (n,), (wgs,))
        cl.enqueue_nd_range_kernel(q, k_bgen, (n,), (wgs,))
        cl.enqueue_nd_range_kernel(q, k_bnorm, (wgs,), (wgs,))
        b_norm = float(bnorm.get()[0])
        if b_norm < 1e-30:
            x.fill(0.0)
            return x, 0, 0

        cl.enqueue_nd_range_kernel(q, k_mdot, (n,), (wgs,))
        cl.enqueue_nd_range_kernel(q, k_init_dot, (n,), (wgs,))
        k_rbeta.set_arg(5, f32(tol * tol * b_norm * b_norm))
        cl.enqueue_nd_range_kernel(q, k_rbeta, (wgs,), (wgs,))

        iters = 0
        status_np = status.get()
        if int(status_np[0]) == 0 and int(status_np[1]) == 0:
            for _ in range(maxiter):
                iters += 1
                cl.enqueue_nd_range_kernel(q, k_mdot, (n,), (wgs,))
                cl.enqueue_nd_range_kernel(q, k_ralpha, (wgs,), (wgs,))
                cl.enqueue_nd_range_kernel(q, k_xrz_dot, (n,), (wgs,))
                cl.enqueue_nd_range_kernel(q, k_rbeta, (wgs,), (wgs,))
                cl.enqueue_nd_range_kernel(q, k_pbeta, (n,), (wgs,))
                if iters % 16 == 0:
                    status_np = status.get()
                    if int(status_np[0]) != 0 or int(status_np[1]) != 0:
                        break

        cl.enqueue_nd_range_kernel(q, k_final, (n,), (wgs,))
        info = 1 if int(status_np[1]) != 0 else (0 if int(status_np[0]) != 0 else 1)
        return x, info, iters

    def sa_solve(self, u_cl, v_cl, w_cl, solid_cl, d_wall_cl, dx, dy, dz, nu_mol, iterations, nx, ny, nz):
        """Spalart-Allmaras turbulans viskozitesi — GPU'da fp64.

        turbulence_models.solve_spalart_allmaras (numpy yolu) ile birebir
        matematik; nu_eff = nu_mol + clip(nu_tilde*fv1, 0, 1000) dondurur
        (float32, 1D ravel, solid hucrelerde nu_mol). Iterasyon sayisi CPU
        yoluyla ayni min_spacing kuralini uygular.
        """
        from constants import SA_CB1, SA_CB2, SA_CV1, SA_CW1, SA_CW2, SA_CW3, SA_SIGMA

        q = self.queue
        n = nx * ny * nz
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
        dt_sa = min(dx, dy, dz) ** 2 / max(6.0 * nu_mol, 1e-6)
        dt_sa = min(dt_sa, 1e-3)

        nu_in = cl_array.empty(q, (n,), dtype=np.float64)
        nu_out = cl_array.empty(q, (n,), dtype=np.float64)
        nu_eff = cl_array.empty(q, (n,), dtype=np.float32)

        k_init = self._kernel("sa_init")
        k_step = self._kernel("sa_step")
        k_fin = self._kernel("sa_finalize")

        k_init.set_scalar_arg_dtypes([None, None, np.float64, np.int32])
        k_init.set_arg(0, nu_in.data)
        k_init.set_arg(1, solid_cl.data)
        k_init.set_arg(2, np.float64(nu_bnd))
        k_init.set_arg(3, np.int32(n))
        cl.enqueue_nd_range_kernel(q, k_init, (n,), None)

        f64 = np.float64
        k_step.set_scalar_arg_dtypes([None] * 7 + [f64] * 14 + [np.int32] * 3)
        k_step.set_arg(2, u_cl.data)
        k_step.set_arg(3, v_cl.data)
        k_step.set_arg(4, w_cl.data)
        k_step.set_arg(5, d_wall_cl.data)
        k_step.set_arg(6, solid_cl.data)
        k_step.set_arg(7, f64(dx))
        k_step.set_arg(8, f64(dy))
        k_step.set_arg(9, f64(dz))
        k_step.set_arg(10, f64(dt_sa))
        k_step.set_arg(11, f64(nu_mol))
        k_step.set_arg(12, f64(SA_CB1))
        k_step.set_arg(13, f64(SA_CB2))
        k_step.set_arg(14, f64(1.0 / SA_SIGMA))
        k_step.set_arg(15, f64(SA_CW1))
        k_step.set_arg(16, f64(SA_CW2))
        k_step.set_arg(17, f64(SA_CW3**6))
        k_step.set_arg(18, f64(SA_CV1**3))
        k_step.set_arg(19, f64(nu_max))
        k_step.set_arg(20, f64(nu_bnd))
        k_step.set_arg(21, np.int32(nx))
        k_step.set_arg(22, np.int32(ny))
        k_step.set_arg(23, np.int32(nz))

        for _ in range(iterations):
            k_step.set_arg(0, nu_out.data)
            k_step.set_arg(1, nu_in.data)
            cl.enqueue_nd_range_kernel(q, k_step, (n,), None)
            nu_in, nu_out = nu_out, nu_in

        k_fin.set_scalar_arg_dtypes([None, None, None, f64, f64, np.int32])
        k_fin.set_arg(0, nu_eff.data)
        k_fin.set_arg(1, nu_in.data)
        k_fin.set_arg(2, solid_cl.data)
        k_fin.set_arg(3, f64(nu_mol))
        k_fin.set_arg(4, f64(SA_CV1**3))
        k_fin.set_arg(5, np.int32(n))
        cl.enqueue_nd_range_kernel(q, k_fin, (n,), None)
        return nu_eff
