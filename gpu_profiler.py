import os
import time

import argparse

ap = argparse.ArgumentParser()
ap.add_argument("--device", default="auto")
args, _ = ap.parse_known_args()

os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["AV_OPENCL_ALLOW_CPU"] = "1"

import numpy as np
import gpu_backend
import cl_backend
import cl_solver
import pyopencl.array as cl_array

from validation.sphere_stl import sphere_stl
from solver import create_voxel_mask, solve_potential_flow, POISSON_TOL

nx, ny, nz = 40, 20, 20
dx = dy = dz = 0.05
inv_dx, inv_dy, inv_dz = 1.0 / dx, 1.0 / dy, 1.0 / dz
inv_dx2, inv_dy2, inv_dz2 = inv_dx * inv_dx, inv_dy * inv_dy, inv_dz * inv_dz
n_total = nx * ny * nz
nu_mol = 1.5e-5

verts, i_idx, j_idx, k_idx = sphere_stl(0.25, (-1.0, 0.0, 0.0), n_lat=24, n_lon=32)
mask, solid_fraction, ok = create_voxel_mask(verts, i_idx, j_idx, k_idx, nx, ny, nz, (4.0, 2.0, 2.0))
u_np, v_np, w_np = solve_potential_flow(
    mask, 10.0, nx, ny, nz, solid_fraction=solid_fraction, tunnel_dims=(4.0, 2.0, 2.0)
)
u_np = np.asarray(u_np).copy()
v_np = np.asarray(v_np).copy()
w_np = np.asarray(w_np).copy()
u_np[mask] = 0.0
v_np[mask] = 0.0
w_np[mask] = 0.0

mask_eff = np.asarray(mask) | (np.asarray(solid_fraction) >= 1.0)
vol_np = (1.0 - np.asarray(solid_fraction)).clip(0.0, 1.0).astype(np.float32).ravel()

gpu_backend.detect_gpu(args.device)
slvr = cl_solver.OpenCLSolver(cl_backend._ctx, cl_backend._queue)
q = slvr.queue

u_cl = slvr.to_device(u_np.astype(np.float32).ravel())
v_cl = slvr.to_device(v_np.astype(np.float32).ravel())
w_cl = slvr.to_device(w_np.astype(np.float32).ravel())
mask_cl = slvr.to_device(mask_eff.astype(np.float32).ravel())
inv_mask_cl = slvr.to_device((1.0 - mask_eff.astype(np.float32).ravel()))
vol_cl = slvr.to_device(vol_np)
nu_eff_cl = slvr.to_device(np.full(n_total, float(nu_mol), dtype=np.float32))
rhs_cl = cl_array.zeros(q, (n_total,), dtype=np.float32)

diag = 2.0 * (inv_dx2 + inv_dy2 + inv_dz2) * np.ones(n_total, dtype=np.float64)
ox, oy, oz = ny * nz, nz, 1
nb_offsets = [ox, -ox, oy, -oy, oz, -oz]
nb_coefs = [inv_dx2, inv_dx2, inv_dy2, inv_dy2, inv_dz2, inv_dz2]
i3 = np.broadcast_to(np.arange(nx)[:, None, None], (nx, ny, nz)).ravel()
j3 = np.broadcast_to(np.arange(ny)[None, :, None], (nx, ny, nz)).ravel()
k3 = np.broadcast_to(np.arange(nz)[None, None, :], (nx, ny, nz)).ravel()
guards = np.stack([i3 < nx - 1, i3 > 0, j3 < ny - 1, j3 > 0, k3 < nz - 1, k3 > 0], axis=1)
for k in range(6):
    nb = np.clip(np.arange(n_total) + nb_offsets[k], 0, n_total - 1)
    diag -= nb_coefs[k] * (guards[:, k] & mask_eff.ravel()[nb])
diag[mask_eff.ravel()] = 1.0
inv_diag_cl = slvr.to_device((1.0 / diag).astype(np.float32).ravel())
p_init_cl = slvr.to_device(np.zeros(n_total, dtype=np.float32))

dt_f = np.float32(2.66e-4)
half_dt_f = np.float32(0.5) * dt_f
NREP = 20


def bench(name, fn, n=NREP):
    fn()
    q.finish()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    q.finish()
    dt = (time.perf_counter() - t0) / n
    print(f"{name:35s} {dt * 1000:10.3f} ms")
    return dt


ns_rhs_ms = bench(
    "ns_rhs (2x stage)",
    lambda: slvr.ns_rhs(
        u_cl, v_cl, w_cl, nu_eff_cl, vol_cl, inv_dx, inv_dy, inv_dz, inv_dx2, inv_dy2, inv_dz2, nx, ny, nz
    ),
)
divergence_ms = bench(
    "divergence_rhs",
    lambda: slvr.divergence_rhs(
        u_cl, v_cl, w_cl, rhs_cl, mask_cl, inv_dx, inv_dy, inv_dz, 1.0 / dt_f, nx, ny, nz
    ),
)
p_out = p_init_cl.copy()
poisson_ms = bench(
    "poisson_cg_device (tol 1e-6)",
    lambda: slvr.poisson_cg_device(
        rhs_cl,
        p_out,
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
    ),
)
bench(
    "poisson_cg_host (eski, karsilastirma)",
    lambda: slvr.poisson_cg(
        rhs_cl,
        p_out,
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
    ),
)
project_ms = bench(
    "project_correct",
    lambda: slvr.project_correct(u_cl, v_cl, w_cl, p_out, mask_cl, dt_f, inv_dx, inv_dy, inv_dz, nx, ny, nz),
)
tunnel_ms = bench("apply_tunnel_wall_bc", lambda: slvr.apply_tunnel_wall_bc(u_cl, v_cl, w_cl, nx, ny, nz))
inlet_ms = bench("apply_inlet_bc", lambda: slvr.apply_inlet_bc(u_cl, v_cl, w_cl, 10.0, ny, nz))
outlet_ms = bench("apply_outlet_bc", lambda: slvr.apply_outlet_bc(u_cl, v_cl, w_cl, p_out, nx, ny, nz))
absmax3_ms = bench("dt abs_max3 (x3)", lambda: slvr.abs_max3(u_cl, v_cl, w_cl, 0.0, 0.0, 0.0, n_total))
bench(
    "max/min x6 (eski, karsilastirma)",
    lambda: (
        max(cl_array.max(u_cl).get(), -cl_array.min(u_cl).get()),
        max(cl_array.max(v_cl).get(), -cl_array.min(v_cl).get()),
        max(cl_array.max(w_cl).get(), -cl_array.min(w_cl).get()),
    ),
)
bench("to_numpy u (turbulans)", lambda: slvr.to_numpy(u_cl))

from solver import _build_wall_distance_cache  # noqa: E402

d_wall_np = _build_wall_distance_cache(mask_eff, dx, dy, dz, np)
d_wall_cl = cl_array.to_device(q, d_wall_np.astype(np.float64).ravel())
sa_ms = bench(
    "sa_solve (80 iter, fp64)",
    lambda: slvr.sa_solve(u_cl, v_cl, w_cl, mask_cl, d_wall_cl, dx, dy, dz, nu_mol, 60, nx, ny, nz),
    n=3,
)

# ── Gizli maliyet olcutleri ────────────────────────────────────────────────
import pyopencl as cl

prog = cl.Program(slvr.ctx, "__kernel void noop(int n){ if (get_global_id(0)==0) { n=0; } }").build()
noop = prog.noop


def enq(n):
    for _ in range(n):
        noop(q, (128,), None, np.int32(1))
    q.finish()


bench("noop enqueue (per-call us)", lambda: enq(500), n=3)


def alloc20():
    cl_array.empty(q, (16000,), dtype=np.float32)
    cl_array.empty(q, (16000,), dtype=np.float32)
    cl_array.empty(q, (16000,), dtype=np.float32)
    cl_array.empty(q, (16000,), dtype=np.float32)
    cl_array.empty(q, (16000,), dtype=np.float32)
    q.finish()


bench("cl_array.empty x20 (16000)", alloc20, n=5)


def get500():
    b = cl_array.zeros(q, (1,), dtype=np.float32)
    for _ in range(500):
        b.get()


bench("scalar get x500", get500, n=3)

# ── 1 adim tahmini (olcudeki kalemler; sa_solve 3 adimda 1, tunnel 2x, ─────
#    inlet 2x, abs_max3 x3. Enqueue overhead / mask / clamp / warm haric.)
t = 2 * ns_rhs_ms + divergence_ms + poisson_ms + project_ms
t += 2 * tunnel_ms + 2 * inlet_ms + outlet_ms + absmax3_ms
t += sa_ms / 3.0
print(f"\n1 adim tahmini (2 rhs + div + poisson + proj + 3BC + dt + sa/3): {t * 1000:.1f} ms")
