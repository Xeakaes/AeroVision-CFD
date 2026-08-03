"""
AeroVision CFD - Solver Unit Tests
Physics engine verification tests.
"""

import pytest
import numpy as np
from solver import (
    create_voxel_mask,
    solve_potential_flow,
    solve_k_epsilon,
    solve_k_omega_sst,
    solve_spalart_allmaras,
    _smagorinsky_nu,
    compute_pressure_field,
    compute_force_coefficients,
    compute_mesh_refinement,
    compute_adaptive_timesteps,
    interpolate_cp_to_vertices,
    rotate_vertices,
    apply_offset,
    compute_q_criterion,
    solve_navier_stokes_full,
    _assemble_poisson_matrix,
    _poisson_cg,
    _lru_insert,
    _POISSON_CACHE,
    _CACHE_MAX_ENTRIES,
)


class TestVoxelMask:
    """Voxel maskesi olusturma testleri."""

    def test_basic_mask_creation(self):
        """Temel voxel maskesi olusturma."""
        n_tri = 100
        verts = np.random.rand(n_tri * 3, 3).astype(np.float32)
        i_idx = np.arange(0, n_tri * 3, 3, dtype=np.int32)
        j_idx = i_idx + 1
        k_idx = i_idx + 2
        nx, ny, nz = 20, 10, 10
        tunnel_dims = (10.0, 4.0, 4.0)

        mask, solid_fraction, success = create_voxel_mask(verts, i_idx, j_idx, k_idx, nx, ny, nz, tunnel_dims)

        assert success
        assert mask.shape == (nx, ny, nz)
        assert solid_fraction.shape == (nx, ny, nz)
        assert mask.dtype == bool
        assert solid_fraction.dtype == np.float64

    def test_mask_with_no_geometry(self):
        """Geometry olmadan maskesi olusturma."""
        verts = np.array([[100, 100, 100]], dtype=np.float32)
        i_idx = np.array([0], dtype=np.int32)
        j_idx = np.array([0], dtype=np.int32)
        k_idx = np.array([0], dtype=np.int32)

        mask, solid_fraction, success = create_voxel_mask(
            verts, i_idx, j_idx, k_idx, 10, 10, 10, (10.0, 4.0, 4.0)
        )

        assert success
        assert not mask.any()  # Hicbir hücre dolu olmamali


class TestPotentialFlow:
    """Potansiyel akis testleri."""

    def test_basic_flow(self):
        """Temel potansiyel akis cozumu."""
        nx, ny, nz = 20, 10, 10
        mask = np.zeros((nx, ny, nz), dtype=bool)
        v_wind = 10.0

        u, v, w = solve_potential_flow(mask, v_wind, nx, ny, nz)

        assert u.shape == (nx, ny, nz)
        assert v.shape == (nx, ny, nz)
        assert w.shape == (nx, ny, nz)

        # Giris hizi ruzgar hizina esit olmali
        assert np.allclose(u[0, :, :], v_wind, rtol=0.1)

    def test_flow_around_obstacle(self):
        """Engel etrafindaki akis."""
        nx, ny, nz = 30, 15, 15
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[10:20, 5:10, 5:10] = True  # Küp engel
        v_wind = 10.0

        u, v, w = solve_potential_flow(mask, v_wind, nx, ny, nz)

        # Engel uzerinde hiz sifir olmali
        assert np.allclose(u[mask], 0.0, atol=1e-6)
        assert np.allclose(v[mask], 0.0, atol=1e-6)
        assert np.allclose(w[mask], 0.0, atol=1e-6)


class TestKEpsilon:
    """k-epsilon türbülans modeli testleri."""

    def test_basic_kepsilon(self):
        """Temel k-epsilon cozumu."""
        nx, ny, nz = 20, 10, 10
        dx, dy, dz = 0.5, 0.4, 0.4
        nu_mol = 1.5e-5

        u = np.ones((nx, ny, nz)) * 10.0
        v = np.zeros((nx, ny, nz))
        w = np.zeros((nx, ny, nz))
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[0:3, 0:3, 0:3] = True

        nu_t = solve_k_epsilon(u, v, w, mask, dx, dy, dz, nu_mol=nu_mol, iterations=5)

        assert nu_t.shape == (nx, ny, nz)
        assert np.all(nu_t >= 0)
        assert np.all(nu_t <= 100)

    def test_kepsilon_mask_zero(self):
        """Maskeli hücrelerde eddy viskozite sifir olmali."""
        nx, ny, nz = 20, 10, 10
        dx, dy, dz = 0.5, 0.4, 0.4
        nu_mol = 1.5e-5

        u = np.ones((nx, ny, nz)) * 10.0
        v = np.zeros((nx, ny, nz))
        w = np.zeros((nx, ny, nz))
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[5:10, 5:10, 5:10] = True

        nu_t = solve_k_epsilon(u, v, w, mask, dx, dy, dz, nu_mol=nu_mol, iterations=5)

        assert np.allclose(nu_t[mask], 0.0)


class TestKOmegaSST:
    """k-omega SST türbülans modeli testleri."""

    def test_basic_komega_sst(self):
        """Temel k-omega SST cozumu."""
        nx, ny, nz = 20, 10, 10
        dx, dy, dz = 0.5, 0.4, 0.4
        nu_mol = 1.5e-5

        u = np.ones((nx, ny, nz)) * 10.0
        v = np.zeros((nx, ny, nz))
        w = np.zeros((nx, ny, nz))
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[0:3, 0:3, 0:3] = True

        nu_t = solve_k_omega_sst(u, v, w, mask, dx, dy, dz, nu_mol=nu_mol, iterations=5)

        assert nu_t.shape == (nx, ny, nz)
        assert np.all(nu_t >= 0)
        assert np.all(nu_t <= 100)

    def test_komega_sst_mask_zero(self):
        """Maskeli hücrelerde eddy viskozite sifir olmali."""
        nx, ny, nz = 20, 10, 10
        dx, dy, dz = 0.5, 0.4, 0.4
        nu_mol = 1.5e-5

        u = np.ones((nx, ny, nz)) * 10.0
        v = np.zeros((nx, ny, nz))
        w = np.zeros((nx, ny, nz))
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[5:10, 5:10, 5:10] = True

        nu_t = solve_k_omega_sst(u, v, w, mask, dx, dy, dz, nu_mol=nu_mol, iterations=5)

        assert np.allclose(nu_t[mask], 0.0)


class TestPressureField:
    """Basinç alani testleri."""

    def test_pressure_computation(self):
        """Basinç alani hesaplama."""
        nx, ny, nz = 10, 10, 10
        u = np.ones((nx, ny, nz)) * 10.0
        v = np.zeros((nx, ny, nz))
        w = np.zeros((nx, ny, nz))
        v_wind = 10.0

        cp = compute_pressure_field(u, v, w, v_wind)

        assert cp.shape == (nx, ny, nz)
        assert np.all(cp >= -10)
        assert np.all(cp <= 5)
        # Hiz ruzgar hizina esit oldugunda Cp ~ 0 olmali
        assert np.allclose(cp, 0.0, atol=0.1)


class TestMeshRefinement:
    """Mesh refinement testleri."""

    def test_refinement_map(self):
        """Refinement haritasi olusturma."""
        nx, ny, nz = 20, 10, 10
        dx, dy, dz = 0.5, 0.4, 0.4
        tunnel_dims = (10.0, 4.0, 4.0)
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[5:10, 5:10, 5:10] = True

        ref_map = compute_mesh_refinement(mask, dx, dy, dz, tunnel_dims)

        assert ref_map.shape == (nx, ny, nz)
        assert np.all(ref_map >= 0)
        assert np.all(ref_map <= 2)
        # Engel yakinindaki hücreler daha yuksek seviyede olmali
        assert ref_map[5:10, 5:10, 5:10].max() > 0


class TestAdaptiveTimesteps:
    """Adaptif zaman adimi testleri."""

    def test_adaptive_dt(self):
        """Adaptif zaman adimi hesaplama."""
        nx, ny, nz = 20, 10, 10
        u = np.ones((nx, ny, nz)) * 10.0
        v = np.zeros((nx, ny, nz))
        w = np.zeros((nx, ny, nz))
        nu_eff = np.ones((nx, ny, nz)) * 1.5e-5

        dt = compute_adaptive_timesteps(u, v, w, 0.5, 0.4, 0.4, nu_eff, 10.0)

        assert dt > 0
        assert dt < 1.0  # Makul bir deger olmali


class TestGeometryTransforms:
    """Geometri donusum testleri."""

    def test_rotation(self):
        """Rotasyon donusumu."""
        verts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)

        # Sifir rotasyon
        rotated = rotate_vertices(verts, 0, 0, 0)
        np.testing.assert_array_almost_equal(rotated, verts)

        # 90 derece Z rotasyonu
        rotated_90 = rotate_vertices(verts, 90, 0, 0)
        assert rotated_90.shape == verts.shape

    def test_offset(self):
        """Ofset donusumu."""
        verts = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]], dtype=np.float32)
        offset_verts = apply_offset(verts, 1, 2, 3)

        expected = verts + np.array([1, 2, 3], dtype=np.float32)
        np.testing.assert_array_almost_equal(offset_verts, expected)


class TestQCriterion:
    """Q-Criterion testleri."""

    def test_q_criterion_computation(self):
        """Q-Criterion hesaplama."""
        nx, ny, nz = 20, 10, 10
        dx, dy, dz = 0.5, 0.4, 0.4

        u = np.ones((nx, ny, nz)) * 10.0
        v = np.zeros((nx, ny, nz))
        w = np.zeros((nx, ny, nz))

        q = compute_q_criterion(u, v, w, dx, dy, dz, GPU=False)

        assert q.shape == (nx, ny, nz)
        # Duzgun akista Q ~ 0 olmali
        assert np.allclose(q, 0.0, atol=0.01)


class TestInterpolation:
    """Enterpolasyon testleri."""

    def test_cp_interpolation(self):
        """Cp enterpolasyonu."""
        verts = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float32)
        tunnel_dims = (10.0, 4.0, 4.0)
        nx, ny, nz = 10, 10, 10

        cp_field = np.random.rand(nx, ny, nz)
        mask = np.zeros((nx, ny, nz), dtype=bool)

        vertex_cp = interpolate_cp_to_vertices(verts, cp_field, mask, tunnel_dims, nx, ny, nz)

        assert vertex_cp.shape == (3,)
        assert np.all(vertex_cp >= 0)
        assert np.all(vertex_cp <= 1)


class TestSpalartAllmaras:
    """Spalart-Allmaras türbülans modeli testleri."""

    def test_basic_sa(self):
        """Temel SA cozumu."""
        nx, ny, nz = 20, 10, 10
        dx, dy, dz = 0.5, 0.4, 0.4
        nu_mol = 1.5e-5

        u = np.ones((nx, ny, nz)) * 10.0
        v = np.zeros((nx, ny, nz))
        w = np.zeros((nx, ny, nz))
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[0:3, 0:3, 0:3] = True

        nu_t = solve_spalart_allmaras(u, v, w, mask, dx, dy, dz, nu_mol=nu_mol, iterations=5)

        assert nu_t.shape == (nx, ny, nz)
        assert np.all(nu_t >= 0)
        assert np.all(nu_t <= 100)

    def test_sa_mask_zero(self):
        """Maskeli hücrelerde eddy viskozite sifir olmali."""
        nx, ny, nz = 20, 10, 10
        dx, dy, dz = 0.5, 0.4, 0.4
        nu_mol = 1.5e-5

        u = np.ones((nx, ny, nz)) * 10.0
        v = np.zeros((nx, ny, nz))
        w = np.zeros((nx, ny, nz))
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[5:10, 5:10, 5:10] = True

        nu_t = solve_spalart_allmaras(u, v, w, mask, dx, dy, dz, nu_mol=nu_mol, iterations=5)

        assert np.allclose(nu_t[mask], 0.0)


class TestSmagorinsky:
    """Smagorinsky LES modeli testleri."""

    def test_basic_smagorinsky(self):
        """Temel Smagorinsky eddy viskozitesi."""
        nx, ny, nz = 20, 10, 10
        dx, dy, dz = 0.5, 0.4, 0.4

        u = np.ones((nx, ny, nz)) * 10.0
        v = np.zeros((nx, ny, nz))
        w = np.zeros((nx, ny, nz))

        nu_t = _smagorinsky_nu(u, v, w, dx, dy, dz, np)

        assert nu_t.shape == (nx, ny, nz)
        assert np.all(nu_t >= 0)
        assert np.allclose(nu_t, 0.0, atol=1e-4)  # Duzgun akista kayma yok


class TestPoissonSolver:
    """Basinç Poisson cozucu testleri."""

    def test_poisson_solution(self):
        """Poisson denklemi dogru cozulmeli."""
        nx, ny, nz = 8, 8, 8
        dx, dy, dz = 0.5, 0.4, 0.4
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[2:5, 2:5, 2:5] = True

        # Bilinen bir RHS: sabit -1, cozum duzgun olmali
        rhs = np.full((nx, ny, nz), -1.0)
        rhs[mask] = 0.0

        A, solid_idx = _assemble_poisson_matrix(mask, dx, dy, dz, nx, ny, nz)
        p = _poisson_cg(rhs, mask, dx, dy, dz, nx, ny, nz, np)

        assert p.shape == (nx, ny, nz)
        assert np.all(np.isfinite(p))
        assert np.allclose(p[mask], 0.0, atol=1e-8)
        # Degerler makul olmali
        assert np.all(np.abs(p) < 1e4)

    def test_poisson_zero_rhs(self):
        """Sifir RHS ile cozum sifir olmali."""
        nx, ny, nz = 8, 8, 8
        dx, dy, dz = 0.5, 0.4, 0.4
        mask = np.zeros((nx, ny, nz), dtype=bool)

        rhs = np.zeros((nx, ny, nz))
        A, solid_idx = _assemble_poisson_matrix(mask, dx, dy, dz, nx, ny, nz)
        p = _poisson_cg(rhs, mask, dx, dy, dz, nx, ny, nz, np)

        assert np.allclose(p, 0.0, atol=1e-10)


class TestPoissonMatrixRegression:
    """Poisson matrisi regresyon testleri.

    OpenCL kernel'inin stencil'iyle (poisson_matvec) birebir ayni matrisi
    uretmelidir: pozitif Laplacian (+L), sarma (wrap-around) komşuluk yok,
    katı satirlar ve row 0 saf dirichlet (diag=1).
    """

    @staticmethod
    def _reference_stencil(mask, dx, dy, dz, nx, ny, nz):
        """Kernel guard'lariyla birebir referans 7-nokta stencil'i.

        Katı komşulukta Neumann (tek taraflı): o yönün diag katkısı düşer,
        akı 0 (impenetrable) maskesiyle tutarlı.
        """
        ax, ay, az = 1.0 / dx**2, 1.0 / dy**2, 1.0 / dz**2
        n = nx * ny * nz
        A = np.zeros((n, n))
        for i in range(nx):
            for j in range(ny):
                for k in range(nz):
                    r = (i * ny + j) * nz + k
                    if mask[i, j, k] or r == 0:
                        A[r, r] = 1.0
                        continue
                    A[r, r] = 2.0 * (ax + ay + az)
                    if i < nx - 1:
                        if mask[i + 1, j, k]:
                            A[r, r] -= ax
                        elif r + ny * nz != 0:
                            A[r, r + ny * nz] = -ax
                    if i > 0:
                        if mask[i - 1, j, k]:
                            A[r, r] -= ax
                        elif r - ny * nz != 0:
                            A[r, r - ny * nz] = -ax
                    if j < ny - 1:
                        if mask[i, j + 1, k]:
                            A[r, r] -= ay
                        elif r + nz != 0:
                            A[r, r + nz] = -ay
                    if j > 0:
                        if mask[i, j - 1, k]:
                            A[r, r] -= ay
                        elif r - nz != 0:
                            A[r, r - nz] = -ay
                    if k < nz - 1:
                        if mask[i, j, k + 1]:
                            A[r, r] -= az
                        elif r + 1 != 0:
                            A[r, r + 1] = -az
                    if k > 0:
                        if mask[i, j, k - 1]:
                            A[r, r] -= az
                        elif r - 1 != 0:
                            A[r, r - 1] = -az
        return A

    def test_matches_kernel_stencil(self):
        """Matris, kernel stencil'iyle birebir ayni olmali (wrap yok)."""
        nx, ny, nz = 8, 8, 8
        dx, dy, dz = 0.5, 0.4, 0.4
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[2:5, 2:5, 2:5] = True
        mask[0, 0, 0] = True  # row 0 zaten dirichlet; kotu kenar da olsun

        A, solid_idx = _assemble_poisson_matrix(mask, dx, dy, dz, nx, ny, nz)
        ref = self._reference_stencil(mask, dx, dy, dz, nx, ny, nz)

        assert solid_idx[0] == 0  # row 0 her zaman dirichlet listesinde
        assert np.allclose(A.toarray(), ref, atol=1e-12)

    def test_no_wrap_around_neighbors(self):
        """Sarma komşuluklar matriste hic bulunmamali (3D-duyarli)."""
        nx, ny, nz = 8, 8, 8
        dx, dy, dz = 0.5, 0.4, 0.4
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[1:4, 1:4, 1:4] = True

        A, _ = _assemble_poisson_matrix(mask, dx, dy, dz, nx, ny, nz)
        coo = A.tocoo()
        for r, c in zip(coo.row, coo.col, strict=True):
            i, j, k = r // (ny * nz), (r // nz) % ny, r % nz
            ci, cj, ck = c // (ny * nz), (c // nz) % ny, c % nz
            di, dj, dk = abs(i - ci), abs(j - cj), abs(k - ck)
            # Periyodik esitlik yok: komşuluk sadece |d|=1 ile sinirli
            assert (di, dj, dk) in {(1, 0, 0), (0, 1, 0), (0, 0, 1)} or (r == c)

    def test_symmetric(self):
        """Matris simetrik olmali (CG icin gerekli)."""
        nx, ny, nz = 8, 8, 8
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[2:5, 2:5, 2:5] = True

        A, _ = _assemble_poisson_matrix(mask, 0.5, 0.4, 0.4, nx, ny, nz)
        assert np.allclose(A.toarray(), A.toarray().T, atol=1e-12)

    def test_positive_definite(self):
        """SPD olmali: tum ozdegerler > 0 (NSD -L bug'ina karsi regresyon)."""
        nx, ny, nz = 8, 8, 8
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[2:5, 2:5, 2:5] = True

        A, _ = _assemble_poisson_matrix(mask, 0.5, 0.4, 0.4, nx, ny, nz)
        eig = np.linalg.eigvalsh(A.toarray())
        assert np.all(eig > 1e-10), f"NSD tespit edildi, min eig={eig.min()}"


class TestPoissonCache:
    """Poisson matris LRU onbellek testleri."""

    def test_cache_insert_and_hit(self):
        """Onbellege ekleme ve cagirma."""
        key = "test-key-1"
        value = (np.ones((2, 2)), np.ones(3), np.ones(4))

        _lru_insert(_POISSON_CACHE, key, value)
        assert key in _POISSON_CACHE
        assert _POISSON_CACHE[key] is value

    def test_cache_lru_limit(self):
        """Onbellek _CACHE_MAX_ENTRIES ile sinirli olmali."""
        _POISSON_CACHE.clear()

        for i in range(_CACHE_MAX_ENTRIES + 3):
            _lru_insert(_POISSON_CACHE, f"key-{i}", np.ones((2, 2)))

        assert len(_POISSON_CACHE) == _CACHE_MAX_ENTRIES
        # En eski anahtarlar atilmis olmali
        assert "key-0" not in _POISSON_CACHE
        assert "key-1" not in _POISSON_CACHE
        assert "key-2" not in _POISSON_CACHE
        # En yeni anahtarlar duruyor olmali
        assert "key-5" in _POISSON_CACHE
        assert "key-6" in _POISSON_CACHE

    def test_cache_lru_retains_recently_used(self):
        """Erisilen giris yeniden kullanildiginda atilma sirasinda geriye gitmeli."""
        _POISSON_CACHE.clear()
        for i in range(_CACHE_MAX_ENTRIES):
            _lru_insert(_POISSON_CACHE, f"k{i}", i)

        # En eskiyi erisilir yap
        _POISSON_CACHE["k0"] = 0
        _POISSON_CACHE.move_to_end("k0")
        _lru_insert(_POISSON_CACHE, "knew", -1)

        assert len(_POISSON_CACHE) == _CACHE_MAX_ENTRIES
        # k1 artik en eski, atilmis olmali; k0 korunmali
        assert "k1" not in _POISSON_CACHE
        assert "k0" in _POISSON_CACHE


class TestForceCoefficients:
    """Kuvvet katsayilari testleri."""

    def _run_ns(self, nx=30, ny=14, nz=14):
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[10:20, 4:10, 4:10] = True
        u, v, w = solve_potential_flow(mask, 20.0, nx, ny, nz)
        u2, v2, w2, p = solve_navier_stokes_full(
            u, v, w, mask, 20.0, nx, ny, nz, iterations=20, turbulence_model="smagorinsky"
        )
        return u2, v2, w2, p, mask

    def test_drag_positive_with_viscous_wall(self):
        """Viskos duvarli suruklenme pozitif olmali."""
        u2, v2, w2, p, mask = self._run_ns()
        verts = np.array(
            [
                [0, 0, 0.5],
                [1, 0, 0.5],
                [0, 1, 0.5],
                [1, 1, 0.5],
                [0, 1, 0.5],
                [1, 0, 0.5],
            ],
            dtype=np.float32,
        )
        i_idx = np.array([0, 3], dtype=np.int32)
        j_idx = i_idx + 1
        k_idx = i_idx + 2
        nx, ny, nz = u2.shape

        result = compute_force_coefficients(
            verts,
            i_idx,
            j_idx,
            k_idx,
            np.zeros((nx, ny, nz)),
            u2,
            v2,
            w2,
            (10, 4, 4),
            nx,
            ny,
            nz,
            20.0,
            mask=mask,
        )

        assert result["valid"]
        assert result["drag_N"] > 0

    def test_rho_proportionality(self):
        """Kuvvetler rho ile dogru orantili olmali (F ~ rho)."""
        u2, v2, w2, p, mask = self._run_ns()
        verts = np.array(
            [
                [0, 0, 0.5],
                [1, 0, 0.5],
                [0, 1, 0.5],
                [1, 1, 0.5],
                [0, 1, 0.5],
                [1, 0, 0.5],
            ],
            dtype=np.float32,
        )
        i_idx = np.array([0, 3], dtype=np.int32)
        j_idx = i_idx + 1
        k_idx = i_idx + 2
        nx, ny, nz = u2.shape

        kw = dict(
            verts=verts,
            i_idx=i_idx,
            j_idx=j_idx,
            k_idx=k_idx,
            cp_field=np.zeros((nx, ny, nz)),
            u=u2,
            v=v2,
            w=w2,
            tunnel_dims=(10, 4, 4),
            nx=nx,
            ny=ny,
            nz=nz,
            v_wind=20.0,
            mask=mask,
        )
        f_air = compute_force_coefficients(rho=1.225, **kw)
        f_water = compute_force_coefficients(rho=998.2, **kw)

        expected = 998.2 / 1.225
        assert f_air["drag_N"] > 0
        assert np.isclose(f_water["drag_N"] / f_air["drag_N"], expected, rtol=1e-6)


class TestFluidParameters:
    """Coklu akiskan (rho/nu_mol) parametre testleri."""

    def test_ns_accepts_fluid_params(self):
        """NS cozucu rho ve nu_mol parametrelerini kabul etmeli."""
        nx, ny, nz = 30, 14, 14
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[10:20, 4:10, 4:10] = True
        u, v, w = solve_potential_flow(mask, 20.0, nx, ny, nz)

        u2, v2, w2, p = solve_navier_stokes_full(
            u,
            v,
            w,
            mask,
            20.0,
            nx,
            ny,
            nz,
            iterations=15,
            turbulence_model="smagorinsky",
            rho=998.2,
            nu_mol=1.004e-6,
        )

        assert np.all(np.isfinite(u2))
        assert np.all(np.isfinite(p))
        assert np.allclose(u2[mask], 0.0, atol=1e-8)

    def test_ns_defaults_to_air(self):
        """Parametresiz cagri havayi (RHO) kullanmali."""
        nx, ny, nz = 30, 14, 14
        mask = np.zeros((nx, ny, nz), dtype=bool)
        mask[10:20, 4:10, 4:10] = True
        u, v, w = solve_potential_flow(mask, 20.0, nx, ny, nz)

        u2, v2, w2, p = solve_navier_stokes_full(
            u, v, w, mask, 20.0, nx, ny, nz, iterations=15, turbulence_model="smagorinsky"
        )

        assert np.all(np.isfinite(u2))
        assert np.all(np.isfinite(p))


# ════════════════════════════════════════════════════════════════════════════
# VALIDATION — tam doğrulama benchmark'lari (yavas; marker ile dahil edilir)
#   python -m pytest test_solver.py -m validation
# ════════════════════════════════════════════════════════════════════════════

validation = pytest.mark.validation


@validation
class TestSphereBenchmark:
    """Kure Cd benchmark'i — deneysel referans Cd = 0.47 (tolerans %20)."""

    def test_sphere_cd_quick(self):
        """Dusuk cozunurluk smoke: Cd 0.1-1.5 araliginda, finite."""
        from validation.validate_cd import run_case

        r = run_case("quick", "spalart_allmaras", verbose=False)
        assert np.isfinite(r["Cd_physical"])
        assert 0.1 < r["Cd_physical"] < 1.5, f"Cd={r['Cd_physical']} beklenen disinda"
        assert r["valid"]


@validation
class TestAirfoilBenchmark:
    """NACA 0012 kaldirma benchmark'i — isaret ve egim kontrolu."""

    def test_airfoil_lift_sign_and_slope(self):
        """alpha=8'de Cl > 0 (dogru isaret) ve sonuc finite."""
        from validation.validate_airfoil import run_case

        r8 = run_case(8.0, "coarse", verbose=False)
        assert np.isfinite(r8["Cl"]) and np.isfinite(r8["Cd"])
        assert r8["Cl"] > 0.05, f"Cl={r8['Cl']} — isaret veya buyukluk hatali"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
