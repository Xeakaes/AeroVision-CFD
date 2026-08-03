"""
AeroVision CFD - Fizik Sabitleri ve Onbellekler

Tum CFD modulleri tarafindan paylasilan sabitler ve sinirli LRU onbellekler.
Bu modul hicbir CFD modulune bagimli degildir (saf veri + yardimci fonksiyon).
"""

from collections import OrderedDict

# ── Fizik sabitleri ────────────────────────────────────────────────────────
TUNNEL_DIMS = (10.0, 4.0, 4.0)  # tx, ty, tz  [m]
RHO = 1.225  # hava yoğunluğu [kg/m³]

# ── Smagorinsky LES ────────────────────────────────────────────────────────
C_SMAG = 0.17  # Smagorinsky sabiti

# ── Spalart-Allmaras sabitleri ─────────────────────────────────────────────
SA_CB1 = 0.1355  # Üretim terimi sabiti
SA_CB2 = 0.622  # Difüzyon sabiti
SA_SIGMA = 2.0 / 3.0  # Prandtl benzeri sabiti
SA_KAPPA = 0.41  # Von Karman sabiti
SA_CW1 = SA_CB1 / SA_KAPPA**2 + (1.0 + SA_CB2) / SA_SIGMA
SA_CW2 = 0.3
SA_CW3 = 2.0
SA_CV1 = 7.1
SA_CV2 = 0.7
SA_CV3 = 0.9
SA_CV4 = 0.8

# ── k-epsilon türbülans modeli sabitleri (Launder & Spalding 1974) ─────────
KE_CMU = 0.09  # ν_t = C_μ * k² / ε
KE_C1 = 1.44  # Üretim sabiti
KE_C2 = 1.92  # Dağılma sabiti
KE_SIGMA_K = 1.0  # k difüzyon Prandtl sayısı
KE_SIGMA_E = 1.3  # ε difüzyon Prandtl sayısı
KE_WALL_K = 0.0  # Duvarlarda k (sıfır)
KE_WALL_E = 1e-10  # Duvarlarda ε (çok küçük, 0/0 önlemek için)

# ── k-omega SST sabitleri (Menter 1994) ────────────────────────────────────
KO_BETA1 = 0.075
KO_BETA2 = 0.0828
KO_SIGMA_K1 = 0.85
KO_SIGMA_W1 = 0.50
KO_SIGMA_K2 = 1.0
KO_SIGMA_W2 = 0.856
KO_A1 = 0.31
KO_BETA_STAR = 0.09

# ── Basınç Poisson çözücü ──────────────────────────────────────────────────
POISSON_TOL = 1e-6
POISSON_MAXIT = 800

# ── Sınırlı LRU önbellekler — bellek sızıntısını önler (maksimum giriş sayısı)
_CACHE_MAX_ENTRIES = 4
_POISSON_CACHE = OrderedDict()
_WALL_DISTANCE_CACHE = OrderedDict()


def _lru_insert(cache, key, value):
    """Sınırlı LRU önbellek yazma — kapasite aşılınca en eski girişi at."""
    cache[key] = value
    cache.move_to_end(key)
    while len(cache) > _CACHE_MAX_ENTRIES:
        cache.popitem(last=False)
