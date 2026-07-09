#!/usr/bin/env python3
"""
reproduce.py — full reproduction pipeline for
"A Two-Channel Accretion Mechanism for the Evolving Dark-Energy Signal" (Green 2026)

Usage:
    python3 reproduce.py quick          # DR1: evaluate chi2 at recorded best-fit points (~2 min)
    python3 reproduce.py full           # DR1: re-run every optimisation from scratch (~15-30 min)
    python3 reproduce.py --dr2 quick    # same, against DESI DR2 BAO (arXiv:2503.14738)
    python3 reproduce.py --dr2 full     # same, DR2, full cold-start

The ONLY DR1-vs-DR2 difference is the DESI BAO block; Planck priors and Pantheon+ SNe are
identical. Each run auto-writes its console output to log-reproduce-dr{1,2}-{quick,full}.txt.

Data are downloaded on first run into ./cache (Pantheon+ ~34 MB total).
See VALIDATION-README.md for exact provenance, all approximations, optimizer
settings, target numbers, and recommended pass tolerances.

Environment of record: Python 3.12, numpy/scipy/pandas as pinned in requirements.txt.
No random numbers are used anywhere; results are deterministic given starts/settings.
"""
import sys, os, urllib.request
import warnings
from scipy.integrate import IntegrationWarning

import numpy as np
import pandas as pd
from scipy.integrate import quad
from scipy.optimize import minimize
from scipy.interpolate import interp1d

# Quiet harmless numerical noise from quad integrations in edge parameter space
warnings.filterwarnings('ignore', category=IntegrationWarning, message='.*subdivisions.*')
warnings.filterwarnings('ignore', category=IntegrationWarning, message='.*roundoff error.*')

# --- data-release flag: --dr2 switches ONLY the DESI BAO block (defined below) ---
DR2   = '--dr2' in sys.argv
_args = [a for a in sys.argv[1:] if a != '--dr2']
MODE  = _args[0] if _args else "quick"
REL   = 'dr2' if DR2 else 'dr1'

class _Tee:
    """Write to console and a log file at once, so each run self-documents its output."""
    def __init__(self, *streams): self._streams = streams
    def write(self, s):
        for st in self._streams: st.write(s)
        return len(s)
    def flush(self):
        for st in self._streams: st.flush()

# ----------------------------------------------------------------------
# Constants and fixed quantities (see README §"Fixed constants")
# ----------------------------------------------------------------------
c      = 299792.458          # km/s
Ob_h2  = 0.02236             # FIXED (not a Gaussian prior)
Og_h2  = 2.47e-5             # photons
Or_h2  = 4.15e-5             # total radiation incl. neutrinos (Neff ~ 3.046 implicit)
z_star = 1089.80             # FIXED recombination redshift (not recomputed per model)
z_drag = 1059.94             # FIXED drag epoch
CMB    = {'lA': (301.47, 0.09), 'R': (1.7502, 0.0046)}   # adopted compressed values, DIAGONAL

# DESI BAO block -- the ONLY DR1-vs-DR2 difference in the whole pipeline (--dr2 flag).
# (z_eff, DM/rd, sigma, DH/rd, sigma). D_M-D_H correlations NOT used (diagonal likelihood).
if DR2:
    # DESI DR2 BAO -- Table 4 of arXiv:2503.14738v3 (Phys. Rev. D 112, 083515). ANISOTROPIC
    # tracers only, diagonal likelihood -- SAME methodology as the DR1 analysis (which also used
    # only anisotropic tracers). The isotropic BGS point (z=0.295, D_V/rd=7.942+/-0.075) is
    # omitted for that parallelism. Values verified against the published v3 table 2026-07-09.
    BAO = [(0.510, 13.588, 0.167, 21.863, 0.425),   # LRG1
           (0.706, 17.351, 0.177, 19.455, 0.330),   # LRG2
           (0.934, 21.576, 0.152, 17.641, 0.193),   # LRG3+ELG1
           (1.321, 27.601, 0.318, 14.176, 0.221),   # ELG2
           (1.484, 30.512, 0.760, 12.817, 0.516),   # QSO
           (2.330, 38.988, 0.531,  8.632, 0.101)]   # Lya
else:
    # DESI DR1 BAO (Adame et al. 2025, JCAP 02:021; arXiv:2404.03002)
    BAO = [(0.51, 13.62, 0.25, 20.98, 0.61),
           (0.71, 16.85, 0.32, 20.08, 0.60),
           (0.93, 21.71, 0.28, 17.88, 0.35),
           (1.32, 27.79, 0.69, 13.82, 0.42),
           (2.33, 39.71, 0.94,  8.52, 0.17)]

PANTHEON_DAT = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
                "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES.dat")
PANTHEON_COV = ("https://raw.githubusercontent.com/PantheonPlusSH0ES/DataRelease/main/"
                "Pantheon%2B_Data/4_DISTANCES_AND_COVAR/Pantheon%2BSH0ES_STAT%2BSYS.cov")

# ----------------------------------------------------------------------
# Recorded best-fit points (console output of the original session)
# ----------------------------------------------------------------------
REC = {
  'cons_LCDM':  dict(p=[0.308, 68.1],                          chi2=17.6),
  'cons_w0wa':  dict(p=[0.32, 66.0, -0.19, -2.45],             chi2=12.9),   # Om,H0 approximate; polish
  'cons_C':     dict(p=[0.3381, 65.212, 1.2152, 2.568, 1.0],           chi2=13.8),  # kappa fixed at physical ceiling (see run_cell)
  'full_LCDM':  dict(p=[0.31110763, 67.84800502],              chi2=1409.4),
  'full_w0wa':  dict(p=[0.31319129, 67.72943393, -0.83540666, -0.63897075], chi2=1403.4),
  'full_C':     dict(p=[0.313, 67.8, 0.37, 1.78, 0.85],        chi2=1403.3), # rounded; polish
}

# ----------------------------------------------------------------------
# Data acquisition and SN likelihood preparation
# ----------------------------------------------------------------------
def get_sn():
    os.makedirs("cache", exist_ok=True)
    if not os.path.exists("cache/sn_like.npz"):
        for url, fn in [(PANTHEON_DAT, "cache/pantheon.dat"), (PANTHEON_COV, "cache/pantheon.cov")]:
            if not os.path.exists(fn):
                print(f"downloading {fn} ...")
                urllib.request.urlretrieve(url, fn)
        df = pd.read_csv("cache/pantheon.dat", sep=r"\s+")
        with open("cache/pantheon.cov") as f:
            N = int(f.readline()); Cfull = np.loadtxt(f).reshape(N, N)
        # EXACT cut logic (fingerprint: 1580 SNe): zHD > 0.01 AND IS_CALIBRATOR == 0
        mask = (df["zHD"].values > 0.01) & (df["IS_CALIBRATOR"].values == 0)
        z  = df["zHD"].values[mask]           # zHD used BOTH in D_C(z) and the (1+z) factor
        mb = df["m_b_corr"].values[mask]      # SALT2 m_b_corr column
        Cs = Cfull[np.ix_(mask, mask)]
        Cinv = np.linalg.inv(Cs)
        ones = np.ones(mask.sum())
        np.savez("cache/sn_like.npz", z=z, mb=mb, Cinv=Cinv,
                 CinvO=Cinv @ ones, OCO=ones @ Cinv @ ones, n=mask.sum())
    S = np.load("cache/sn_like.npz")
    print(f"SN likelihood: {len(S['z'])} SNe (expect 1580), full STAT+SYS covariance")
    return S["z"], S["mb"], S["Cinv"], S["CinvO"], float(S["OCO"])

zs = mb = Cinv = CinvO = OCO = None   # loaded lazily in main()
zg = np.concatenate([np.linspace(1e-4, 0.1, 120), np.linspace(0.1, 2.31, 240)[1:]])

# ----------------------------------------------------------------------
# Likelihood blocks
# ----------------------------------------------------------------------
def cs_sound(z):
    R = 3 * Ob_h2 / (4 * Og_h2) / (1 + z)
    return c / np.sqrt(3 * (1 + R))

def chi2_blocks(Hz, Om, H0, with_sn):
    """CMB distance priors + DESI BAO (+ Pantheon+ with analytic offset marginalisation)."""
    x = 0.0
    if with_sn:
        Hgrid = np.array([Hz(z) for z in zg])
        DCg = np.concatenate([[0], np.cumsum(0.5 * (1/Hgrid[1:] + 1/Hgrid[:-1]) * np.diff(zg))]) * c
        DC  = np.interp(zs, zg, DCg)
        dL  = (1 + zs) * DC                              # note: (1+zHD), not (1+z_hel)
        r   = mb - 5 * np.log10(dL)                      # additive const absorbed below
        x  += float(r @ Cinv @ r - (CinvO @ r) ** 2 / OCO)   # exact marginalisation form (no logdet)
    rs  = quad(lambda z: cs_sound(z)/Hz(z), z_star, 1e6, limit=60)[0]
    rd  = quad(lambda z: cs_sound(z)/Hz(z), z_drag, 1e6, limit=60)[0]
    DCs = quad(lambda z: c/Hz(z), 0, z_star, limit=60)[0]
    x += ((np.pi*DCs/rs - CMB['lA'][0]) / CMB['lA'][1])**2
    x += ((np.sqrt(Om)*(H0/c)*DCs - CMB['R'][0]) / CMB['R'][1])**2
    for (z, dm, edm, dh, edh) in BAO:
        DCz = quad(lambda zz: c/Hz(zz), 0, z, limit=50)[0]
        x += ((DCz/rd - dm)/edm)**2 + (((c/Hz(z))/rd - dh)/edh)**2
    return x

# ----------------------------------------------------------------------
# Models
# ----------------------------------------------------------------------
def Hz_LCDM(p):
    Om, H0 = p; h = H0/100; Or = Or_h2/h**2; ODE = 1 - Om - Or
    return lambda z: H0*np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + ODE)

def Hz_w0wa(p):
    Om, H0, w0, wa = p; h = H0/100; Or = Or_h2/h**2; ODE = 1 - Om - Or
    fde = lambda z: (1/(1+z))**(-3*(1+w0+wa)) * np.exp(-3*wa*(1 - 1/(1+z)))
    return lambda z: H0*np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + ODE*fde(z))

def Hz_wCDM(p):
    """Constant-w dark energy (wCDM). ODE dilutes as (1+z)^{3(1+w)}; w=-1 recovers LCDM.
    Added for the Bayesian-evidence model comparison (bayes_evidence_suite.py)."""
    Om, H0, w = p; h = H0/100; Or = Or_h2/h**2; ODE = 1 - Om - Or
    return lambda z: H0*np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + ODE*(1+z)**(3*(1+w)))

def Hz_RVM(p):
    """Running-Vacuum Model: rho_Lambda(H) = (3/8piG)(c0 + nu H^2), the low-energy
    RVM of Sola, Gomez-Valent et al. Single running parameter nu (nu=0 -> LCDM).

    Convention adopted: the vacuum exchanges energy with (pressureless) matter, so
    rho_m ~ a^{-3(1-nu)}, while radiation is separately conserved (rho_r ~ a^{-4}).
    Integrating the Friedmann + local-conservation system (x = ln a) gives the exact,
    algebraic Hubble function

        E^2(a) = A a^{-3xi} + Oc0/xi + [Or0/(1+3nu)] a^{-4},

    with xi = 1 - nu, Oc0 = (1 - Om0 - Or0) - nu the dimensionless c0 term, and
    A = 1 - Oc0/xi - Or0/(1+3nu) fixed by E^2(a=1)=1. At nu=0 this collapses term by
    term to Om0 a^{-3} + Or0 a^{-4} + (1-Om0-Or0), i.e. LCDM (verified in the sanity
    check). The prior keeps |nu| small (see bayes_evidence_suite.py), so xi and 1+3nu
    stay well away from their singular points nu=1 and nu=-1/3."""
    Om, H0, nu = p; h = H0/100; Or = Or_h2/h**2
    xi = 1.0 - nu
    Oc0 = (1.0 - Om - Or) - nu
    A = 1.0 - Oc0/xi - Or/(1.0 + 3.0*nu)
    return lambda z: H0*np.sqrt(A*(1+z)**(3*xi) + Oc0/xi + (Or/(1.0 + 3.0*nu))*(1+z)**4)

# ----------------------------------------------------------------------
# Thawing quintessence — exponential potential V = V0 exp(-lambda phi / Mp)
# ----------------------------------------------------------------------
_SQRT6 = np.sqrt(6.0)

def _quint_gamma_b(Nhalf, Om, Or):
    """Background index gamma_b(N)=1+w_b of the matter+radiation mix on a half-step grid,
    interpolating radiation (4/3) to matter (1). Depends only on (N, Om, Or) -- NOT on the
    field -- so it is precomputed ONCE per solve and reused across every shooting iteration."""
    e3 = np.exp(-3.0*Nhalf); e4 = np.exp(-4.0*Nhalf)
    return (Om*e3 + (4.0/3.0)*Or*e4) / (Om*e3 + Or*e4)

def _quint_integrate(gb_half, h, lam, y_i, steps):
    """Fixed-step RK4 of the Copeland-Liddle-Wands (x,y) system from the field-frozen start
    (x_i=0) to today, using the precomputed gamma_b on the half-step grid gb_half (length
    2*steps+1). x=phi_dot/(sqrt6 H Mp), y=sqrt(V)/(sqrt3 H Mp); Omega_phi=x^2+y^2,
    w_phi=(x^2-y^2)/(x^2+y^2). For V=V0 exp(-lam phi/Mp), lambda is constant. No exp() in
    this loop -- that is what makes it fast enough for nested sampling."""
    c = _SQRT6/2.0*lam
    X = np.empty(steps + 1); Y = np.empty(steps + 1)
    x, y = 0.0, y_i; X[0], Y[0] = x, y
    for i in range(steps):
        g1 = gb_half[2*i]; g2 = gb_half[2*i + 1]; g4 = gb_half[2*i + 2]  # g3 == g2 (same N+h/2)
        # k1 at N (gamma=g1)
        acc = 2.0*x*x + g1*(1.0 - x*x - y*y)
        k1x = -3.0*x + c*y*y + 1.5*x*acc;  k1y = -c*x*y + 1.5*y*acc
        # k2 at N+h/2 (gamma=g2)
        xa = x + h/2*k1x; ya = y + h/2*k1y
        acc = 2.0*xa*xa + g2*(1.0 - xa*xa - ya*ya)
        k2x = -3.0*xa + c*ya*ya + 1.5*xa*acc;  k2y = -c*xa*ya + 1.5*ya*acc
        # k3 at N+h/2 (gamma=g2)
        xb = x + h/2*k2x; yb = y + h/2*k2y
        acc = 2.0*xb*xb + g2*(1.0 - xb*xb - yb*yb)
        k3x = -3.0*xb + c*yb*yb + 1.5*xb*acc;  k3y = -c*xb*yb + 1.5*yb*acc
        # k4 at N+h (gamma=g4)
        xc = x + h*k3x; yc = y + h*k3y
        acc = 2.0*xc*xc + g4*(1.0 - xc*xc - yc*yc)
        k4x = -3.0*xc + c*yc*yc + 1.5*xc*acc;  k4y = -c*xc*yc + 1.5*yc*acc
        x += h/6.0*(k1x + 2*k2x + 2*k3x + k4x)
        y += h/6.0*(k1y + 2*k2y + 2*k3y + k4y)
        if y < 0.0: y = 0.0                          # y=sqrt(V) is non-negative by definition
        X[i + 1], Y[i + 1] = x, y
    return X, Y

def solve_quint(Om, H0, lam, N_i=-16.0, steps=1000):
    """Solve thawing exponential quintessence for the fields (Om, H0, lam). Shoots on the
    initial potential amplitude y_i (field starts frozen, x_i=0) so that the field carries
    exactly Omega_phi(today) = 1 - Om - Or -- i.e. flatness closes the model (no free IC).

    `steps` is the RK4 resolution: a speed/accuracy knob. The only accuracy cost is a tiny
    truncation floor on chi2 (the lam=0 case should equal LCDM exactly): measured +0.044 at
    steps=1000, +0.010 at 2000, +0.004 at 3000. 1000 is the default -- that 0.044 is a
    systematic, quintessence-only offset ~25x below the O(1)-chi2 approximations the pipeline
    already makes (diagonal BAO, compressed CMB priors), so it changes no ranking or verdict,
    while ~halving the field-solve cost. Raise it if you want the floor smaller.
    Returns (a_grid, Omega_phi(a), w_phi(a), E2(a)) with a ascending.

    Reconstruction identity: since Omega_phi = rho_phi/rho_tot and matter+radiation are
    standard, E^2(a) = (Om a^-3 + Or a^-4)/(1 - Omega_phi(a)); this gives E^2(1)=1 exactly.
    lam=0 is the frozen field (w=-1 identically) and reproduces LCDM up to the small
    step-size truncation floor documented above."""
    from scipy.optimize import brentq
    h = H0/100.0; Or = Or_h2/h**2
    ODE0 = 1.0 - Om - Or
    Ns = np.linspace(N_i, 0.0, steps + 1); hN = Ns[1] - Ns[0]
    Nhalf = np.linspace(N_i, 0.0, 2*steps + 1)
    gb_half = _quint_gamma_b(Nhalf, Om, Or)          # precomputed once, reused every shot
    def miss(log_yi):
        X, Y = _quint_integrate(gb_half, hN, lam, np.exp(log_yi), steps)
        return (X[-1]**2 + Y[-1]**2) - ODE0
    # y_i is tiny (field negligible at a~1e-7); bracket log(y_i) generously and shoot tightly
    # (loose shooting biases Omega_phi,0 and hence every distance -- the CMB priors are 0.03%).
    log_yi = brentq(miss, -45.0, 0.0, xtol=1e-6, rtol=1e-10)
    X, Y = _quint_integrate(gb_half, hN, lam, np.exp(log_yi), steps)
    a = np.exp(Ns)
    Ophi = X*X + Y*Y
    wphi = np.where(Ophi > 1e-30, (X*X - Y*Y)/np.where(Ophi > 1e-30, Ophi, 1.0), -1.0)
    E2 = (Om*a**-3 + Or*a**-4) / (1.0 - Ophi)
    return a, Ophi, wphi, E2

def Hz_quint(p):
    """Thawing quintessence, exponential potential (lambda=0 -> LCDM). Added for the
    Bayesian-evidence / benchmark comparison. Quintessence obeys w >= -1 at all times --
    it structurally cannot produce the phantom (w<-1) phase the DESI signal prefers, which
    is the physically meaningful contrast against the two-channel mechanism."""
    Om, H0, lam = p
    a, Ophi, wphi, E2 = solve_quint(Om, H0, lam)
    lna = np.log(a); Egrid = np.sqrt(E2)             # lna ascending -> np.interp valid
    # np.interp (not interp1d) for low per-call overhead: chi2_blocks queries H(z) at
    # hundreds of z per evaluation. Outside the grid np.interp clamps to the edge values.
    return lambda z: H0*float(np.interp(np.log(1.0/(1.0+z)), lna, Egrid))

def build_rho(mu0, n, kap, N=1200):
    """RK4 integration of Eq.(3), x = ln a from 0 down to -9.2 (a ~ 1e-4); rho held for x < -9.2."""
    xs = np.linspace(0, -9.2, N); h = xs[1] - xs[0]
    Mt = lambda x: np.exp(-mu0*(1 - np.exp(n*x))/n)
    f  = lambda x, r: mu0*np.exp(n*x)*(kap*Mt(x)*np.exp(-3*x) - 2*r)
    vals = np.empty(N); r = 1.0; vals[0] = 1.0
    for i in range(N-1):
        x = xs[i]
        k1 = f(x, r); k2 = f(x+h/2, r+h/2*k1); k3 = f(x+h/2, r+h/2*k2); k4 = f(x+h, r+h*k3)
        r = max(r + h/6*(k1 + 2*k2 + 2*k3 + k4), 1e-12); vals[i+1] = r
    return interp1d(xs[::-1], vals[::-1], bounds_error=False, fill_value=(vals[-1], 1.0))

def Hz_C(p):
    Om, H0, mu0, n, kap = p
    rint = build_rho(mu0, n, kap)
    h = H0/100; Or = Or_h2/h**2; ODE = 1 - Om - Or
    return lambda z: H0*np.sqrt(Om*(1+z)**3 + Or*(1+z)**4 + ODE*float(rint(np.log(1/(1+z)))))

BOUNDS = dict(Om=(0.2, 0.45), H0=(60, 76), w0=(-2.5, 0), wa=(-4, 3),
              mu0=(0, 3), n=(0.5, 12), kap=(0, 40))

def chi2(model, p, with_sn):
    if model == 'LCDM':
        Om, H0 = p
        if not (BOUNDS['Om'][0] < Om < BOUNDS['Om'][1] and BOUNDS['H0'][0] < H0 < BOUNDS['H0'][1]): return 1e9
        return chi2_blocks(Hz_LCDM(p), Om, H0, with_sn)
    if model == 'w0wa':
        Om, H0, w0, wa = p
        ok = (BOUNDS['Om'][0] < Om < BOUNDS['Om'][1] and BOUNDS['H0'][0] < H0 < BOUNDS['H0'][1]
              and BOUNDS['w0'][0] < w0 < BOUNDS['w0'][1] and BOUNDS['wa'][0] < wa < BOUNDS['wa'][1])
        if not ok: return 1e9
        return chi2_blocks(Hz_w0wa(p), Om, H0, with_sn)
    if model == 'C':
        Om, H0, mu0, n, kap = p
        ok = (BOUNDS['Om'][0] < Om < BOUNDS['Om'][1] and BOUNDS['H0'][0] < H0 < BOUNDS['H0'][1]
              and BOUNDS['mu0'][0] <= mu0 < BOUNDS['mu0'][1] and BOUNDS['n'][0] < n < BOUNDS['n'][1]
              and BOUNDS['kap'][0] <= kap < BOUNDS['kap'][1])
        if not ok: return 1e9
        return chi2_blocks(Hz_C(p), Om, H0, with_sn)

def fit(model, starts, with_sn, fatol, maxiter):
    best = None
    for s in starts:
        r = minimize(lambda p: chi2(model, p, with_sn), s, method='Nelder-Mead',
                     options={'fatol': fatol, 'maxiter': maxiter})
        if best is None or r.fun < best.fun: best = r
    return best

# ----------------------------------------------------------------------
# Original optimiser settings (used verbatim in 'full' mode)
# ----------------------------------------------------------------------
SETTINGS = {
 'cons_LCDM': dict(model='LCDM', starts=[[0.315, 67.5]], sn=False, fatol=1e-3, maxiter=400),
 'cons_w0wa': dict(model='w0wa', starts=[[0.32, 66, -0.8, -0.8]], sn=False, fatol=1e-3, maxiter=600),
 'cons_C':    dict(model='C', starts=[[0.315, 67.5, 0.5, 5.0, 8.0], [0.31, 66.5, 0.8, 4.0, 4.0]],
                   sn=False, fatol=0.02, maxiter=500),
 'full_LCDM': dict(model='LCDM', starts=[[0.315, 67.5]], sn=True, fatol=0.01, maxiter=400),
 'full_w0wa': dict(model='w0wa', starts=[[0.32, 67, -0.85, -0.6], [0.31, 68, -0.75, -0.9]],
                   sn=True, fatol=0.01, maxiter=500),
 'full_C':    dict(model='C', starts=[[0.32, 67.0, 1.1, 2.4, 1.0], [0.315, 67.5, 0.6, 3.5, 1.2]],
                   sn=True, fatol=0.02, maxiter=320),
}

def run_cell(key):
    S = SETTINGS[key]
    # Conservative two-channel: kappa is a conversion efficiency (kappa<=1). The free-kappa
    # conservative fit (no SN) descends into a shallow valley that keeps dropping PAST the
    # physical ceiling into unphysical kappa>1 (13.89 at kappa=1.006, 13.44 at 1.21, ... --
    # optimiser-dependent, not reproducible). So we report the physically meaningful value at
    # the ceiling: fit (Om,H0,mu0,n) with kappa FIXED at 1.0 -- identical to the kappa=1 point
    # of the kappa-profile below (chi2~13.8), deterministic and physical. NOTE: the FULL vector
    # (full_C) keeps kappa free up to 40 and the DATA select kappa~0.85 -- that key result is
    # untouched; only this SN-free conservative cell is pinned to the physical ceiling.
    if key == 'cons_C':
        r = minimize(lambda q: chi2('C', [q[0], q[1], q[2], q[3], 1.0], False),
                     [0.335, 65.6, 1.15, 2.4], method='Nelder-Mead',
                     options={'fatol': 0.02, 'maxiter': 400})
        return r.fun, np.array([r.x[0], r.x[1], r.x[2], r.x[3], 1.0])
    if MODE == 'quick':
        p = REC[key]['p']
        # Conservative cells are re-polished (see README: the original conservative column
        # used quad limit=150; this package standardises on the full-vector configuration,
        # limit=60, under which minima VALUES are stable but their locations shift slightly).
        # Full-vector LCDM/w0wa are evaluated exactly at the recorded (DR1) vectors -- but under
        # --dr2 those recorded points are NOT the DR2 optima, so on DR2 we re-polish full_LCDM/
        # full_w0wa too (otherwise quick would report the DR1-point chi2 on DR2 data). DR1
        # behaviour is unchanged (the recorded vectors ARE the DR1 optima).
        if key in ('cons_LCDM', 'cons_w0wa', 'full_C') or (DR2 and key in ('full_LCDM', 'full_w0wa')):
            r = fit(S['model'], [p], S['sn'], S['fatol'], 150)
            return r.fun, r.x
        return chi2(S['model'], p, S['sn']), np.array(p)
    r = fit(S['model'], S['starts'], S['sn'], S['fatol'], S['maxiter'])
    return r.fun, r.x

def main():
  global zs, mb, Cinv, CinvO, OCO
  _logf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                            f"log-reproduce-{REL}-{MODE}.txt"), 'w', encoding='utf-8')
  sys.stdout = _Tee(sys.__stdout__, _logf)
  print(f"# log-reproduce-{REL}-{MODE}.txt  --  auto-generated by: "
        f"python3 reproduce.py {'--dr2 ' if DR2 else ''}{MODE}  (deterministic; stack pinned in requirements.txt)")
  zs, mb, Cinv, CinvO, OCO = get_sn()
  if DR2:
      print(f"\n=== MODE: {MODE}  (DESI DR2 BAO) ===  (DR1 targets for reference: cons 17.6/12.9/13.8 ; full 1409.4/1403.4/1403.3 -- DR2 values WILL differ)")
  else:
      print(f"\n=== MODE: {MODE}  (DESI DR1 BAO) ===  (targets: cons 17.6 / 12.9 / 13.8 ; full 1409.4 / 1403.4 / 1403.3)")
  results = {}
  for key in ['cons_LCDM', 'cons_w0wa', 'cons_C', 'full_LCDM', 'full_w0wa', 'full_C']:
      f, p = run_cell(key)
      results[key] = (f, p)
      print(f"{key:>10}: chi2 = {f:8.2f}   params = {np.array2string(p, precision=4)}")

  # ----------------------------------------------------------------------
  # Shape outputs from the full-vector best fit  (targets: w(z=1)=-1.14, cross z=0.41, w0=-0.86)
  # ----------------------------------------------------------------------
  Om, H0, mu0, n, kap = results['full_C'][1]
  rint = build_rho(mu0, n, kap)
  xsh = np.linspace(np.log(1/3.1), 0, 300)
  rho = np.array([float(rint(x)) for x in xsh])
  weff = -1 - np.gradient(np.log(rho), xsh)/3
  zc = next((1/np.exp(xsh[i]) - 1 for i in range(len(xsh)-1, 0, -1)
             if weff[i] > -1 and weff[i-1] <= -1), None)
  ipk = int(np.argmax(rho))
  print(f"\nshape: w_eff(z=1) = {weff[np.argmin(abs(xsh - np.log(0.5)))]:.3f}, "
        f"w_eff(0) = {weff[-1]:.3f}, crossing z = {zc:.3f}, "
        f"rho peak z = {1/np.exp(xsh[ipk]) - 1:.3f}")

  # ----------------------------------------------------------------------
  # Kappa profile on the conservative vector (targets: 16.2 / 13.8 / 17.2 at kappa = 0.5 / 1 / 3)
  # ----------------------------------------------------------------------
  if DR2:
      print("\nkappa profile (DR2 vector; DR1 reference 16.2 / 13.8 / 17.2 at kappa=0.5/1/3 -- DR2 differs; bounded above by LCDM via mu0->0):")
  else:
      print("\nkappa profile (targets 16.2 / 13.8 / 17.2 at kappa=0.5/1/3; bounded above by LCDM 17.6 via mu0->0):")
  for kapf in [0.5, 1.0, 3.0]:
      r = minimize(lambda p: chi2('C', [p[0], p[1], p[2], p[3], kapf], False),
                   [0.335, 65.6, 1.15, 2.4], method='Nelder-Mead',
                   options={'fatol': 0.05, 'maxiter': 300})
      print(f"  kappa = {kapf}: chi2 = {r.fun:6.1f}")

  # ----------------------------------------------------------------------
  # Section 5 mock test (targets: w0 = -0.97, wa = -0.05, implied crossing z ~ 1.2)
  # kappa=0 (dilution-only) limit: rho(a) = exp(2*mu0*(1-a^n)/n), closed form, w >= -1 always.
  # ----------------------------------------------------------------------
  def Hz_B(Om, H0, mu0f, nf):
      h = H0/100; Or = Or_h2/h**2; ODE = 1 - Om - Or
      return lambda z: H0*np.sqrt(Om*(1+z)**3 + Or*(1+z)**4
                                  + ODE*np.exp(2*mu0f*(1 - (1/(1+z))**nf)/nf))

  def observ(Hz, Om, H0):
      rs  = quad(lambda z: cs_sound(z)/Hz(z), z_star, 1e6, limit=60)[0]
      rd  = quad(lambda z: cs_sound(z)/Hz(z), z_drag, 1e6, limit=60)[0]
      DCs = quad(lambda z: c/Hz(z), 0, z_star, limit=60)[0]
      o = [np.pi*DCs/rs, np.sqrt(Om)*(H0/c)*DCs]
      for (z, *_ ) in BAO:
          DCz = quad(lambda zz: c/Hz(zz), 0, z, limit=50)[0]
          o += [DCz/rd, (c/Hz(z))/rd]
      return np.array(o)

  errs = np.array([CMB['lA'][1], CMB['R'][1]] + [e for b in BAO for e in (b[2], b[4])])
  data = np.array([CMB['lA'][0], CMB['R'][0]] + [v for b in BAO for v in (b[1], b[3])])

  allowed = None
  for mu0f in [0.2, 0.35]:
      r = minimize(lambda p: 1e9 if not (0.2 < p[0] < 0.45 and 60 < p[1] < 76 and 0.3 < p[2] < 12)
                   else float(np.sum(((observ(Hz_B(p[0], p[1], mu0f, p[2]), p[0], p[1]) - data)/errs)**2)),
                   [0.31, 68.5, 3.0], method='Nelder-Mead', options={'fatol': 0.02, 'maxiter': 250})
      print(f"\nallowed thawing model mu0={mu0f}: chi2 = {r.fun:.1f} (LCDM baseline 17.6)")
      if r.fun - 17.6 < 2.5 and allowed is None:
          allowed = (r.x[0], r.x[1], mu0f, r.x[2])
  mock = observ(Hz_B(*allowed), allowed[0], allowed[1])
  rM = minimize(lambda p: 1e9 if not (0.2 < p[0] < 0.45 and 60 < p[1] < 76 and -2.5 < p[2] < 0 and -4 < p[3] < 3)
                else float(np.sum(((observ(Hz_w0wa(p), p[0], p[1]) - mock)/errs)**2)),
                [allowed[0], allowed[1], -0.85, -0.4], method='Nelder-Mead',
                options={'fatol': 0.005, 'maxiter': 500})
  w0m, wam = rM.x[2], rM.x[3]
  ac = 1 + (1 + w0m)/wam if wam < 0 else None
  print(f"w0wa fit to kappa=0 mock: w0 = {w0m:.2f}, wa = {wam:.2f}"
        + (f", implied SPURIOUS crossing z = {1/ac - 1:.2f}" if ac and 0 < ac < 1 else ", no crossing"))
  print("\ndone. Compare against targets in VALIDATION-README.md (pass tolerances stated there).")
  sys.stdout = sys.__stdout__
  _logf.close()


if __name__ == "__main__":
    main()
