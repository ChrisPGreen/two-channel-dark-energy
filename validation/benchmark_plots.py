#!/usr/bin/env python3
"""
benchmark_plots.py -- Step C of the benchmark suite: cross-model comparison figures.

For every model in the suite (LCDM, wCDM, RVM, quintessence, w0wa, two-channel), this
fits the best-fit parameters on the SAME data vector used everywhere else (Planck CMB
distance priors + DESI DR1 BAO + full Pantheon+), then draws four comparison panels:

  (1) w_eff(z)           reconstructed dark-energy equation of state
  (2) H(z)/H_LCDM - 1    expansion-rate residual vs LCDM (per cent)
  (3) D_M(z)/D_M,LCDM-1  comoving-distance residual vs LCDM (per cent), DESI BAO z's marked
  (4) f*sigma8(z)        linear growth (fiducial sigma8_0, GR, no DE clustering)

w_eff is reconstructed UNIFORMLY for every model as
    w_eff(a) = -1 - (1/3) d ln rho_de / d ln a,   rho_de = E^2 - Om a^-3 - Or a^-4,
i.e. what an observer assuming standard matter would infer -- so all curves sit on the
same footing. Quintessence's w stays >= -1 by construction; the two-channel and w0wa
curves dip below -1 (phantom) and cross back, which quintessence structurally cannot do.
That contrast is the scientific point of the figure.

The growth panel uses standard GR sub-horizon growth with NO dark-energy clustering
(a level-1 / background approximation, matching the rest of this package); it is
indicative, not a substitute for a Boltzmann-code fsigma8.

Outputs (results/):
  benchmark_plots.png / .pdf   the 4-panel figure
  best_fits.json               best-fit params + chi2 per model (cached; --refit to redo)
  derived_wz.csv               w_eff/H/D_M/fsigma8 per model on the z grid

Run:  python3 benchmark_plots.py            # uses cached best fits if present
      python3 benchmark_plots.py --refit    # re-optimise every model first
Requires: matplotlib (plus the reproduce.py stack).
"""
import os, sys, json
import numpy as np
from scipy.optimize import minimize
from scipy.integrate import solve_ivp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import reproduce as R

RESULTS = "results"
os.makedirs(RESULTS, exist_ok=True)
DR2 = getattr(R, 'DR2', False)          # --dr2 flag (reproduce picks it up from the shared argv)
SUF = '_dr2' if DR2 else ''
REL = 'dr2' if DR2 else 'dr1'
CACHE = os.path.join(RESULTS, f"best_fits{SUF}.json")
REFIT = "--refit" in sys.argv

# Colour-blind-safe (Wong) palette + distinct line styles so the panels also read in B/W.
MODELS = [
    dict(key='LCDM',  label=r'$\Lambda$CDM',      build=R.Hz_LCDM,  start=[0.311, 67.85],
         bounds=[(0.2, 0.45), (60, 76)],                          color='k',       ls='--', lw=1.6),
    dict(key='wCDM',  label='wCDM',               build=R.Hz_wCDM,  start=[0.316, 67.1, -0.97],
         bounds=[(0.2, 0.45), (60, 76), (-2.5, 0)],               color='#E69F00', ls='-.', lw=1.6),
    dict(key='RVM',   label='RVM (running vac.)', build=R.Hz_RVM,   start=[0.307, 67.3, 0.0016],
         bounds=[(0.2, 0.45), (60, 76), (-0.05, 0.05)],           color='#56B4E9', ls=':',  lw=1.8),
    dict(key='quint', label='Quintessence',       build=R.Hz_quint, start=[0.318, 67.0, 0.65],
         bounds=[(0.2, 0.45), (60, 76), (0.0, 2.0)],              color='#009E73', ls=(0, (3, 1, 1, 1)), lw=1.8),
    dict(key='w0wa',  label='$w_0w_a$ (CPL)',     build=R.Hz_w0wa,  start=[0.313, 67.7, -0.835, -0.64],
         bounds=[(0.2, 0.45), (60, 76), (-2.5, 0), (-4, 3)],      color='#0072B2', ls='-',  lw=1.6),
    dict(key='C',     label='Two-channel',        build=R.Hz_C,     start=[0.313, 67.78, 0.375, 1.78, 0.851],
         bounds=[(0.2, 0.45), (60, 76), (0, 3), (0.5, 12), (0, 40)], color='#D55E00', ls='-', lw=2.6),
]
BYKEY = {m['key']: m for m in MODELS}
DESI_Z = ([0.510, 0.706, 0.934, 1.321, 1.484, 2.330] if DR2
          else [0.51, 0.71, 0.93, 1.32, 2.33])   # DESI effective redshifts (panel 3 shading)


def chi2_of(m, p):
    for (lo, hi), v in zip(m['bounds'], p):
        if not (lo <= v <= hi):
            return 1e9
    try:
        return R.chi2_blocks(m['build'](list(p)), p[0], p[1], with_sn=True)
    except Exception:
        return 1e9


def fit_all():
    """Best-fit each model on the full vector; cache to JSON so re-runs are instant."""
    if os.path.exists(CACHE) and not REFIT:
        with open(CACHE) as f:
            cached = json.load(f)
        if set(cached) >= {m['key'] for m in MODELS}:
            print(f"using cached best fits ({CACHE}); pass --refit to re-optimise")
            return cached
    fits = {}
    for m in MODELS:
        r = minimize(lambda p: chi2_of(m, p), m['start'], method='Nelder-Mead',
                     options={'fatol': 0.01, 'maxiter': 1000})
        fits[m['key']] = dict(p=list(r.x), chi2=float(r.fun))
        print(f"  {m['label']:<22} chi2 = {r.fun:8.2f}   p = {np.array2string(r.x, precision=4)}")
    with open(CACHE, "w") as f:
        json.dump(fits, f, indent=2)
    return fits


def Or_of(H0):
    return R.Or_h2 / (H0 / 100.0) ** 2


def H_on(m, p, z):
    Hz = m['build'](list(p))
    return np.array([Hz(zz) for zz in z])


def w_eff_on(m, p):
    """Uniform reconstructed w(a) on an a-ascending grid (z ~ 3 -> 0). rho_de is the
    non-(matter+radiation) part of E^2; w = -1 - (1/3) d ln rho_de / d ln a."""
    Om, H0 = p[0], p[1]; Or = Or_of(H0)
    # cap at z=2: beyond it the DE density is a small residual of E^2 (noisy reconstruction)
    # and the two-channel's deep-past phantom tail would squash the w=-1 crossing region.
    a = np.linspace(1.0 / (1.0 + 2.0), 1.0, 400)
    z = 1.0 / a - 1.0
    E2 = (H_on(m, p, z) / H0) ** 2
    rho_de = E2 - Om * a ** -3 - Or * a ** -4
    lna = np.log(a)
    w = -1.0 - np.gradient(np.log(np.clip(rho_de, 1e-12, None)), lna) / 3.0
    return z, w, rho_de > 0


def growth_fs8(m, p, z_out, sigma8_0=0.81):
    """Linear growth f*sigma8(z): GR, sub-horizon, no DE clustering (level-1)."""
    Om, H0 = p[0], p[1]; Or = Or_of(H0)
    Nd = np.linspace(-7.0, 0.0, 700); ad = np.exp(Nd); zd = 1.0 / ad - 1.0
    E2 = (H_on(m, p, zd) / H0) ** 2
    dlnE2 = np.gradient(np.log(E2), Nd)
    Om_a = Om * ad ** -3 / E2
    dl = np.interp;
    def rhs(N, y):
        return [y[1], -(2.0 + 0.5 * dl(N, Nd, dlnE2)) * y[1] + 1.5 * dl(N, Nd, Om_a) * y[0]]
    sol = solve_ivp(rhs, [Nd[0], 0.0], [np.exp(Nd[0]), np.exp(Nd[0])], t_eval=None,
                    dense_output=True, rtol=1e-7, atol=1e-9)
    N_out = np.log(1.0 / (1.0 + z_out))
    D = sol.sol(N_out)[0]; Dp = sol.sol(N_out)[1]
    D0 = sol.sol(0.0)[0]
    f = Dp / D
    sigma8 = sigma8_0 * D / D0
    return f * sigma8


def main():
    _logf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), f"log-benchmark-{REL}.txt"), 'w', encoding='utf-8')
    sys.stdout = R._Tee(sys.__stdout__, _logf)
    print(f"# log-benchmark-{REL}.txt -- auto-generated by benchmark_plots.py (DESI {'DR2' if DR2 else 'DR1'} full-vector fits)")
    R.zs, R.mb, R.Cinv, R.CinvO, R.OCO = R.get_sn()
    print("\nFitting all models on the full data vector (Planck priors + DESI BAO + Pantheon+):")
    fits = fit_all()

    zc = np.linspace(0.0, 3.0, 300)       # w, H, D_M panels
    zg = np.linspace(0.02, 2.0, 120)      # fsigma8 panel

    # LCDM reference curves for residual panels
    pL = fits['LCDM']['p']
    HL = H_on(BYKEY['LCDM'], pL, zc)
    DML = R.c * np.concatenate([[0], np.cumsum(0.5 * (1 / HL[1:] + 1 / HL[:-1]) * np.diff(zc))])

    csv_rows = []
    fig, ax = plt.subplots(2, 2, figsize=(12, 9))
    (a1, a2), (a3, a4) = ax

    for m in MODELS:
        p = fits[m['key']]['p']
        style = dict(color=m['color'], ls=m['ls'], lw=m['lw'], label=m['label'])

        # (1) w_eff(z)
        zw, w, ok = w_eff_on(m, p)
        a1.plot(zw, w, **style)

        # (2)+(3) H and D_M residuals vs LCDM
        H = H_on(m, p, zc)
        DM = R.c * np.concatenate([[0], np.cumsum(0.5 * (1 / H[1:] + 1 / H[:-1]) * np.diff(zc))])
        a2.plot(zc, 100.0 * (H / HL - 1.0), **style)
        with np.errstate(invalid='ignore', divide='ignore'):
            dm_res = 100.0 * (DM / DML - 1.0)
        a3.plot(zc[1:], dm_res[1:], **style)

        # (4) f sigma8
        fs8 = growth_fs8(m, p, zg)
        a4.plot(zg, fs8, **style)

        for zz, ww in zip(zw[::40], w[::40]):
            csv_rows.append((m['key'], round(float(zz), 4), round(float(ww), 4)))

    a1.axhline(-1.0, color='0.6', lw=0.8, zorder=0)
    a1.set_xlabel('redshift $z$'); a1.set_ylabel(r'$w_{\rm eff}(z)$')
    a1.set_title('(1) Reconstructed equation of state')
    a1.text(0.02, -1.06, 'phantom  ($w<-1$)', fontsize=8, color='0.4')
    a1.legend(fontsize=8, ncol=2, loc='best')

    a2.axhline(0.0, color='0.6', lw=0.8, zorder=0)
    a2.set_xlabel('redshift $z$'); a2.set_ylabel(r'$H/H_{\Lambda{\rm CDM}}-1$  [%]')
    a2.set_title('(2) Expansion-rate residual vs $\\Lambda$CDM')

    for zd in DESI_Z:
        a3.axvline(zd, color='0.85', lw=0.8, zorder=0)
    a3.axhline(0.0, color='0.6', lw=0.8, zorder=0)
    a3.set_xlabel('redshift $z$'); a3.set_ylabel(r'$D_M/D_{M,\Lambda{\rm CDM}}-1$  [%]')
    a3.set_title('(3) Distance residual vs $\\Lambda$CDM (DESI $z$ shaded)')

    a4.set_xlabel('redshift $z$'); a4.set_ylabel(r'$f\sigma_8(z)$')
    a4.set_title('(4) Growth $f\\sigma_8(z)$  (GR, no DE clustering)')

    fig.suptitle('Benchmark suite: two-channel mechanism vs standard dark-energy models\n'
                 f'(all fit on Planck distance priors + DESI {"DR2" if DR2 else "DR1"} BAO + Pantheon+)', fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    png = os.path.join(RESULTS, f"benchmark_plots{SUF}.png")
    pdf = os.path.join(RESULTS, f"benchmark_plots{SUF}.pdf")
    fig.savefig(png, dpi=150); fig.savefig(pdf)
    print(f"\nwrote {png}\nwrote {pdf}")

    with open(os.path.join(RESULTS, f"derived_wz{SUF}.csv"), "w") as f:
        f.write("model,z,w_eff\n")
        for r in csv_rows:
            f.write(f"{r[0]},{r[1]},{r[2]}\n")
    print(f"wrote {os.path.join(RESULTS, f'derived_wz{SUF}.csv')}")

    print("\nbest-fit chi2 (lower = better fit; not the Bayesian ranking -- see "
          "bayes_evidence_suite.py):")
    for m in MODELS:
        print(f"  {m['label']:<22} chi2 = {fits[m['key']]['chi2']:8.2f}")
    sys.stdout = sys.__stdout__
    _logf.close()


if __name__ == "__main__":
    main()
