#!/usr/bin/env python3
"""
bayes_evidence_suite.py — Nested-Sampling evidence for a FULL model comparison,
+ multiprocessing. The two-channel kappa prior is a switch, KAP_MAX, defaulting to the
PHYSICAL efficiency bound [0,1] (kappa is a conversion efficiency, so kappa<=1); set
KAP_MAX = 40.0 only for the explicit unphysical stress test.

Computes the formal Bayesian evidence ln(Z) on the same data vector (Planck CMB
distance priors + DESI DR1 BAO + full Pantheon+ SN, analytic offset marginalised)
for six expansion histories, and reports, in Bayesian model-selection terms, whether the
data favour the two-channel accretion mechanism, disfavour it, or are inconclusive
(indistinguishable) relative to each rival, graded on the Jeffreys scale:

    LCDM   (2 params)  cosmological constant, w = -1                 [baseline]
    wCDM   (3 params)  constant equation of state w
    RVM    (3 params)  running-vacuum model, rho_Lambda = c0 + nu H^2
    quint  (3 params)  thawing quintessence, exponential potential (w >= -1 always)
    w0wa   (4 params)  CPL empirical parameterisation
    C      (5 params)  two-channel accretion mechanism  <-- REFERENCE

Prior posture (see KAP_MAX below): kappa is a conversion EFFICIENCY and cannot exceed 1,
so the physically honest prior is kappa in [0, 1] -- this is the DEFAULT and the headline
result. Widening to [0, 40] (KAP_MAX=40) is available as an explicit robustness/STRESS
TEST only: it puts prior mass on impossible efficiencies and charges an Occam penalty that
is a numerical artefact, not a physical cost. The first-pass narrow run (bayes_evidence.py)
gave Delta ln Z = +0.356 vs w0wa under [0, 1]; this script reproduces that physical setup
for all six models at once.

Hardware: tuned for an 8-physical-core CPU (e.g. Ryzen 7 5800X, 16 logical
threads). Uses 8 worker processes (physical cores) -- for this CPU-bound workload
(RK4 integration, scipy.quad), SMT/hyperthreading gives little extra throughput,
so 8 is the right number, not 16.

Requires: pip install dynesty
Run via:  python3 bayes_evidence_suite.py            # DR1, nlive = 500 (production)
          python3 bayes_evidence_suite.py 100        # DR1, smaller nlive, faster smoke run
          python3 bayes_evidence_suite.py --dr2      # DR2 BAO (arXiv:2503.14738), nlive = 500
Output is auto-written to log-evidence-dr{1,2}-suite.txt (results only; the dynesty
progress bar stays on the console). Several HOURS at nlive=500 — run it yourself, not inline.
"""

import sys, os
import numpy as np
import warnings
from multiprocessing import Pool
from dynesty import NestedSampler

import reproduce

TEST_ZS = np.array([0.0, 0.5, 1.0, 2.33, 10.0, 100.0, 1089.80])
N_CORES = 8       # physical cores on the 5800X; do NOT use 16 (logical/SMT) here
# --dr2 switches the DESI BAO block (via the shared argv, picked up by reproduce); the optional
# positional arg is nlive. So:  bayes_evidence_suite.py [nlive] [--dr2]  (order-independent).
DR2     = '--dr2' in sys.argv
_args   = [a for a in sys.argv[1:] if a != '--dr2']
NLIVE   = int(_args[0]) if _args else 500
REL     = 'dr2' if DR2 else 'dr1'

# RVM running parameter prior: nu in [-NU_HALFWIDTH, +NU_HALFWIDTH], nu=0 -> LCDM.
# NOTE (prior sensitivity): the data pin the running tightly -- the joint fit prefers
# nu ~ +0.0016 and chi2 degrades steeply away from it (the CMB sound horizon is
# measured to ~0.03%, so |nu| ~ 0.01 already shifts it by many sigma). Bayesian
# evidence therefore charges the RVM a large Occam penalty proportional to
# ln(prior width / posterior width): the WIDER this prior, the WORSE the RVM's
# ln(Z), for the same best fit. 0.05 is generous (~30x the data-preferred value on
# each side) yet keeps xi=1-nu and 1+3nu well away from their singular points.
# If the paper adopts a tighter, theory-motivated running prior, shrink this and
# the RVM's evidence rises accordingly -- report the value WITH its prior.
NU_HALFWIDTH = 0.05

# Two-channel efficiency prior: kappa in [0, KAP_MAX].
# PHYSICAL DEFAULT = 1.0. kappa is a conversion efficiency (fraction of infalling rest
# energy that becomes interior vacuum) and CANNOT exceed 1 by construction, so [0, 1] is
# the physically honest prior -- not an "artificially narrow" one. A wider box just puts
# prior mass on impossible values and charges the model an Occam penalty (~ln(KAP_MAX))
# that is a numerical artefact, not a real complexity cost; it also creates a thin-sliver
# posterior in a big empty box that the sampler explores very inefficiently.
# Set KAP_MAX = 40.0 ONLY for the explicit robustness/stress test ("even allowing
# unphysical efficiencies up to 40, the evidence degrades"), never as the headline.
KAP_MAX = 1.0

# =====================================================================
# Per-worker data guard.
#
# Each worker process, under Windows' 'spawn' start method, is a FRESH Python
# interpreter that re-imports reproduce.py from scratch -- it does NOT inherit
# the SN data (zs, mb, Cinv, CinvO, OCO) loaded into the main process's memory.
# On Linux/Mac ('fork') workers DO inherit it, so this bug is platform-specific
# and silent: without the guard every worker's zs/mb/... are None, every
# likelihood call throws, is caught below, and silently returns -inf -- the
# sampler then spins forever unable to find a single valid live point. This
# lazy per-worker load, guarded to run once per process, fixes it on both.
# =====================================================================

def _ensure_data():
    if reproduce.zs is None:
        (reproduce.zs, reproduce.mb, reproduce.Cinv,
         reproduce.CinvO, reproduce.OCO) = reproduce.get_sn()


def _finish(Hz, Om, H0):
    """Shared tail: viability guard at TEST_ZS, then the full chi2 -> log-like.

    Genuine numerical pathology in the MODEL evaluation -- overflow, invalid sqrt,
    divide-by-zero (all RuntimeWarnings) -- is promoted to an error so the point is
    rejected (-inf). Then the returned chi2 is explicitly checked to be finite.

    We do NOT promote scipy's quad 'roundoff'/'subdivisions' IntegrationWarnings to
    errors. Those are benign tolerance notices: quad still returns a value accurate
    to ~7 digits, far beyond what the fit needs. reproduce.py already silences them
    module-wide (IntegrationWarning is a UserWarning, not a RuntimeWarning, so the
    filter below leaves them alone). Promoting them WOULD reject good evaluations --
    including the two-channel model's own best-fit point (chi2=1403.26), where the
    RK4-interpolant + quad legitimately trips the roundoff notice on scipy >= 1.11 --
    silently biasing that model's evidence downward. The finite-chi2 check is the
    real safeguard against unusable integrations."""
    for z_test in TEST_ZS:
        val = Hz(z_test)
        if not np.isfinite(val) or val <= 0:
            return -np.inf
    with warnings.catch_warnings():
        warnings.filterwarnings('error', category=RuntimeWarning)
        chi2_val = reproduce.chi2_blocks(Hz, Om, H0, with_sn=True)
    if not np.isfinite(chi2_val):
        return -np.inf
    return -0.5 * chi2_val

# =====================================================================
# Prior transforms (uniform hypercube -> physical parameters).
# Om and H0 share IDENTICAL priors across every model, so the comparison is
# fair: models differ only in their dark-energy sector and its parameter volume.
# Each must be a top-level function to be picklable for multiprocessing.
# =====================================================================

def prior_transform_LCDM(u):
    p = np.empty_like(u)
    p[0] = 0.2 + 0.25 * u[0]    # Om: [0.2, 0.45]
    p[1] = 60.0 + 16.0 * u[1]   # H0: [60, 76]
    return p

def prior_transform_wCDM(u):
    p = np.empty_like(u)
    p[0] = 0.2 + 0.25 * u[0]    # Om: [0.2, 0.45]
    p[1] = 60.0 + 16.0 * u[1]   # H0: [60, 76]
    p[2] = -2.5 + 2.5 * u[2]    # w:  [-2.5, 0]  (same width as w0wa's w0)
    return p

def prior_transform_RVM(u):
    p = np.empty_like(u)
    p[0] = 0.2 + 0.25 * u[0]    # Om: [0.2, 0.45]
    p[1] = 60.0 + 16.0 * u[1]   # H0: [60, 76]
    p[2] = -NU_HALFWIDTH + 2.0 * NU_HALFWIDTH * u[2]   # nu: [-NU_HALFWIDTH, +NU_HALFWIDTH]; nu=0 -> LCDM
    return p

def prior_transform_quint(u):
    p = np.empty_like(u)
    p[0] = 0.2 + 0.25 * u[0]    # Om:  [0.2, 0.45]
    p[1] = 60.0 + 16.0 * u[1]   # H0:  [60, 76]
    p[2] = 0.0 + 2.0 * u[2]     # lam: [0, 2]  exponential-potential slope; lam=0 -> LCDM.
    #   lam^2 < 2 (lam < ~1.41) is the accelerating field-dominated regime; the box runs a
    #   little past it so the data, not the prior edge, decide. Larger lam only raises w today.
    return p

def prior_transform_w0wa(u):
    p = np.empty_like(u)
    p[0] = 0.2 + 0.25 * u[0]    # Om: [0.2, 0.45]
    p[1] = 60.0 + 16.0 * u[1]   # H0: [60, 76]
    p[2] = -2.5 + 2.5 * u[2]    # w0: [-2.5, 0]
    p[3] = -4.0 + 7.0 * u[3]    # wa: [-4, 3]
    return p

def prior_transform_C_wide(u):
    p = np.empty_like(u)
    p[0] = 0.2 + 0.25 * u[0]    # Om:  [0.2, 0.45]
    p[1] = 60.0 + 16.0 * u[1]   # H0:  [60, 76]
    p[2] = 0.0 + 3.0 * u[2]     # mu0: [0, 3]
    p[3] = 0.5 + 11.5 * u[3]    # n:   [0.5, 12]
    p[4] = 0.0 + KAP_MAX * u[4]   # kap: [0, KAP_MAX]; physical efficiency bound KAP_MAX=1.0
    return p

# =====================================================================
# Log-likelihoods (top-level, picklable). Each builds its model's Hz from
# reproduce.py and defers the identical data comparison to _finish().
# =====================================================================

def loglike_LCDM(p):
    _ensure_data()
    try:
        return _finish(reproduce.Hz_LCDM(p), p[0], p[1])
    except Exception:
        return -np.inf

def loglike_wCDM(p):
    _ensure_data()
    try:
        return _finish(reproduce.Hz_wCDM(p), p[0], p[1])
    except Exception:
        return -np.inf

def loglike_RVM(p):
    _ensure_data()
    try:
        return _finish(reproduce.Hz_RVM(p), p[0], p[1])
    except Exception:
        return -np.inf

def loglike_quint(p):
    _ensure_data()
    try:
        return _finish(reproduce.Hz_quint(p), p[0], p[1])
    except Exception:
        return -np.inf

def loglike_w0wa(p):
    _ensure_data()
    try:
        return _finish(reproduce.Hz_w0wa(p), p[0], p[1])
    except Exception:
        return -np.inf

def loglike_C(p):
    _ensure_data()
    try:
        return _finish(reproduce.Hz_C(p), p[0], p[1])
    except Exception:
        return -np.inf

# =====================================================================
# Model registry. 'C' (two-channel) is the reference the table is anchored on.
# =====================================================================

REFERENCE = 'C'
# Every model shares Om in [0.2, 0.45] and H0 in [60, 76]; 'prior' lists only the
# EXTRA dark-energy-sector parameters, so a referee can see at a glance that no model
# was handed a narrower box than another (cf. the benchmark-suite fairness point).
MODELS = [
    dict(key='LCDM', label='LCDM (w=-1)',       ndim=2, pt=prior_transform_LCDM,   ll=loglike_LCDM,
         prior='(none)'),
    dict(key='wCDM', label='wCDM (const w)',    ndim=3, pt=prior_transform_wCDM,   ll=loglike_wCDM,
         prior='w[-2.5,0]'),
    dict(key='RVM',  label='RVM (running vac)', ndim=3, pt=prior_transform_RVM,    ll=loglike_RVM,
         prior=f'nu[-{NU_HALFWIDTH:g},{NU_HALFWIDTH:g}]'),
    dict(key='quint',label='Quintessence',      ndim=3, pt=prior_transform_quint,  ll=loglike_quint,
         prior='lambda[0,2]'),
    dict(key='w0wa', label='w0wa (CPL)',        ndim=4, pt=prior_transform_w0wa,   ll=loglike_w0wa,
         prior='w0[-2.5,0], wa[-4,3]'),
    dict(key='C',    label='Two-Channel',       ndim=5, pt=prior_transform_C_wide, ll=loglike_C,
         prior=f'mu0[0,3], n[0.5,12], kap[0,{KAP_MAX:g}]'),
]


def jeffreys(abs_delta):
    """Jeffreys' scale bands for |Delta ln Z|."""
    if abs_delta < 1.0:  return "INCONCLUSIVE"
    if abs_delta < 2.5:  return "POSITIVE"
    if abs_delta < 5.0:  return "STRONG"
    return "DECISIVE"


def print_priors():
    """Show every model's prior box up front so the comparison is manifestly fair
    (the shared Om/H0 box plus each model's extra dark-energy parameters)."""
    print("Priors (shared: Om in [0.2, 0.45], H0 in [60, 76]; extra DE params below):")
    for m in MODELS:
        print(f"  {m['label']:<20} k={m['ndim']}   {m['prior']}")
    print()


# =====================================================================
# TABLE RENDERING
# =====================================================================

def print_tables(results):
    """Render the evidence ranking (Table 1) and the two-channel-vs-rivals
    comparison (Table 2) from a {key: (logz, logzerr)} dict. Factored out so the
    table/verdict logic can be unit-tested without a full sampling run."""
    # ---------------------------------------------------------------
    # Table 1: evidence ranking (best evidence first)
    # ---------------------------------------------------------------
    ranked = sorted(MODELS, key=lambda m: results[m['key']][0], reverse=True)
    best_logz = results[ranked[0]['key']][0]

    print("=" * 78)
    print("  TABLE 1 — EVIDENCE RANKING")
    print("=" * 78)
    print(f"{'model':<20}{'k':>3}{'chi2min':>10}{'ln(Z)':>12}{'+/-':>7}{'d lnZ vs best':>15}")
    print("-" * 78)
    for m in ranked:
        lz, le, c2 = results[m['key']]
        star = "  <--best" if m['key'] == ranked[0]['key'] else ""
        ref  = " (ref)" if m['key'] == REFERENCE else ""
        print(f"{m['label']:<20}{m['ndim']:>3}{c2:>10.2f}{lz:>12.3f}{le:>7.3f}"
              f"{lz - best_logz:>15.3f}{star}{ref}")
    print("-" * 78)
    print("chi2min = best-fit chi2 (frequentist); ln(Z) = Bayesian evidence, Occam-")
    print("penalised for prior volume. d lnZ is measured DOWN from the best model.")
    print("Note the two axes can disagree: a lower chi2min can still lose on ln(Z) if")
    print("it cost extra parameters/prior volume to get there.\n")

    # ---------------------------------------------------------------
    # Table 2: Bayesian model selection, two-channel vs each rival (favour/disfavour/inconclusive)
    # ---------------------------------------------------------------
    logz_C, err_C, _ = results[REFERENCE]
    print("=" * 72)
    print(f"  TABLE 2 — BAYESIAN MODEL SELECTION: TWO-CHANNEL vs EACH RIVAL")
    print(f"  Delta ln Z = ln Z(two-channel) - ln Z(rival)   [+ favours two-channel]")
    print(f"  B = Bayes factor (posterior odds under equal model priors)")
    print("=" * 72)
    print(f"{'rival':<20}{'d lnZ':>9}{'+/-':>7}{'B':>10}  {'evidence (Jeffreys)':<28}")
    print("-" * 72)
    # 'even' = evidence too weak to prefer either model; 'fav'/'dis' = data favour /
    # disfavour the two-channel mechanism relative to the rival, graded by strength.
    tally = {'fav': [], 'even': [], 'dis': []}
    for m in MODELS:
        if m['key'] == REFERENCE:
            continue
        lz, le, _ = results[m['key']]
        delta = logz_C - lz
        cerr  = np.sqrt(err_C**2 + le**2)
        bayesf = np.exp(min(abs(delta), 700))   # guard exp overflow on huge gaps
        band  = jeffreys(abs(delta))            # INCONCLUSIVE / POSITIVE / STRONG / DECISIVE
        if band == "INCONCLUSIVE":
            outcome, tag = 'even', "inconclusive (indistinguishable)"
        elif delta > 0:
            outcome, tag = 'fav', f"favours two-channel ({band.lower()})"
        else:
            outcome, tag = 'dis', f"favours rival ({band.lower()})"
        tally[outcome].append(m['label'])
        print(f"{m['label']:<20}{delta:>9.3f}{cerr:>7.3f}{bayesf:>10.2f}  {tag:<28}")
    print("-" * 72)

    # ---------------------------------------------------------------
    # Verdict summary (Bayesian model-selection language)
    # ---------------------------------------------------------------
    def _fmt(names):
        return ", ".join(names) if names else "none"
    print("\nSUMMARY — Bayesian evidence for the two-channel mechanism (wide kappa prior):")
    print(f"  data favour two-channel over    : {_fmt(tally['fav'])}")
    print(f"  indistinguishable from          : {_fmt(tally['even'])}")
    print(f"  data disfavour two-channel vs   : {_fmt(tally['dis'])}")
    print("\nJeffreys scale on |Delta ln Z|: < 1 inconclusive; 1-2.5 positive; 2.5-5")
    print("strong; > 5 decisive. Being evidentially indistinguishable from the empirical")
    print("w0wa form is the intended result -- the mechanism reproduces the evolving-dark-")
    print("energy preference at the same fit quality, but from a physical origin for w(z).")
    print("=" * 72 + "\n")

    if KAP_MAX <= 1.0:
        print(f"PRIOR: kappa in [0, {KAP_MAX:g}] -- the PHYSICAL efficiency bound (headline result).")
        print("Cross-check Delta ln Z(two-channel - w0wa) against the first-pass")
        print("bayes_evidence.py value of +0.356 (same physical prior).")
        print("To run the unphysical stress test, set KAP_MAX = 40.0 at the top and re-run;")
        print("the two-channel evidence should DROP by ~ln(40)~3.7 (Occam on impossible volume).")
    else:
        print(f"PRIOR: kappa in [0, {KAP_MAX:g}] -- this is the UNPHYSICAL STRESS TEST (kappa>1 is")
        print("impossible). Report the KAP_MAX=1.0 physical run as the headline; use this only")
        print("to show the evidence is prior-sensitive when the physical bound is dropped.")


# =====================================================================
# RUNNER
# =====================================================================

if __name__ == "__main__":
    _logf = open(os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              f"log-evidence-{REL}-suite.txt"), 'w', encoding='utf-8')
    sys.stdout = reproduce._Tee(sys.__stdout__, _logf)
    print(f"# log-evidence-{REL}-suite.txt -- auto-generated by bayes_evidence_suite.py "
          f"(nlive={NLIVE}{', --dr2' if DR2 else ''}); dynesty progress bar -> console only, results captured here")
    # Load the Pantheon+ SN data ONCE in the main process. On Linux/Mac the Pool
    # workers fork from here and inherit the populated reproduce module state,
    # avoiding each of the 8 workers re-downloading/re-parsing the 33 MB covariance.
    (reproduce.zs, reproduce.mb, reproduce.Cinv,
     reproduce.CinvO, reproduce.OCO) = reproduce.get_sn()
    print(f"SN data loaded in main process: {len(reproduce.zs)} SNe (expect 1580)")

    _kap_tag = "PHYSICAL BOUND" if KAP_MAX <= 1.0 else "WIDE / STRESS TEST"
    print("\n" + "=" * 68)
    print(f"  NESTED-SAMPLING EVIDENCE — MULTI-MODEL COMPARISON ({_kap_tag})")
    print(f"  kappa prior: [0, {KAP_MAX:g}]  |  nlive: {NLIVE}  |  workers: {N_CORES}")
    print(f"  data: Planck distance priors + DESI {'DR2' if DR2 else 'DR1'} BAO + Pantheon+ (1580 SNe)")
    print("=" * 68 + "\n")

    print_priors()

    results = {}   # key -> (logz, logzerr, chi2min)
    with Pool(processes=N_CORES) as pool:
        for i, m in enumerate(MODELS, 1):
            print(f"Executing run {i}/{len(MODELS)}: {m['label']} "
                  f"(ndim={m['ndim']}) ...")
            # Sampler: ONE method per comparison table, so the six models are compared like-for-like.
            # DR1 keeps dynesty's published default ('auto' -> uniform), which converges for all six.
            # On DR2 the two-channel posterior STALLS uniform, so uniform is not viable there -- hence
            # the WHOLE DR2 table moves to robust slice sampling ('rslice'), applied to ALL six models
            # (not just two-channel) to keep the DR2 evidences mutually consistent. rslice and uniform
            # are unbiased estimators of the SAME evidence, so DR1's published numbers are unchanged
            # and each table is internally consistent. Cost: rslice is slower per point on the low-dim
            # models, accepted as the price of a fair DR2 comparison. (Evidence path only; figures use
            # the frequentist fits.)
            _sample = 'rslice' if DR2 else 'auto'
            sampler = NestedSampler(m['ll'], m['pt'], ndim=m['ndim'],
                                    nlive=NLIVE, sample=_sample,
                                    pool=pool, queue_size=N_CORES)
            sampler.run_nested(print_progress=True)
            res = sampler.results
            logz, logzerr = res.logz[-1], res.logzerr[-1]
            # chi2min = -2 x the peak log-likelihood the sampler visited: a by-product
            # estimate of the best fit, for the frequentist column alongside ln(Z).
            chi2min = -2.0 * float(np.max(res.logl))
            results[m['key']] = (logz, logzerr, chi2min)
            print(f"--> {m['label']}: ln(Z) = {logz:.3f} +/- {logzerr:.3f}"
                  f"   chi2min = {chi2min:.2f}\n")

    print_tables(results)
    sys.stdout = sys.__stdout__
    _logf.close()
