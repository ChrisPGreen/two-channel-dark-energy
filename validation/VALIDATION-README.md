# Validation package — Two-Channel Accretion Mechanism (Green 2026)

**Release v1.1.** This package corresponds to release v1.1 — it adds the Bayesian evidence
suite and the DESI DR2 cross-check to the v1.0.2 baseline (the core mechanism, the headline
fit, and the forced-ordering theorem are unchanged). Reproduce against
`log-reproduce-dr1-full.txt`; the manuscript, this package, and that reference run are the
same release.

This package answers the independent-validation checklist point by point. The governing
principle throughout: **what is documented here is what the code actually did**, including
every approximation. To match the paper's numbers you must replicate the approximations
first; upgrading them (correlations, per-model recombination, etc.) is a valuable *extension*
but will not reproduce the fingerprints.

**The reference run is `reproduce.py full`**, captured in `log-reproduce-dr1-full.txt`
— reproduce against that. Full mode cold-starts every optimisation from the original starting
points and is authoritative wherever it differs from the shorter `quick` mode.

See **§0 below** for the full file manifest — all five scripts, the `--dr2` flag, and every
auto-written log — with the exact command to run each.

---

## 0. File manifest (what each file is, and which data release)

The **only** distinction between DR1 and DR2 is the DESI BAO block; Planck distance priors and
the full Pantheon+ sample are identical in both. DR1 is the paper's primary analysis; DR2 is a
robustness cross-check, selected at run time with a **`--dr2`** flag — there are no separate
DR2 scripts. **Every script auto-writes its own log** (named below); no manual console
copy-paste, and re-running overwrites the log deterministically.

**Scripts (5)**

| Script | What it does | How to run |
|---|---|---|
| `reproduce.py` | Background pipeline: ΛCDM / w0–wa / two-channel fits (conservative + full vectors), shape, κ-profile, mock test | `reproduce.py [quick\|full] [--dr2]` |
| `bayes_evidence.py` | Nested-sampling ln Z, two-channel vs w0–wa pair | `bayes_evidence.py [--dr2]` |
| `bayes_evidence_suite.py` | Nested-sampling ln Z, six-model suite (ΛCDM, wCDM, RVM, quintessence, w0–wa, two-channel) | `bayes_evidence_suite.py [nlive] [--dr2]` |
| `benchmark_plots.py` | "Step C" 4-panel model-comparison figure (w_eff, H & D_M residuals, fσ8) → `results/` | `benchmark_plots.py [--dr2] [--refit]` |
| `camb_checks.py` | Perturbation-level CAMB/PPF checks on the best-fit w(a) | `camb_checks.py` (needs `camb`) |

**Auto-written logs**

| Log | Produced by | Release |
|---|---|---|
| `log-reproduce-dr1-quick.txt` / `-full.txt` | `reproduce.py [quick\|full]` (`-full` is **the reference run**) | DR1 |
| `log-reproduce-dr2-quick.txt` / `-full.txt` | `reproduce.py --dr2 [quick\|full]` | DR2 |
| `log-evidence-dr1-pair.txt` | `bayes_evidence.py` (two-channel vs w0–wa) | DR1 |
| `log-evidence-dr1-suite.txt` | `bayes_evidence_suite.py` (six-model suite) | DR1 |
| `log-evidence-dr2-suite.txt` | `bayes_evidence_suite.py --dr2` (six-model suite, DR2) | DR2 |
| `log-benchmark-dr1.txt` / `log-benchmark-dr2.txt` | `benchmark_plots.py [--dr2]` (console summary; figure → `results/`) | DR1 / DR2 |
| `log-camb-checks.txt` | `camb_checks.py` | DR1 |
| `results/benchmark_plots{,_dr2}.{pdf,png}`, `results/derived_wz{,_dr2}.csv` | `benchmark_plots.py [--dr2]` | DR1 / DR2 |
| `requirements.txt` | pinned dependency stack | — |

**Optional / not generated:** `log-evidence-dr2-pair.txt` (`bayes_evidence.py --dr2`) — the DR2
single-pair cross-check, not run by default. Figure 1 (`weff_history.pdf`) has no generator
script in this package.

---

## 1. The code

Everything is in `reproduce.py`, self-contained (no helper modules): the χ² for each data
block, the RK4 integrator for Eq. (3), the ΛCDM / w0–wa baselines, the driver, the κ profile,
and the §5 mock test. `python3 reproduce.py full` re-runs every optimisation from the original
starting points; `quick` evaluates at the recorded best-fit points (with a short polish where
the recorded vector was rounded) and finishes in ~2 minutes. Two extra baselines used only by
the evidence comparison, `Hz_wCDM` (constant w) and `Hz_RVM` (running vacuum), also live in
`reproduce.py`. The Bayesian evidence pipeline is separate (§9): `bayes_evidence.py` (narrow κ
prior, two-channel vs w0–wa) and `bayes_evidence_suite.py` (full six-model table; physical
κ ∈ [0, 1] prior by default, `KAP_MAX = 40` for the stress test).
Step C comparison figures across all models come from `benchmark_plots.py` (§10).

## 2. Data provenance — exact

**Pantheon+.** Repo `PantheonPlusSH0ES/DataRelease`, branch `main`, files
`Pantheon+_Data/4_DISTANCES_AND_COVAR/Pantheon+SH0ES.dat` and
`Pantheon+SH0ES_STAT+SYS.cov` (the **stat+sys** covariance; file size 33,284,960 bytes,
first line `1701`). Cut logic, exactly: `(zHD > 0.01) & (IS_CALIBRATOR == 0)` → **1580 SNe**
(the fingerprint). Magnitude column: `m_b_corr`. Redshift: `zHD` used **both** in the comoving
distance and in the (1+z) luminosity-distance factor — a deliberate simplification (professional
analyses use 1+z_hel in the prefactor); it is identical across all three models and largely
cancels in comparisons, but you must replicate it to match χ² absolutely. Distances via
trapezoidal cumulative integration on the fixed grid
`concat(linspace(1e-4, 0.1, 120), linspace(0.1, 2.31, 240)[1:])`, then interpolation to each zHD.

**Offset marginalisation, exact form.** With residuals r = m_b_corr − 5·log10(dL) (any
additive constant, including the −25 and the H0/M_B degeneracy, absorbed):
χ²_SN = rᵀC⁻¹r − (1ᵀC⁻¹r)²/(1ᵀC⁻¹1). No log-determinant term is included (it is
parameter-independent here).

**DESI BAO.** DESI DR1 (Adame et al. 2025, JCAP 02:021; arXiv:2404.03002), five tracers:

| z_eff | D_M/r_d | σ | D_H/r_d | σ |
|---|---|---|---|---|
| 0.51 | 13.62 | 0.25 | 20.98 | 0.61 |
| 0.71 | 16.85 | 0.32 | 20.08 | 0.60 |
| 0.93 | 21.71 | 0.28 | 17.88 | 0.35 |
| 1.32 | 27.79 | 0.69 | 13.82 | 0.42 |
| 2.33 | 39.71 | 0.94 |  8.52 | 0.17 |

**The D_M–D_H correlation coefficients were NOT used** — the BAO likelihood is diagonal.
This is a disclosed approximation; DR1 publishes the correlations (~ −0.4), and using them
will shift χ² by O(1) and best-fit parameters slightly. Replicate the diagonal form first.

**Planck distance priors.** Adopted compressed values, **diagonal** (no ℓ_A–R correlation,
no ω_b row in the Gaussian): ℓ_A = 301.47 ± 0.09, R = 1.7502 ± 0.0046. These are adopted
values in the style of Chen, Huang & Wang (2019); use these exact numbers. ω_b is **fixed**,
not a prior (below). The shift parameter is computed as R = √(Ω_m)·(H0/c)·D_C(z*), and
ℓ_A = π·D_C(z*)/r_s(z*), with r_s integrated from z* to 10⁶ using the standard
c_s(z) = c/√(3(1+R_b)), R_b = 3ω_b/(4ω_γ)/(1+z).

**Fixed constants (never varied, never recomputed per model).**
ω_b = 0.02236; ω_γ = 2.47×10⁻⁵; ω_r = 4.15×10⁻⁵ (total radiation, i.e. N_eff ≈ 3.046
folded in and treated as fully relativistic at all times; neutrino mass zero);
z* = 1089.80; z_drag = 1059.94 (both **fixed**, not recomputed from the cosmology — a
disclosed approximation); spatial flatness exact; quad limits 60 (r_s, r_d, D_C(z*)) and
50 (BAO distances), default scipy tolerances.

**Free parameters and bounds (hard rejection outside).**
Ω_m ∈ (0.2, 0.45); H0 ∈ (60, 76); μ0 ∈ [0, 3); n ∈ (0.5, 12); κ ∈ [0, 40);
w0 ∈ (−2.5, 0); wa ∈ (−4, 3).

**Model integrator.** Eq. (3) by classical RK4 in x = ln a on `linspace(0, −9.2, 1200)`
(descending), ρ(1) = 1, floored at 10⁻¹², held constant for x < −9.2. M_p(a) closed form.
(One original run used N = 800 for the κ profile and CAMB table; the N = 800 ↔ 1200
difference is < 0.05 in χ².)

## 3. Optimizer settings

`scipy.optimize.minimize`, method **Nelder–Mead**, deterministic (no seed anywhere).
Starts, fatol, maxiter per cell are in `SETTINGS` inside `reproduce.py`, verbatim from the
original runs (e.g. full-vector two-channel: starts [0.32, 67.0, 1.1, 2.4, 1.0] and
[0.315, 67.5, 0.6, 3.5, 1.2], fatol 0.02, maxiter 320, best of the two kept). Nelder–Mead
convergence wobble is the dominant reproducibility noise: expect a few ×0.1 in χ².

## 4. Target numbers and pass tolerances

Targets (paper Table 1 and §4–§5): conservative 17.6 / 12.9 / 13.8; full 1409.4 / 1403.4 /
1403.3; best fit μ0 ≈ 0.37, n ≈ 1.8, κ ≈ 0.85; κ profile 16.2 / 13.8 / 17.2 at κ = 0.5/1/3 (bounded above by ΛCDM's 17.6 via μ0 → 0);
mock refit w0 = −0.97, wa = −0.05, implied crossing z ≈ 1.2; shape w_eff(z=1) = −1.14,
crossing z ≈ 0.40, w_eff(0) = −0.86. These are the full-mode numbers the manuscript prints;
§7 explains the environment-sensitivity trap and the κ-profile ceiling structure, and §8 is
the authoritative full-mode reference summary. The **full-mode console log**
(`log-reproduce-dr1-full.txt`) gives the unrounded values actually produced (e.g.
full-vector ΛCDM 1409.38, w0–wa 1403.40 at w0 = −0.8354, wa = −0.639) — reproduce against those.

**Recommended pass tolerances** (your 0.1 instinct is right for the *evaluated* cells but too
tight for independently re-optimised ones): χ² within **±0.3** per cell across environments
(quad tolerances + NM wobble); if you evaluate at the console-log parameter vectors rather
than re-optimising, demand **±0.05**. Parameters: Ω_m ±0.005, H0 ±0.3, κ ±0.05, and
(μ0, n) jointly — note μ0 and n are partially degenerate along fixed late-time feeding, so
compare the *shape outputs* (crossing z ±0.02, w_eff(0) ±0.01, w_eff(1) ±0.02) rather than
μ0, n individually if they disagree at face value. Mock test: w0 ±0.02, wa ±0.05.
Anything outside these bands is a genuine discrepancy worth chasing.

## 5. CAMB stretch goal

Separate job, `camb_checks.py`; do the background pipeline first, as you suggested.
Environment of record **camb==1.6.6**. Configuration: `DarkEnergyPPF` with
`set_w_a_table(a, w)`, table built from the Eq. (3) integrator on the same x-grid, prepended
with (a = 10⁻⁶, w = −1), and **w capped at max(w, −15)** — the cap acts only where
Ω_DE < 10⁻⁹ and was verified to leave all background observables unchanged (the uncapped
early w spikes negative as ρ → 0; it is dynamically irrelevant there but hostile to the
integrator). Cosmology: mnu = 0, omk = 0, As = 2.1×10⁻⁹, ns = 0.965, tau = 0.054,
ombh2 = 0.02236, omch2 = Ω_m·h² − ombh2. Cutoff runs: hard cutoff in a tabulated primordial
spectrum (`set_initial_power_table`, 3000 log points over k = 10⁻⁶·⁵–10 Mpc⁻¹, power floored
at 10⁻³⁰·As below k_cut), with R_s = 1.66×10²⁶ m = 5379 Mpc comoving and the two conventions
k_cut = 1/R_s and π/R_s.

## 6. Known approximations, complete list (the "missing pieces" you asked about)

1. BAO likelihood diagonal (no D_M–D_H correlations).
2. CMB prior diagonal, two observables only, ω_b fixed rather than a third Gaussian row.
3. z* and z_drag fixed, not recomputed per cosmology.
4. (1+zHD) rather than (1+z_hel) in the SN luminosity-distance prefactor.
5. Radiation density fixed via ω_r (neutrinos massless, always relativistic).
6. Background-level only; the perturbation check assumes the minimal interaction completion
   (homogeneous transfer, standard sound speed) via PPF.
7. Nelder–Mead point estimates only — no MCMC, no marginalised uncertainties; quoted
   parameter values are best-fit, not posterior means.
8. No fixed seed because nothing is stochastic.

Items 1–4 are the ones that will bite an independent implementation that "improves" them
before first matching them. All are stated in or consistent with the paper's own framing of
the analysis as a screening test.

---

## 7. Verification notes and traps (read before comparing)

Two things can make an independent re-run disagree with the numbers above even when nothing is
wrong. Both are understood; neither is an error.

**Environment sensitivity (the biggest trap).** The optimiser's minimum *value* is stable to
~±0.1 across integrator settings, but the minimum *location* shifts to compensate — so
evaluating a *recorded* parameter vector under different quad settings can mislead. Concretely,
the original conservative column was computed with quad limit = 150; this package standardises
on the full-vector configuration (limit = 60). At fixed parameters that difference shifts χ² by
O(1): the recorded conservative ΛCDM point evaluates to 18.84 under limit-60, while the
re-optimised minimum lands at 17.62, on target. **Rule:** compare re-optimised minimum
*values*, or evaluate the full-vector cells at the exact vector in §8 — do not evaluate a
rounded recorded vector under a different integrator and expect the recorded χ².

**Nelder–Mead stranding.** With finite iteration caps, Nelder–Mead can terminate before
convergence, especially on the κ-profile cells where the likelihood is shallow. A
partially-converged κ = 3 run can strand well above the true value; the full cold-start run
reaches the converged profile. If a re-run lands high at κ = 3, suspect a stranding and
lengthen the leash before treating it as a discrepancy.

**The κ-profile structure (why it looks the way it does).** The profile has a structural
ceiling: at any κ, μ0 → 0 switches the mechanism off and recovers exact ΛCDM (χ² = 17.6), so
the profile *cannot* exceed 17.6 — verified in the log. The converged profile is a shallow
valley bounded above by that ceiling: ≈17.6 as κ → 0, **16.2** at κ = 0.5, **13.8** at
κ ≈ 1, and **17.2** by κ = 3, where the optimiser retreats to near-zero feeding. Interpretation:
the full improvement over ΛCDM is available only in the neighbourhood of the physical budget;
away from it the fit degrades toward, and is capped by, the ΛCDM value as feeding shuts down.
Validators should reproduce this valley-with-ceiling shape; a value materially above 17.6 at
any κ is a stranding, not a feature.

**Scripting note.** `Hz_B` and `observ` live inside `main()`; for scripted use import
`build_rho`, `chi2`, `chi2_blocks` and the module constants (the log's continuation blocks
show working examples).


---

## 8. Reference run — full mode (`reproduce.py full`)

The authoritative numbers, from a complete cold-start `full` run
(`log-reproduce-dr1-full.txt`), all optimisations from the original starting points:

- **Conservative vector** (ΛCDM / w0–wa / two-channel): 17.62 / 12.85 / **13.80**
- **Full vector**: 1409.38 / 1403.40 / **1403.26**
- **Two-channel full-vector best fit**: (Ω_m, H0, μ0, n, κ) =
  (0.3130, 67.7826, 0.3749, 1.7807, 0.8508) → χ² = 1403.26
  (evaluate here for the ±0.05 strict-comparison tolerance)
- **Shape**: w_eff(z=1) = −1.141, crossing z = 0.400, ρ-peak z = 0.406, w_eff(0) = −0.858
- **κ profile**: 16.2 / 13.8 / 17.2 at κ = 0.5 / 1 / 3
- **Mock test**: w0 = −0.97, wa = −0.05, spurious crossing z = 1.21

The conservative two-channel cell is reported at the **physical efficiency ceiling κ = 1.0**
(χ² = 13.80), fixing κ = 1 and optimising (Ω_m, H0, μ0, n) — identical to the κ = 1 point of the
profile above, and **deterministic in both quick and full modes**. It is quoted instead of the
free-κ conservative optimum because κ is a conversion efficiency (κ ≤ 1) and the free-κ
conservative fit descends *past* the ceiling into unphysical territory — a shallow valley that
keeps dropping (13.89 at κ = 1.006, 13.44 at κ = 1.21, 12.97 at κ = 1.57), so the free optimum is
optimiser-dependent and not reproducible. The manuscript prints 13.8 (Table 1, §5). The
**full-vector** best fit, by contrast, keeps κ free (allowed up to 40) and the data select
κ ≃ 0.85 (0.851 rounded) — comfortably inside the physical budget, and the paper's key result.
**Reproduce against these numbers.**

## 9. Multi-model Bayesian evidence (`bayes_evidence_suite.py`)

`bayes_evidence.py` computes the formal evidence ln(Z) by nested sampling (dynesty) for the
two-channel mechanism vs w0–wa under the physical κ ∈ [0, 1] prior: Δln Z(C − w0wa)
= **+2.69** (`log-evidence-dr1-pair.txt`), cross-validating the six-model suite's +2.50 below.
(The pre-guard-fix first pass gave a biased +0.356; see the honesty flags at the end of this section.)

`bayes_evidence_suite.py` extends this to a **full six-model comparison** on the same full data
vector (Planck distance priors + DESI DR1 BAO + Pantheon+). All six share identical
Ω_m ∈ [0.2, 0.45] and H0 ∈ [60, 76] priors (so only the dark-energy sector and its parameter
volume differ). The two-channel κ prior is set by the top-level constant `KAP_MAX`:

- **`KAP_MAX = 1.0` (default — the headline result).** κ is a conversion *efficiency* and cannot
  exceed 1, so [0, 1] is the physically honest prior, not an "artificially narrow" one. Report this.
- **`KAP_MAX = 40.0` (stress test only).** Widening to [0, 40] — the range the paper's *frequentist*
  point-fits let κ float over, to show the data pull it back to ≈ 0.85 unforced — puts prior mass on
  impossible efficiencies (κ > 1) and charges an Occam penalty of ≈ ln 40 ≈ 3.7 that is a numerical
  **artefact**, not a physical cost. It also makes the two-channel posterior a thin sliver in a
  mostly-empty box, so the sampler explores it very inefficiently (much slower). Use it only to show
  the evidence is prior-sensitive when the physical bound is dropped — never as the headline.

The six expansion histories:

| Model | Free params | DE sector | best-fit χ² (full vector) |
|---|---|---|---|
| ΛCDM | Ω_m, H0 | w = −1 (constant) | 1409.38 |
| wCDM | + w | w constant, prior [−2.5, 0] | 1408.01 (w ≃ −0.97) |
| RVM | + ν | ρ_Λ = c0 + ν H², prior ν ∈ [−0.05, 0.05] | 1407.75 (ν ≃ +0.0016) |
| Quintessence | + λ | thawing, V = V0 e^{−λφ/Mp}, prior λ ∈ [0, 2] | 1406.85 (λ ≃ 0.66; 1406.80 at steps=2000) |
| w0–wa | + w0, wa | CPL, priors w0 ∈ [−2.5, 0], wa ∈ [−4, 3] | 1403.40 |
| Two-channel | + μ0, n, κ | mechanism, κ ∈ [0, 1] (physical; [0, 40] = stress test) | 1403.26 |

The χ² ordering is itself the story: the models that **can** cross into the phantom regime
(w0–wa, two-channel) fit best; quintessence — which **cannot** (w ≥ −1 always) — sits with
ΛCDM/wCDM/RVM, a full Δχ² ≈ 3–6 behind. The two-channel mechanism captures the phantom-crossing
preference through a physical mechanism rather than a fitting function.

**Evidence result (physical κ ∈ [0, 1] prior, nlive = 500).** ln(Z): quintessence −712.43,
two-channel −712.53, ΛCDM −712.55, w0–wa −715.03, wCDM −715.22, RVM −715.24 (all ± ~0.24 to
0.28). There is a **three-way tie at the top** — quintessence, two-channel and ΛCDM sit within
0.12, well inside the errors — and then a ~2.5 gap down to the rest. So the two-channel mechanism
is **evidentially indistinguishable from ΛCDM (Δln Z = +0.02) and quintessence (−0.10)** and
**favoured over w0–wa (+2.50, "positive"), wCDM (+2.69) and RVM (+2.71, "strong")**. Read
carefully: two-channel has the **joint-best** evidence — statistically tied with ΛCDM and
quintessence — and is favoured over the w0–wa/wCDM/RVM alternatives. It does **not** uniquely
lead (quintessence ties it, marginally higher; and only these six models were tested). The tie
with ΛCDM is the correct verdict for a ~2σ signal, where the data cannot yet select evolving DE
over the cosmological constant. Notably it beats w0–wa despite one **more** parameter
and an identical χ²min: its parameters cost ~1 each in Occam vs w0–wa's ~2.7 each (better
constrained relative to their priors; κ physically bounded).

Two honesty flags: (i) this Δln Z(C − w0wa) ≈ +2.5 **supersedes** the first-pass value +0.356 —
the ~+2 rise traces to the guard-bug fix (below) and is now **confirmed**: the guard-fixed
`bayes_evidence.py` (single-threaded, only the guard changed) independently gives +2.69, matching
the suite; the two-channel ln Z rose +2.3 while w0–wa moved <0.05 (guard bug is two-channel-
specific). Combined Δln Z ≈ **+2.6 ± 0.4 — positive, at the edge of strong** (the two runs
straddle the 2.5 Jeffreys boundary, so quote the number, not the single-run "strong" label).
(ii) The margin over w0–wa is **prior-sensitive** (w0–wa's w0/wa priors are wide); the ties with
ΛCDM/quintessence are the more robust statement. The paper's case still rests on the
**mechanism**, the **fit-match with w0–wa**, and the **forced-ordering theorem** — the evidence is
now a supporting result, not the headline.

Output: a **priors summary** (shared Ω_m/H0 box + each model's extra DE parameters, printed up
front so the comparison is manifestly fair); **Table 1** ranks all six by ln(Z) with both the
frequentist **χ²min** (= −2 × peak log-likelihood the sampler visited — a hair ABOVE the optimizer
best-fit χ² in the table above, e.g. two-channel 1403.43 vs 1403.26, since nested sampling doesn't
land exactly on the peak; quote the optimizer values as the best fits) and the Bayesian ln(Z)
side by side — the two axes can disagree, e.g. RVM reaches a lower χ²min than ΛCDM yet loses on
ln(Z) once its wide-ν Occam penalty is charged; **Table 2** reports, per rival, Δln Z =
ln Z(two-channel) − ln Z(rival), the Bayes factor B, and the verdict in model-selection
terms — the data **favour** the mechanism, **disfavour** it, or are **inconclusive
(indistinguishable)** relative to that rival — graded on the Jeffreys band. Run `python3 bayes_evidence_suite.py` (production nlive = 500) or pass a smaller
nlive as `argv[1]` for a faster, noisier smoke run.

**DR2 evidence result (`bayes_evidence_suite.py --dr2`, physical κ ∈ [0, 1], nlive = 500;
`log-evidence-dr2-suite.txt`).** Repeating the six-model comparison on the DESI DR2 BAO vector
sharpens the DR1 picture. ln(Z): two-channel −711.34, quintessence −711.46, ΛCDM −712.37,
w0–wa −713.64, RVM −713.89, wCDM −715.31 (all ± ~0.22 to 0.29). The two-channel mechanism is now
the **evidence-preferred model of all six** — strongly favoured over wCDM (**+3.97**) and RVM
(**+2.55**), positively over w0–wa (**+2.30**), and nominally favoured over ΛCDM (**+1.02**); only
quintessence remains indistinguishable (+0.12); disfavoured by none. Where DR1 left it level with
ΛCDM and a hair behind quintessence, DR2 places it best of the six — consistent with the frequentist
Δχ²(ΛCDM − C) rising from ≈ 6 (DR1) to ≈ 10 (DR2). **Numerical note:** on DR2 the two-channel
posterior is narrow enough to stall dynesty's default proposal, so the DR2 evidences use slice
sampling (`sample='rslice'`) across all six models — an integrator choice that does not bias ln(Z);
the DR1 evidences use the default. This is the paper's Appendix *"Against DESI DR2 data"*.

**RVM background.** Running-vacuum model (Solà, Gómez-Valent et al.), ρ_Λ(H) = (3/8πG)(c0 +
ν H²). Convention: vacuum exchanges energy with pressureless matter (ρ_m ∝ a^{−3(1−ν)}),
radiation separately conserved (ρ_r ∝ a^{−4}). Integrating Friedmann + local conservation gives
the exact algebraic E²(a) = A a^{−3ξ} + Ω_c0/ξ + [Ω_r0/(1+3ν)] a^{−4}, with ξ = 1 − ν,
Ω_c0 = (1 − Ω_m0 − Ω_r0) − ν, A fixed by E²(1) = 1. At ν = 0 it collapses term-by-term to ΛCDM
(verified to ~1e−16).

**Quintessence background.** Thawing scalar field with an exponential potential
V = V0 exp(−λφ/Mp). `solve_quint` integrates the Copeland–Liddle–Wands autonomous system
(variables x = φ̇/√6 H Mp, y = √V/√3 H Mp) in e-folds with a time-dependent background index
γ_b(N) that carries the radiation→matter transition, starting from a frozen field (x_i = 0) and
**shooting on the initial potential amplitude** so the field carries exactly Ω_φ(today) =
1 − Ω_m − Ω_r (flatness closes the model — the initial condition is not a free parameter). Then
E²(a) = (Ω_m0 a^{−3} + Ω_r0 a^{−4})/(1 − Ω_φ(a)). λ = 0 is the frozen field (w = −1) and
reproduces ΛCDM up to a small step-size truncation floor: Δχ² ≈ +0.044 at the default 1000 RK4
steps (+0.010 at 2000, +0.004 at 3000). That floor is a systematic, quintessence-only offset
~25× below the O(1)-χ² approximations the pipeline already discloses (diagonal BAO, compressed
CMB priors), so it changes no ranking or favour/disfavour/inconclusive verdict; raise `steps` in
`solve_quint` if you want it smaller. The physically important property: **quintessence obeys
w ≥ −1 at all times** — it cannot produce the phantom (w < −1) phase the DESI signal prefers, so
it fits a Δχ² ≈ 3 worse than w0–wa / two-channel. That structural limitation, not a numerical
accident, is why it is in the suite.

**Two traps for this script specifically.**

1. *Benign integration warnings are NOT rejected.* The likelihood guard promotes genuine
   `RuntimeWarning`s (overflow/invalid — real numerical pathology) to −inf, but leaves scipy's
   `quad` `roundoff`/`subdivisions` `IntegrationWarning`s alone (`reproduce.py` silences them
   module-wide). On scipy ≥ 1.11 the RK4-interpolant + `quad` trips the roundoff notice **at the
   two-channel model's own best-fit point** — the returned χ² (1403.26) is still accurate to ~7
   digits. An earlier guard promoted that notice to −inf, silently rejecting the model's own
   best region and biasing its evidence downward. A finite-χ² check is the real safeguard.

2. *RVM evidence is prior-sensitive — report ν's prior with it.* The CMB sound horizon is
   pinned to ~0.03%, so the running is tightly constrained (data prefer ν ≃ +0.0016; χ² degrades
   steeply away from it). The evidence charges an Occam penalty ∝ ln(prior width / posterior
   width): a wider ν prior → lower RVM ln(Z), for the *same* best fit. The prior half-width is a
   documented top-level constant `NU_HALFWIDTH` (default 0.05); a tighter, theory-motivated
   running prior raises the RVM's standing. This is genuine prior dependence, not an artefact.

## 10. Benchmark comparison figures — Step C (`benchmark_plots.py`)

Fits every suite model on the same full data vector and writes a 4-panel figure to `results/`
(`benchmark_plots.png` / `.pdf`), plus `best_fits.json` (cached; `--refit` re-optimises) and
`derived_wz.csv`. Panels: **(1)** reconstructed w(z); **(2)** H(z)/H_ΛCDM − 1; **(3)**
D_M(z)/D_M,ΛCDM − 1 with DESI redshifts marked; **(4)** growth f σ8(z).

- **Uniform w(z) reconstruction.** For every model, w_eff(a) = −1 − (1/3) d ln ρ_de/d ln a with
  ρ_de ≡ E²(a) − Ω_m0 a^{−3} − Ω_r0 a^{−4} — i.e. what an observer assuming *standard* matter
  would infer, applied identically to all models so the curves are comparable. Two consequences to
  read correctly: (a) for RVM, where matter itself runs, the small late-time drift below w = −1 is
  the running showing up in a standard-matter reconstruction, not a true phantom vacuum; (b) at
  high z, where the DE density is a small residual of E², the reconstruction is noisy (visible as
  faint wiggle on the quintessence curve) — the panel is capped at z = 2 for this reason. The
  evidence/χ² numbers use the exact solvers, never this reconstruction.
- **Growth panel is level-1.** f σ8 uses standard GR sub-horizon growth with **no** dark-energy
  clustering and a fiducial σ8,0 = 0.81 — indicative only, consistent with the paper's
  background-level scope (a Boltzmann-code f σ8 is the separate `camb_checks.py` direction).

Requires `matplotlib`. Runtime is dominated by the one-off best-fit optimisation of all six
models (~minutes, mostly quintessence); it is cached, so re-plotting is instant.