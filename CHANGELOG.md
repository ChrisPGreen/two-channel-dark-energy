# Changelog

All notable changes to the paper and its validation package. The last public
release is **v1.0.2** (https://github.com/ChrisPGreen/two-channel-dark-energy/tree/v1.0.2).

## v1.1 — July 2026

v1.1 adds Bayesian model selection and a DESI DR2 robustness cross-check. **The core
mechanism, the forced-ordering theorem, and the headline full-vector fit (χ² = 1403.3,
matching the empirical w0–wa form, Δχ² ≈ 6 below ΛCDM) are unchanged from v1.0.2.**

### Added
- **Bayesian model selection.** A six-model nested-sampling evidence comparison
  (`bayes_evidence_suite.py`): the two-channel mechanism against ΛCDM, constant-*w* (wCDM),
  running-vacuum (RVM), thawing quintessence, and the empirical w0–wa (CPL) form, on one common
  data vector under the physical efficiency prior κ ∈ [0, 1]. The mechanism is **favoured over
  the w0–wa, wCDM, and running-vacuum alternatives** (Δln Z ≈ +2.5 to +2.7) and **evidentially
  indistinguishable from ΛCDM and thawing quintessence**; disfavoured by none. Cross-validated by
  an independent single-comparison run (`bayes_evidence.py`). Presented in the paper's **Appendix D**
  (DESI DR1) and co-headlined in the abstract and introduction; the body argument (mechanism,
  frequentist fit, forced-ordering theorem) is unchanged from v1.0.2.
- **Model-comparison figure** (`benchmark_plots.py`): reconstructed w(z), H(z)/D_M(z) residuals
  vs ΛCDM, and fσ8(z) growth for all six models. Added to the paper.
- **DESI DR2 robustness cross-check** (`--dr2` flag). The whole pipeline can be run against the
  DESI DR2 BAO (arXiv:2503.14738); the evolving-dark-energy preference strengthens (Δχ² ≈ 6 → 10
  below ΛCDM, and the two-channel becomes the evidence-preferred model), reported in a dedicated
  appendix. Planck priors and the Pantheon+ supernovae are identical to the DR1 analysis; only
  the BAO block changes.
- Literature-novelty analysis (paper), an explicit DR1/DR2 data-provenance statement, a version
  note, and the concept DOI on the title page.

### Changed
- Abstract compressed, with the evidence result foregrounded.
- The **conservative-vector** two-channel χ² is now quoted at the physical efficiency ceiling
  κ = 1 (**13.8**, was 13.9) — a reproducible, physically-meaningful value. The headline
  full-vector fit, model rankings, and all conclusions are unchanged.
- Validation package reorganised: DR1 and DR2 are unified behind a single `--dr2` flag (five
  scripts: `reproduce.py`, `bayes_evidence.py`, `bayes_evidence_suite.py`, `benchmark_plots.py`,
  `camb_checks.py`); every script now auto-writes its own log (`log-<tool>-dr{1,2}-<variant>.txt`);
  and `VALIDATION-README.md` gains a §0 file manifest with the exact command for each script.
- Anisotropy/novelty wording tightened: the preferred cosmic axis is presented as *permitted* by
  the geometry (given rotating/uneven accretion), not forced.

### Fixed
- **Evidence-integration guard bug.** Both evidence scripts had promoted scipy `quad`'s benign
  roundoff/subdivision warnings to −∞; on scipy ≥ 1.11 this fired at the two-channel model's own
  best-fit region and silently biased its evidence **low**. Corrected (reject only genuine
  runtime warnings + a finite-χ² check). The two-channel-vs-w0–wa evidence moves from an
  artifactual Δln Z ≈ +0.36 to a robust **≈ +2.6** (positive, edge of strong), confirmed by two
  independent code paths.
- Paper: Table 1 and Figure 1 float placement (no longer split across paragraphs).

### Unchanged
- The two-channel mechanism, the forced-ordering theorem, the headline full-vector χ² fit, and
  all physical conclusions.

## v1.0.2 — 2026-07 (last public release)
- Baseline: the two-channel mechanism, forced-ordering theorem, DR1 fit, GR appendices, and the
  reproduction package as archived on Zenodo / GitHub `tree/v1.0.2`.

## v1.0.1 — 2026-07-06
- VALIDATION-README.md corrections.

## v1.0 — 2026-07-05
- Initial Release
