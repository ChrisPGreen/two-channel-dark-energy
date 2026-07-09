# A Two-Channel Accretion Mechanism for Evolving Dark Energy

Christopher P. Green — Independent Researcher — research@wilson-green.com

A physical mechanism for evolving dark energy, as preferred by recent combined DESI analyses:
two competing channels driven by a single accretion flux onto a parent black hole whose
interior is the observable universe. The mechanism matches the empirical w0–wa description
of Planck + DESI DR1 BAO + Pantheon+ data to within rounding (chi2 1403.3 vs 1403.4; LCDM 1409.4),
with a provably forced rise–peak–decline ordering and a fitted conversion efficiency inside
the physical energy budget (kappa ~ 0.85). A six-model nested-sampling comparison on the same
vector finds the mechanism statistically tied with LCDM and thawing quintessence and favoured
over the w0–wa, wCDM, and running-vacuum alternatives (Delta lnZ ~ +2.5 to +2.7; paper Appendix D,
validation/VALIDATION-README.md Sec. 9). The main analysis uses DESI DR1 BAO; __a full refit against
DESI DR2 (paper Appendix E) strengthens the preference (Delta chi2 ~ 6 -> 10 below LCDM) and makes
the mechanism the evidence-preferred model of the six__. The 2.8–4.2 sigma figure quoted in the paper
is DESI's own DR2 field-level significance.

## Contents

- `Green_two_channel_dark_energy.pdf` — the paper (incl. Appendices A–E:
  Vaidya injection channel, Misner–Sharp dilution channel, scope of the junction
  analysis and the rapid-conversion limit, Bayesian model selection on DESI DR1,
  and confrontation with DESI DR2).
- `latex-source/` — `main.tex`, `references.bib`, `figures/weff_history.pdf`.
  Compile: `pdflatex → bibtex → pdflatex → pdflatex` (or `latexmk -pdf`).
- `validation/` — complete reproduction package:
  - `reproduce.py` — self-contained background pipeline (data download, all likelihoods, all
    fits). `python3 reproduce.py quick` (~2 min) or `full` (~15–30 min); add `--dr2` for the
    DESI DR2 cross-check.
  - `bayes_evidence.py`, `bayes_evidence_suite.py` — Bayesian evidence (ln Z) by nested
    sampling; `benchmark_plots.py` — the model-comparison figure; `camb_checks.py` —
    perturbation-level checks via CAMB 1.6.6. Each script auto-writes its own `log-*.txt`.
  - `VALIDATION-README.md` — §0 file manifest (every script, the `--dr2` flag, every log) plus
    exact data provenance, every approximation disclosed, target numbers, and pass tolerances.
  - `requirements.txt`

## Reproducing the headline numbers

    pip install -r validation/requirements.txt
    python3 validation/reproduce.py quick

Targets: chi2 = 17.6 / 12.9 / 13.8 (Planck priors + DESI BAO) and
1409.4 / 1403.4 / 1403.3 (adding 1580 Pantheon+ SNe); crossing z ≈ 0.40;
kappa ≈ 0.85. See VALIDATION-README.md for tolerances before flagging discrepancies.

## Provenance and AI disclosure

The hypothesis, conceptual development, research direction, and all analytical and interpretive
decisions are the author's own. Claude (Anthropic) served as an instrument of that direction —
formalising the mathematics, running the computations, and drafting the text — and equally as
an adversary, pressuring the author's proposals on physical and statistical grounds so that ideas
were retained only after surviving challenge. The general-relativistic scoping of Appendices A–C,
and an independent reproduction of the numerical pipeline as a validation check, were developed
in the same directed, adversarial dialogue with ChatGPT (OpenAI). All scientific interpretations
and conclusions remain the author's sole responsibility.
