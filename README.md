# A Two-Channel Accretion Mechanism for Evolving Dark Energy

Christopher P. Green — Independent Researcher — research@wilson-green.com

A physical mechanism for evolving dark energy, as preferred by recent combined DESI analyses:
two competing channels driven by a single accretion flux onto a parent black hole whose
interior is the observable universe. The mechanism matches the empirical w0–wa description
of Planck + DESI BAO + Pantheon+ data to within rounding (chi2 1403.3 vs 1403.4; LCDM 1409.4),
with a provably forced rise–peak–decline ordering and a fitted conversion efficiency inside
the physical energy budget (kappa ~ 0.85).

## Contents

- `Green_two_channel_dark_energy.pdf` — the paper (13 pp., incl. Appendices A–C:
  Vaidya injection channel, Misner–Sharp dilution channel, scope of the junction
  analysis and the rapid-conversion limit).
- `latex-source/` — `main.tex`, `references.bib`, `figures/weff_history.pdf`.
  Compile: `pdflatex → bibtex → pdflatex → pdflatex` (or `latexmk -pdf`).
- `validation/` — complete reproduction package:
  - `reproduce.py` — self-contained pipeline (data download, all likelihoods, all fits).
    `python3 reproduce.py quick` (~2 min) or `full` (~15–30 min).
  - `camb_checks.py` — perturbation-level checks via CAMB 1.6.6 (PPF, tabulated w(a)).
  - `VALIDATION-README.md` — exact data provenance, every approximation disclosed,
    optimizer settings, target numbers, pass tolerances.
  - `console-log-verification.txt` — genuine run in the environment of record.
  - `requirements.txt`

## Reproducing the headline numbers

    pip install -r validation/requirements.txt
    python3 validation/reproduce.py quick

Targets: chi2 = 17.6 / 12.9 / 13.7 (Planck priors + DESI BAO) and
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
