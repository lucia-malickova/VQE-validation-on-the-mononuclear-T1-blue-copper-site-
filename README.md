# VQE Validation on the Mononuclear T1 (Blue-Copper) Site

Code and data accompanying **"VQE Validation on the Mononuclear T1 (Blue-Copper)
Site: A Stepping Stone to the Multi-Copper Laccase Cluster"**
([arXiv:2609.20439](https://arxiv.org/abs/2609.20439), v2).

## Important note on v1 vs v2

Version 1 of the paper reported a headline ADAPT-VQE result (4.03 mEh) that was
computed against an **unphysical spin-quartet (S=3/2) reference state**, not
the physical spin-doublet (S=1/2) ground state that this Cu(II) system
actually has. The bug: restricting the calculation to the correct electron
count and correct net spin projection (M_S) does **not** by itself guarantee
the correct total spin (S) — the sector contains both the physical doublet
and an unphysical, lower-energy quartet, and unconstrained optimisation
converges to the latter. This affected both the reference energy and the
VQE results (ADAPT-VQE and the hardware-efficient ansatz alike).

**Version 2 fixes this** by explicitly projecting onto the exact S=1/2
subspace (`spin_fci.py`) and re-verifying every number in the paper against
it. See Sections 2.2 and 2.4 of the paper for the full explanation. The
scripts in this repository are the v2, spin-corrected versions — running them
is how you can independently verify every number in Tables 1 and 2 of the
paper yourself.

## Reproducing the paper's results

All scripts read the same active-space Hamiltonian, `artifacts/prod/active.FCIDUMP`
(CAS(15e,9o), built by `build_hamiltonian.py` from the geometry in `geometry.py`).

| Paper result | Script | Command |
|---|---|---|
| Table 1: doublet/quartet reference energies (Eq. 2–3) | `spin_fci.py` (library) | used internally by the scripts below |
| Table 1 + Fig. 3: ADAPT-VQE, exact spin projection (1.57 mEh, 123 ops, ⟨S²⟩=0.75) | `run_adapt_doublet.py` | `python run_adapt_doublet.py artifacts/prod/active.FCIDUMP` |
| Table 2: ADAPT-VQE circuit depth/CNOT (20,671 / 15,879) | `circuit_metrics_doublet.py` | run after `run_adapt_doublet.py` (reads `adapt_doublet_spin_pure.json`) |
| Table 1 + Fig. 1/2: HEA reps=1/2, spin-penalised (430.6 / 370.6 mEh) | `hea_spin_constrained.py` | `python hea_spin_constrained.py artifacts/prod/active.FCIDUMP --reps 1` (and `--reps 2`) |
| Section 3.3 claim: unconstrained HEA drifts to quartet (⟨S²⟩≈1.75–3.75) | `hea_spin_constrained.py --mu 0.0` | `python hea_spin_constrained.py artifacts/prod/active.FCIDUMP --reps 2 --mu 0.0` (reproduces the pre-correction, contaminated ~167 mEh number, now understood to be ~33% quartet-contaminated) |
| Section 6: noise-sensitivity benchmark (350.5 mEh, at converged parameters) | `noise_benchmark_corrected.py` | run after `hea_spin_constrained.py --reps 1` (reads `hea_reps1_spin_constrained.json`) |
| Fig. 4: classical-scaling wall | `classical_wall.py` | `python classical_wall.py` |
| Section 3.1: methionine-S exclusion check (<0.1 mEh CASSCF shift) | `which_sulfur.py` | `python which_sulfur.py` |

Every one of the scripts above performs its own internal consistency checks
(finite-value assertions, spin-purity assertions, parameter-count checks
against the published circuit sizes) and will raise an error rather than
silently produce a bad number.

## Core files

- **`spin_fci.py`** — Shared library: FCIDUMP parsing, determinant-sector
  Hamiltonian construction, and the exact S=1/2 projection (`ker(Ŝ₊)`) that
  the whole correction is built on. No Qiskit dependency.
- **`build_hamiltonian.py`**, **`geometry.py`**, **`config.py`** — Step 1:
  ROHF → AVAS active space → CASSCF → integrals export
  (`artifacts/prod/integrals.npz`, `artifacts/prod/active.FCIDUMP`).
- **`run_adapt_doublet.py`** — Spin-pure ADAPT-VQE (exact projection method,
  Section 2.4). This is the corrected replacement for the script that
  produced the flawed v1 result.
- **`hea_spin_constrained.py`** — Hardware-efficient ansatz (HEA), re-optimised
  with an explicit S²-penalty (soft constraint, since HEA is a real circuit
  and can't be trivially restricted to an abstract subspace the way ADAPT-VQE
  can). Pass `--mu 0.0` to reproduce the original, unconstrained/contaminated
  behaviour for comparison.
- **`circuit_metrics_doublet.py`** — Rebuilds the ADAPT-VQE circuit from a
  saved checkpoint and reports transpiled depth/CNOT count (Table 2).
- **`noise_benchmark_corrected.py`** — Depolarising-noise simulation of the
  HEA reps=1 circuit *at its actual converged parameters* (not
  representative/random ones — see Section 6 of the paper for why that
  distinction matters). Requires `qiskit-aer`.
- **`classical_wall.py`** — Standalone figure generator for the classical-FCI
  memory-scaling wall (Fig. 4); not affected by the spin bug.
- **`which_sulfur.py`** — Checks that excluding the axial methionine sulphur
  from the active space is justified (<0.1 mEh CASSCF energy shift).

## `archive/`

Superseded scripts and data from earlier development: one-off diagnostic
"probe" scripts, redundant/dead-end ADAPT-VQE and UCCSD attempts, old plots
with hardcoded pre-correction numbers, and Hamiltonians for active-space
variants that weren't used in the final paper. Kept for transparency (and
because git history isn't always convenient to browse), not because they
should be run.

## `future_work/`

Scripts and data for the polymer/plastic-substrate interaction and PES-scan
workstream. This is a **separate, not-yet-validated** extension of the T1
benchmark (see Section 7 of the paper, "From Validation to the Target
Problem") — do not treat results from these scripts as validated the way the
top-level scripts are.

## Installation

```bash
pip install -r requirements.txt
```

## Data and code integrity

If you find a discrepancy between a number in the paper and what these
scripts produce, please open an issue — that is exactly the kind of check
this repository exists to make possible.
