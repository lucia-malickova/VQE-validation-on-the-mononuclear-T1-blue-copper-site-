# T1 blue-copper VQE benchmark

Modular pipeline for the EuroHPC emulator benchmark. Built so the expensive
PySCF step runs **once** and persists to disk; the quantum part reloads the
integrals, so you iterate on the VQE cheaply without ever re-running the build.

## Files
- `config.py` — one switch, `MODE = "test"` / `"prod"`. Controls scale, basis, ansatz.
- `geometry.py` — the T1 blue-copper model geometry (swap in real coords later).
- `build_hamiltonian.py` — **step 1**: ROHF → active space → CASSCF → integrals (`artifacts/integrals.npz`, `active.FCIDUMP`).
- `run_vqe.py` — **step 2**: reload integrals → ideal + noisy VQE → `artifacts/vqe_history.npz`.
- `plot_report.py` — **step 3**: convergence plot + `benchmark_report.txt`.
- `job.sh` — SLURM wrapper for the emulator.

## Run order (locally, Mac)
```bash
pip install -r requirements.txt
python build_hamiltonian.py
python run_vqe.py
python plot_report.py
```

## Test first, then scale
1. Leave `MODE = "test"` → small manual CAS(3,3), ~6 qubits, seconds. This only
   proves the chain is wired (incl. the open-shell doublet path). **Don't read
   scientific meaning into the test run** — at this size the ideal VQE trivially
   matches CASSCF.
2. Once it runs clean end-to-end, set `MODE = "prod"` → AVAS(Cu 3d), def2-SVP,
   ~10 qubits, hardware-efficient ansatz. This is the real benchmark: the ideal
   run is a genuine test, and the noisy run gives the honest feasibility signal
   (expect it to land **above** chemical accuracy — that's a legitimate result).

## Notes / things to confirm before the report
- **AVAS API**: `pyscf.mcscf.avas.avas` returns `(ncas, nelecas, mo)`. If your
  PySCF version differs, adjust `select_active_space` in `build_hamiltonian.py`.
- **Native gate set**: `BASIS_GATES` is a CZ-based stand-in for Euro-Q-Exa (IQM).
  Confirm the exact native set/connectivity with the hosting entity for deliverable 4.
- **Ansatz honesty**: `prod` uses a hardware-efficient ansatz because UCCSD at
  ~10+ qubits is thousands of CNOTs — infeasible on near-term hardware. Report the
  real depth/2q-gate counts; that gap *is* the resource-realism finding.
- To grow the active space toward the upper end (16–20 qubits), add `"S 3p"` to
  `avas_labels` in the `prod` profile.
