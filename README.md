# T1 Blue-Copper VQE Benchmark & Polymer Interaction

Modular pipeline for the EuroHpc emulator benchmark. Built so the expensive PySCF step runs once and persists to disk; the quantum part reloads the integrals, so you iterate on the VQE cheaply without ever re-running the build.

## Files

- **`config.py`** — one switch, `MODE = "test"` / `"prod"`. Controls scale, basis, ansatz.
- **`geometry.py`** — the T1 blue-copper model geometry (swap in real coords later).
- **`build_hamiltonian.py`** — step 1: ROHF → active space → CASSCF → integrals (`artifacts/integrals.npz`, `active.FCIDUMP`).
- **`run_polymer_interaction.py`** — model for active site and polymer substrate coupling (ROHF integration & FCIDUMP export).
- **`run_pes_scan.py`** — potential energy surface scan generator for polymer approaching the active site (`pes_smooth_step_*.FCIDUMP`).
- **`run_vqe.py`** — step 2: reload integrals → ideal + noisy VQE → `artifacts/vqe_history.npz`.
- **`plot_report.py`** — step 3: convergence plot + `benchmark_report.txt`.
- **`job.sh`** — SLURM wrapper for the emulator.

## Run order (locally, Mac)

```bash
pip install -r requirements.txt
python build_hamiltonian.py
python run_polymer_interaction.py
python run_pes_scan.py
python run_vqe.py
python plot_report.py
