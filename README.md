# T1 Blue-Copper VQE Benchmark & Polymer Interaction

Modular pipeline for the EuroHPC emulator benchmark. Built so the expensive PySCF step runs once and persists to disk; the quantum part reloads the integrals, so you iterate on the VQE cheaply without ever re-running the build.

## Project Structure & Files

- **`config.py`** — Single configuration switch (`MODE = "test"` / `"prod"`). Controls scale, basis sets, and ansatz parameters.
- **`geometry.py`** — The T1 blue-copper model geometry definitions.
- **`build_hamiltonian.py`** — Step 1: ROHF $\rightarrow$ active space $\rightarrow$ CASSCF $\rightarrow$ integrals export (`artifacts/integrals.npz`, `active.FCIDUMP`).
- **`run_polymer_interaction.py`** — Model for active site and polymer substrate coupling (ROHF integration & FCIDUMP export).
- **`run_pes_scan.py`** — Potential Energy Surface (PES) scan generator for the polymer approaching the active site (`pes_smooth_step_*.FCIDUMP`).
- **`run_adapt_pes_analysis.py`** — Diagnostic script to inspect PES scan FCIDUMP files, verifying spatial dimensions, orbital counts, and Jordan-Wigner mapped qubit operators.
- **`run_vqe.py`** — Step 2: Reload integrals $\rightarrow$ ideal + noisy VQE execution $\rightarrow$ `artifacts/vqe_history.npz`.
- **`plot_report.py`** — Step 3: Convergence plots and final `benchmark_report.txt` generation.
- **`job.sh`** — SLURM wrapper script for executing high-performance jobs on HPC infrastructure.

## Requirements & Installation & Execution Order

Install requirements and run the pipeline components in the following logical sequence:

```bash
pip install -r requirements.txt

python build_hamiltonian.py
python run_polymer_interaction.py
python run_pes_scan.py
python run_adapt_pes_analysis.py
python run_vqe.py
python plot_report.py
