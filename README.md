T1 Blue-Copper VQE Benchmark & Polymer Interaction
Modular pipeline for the EuroHPC emulator benchmark. Built so the expensive PySCF step runs once and persists to disk; the quantum part reloads the integrals, so you iterate on the VQE cheaply without ever re-running the build.
Project Structure & Files
config.py — Single configuration switch (MODE = "test" / "prod"). Controls scale, basis sets, and ansatz parameters.
geometry.py — The T1 blue-copper model geometry definitions.
build_hamiltonian.py — Step 1: ROHF → active space → CASSCF → integrals export (artifacts/integrals.npz, active.FCIDUMP).
run_polymer_interaction.py — Model for active site and polymer substrate coupling (ROHF integration & FCIDUMP export).
run_pes_scan.py — Potential Energy Surface (PES) scan generator for the polymer approaching the active site (pes_smooth_step_*.FCIDUMP).
run_adapt_pes_analysis.py — Diagnostic script to inspect PES scan FCIDUMP files, verifying spatial dimensions, orbital counts, and Jordan-Wigner mapped qubit operators.
run_vqe.py — Step 2: Reload integrals → ideal + noisy VQE execution → artifacts/vqe_history.npz.
plot_report.py — Step 3: Convergence plots and final benchmark_report.txt generation.
job.sh — SLURM wrapper script for executing high-performance jobs on HPC infrastructure.
Requirements & Installation
Install the required Python dependencies in your virtual environment:
Bash
pip install -r requirements.txt
Execution Order
Run the pipeline components in the following logical sequence:
Build the Hamiltonian:
Bash
python build_hamiltonian.py
Compute Polymer Interaction:
Bash
python run_polymer_interaction.py
Generate Potential Energy Surface (PES) Scan:
Bash
python run_pes_scan.py
Analyze PES Scan Integrals & Qubit Mappings:
Bash
python run_adapt_pes_analysis.py
Execute VQE Simulation:
Bash
python run_vqe.py
Generate Report & Visualizations:
Bash
python plot_report.py
