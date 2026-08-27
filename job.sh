#!/bin/bash
#SBATCH --job-name=t1_vqe
#SBATCH --output=t1_vqe_%j.out
#SBATCH --error=t1_vqe_%j.err
#SBATCH --nodes=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00

set -euo pipefail   # abort the chain if build or VQE fails (no stale re-plot)

# Emulator run. Set MODE = "prod" in config.py before submitting.
module load python 2>/dev/null || true
# source your venv here if needed:
# source "$HOME/venvs/vqe/bin/activate"

python build_hamiltonian.py   # step 1: integrals (run once)
python run_vqe.py             # step 2: ideal + noisy VQE
python plot_report.py         # step 3: plot + report