#!/usr/bin/env python3
"""Re-optimise HEA (ExcitationPreserving) with an explicit S^2 penalty,
evaluated by projecting the statevector onto the physical (N=15) sector,
reusing the sector/S^2 machinery from spin_fci.py.

See paper Section 2.3 / arXiv:2609.20439v2. Produces the spin-penalised
HEA reps=1/reps=2 rows of Table 1.
"""
import argparse, json, time
import numpy as np
from scipy.optimize import minimize
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit.circuit.library import ExcitationPreserving
from qiskit.quantum_info import Statevector

from spin_fci import (
    read_fcidump, sector_hamiltonian, splus_matrix, spin_square_expectation,
)

ap = argparse.ArgumentParser()
ap.add_argument("fcidump")
ap.add_argument("--reps", type=int, required=True)
ap.add_argument("--mu", type=float, default=8.0,
                 help="S^2-penalty weight; --mu 0.0 reproduces the unconstrained (contaminated) result")
ap.add_argument("--maxiter", type=int, default=150)
args = ap.parse_args()
outname = f"hea_reps{args.reps}_spin_constrained.json"

data = read_fcidump(args.fcidump)
nalpha = (data.nelec + data.ms2) // 2
nbeta = data.nelec - nalpha
norb = data.norb
nso = 2 * norb

basis, Hms = sector_hamiltonian(data, nalpha, nbeta)
index = {int(d): i for i, d in enumerate(basis)}
_, _, Splus = splus_matrix(norb, nalpha, nbeta)

mapper = JordanWignerMapper()
hf_circ = HartreeFock(norb, (nalpha, nbeta), mapper)
ansatz = hf_circ.compose(ExcitationPreserving(nso, reps=args.reps, entanglement="linear"))
nparams = ansatz.num_parameters
print(f"nso={nso} sector_dim={len(basis)} nparams={nparams}", flush=True)
print("CHECK: reps=1 should give nparams=53, reps=2 should give nparams=88 "
      "(matching Table 2 in the paper) -- if not, the ansatz differs from the published one.")

def sector_vector(sv_data):
    v = np.zeros(len(basis), dtype=complex)
    for det, i in index.items():
        v[i] = sv_data[det]
    n2 = float(np.vdot(v, v).real)
    return v, n2

MU_LEAK = 20.0

def evaluate(theta):
    sv = Statevector(ansatz.assign_parameters(theta)).data
    v, n2 = sector_vector(sv)
    if n2 < 1e-8:
        return 1e3, np.nan, np.nan, n2
    v_norm = v / np.sqrt(n2)
    E = float(np.vdot(v_norm, Hms @ v_norm).real)
    s2 = spin_square_expectation(v_norm, Splus, ms=0.5)
    spin_penalty = args.mu * (s2 - 0.75) ** 2
    leak_penalty = MU_LEAK * (1.0 - n2) ** 2
    F = E + spin_penalty + leak_penalty
    return F, E, s2, n2

def objective(theta):
    return evaluate(theta)[0]

t0 = time.time()
rng = np.random.default_rng(42)
x0 = rng.normal(scale=0.1, size=nparams)

history = []
def callback(xk):
    F, E, s2, n2 = evaluate(xk)
    dev = (E - (-2518.989067)) * 1e3
    history.append({"iter": len(history) + 1, "F": F, "E": E, "dev_mEh": dev, "S2": s2, "sector_norm2": n2})
    if len(history) % 5 == 0:
        print(f"  [{len(history)} iters] E={E:.6f} dev={dev:.3f} mEh <S2>={s2:.4f}", flush=True)

res = minimize(objective, x0, method="Powell", callback=callback,
                options={"maxiter": args.maxiter * 20, "xtol": 1e-5, "ftol": 1e-8})
theta = res.x
print(f"optimizer: success={res.success} nit={res.nit} nfev={res.nfev} message={res.message}")

sv = Statevector(ansatz.assign_parameters(theta)).data
v, n2 = sector_vector(sv)
v = v / np.sqrt(n2)
E = float(np.vdot(v, Hms @ v).real)
s2 = spin_square_expectation(v, Splus, ms=0.5)
dev = (E - (-2518.989067)) * 1e3

print(f"\n=== RESULT reps={args.reps} ===")
print(f"E={E:.6f} Eh   dev={dev:.4f} mEh   sector_norm2={n2:.10f}   <S2>={s2:.6f}")
print(f"elapsed={time.time()-t0:.0f}s")

with open(outname, "w") as f:
    json.dump({"reps": args.reps, "nparams": nparams, "energy_Eh": E,
               "dev_mEh": dev, "sector_norm2": n2, "S2": s2,
               "theta": theta.tolist(), "mu": args.mu, "history": history}, f, indent=2)
print("saved", outname)
