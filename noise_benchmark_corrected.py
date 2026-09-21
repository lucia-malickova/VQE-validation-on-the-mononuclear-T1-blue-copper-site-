#!/usr/bin/env python3
"""Noise-sensitivity benchmark using the ACTUAL spin-penalised, converged
HEA reps=1 parameters (from hea_spin_constrained.py), instead of
representative/random ones. Requires: pip install qiskit-aer

See paper Section 6 / arXiv:2609.20439v2.
"""
import json
import numpy as np
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.circuit.library import HartreeFock
from qiskit.circuit.library import ExcitationPreserving
from qiskit.quantum_info import Statevector
from qiskit import transpile
from qiskit_aer import AerSimulator
from qiskit_aer.noise import NoiseModel, depolarizing_error

from spin_fci import read_fcidump, sector_hamiltonian, spin_square_expectation, splus_matrix

data = read_fcidump("artifacts/prod/active.FCIDUMP")
nalpha = (data.nelec + data.ms2) // 2
nbeta = data.nelec - nalpha
norb = data.norb
nso = 2 * norb

basis, Hms = sector_hamiltonian(data, nalpha, nbeta)
index = {int(d): i for i, d in enumerate(basis)}
_, _, Splus = splus_matrix(norb, nalpha, nbeta)

with open("hea_reps1_spin_constrained.json") as f:
    d = json.load(f)
theta = np.array(d["theta"])
print(f"loaded converged theta: reps={d['reps']} saved_E={d['energy_Eh']:.6f} S2={d['S2']:.4f}")

mapper = JordanWignerMapper()
hf_circ = HartreeFock(norb, (nalpha, nbeta), mapper)
ansatz = hf_circ.compose(ExcitationPreserving(nso, reps=1, entanglement="linear"))
bound = ansatz.assign_parameters(theta)

def sector_energy(sv_data):
    v = np.zeros(len(basis), dtype=complex)
    for det, i in index.items():
        v[i] = sv_data[det]
    n2 = float(np.vdot(v, v).real)
    v = v / np.sqrt(n2)
    E = float(np.vdot(v, Hms @ v).real)
    s2 = spin_square_expectation(v, Splus, ms=0.5)
    return E, s2, n2

# ideal (noiseless) energy, sanity check against the saved JSON
sv_ideal = Statevector(bound).data
E_ideal, S2_ideal, n2_ideal = sector_energy(sv_ideal)
print(f"E_ideal (recomputed) = {E_ideal:.6f}  S2={S2_ideal:.4f}  sector_norm2={n2_ideal:.8f}")

# noise model matching the paper: p1=1e-3 (1q), p2=1e-2 (2q), depolarizing
noise_model = NoiseModel()
noise_model.add_all_qubit_quantum_error(depolarizing_error(1e-3, 1), ["rz", "sx", "x"])
noise_model.add_all_qubit_quantum_error(depolarizing_error(1e-2, 2), ["cx"])

transpiled = transpile(bound, basis_gates=["rz", "sx", "x", "cx"], optimization_level=1)
gc = transpiled.count_ops()
print(f"transpiled depth={transpiled.depth()} cx={gc.get('cx',0)}")
transpiled.save_statevector()

sim = AerSimulator(method="statevector", noise_model=noise_model)
shots = 40

energies, s2s = [], []
for k in range(shots):
    result = sim.run(transpiled, shots=1).result()
    sv_data = np.asarray(result.data(0)["statevector"])
    E, s2, n2 = sector_energy(sv_data)
    energies.append(E)
    s2s.append(s2)
    if (k + 1) % 10 == 0:
        print(f"  [{k+1}/{shots}] E={E:.6f} S2={s2:.4f}", flush=True)

E_noisy = float(np.mean(energies))
E_noisy_std = float(np.std(energies))
S2_noisy = float(np.mean(s2s))

print(f"\n=== RESULT ===")
print(f"E_ideal  = {E_ideal:.6f} Eh")
print(f"E_noisy  = {E_noisy:.6f} Eh  (std over {shots} trajectories: {E_noisy_std:.6f})")
print(f"Delta E_noise = {(E_noisy - E_ideal)*1e3:.2f} mEh")
print(f"<S2> ideal={S2_ideal:.4f}  <S2> noisy(avg)={S2_noisy:.4f}")

with open("noise_benchmark_corrected.json", "w") as f:
    json.dump({"E_ideal": E_ideal, "E_noisy": E_noisy, "E_noisy_std": E_noisy_std,
               "delta_E_noise_mEh": (E_noisy-E_ideal)*1e3, "S2_ideal": S2_ideal,
               "S2_noisy_avg": S2_noisy, "shots": shots,
               "transpiled_depth": transpiled.depth(), "cx_count": gc.get("cx",0)}, f, indent=2)
print("saved noise_benchmark_corrected.json")
