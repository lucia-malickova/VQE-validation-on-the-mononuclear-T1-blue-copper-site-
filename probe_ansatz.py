# probe_ansatz.py — je H spravny? a je ansatz(0) naozaj HF?
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import EfficientSU2
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.hamiltonians import ElectronicEnergy
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.circuit.library import HartreeFock
import config

d = np.load(config.INTEGRALS_NPZ)
ncas, e_core = int(d["ncas"]), float(d["e_core"])
n_particles = (int(d["n_alpha"]), int(d["n_beta"]))
mapper = JordanWignerMapper()

qop_fc = mapper.map(fcidump_to_problem(FCIDump.from_file(config.FCIDUMP)).hamiltonian.second_q_op())
qop_rw = mapper.map(ElectronicEnergy.from_raw_integrals(d["h1"], d["h2"]).second_q_op())
Hf = qop_fc.to_matrix(sparse=True).tocsr()
Hr = qop_rw.to_matrix(sparse=True).tocsr()

def E(H, psi): return float((psi.conj() @ (H @ psi)).real) + e_core

hf = HartreeFock(ncas, n_particles, mapper)
psi_hf = Statevector(hf).data

hea = EfficientSU2(2*ncas, su2_gates=["ry"], entanglement="linear", reps=1)
raw = QuantumCircuit(2*ncas); raw.compose(hf, inplace=True); raw.compose(hea, inplace=True)
ans = transpile(raw, basis_gates=config.BASIS_GATES, optimization_level=1)
psi0 = Statevector(ans.assign_parameters(np.zeros(ans.num_parameters))).data

print(f"FCIDUMP  H, HF only     = {E(Hf, psi_hf):.6f} Eh")
print(f"from_raw H, HF only     = {E(Hr, psi_hf):.6f} Eh")
print(f"FCIDUMP  H, HF+HEA(0)   = {E(Hf, psi0):.6f} Eh")
print(f"fidelity |<HF|ans(0)>|^2= {abs(np.vdot(psi_hf, psi0))**2:.6f}   (ma byt 1.0)")