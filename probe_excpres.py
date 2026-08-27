# probe_excpres.py — dava novy ansatz spravny HF start?
import numpy as np
from qiskit import QuantumCircuit, transpile
from qiskit.circuit.library import ExcitationPreserving
from qiskit.quantum_info import Statevector
from qiskit_nature.second_q.mappers import JordanWignerMapper
from qiskit_nature.second_q.formats.fcidump import FCIDump
from qiskit_nature.second_q.formats.fcidump_translator import fcidump_to_problem
from qiskit_nature.second_q.circuit.library import HartreeFock
import config

d = np.load(config.INTEGRALS_NPZ)
ncas, e_core = int(d["ncas"]), float(d["e_core"])
n_particles = (int(d["n_alpha"]), int(d["n_beta"]))
mapper = JordanWignerMapper()

qop = mapper.map(fcidump_to_problem(FCIDump.from_file(config.FCIDUMP)).hamiltonian.second_q_op())
H = qop.to_matrix(sparse=True).tocsr()

hf = HartreeFock(ncas, n_particles, mapper)
psi_hf = Statevector(hf).data

anz = ExcitationPreserving(2*ncas, entanglement="linear", reps=1)
raw = QuantumCircuit(2*ncas); raw.compose(hf, inplace=True); raw.compose(anz, inplace=True)
ansatz = transpile(raw, basis_gates=config.BASIS_GATES, optimization_level=1)
psi0 = Statevector(ansatz.assign_parameters(np.zeros(ansatz.num_parameters))).data

E = lambda psi: float((psi.conj() @ (H @ psi)).real) + e_core
print(f"params            = {ansatz.num_parameters}")
print(f"E(HF)             = {E(psi_hf):.6f} Eh")
print(f"E(HF+ansatz(0))   = {E(psi0):.6f} Eh   (ma sa rovnat E(HF))")
print(f"fidelity          = {abs(np.vdot(psi_hf, psi0))**2:.6f}   (ma byt 1.0)")